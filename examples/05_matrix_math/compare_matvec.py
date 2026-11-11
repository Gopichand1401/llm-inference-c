"""
compare_matvec.py - check gguf_matvec() results against a NumPy reference.

Run build/gguf_matvec_demo first.

Usage:
    python examples/05_matrix_math/compare_matvec.py model.gguf

For every product saved by the C demo, this script dequantizes the whole
weight with gguf-py and computes y = W x in float64.

Two checks per output element:

1. Bit-exact (the pass/fail check). The C code multiplies in float32 and
   adds the products one by one, column 0 first, with no fused multiply-add
   (GCC's default for -std=c11). NumPy reproduces exactly that with
   np.add.accumulate on float32 products, so every output must match to the
   last bit. Any wrong weight, row, block or input element breaks this.

2. Accuracy (informational). Compared with an exact float64 product, the
   normalised rounding error

       err[r] = |y_c[r] - y_64[r]| / sum_c |W[r][c] * x[c]|

   shows how much float32 accumulation costs; expect about 1e-7 to 1e-6.

The bit-exact check applies to gguf_matvec_reference() (build/matvec_c.bin).
The fast kernel gguf_matvec() (build/matvec_fast.bin) sums in a different
order, so it is checked with the accuracy measure only (limit 1e-5).
"""
import struct
import sys

import numpy as np
from gguf.gguf_reader import GGUFReader
from gguf.quants import dequantize

# Sanity bound for the float64 comparison; real values are ~1e-7.
ACCURACY_LIMIT = 1e-4

if len(sys.argv) != 2:
    print("Usage: python compare_matvec.py model.gguf")
    sys.exit(1)

reader = GGUFReader(sys.argv[1])
tensors = {t.name: t for t in reader.tensors}

x_dim = np.fromfile("build/matvec_x_dim.bin", dtype=np.float32)
x_hidden = np.fromfile("build/matvec_x_hidden.bin", dtype=np.float32)


def weight_chunks(t, chunk=1024):
    """Yield (start, stop, W[start:stop] as float32) for tensor t."""
    cols = int(t.shape[0])
    rows = int(t.n_elements) // cols

    if t.tensor_type.name in ("F32", "F16"):
        w = np.asarray(t.data).reshape(rows, cols).astype(np.float32)
        yield 0, rows, w
        return

    raw = np.ascontiguousarray(t.data).reshape(-1).view(np.uint8)
    raw = raw.reshape(rows, -1)
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        w = dequantize(np.ascontiguousarray(raw[start:stop]),
                       t.tensor_type).reshape(stop - start, cols)
        yield start, stop, w.astype(np.float32)


def reference(t, x):
    """Return (sequential float32 W x, float64 W x, sum |W| |x|)."""
    cols = int(t.shape[0])
    rows = int(t.n_elements) // cols
    y32 = np.empty(rows, dtype=np.float32)
    y64 = np.empty(rows, dtype=np.float64)
    scale = np.empty(rows, dtype=np.float64)
    xd = x.astype(np.float64)

    for start, stop, w in weight_chunks(t):
        products = w * x                          # float32 products
        y32[start:stop] = np.add.accumulate(products, axis=1)[:, -1]
        w64 = w.astype(np.float64)
        y64[start:stop] = w64 @ xd
        scale[start:stop] = np.abs(w64) @ np.abs(xd)

    return y32, y64, scale


def read_records(path):
    with open(path, "rb") as f:
        data = f.read()
    records = []
    offset = 0
    while offset < len(data):
        (name_length,) = struct.unpack_from("<I", data, offset)
        offset += 4
        name = data[offset:offset + name_length].decode("utf-8")
        offset += name_length
        rows, cols = struct.unpack_from("<QQ", data, offset)
        offset += 16
        y = np.frombuffer(data, dtype=np.float32, count=rows, offset=offset)
        offset += rows * 4
        records.append((name, rows, cols, y))
    return records


fast_records = {}
try:
    fast_records = {r[0]: r for r in read_records("build/matvec_fast.bin")}
except FileNotFoundError:
    print("(build/matvec_fast.bin not found: fast kernel not checked)")

FAST_LIMIT = 1e-5

with open("build/matvec_c.bin", "rb") as f:
    data = f.read()

offset = 0
failed = False
count = 0

print(f"{'weight':26s} {'type':5s} {'rows':>6s} {'ref bit-exact':>13s} "
      f"{'ref norm err':>12s} {'fast norm err':>13s}")

while offset < len(data):
    (name_length,) = struct.unpack_from("<I", data, offset)
    offset += 4
    name = data[offset:offset + name_length].decode("utf-8")
    offset += name_length
    rows, cols = struct.unpack_from("<QQ", data, offset)
    offset += 16
    y_c = np.frombuffer(data, dtype=np.float32, count=rows, offset=offset)
    offset += rows * 4

    t = tensors[name]
    x = x_hidden if cols == len(x_hidden) else x_dim
    if cols != len(x):
        raise RuntimeError(f"{name}: no input vector of length {cols}")

    y32, y64, scale = reference(t, x)
    bad = int(np.count_nonzero(y_c.view(np.uint32) != y32.view(np.uint32)))
    norm_err = np.abs(y_c.astype(np.float64) - y64) / np.maximum(scale, 1e-30)
    ok = bad == 0 and bool(norm_err.max() < ACCURACY_LIMIT)

    fast_text = "-"
    if name in fast_records:
        y_fast = fast_records[name][3].astype(np.float64)
        fast_err = (np.abs(y_fast - y64) / np.maximum(scale, 1e-30)).max()
        ok = ok and bool(fast_err < FAST_LIMIT)
        fast_text = f"{fast_err:.3e}"

    failed |= not ok
    count += 1

    print(f"{name:26s} {t.tensor_type.name:5s} {rows:6d} "
          f"{('yes' if bad == 0 else f'NO ({bad})'):>13s} "
          f"{norm_err.max():12.3e} {fast_text:>13s}")

print(f"\nProducts compared: {count}")

if count == 0 or failed:
    print("\nMATVEC REFERENCE COMPARISON: FAILED")
    sys.exit(1)

print("\nMATVEC REFERENCE COMPARISON: PASSED")
