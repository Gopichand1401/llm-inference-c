"""
compare_gguf_rows.py - check rows decoded by the C loader against gguf-py.

Run build/gguf_model_demo first; it writes build/gguf_rows.bin.

Usage:
    python examples/04_model_loading/compare_gguf_rows.py model.gguf

Every record in build/gguf_rows.bin is one row of one weight tensor, decoded
by gguf_weight_dequantize_row(). This script decodes the same row with the
official gguf-py code and requires a bit-exact match.
"""
import struct
import sys

import numpy as np
from gguf.gguf_reader import GGUFReader
from gguf.quants import dequantize

if len(sys.argv) != 2:
    print("Usage: python compare_gguf_rows.py model.gguf")
    sys.exit(1)

reader = GGUFReader(sys.argv[1])
tensors = {t.name: t for t in reader.tensors}


def reference_row(t, row):
    """Decode one row of tensor t with gguf-py."""
    row_length = int(t.shape[0])
    n_rows = int(t.n_elements) // row_length
    name = t.tensor_type.name

    if name in ("F32", "F16"):
        values = np.asarray(t.data).reshape(n_rows, row_length)[row]
        return values.astype(np.float32)

    raw = np.ascontiguousarray(t.data).reshape(-1).view(np.uint8)
    raw_row = raw.reshape(n_rows, -1)[row]
    return dequantize(np.ascontiguousarray(raw_row[None, :]),
                      t.tensor_type).reshape(-1).astype(np.float32)


with open("build/gguf_rows.bin", "rb") as f:
    data = f.read()

offset = 0
records = 0
mismatches = 0
types_seen = {}

while offset < len(data):
    (name_length,) = struct.unpack_from("<I", data, offset)
    offset += 4
    name = data[offset:offset + name_length].decode("utf-8")
    offset += name_length
    row, cols = struct.unpack_from("<QQ", data, offset)
    offset += 16
    c_values = np.frombuffer(data, dtype=np.float32, count=cols, offset=offset)
    offset += cols * 4

    t = tensors[name]
    ref = reference_row(t, row)
    records += 1
    types_seen[t.tensor_type.name] = types_seen.get(t.tensor_type.name, 0) + 1

    if len(ref) != cols or not np.array_equal(c_values.view(np.uint32),
                                              ref.view(np.uint32)):
        mismatches += 1
        diff = np.abs(c_values.astype(np.float64) - ref.astype(np.float64))
        print(f"MISMATCH {name} row {row}: max diff {diff.max()}")

print("Rows compared:", records)
print("Rows per type:", dict(sorted(types_seen.items())))
print("Mismatched rows:", mismatches)

if records == 0 or mismatches:
    print("\nGGUF ROW COMPARISON: FAILED")
    sys.exit(1)

print("\nGGUF ROW COMPARISON: PASSED (bit-exact)")
