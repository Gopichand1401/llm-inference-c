"""
compare_q6k_reference.py - compare the C Q6_K decoder against gguf-py.

Run examples/gguf_q6k_real_demo first; it writes the files in build/.

Usage:
    python examples/03_quantization/compare_q6k_reference.py model.gguf [tensor_name]

Checks:
  1. First block: C output must equal gguf.quants.dequantize() exactly.
  2. Last block:  same exact check.
  3. Every row:   sum and sum of squares of all decoded weights in the row,
                  accumulated in float64, must match to ~1e-9. This covers
                  every block of the tensor.
"""
import sys

import numpy as np
from gguf.gguf_reader import GGUFReader
from gguf.quants import dequantize

BLOCK_BYTES = 210
QK_K = 256

if len(sys.argv) not in (2, 3):
    print("Usage: python compare_q6k_reference.py model.gguf [tensor_name]")
    sys.exit(1)

tensor_name = sys.argv[2] if len(sys.argv) == 3 else "output.weight"

reader = GGUFReader(sys.argv[1])
tensor = next((t for t in reader.tensors if t.name == tensor_name), None)

if tensor is None:
    raise RuntimeError(f"{tensor_name} not found")

if tensor.tensor_type.name != "Q6_K":
    raise RuntimeError(f"Expected Q6_K, got {tensor.tensor_type.name}")

shape = [int(x) for x in tensor.shape]          # GGUF order: ne0 first
row_length = shape[0]
rows = int(tensor.n_elements) // row_length

print("Tensor:", tensor.name)
print("Quantization:", tensor.tensor_type.name)
print("GGUF shape (ne0 first):", shape)
print("Rows:", rows, " Row length:", row_length)

raw = np.ascontiguousarray(tensor.data).reshape(-1)
blocks = raw.reshape(-1, BLOCK_BYTES)
print("Blocks:", blocks.shape[0])

failed = False


def compare_block(label, block_bytes, c_path):
    global failed
    reference = dequantize(np.ascontiguousarray(block_bytes[None, :]),
                           tensor.tensor_type).reshape(-1)
    c_values = np.fromfile(c_path, dtype=np.float32)

    if len(c_values) != QK_K or len(reference) != QK_K:
        raise RuntimeError(f"{label}: expected 256 weights from each side")

    diff = np.abs(c_values.astype(np.float64) - reference.astype(np.float64))
    exact = np.array_equal(c_values, reference)

    print(f"\n=== {label} ===")
    print("Index       C value       GGUF value       Difference")
    for i in range(10):
        print(f"{i:3d}   {c_values[i]:14.9f}   {reference[i]:14.9f}   "
              f"{diff[i]:.12f}")
    print("Maximum absolute difference:", diff.max())
    print("Mean absolute difference:", diff.mean())
    print("Bit-exact match:", exact)

    if not exact:
        failed = True


compare_block("First block", blocks[0], "build/q6k_c_first256.bin")
compare_block("Last block", blocks[-1], "build/q6k_c_last256.bin")

# ---------------------------------------------------------------------------
# Every row: dequantize in chunks so memory stays modest.
# ---------------------------------------------------------------------------
c_stats = np.fromfile("build/q6k_c_row_stats.bin", dtype=np.float64)

if c_stats.size != rows * 2:
    raise RuntimeError(
        f"Row stats: expected {rows * 2} float64 values, got {c_stats.size}")

c_stats = c_stats.reshape(rows, 2)
row_bytes = raw.reshape(rows, -1)

ref_stats = np.empty((rows, 2), dtype=np.float64)
chunk = 1024

for start in range(0, rows, chunk):
    stop = min(start + chunk, rows)
    values = dequantize(np.ascontiguousarray(row_bytes[start:stop]),
                        tensor.tensor_type).astype(np.float64)
    values = values.reshape(stop - start, row_length)
    ref_stats[start:stop, 0] = values.sum(axis=1)
    ref_stats[start:stop, 1] = (values * values).sum(axis=1)

stat_diff = np.abs(c_stats - ref_stats)
worst_row = int(np.argmax(stat_diff.max(axis=1)))
rows_ok = np.allclose(c_stats, ref_stats, rtol=1e-9, atol=1e-9)

print("\n=== All rows (sum, sum of squares) ===")
print("Rows compared:", rows)
print("Max |sum diff|:", stat_diff[:, 0].max())
print("Max |sumsq diff|:", stat_diff[:, 1].max())
print("Worst row:", worst_row,
      " C:", c_stats[worst_row], " ref:", ref_stats[worst_row])
print("All rows within tolerance:", rows_ok)

if not rows_ok:
    failed = True

print("\nQ6_K REFERENCE COMPARISON:", "FAILED" if failed else "PASSED")
sys.exit(1 if failed else 0)
