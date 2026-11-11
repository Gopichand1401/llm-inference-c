"""
compare_tensor_report.py - cross-check the C tensor report against gguf-py.

Run build/gguf_tensor_report first; it writes build/gguf_tensor_report.csv.

Usage:
    python examples/04_model_loading/compare_tensor_report.py model.gguf

For every tensor, compares what the C parser found with what the official
GGUFReader reports: name, GGML type, dimensions, storage size in bytes and
absolute file offset.
"""
import csv
import sys

from gguf.gguf_reader import GGUFReader

if len(sys.argv) != 2:
    print("Usage: python compare_tensor_report.py model.gguf")
    sys.exit(1)

reader = GGUFReader(sys.argv[1])

with open("build/gguf_tensor_report.csv", newline="") as f:
    c_rows = list(csv.DictReader(f))

print("C tensors:     ", len(c_rows))
print("gguf-py tensors:", len(reader.tensors))

failed = len(c_rows) != len(reader.tensors)
mismatches = 0

for i, (row, t) in enumerate(zip(c_rows, reader.tensors)):
    n_dims = int(row["n_dims"])
    c_dims = [int(row[f"ne{d}"]) for d in range(n_dims)]
    py_dims = [int(x) for x in t.shape]

    checks = {
        "name": (row["name"], t.name),
        "type": (int(row["type_id"]), int(t.tensor_type)),
        "dims": (c_dims, py_dims),
        "bytes": (int(row["storage_bytes"]), int(t.n_bytes)),
        "offset": (int(row["abs_offset"]), int(t.data_offset)),
    }

    for field, (c_value, py_value) in checks.items():
        if c_value != py_value:
            mismatches += 1
            print(f"MISMATCH tensor {i} {field}: C={c_value} gguf-py={py_value}")

types = {}
for t in reader.tensors:
    types[t.tensor_type.name] = types.get(t.tensor_type.name, 0) + 1

print("\nTypes according to gguf-py:")
for name, n in sorted(types.items()):
    print(f"  {name:6s} {n:4d} tensors")

print("\nField mismatches:", mismatches)

if failed or mismatches:
    print("\nTENSOR REPORT COMPARISON: FAILED")
    sys.exit(1)

print("\nTENSOR REPORT COMPARISON: PASSED")
