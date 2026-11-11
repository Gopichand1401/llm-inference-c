"""
compare_quant_random.py - test every C block decoder against gguf-py on
random blocks. Needs no model file.

Usage (after make gguf-tools):
    python examples/03_quantization/compare_quant_random.py [blocks_per_type]

For each supported type it generates random block bytes (with the FP16
scale fields kept finite, but including negative, tiny and subnormal
scales), decodes them with build/gguf_quant_check and with
gguf.quants.dequantize(), and requires a bit-exact match.
"""
import os
import subprocess
import sys

import numpy as np
from gguf import GGMLQuantizationType as T
from gguf.quants import dequantize

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000

# type -> (bytes per block, offsets of FP16 scale fields)
TYPES = {
    T.Q4_0: (18, [0]),
    T.Q8_0: (34, [0]),
    T.Q4_K: (144, [0, 2]),
    T.Q5_K: (176, [0, 2]),
    T.Q6_K: (210, [208]),
}

rng = np.random.default_rng(2026)
os.makedirs("build", exist_ok=True)
failed = False

for qtype, (size, fp16_offsets) in TYPES.items():
    raw = rng.integers(0, 256, size=(N, size), dtype=np.uint8)
    for off in fp16_offsets:
        raw[:, off + 1] &= 0x7B          # exponent never all ones: no inf/NaN

    raw_path = f"build/random_{qtype.name}.raw"
    out_path = f"build/random_{qtype.name}.f32"
    raw.tofile(raw_path)

    result = subprocess.run(
        ["./build/gguf_quant_check", str(int(qtype)), raw_path, out_path],
        capture_output=True, text=True)
    if result.returncode != 0:
        print(f"{qtype.name:5s}  C decoder FAILED: {result.stderr.strip()}")
        failed = True
        continue

    c_values = np.fromfile(out_path, dtype=np.float32)
    ref = dequantize(raw, qtype).reshape(-1).astype(np.float32)

    exact = c_values.shape == ref.shape and np.array_equal(
        c_values.view(np.uint32), ref.view(np.uint32))
    mismatches = (int(np.count_nonzero(c_values.view(np.uint32) != ref.view(np.uint32)))
                  if c_values.shape == ref.shape else -1)
    failed |= not exact

    print(f"{qtype.name:5s}  blocks {N:6d}  weights {ref.size:9d}  "
          f"bit-exact {'yes' if exact else 'NO'}  mismatches {mismatches}")

print("\nRANDOM BLOCK COMPARISON:", "FAILED" if failed else "PASSED")
sys.exit(1 if failed else 0)
