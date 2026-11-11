
import sys
import numpy as np

from gguf.gguf_reader import GGUFReader
from gguf.quants import dequantize


if len(sys.argv) != 2:
    print("Usage: python compare_q4k_reference.py model.gguf")
    sys.exit(1)

# 1. Load the GGUF model.
reader = GGUFReader(sys.argv[1])

# 2. Find the embedding tensor.
tensor = next(
    (t for t in reader.tensors if t.name == "token_embd.weight"),
    None
)

if tensor is None:
    raise RuntimeError("token_embd.weight not found")

if tensor.tensor_type.name != "Q4_K":
    raise RuntimeError(
        f"Expected Q4_K, got {tensor.tensor_type.name}"
    )

print("Tensor:", tensor.name)
print("Quantization:", tensor.tensor_type.name)

# 3. Extract the first 144-byte Q4_K block.
raw_blocks = tensor.data.reshape(-1, 144)
first_block = np.ascontiguousarray(raw_blocks[:1])

# 4. Dequantize using official GGUF implementation.
reference = dequantize(
    first_block,
    tensor.tensor_type
).reshape(-1)

# 5. Read our C implementation's output.
c_values = np.fromfile(
    "build/q4k_c_first256.bin",
    dtype=np.float32
)

if len(c_values) != 256 or len(reference) != 256:
    raise RuntimeError("Expected exactly 256 weights from each implementation")

# 6. Compare values.
difference = np.abs(c_values - reference)

print("\nFirst 10 weights:")
print("Index       C value       GGUF value       Difference")

for i in range(10):
    print(
        f"{i:3d}   "
        f"{c_values[i]:14.9f}   "
        f"{reference[i]:14.9f}   "
        f"{difference[i]:.12f}"
    )

print("\nMaximum absolute difference:", np.max(difference))
print("Mean absolute difference:", np.mean(difference))

# 7. Final validation.
if np.allclose(c_values, reference, rtol=1e-5, atol=1e-8):
    print("\nQ4_K REFERENCE COMPARISON: PASSED")
else:
    print("\nQ4_K REFERENCE COMPARISON: FAILED")
