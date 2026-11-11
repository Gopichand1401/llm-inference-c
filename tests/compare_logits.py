
import math
import struct

def load_logits(filename):
    with open(filename, "rb") as f:
        data = f.read()

    if len(data) % 4 != 0:
        raise SystemExit(f"FAIL: Invalid file size: {filename}")

    count = len(data) // 4
    return list(struct.unpack(f"<{count}f", data))

reference = load_logits("reference_logits.bin")
custom = load_logits("custom_logits.bin")

if not reference or not custom:
    raise SystemExit("FAIL: A logit file is empty.")

if len(reference) != len(custom):
    raise SystemExit(
        f"FAIL: Different sizes: reference={len(reference)}, "
        f"custom={len(custom)}"
    )

if not all(math.isfinite(x) for x in reference):
    raise SystemExit("FAIL: Reference contains NaN or infinity.")

if not all(math.isfinite(x) for x in custom):
    raise SystemExit("FAIL: Custom output contains NaN or infinity.")

differences = [
    abs(a - b) for a, b in zip(reference, custom)
]

max_diff = max(differences)
mean_diff = sum(differences) / len(differences)

ref_argmax = max(range(len(reference)), key=reference.__getitem__)
custom_argmax = max(range(len(custom)), key=custom.__getitem__)

print("Number of logits:", len(reference))
print("Maximum absolute difference:", max_diff)
print("Mean absolute difference:", mean_diff)
print("Reference argmax:", ref_argmax)
print("Custom argmax:", custom_argmax)

# Equivalent to allclose with rtol=1e-4, atol=1e-5.
matches = all(
    abs(a - b) <= 1e-5 + 1e-4 * abs(b)
    for a, b in zip(reference, custom)
)

if matches:
    print("PASS: logits match the reference.")
else:
    print("FAIL: logits differ from the reference.")
    idx = max(range(len(differences)), key=differences.__getitem__)
    print("Largest difference at token:", idx)
    print("Reference value:", reference[idx])
    print("Custom value:", custom[idx])

