"""
reference_forward.py - independent NumPy LLaMA forward pass for checking C.

Run build/gguf_forward_demo first; it writes build/forward_tokens.bin and
build/forward_logits.bin.

Usage:
    python examples/06_forward_pass/reference_forward.py model.gguf [tokenizer.bin]

What it does:
  1. Reads the configuration and the weights with the official gguf-py
     package (gguf.quants.dequantize), never with our C code.
  2. Runs the standard LLaMA computation for the same tokens, in float64,
     processing all prompt positions at once with a causal mask
     (the C code processes one position at a time with a KV cache, so this
     also checks the KV cache logic).
  3. Compares the logits at every position with the C output.
  4. Optionally checks that tokenizer.bin has the same vocabulary as the
     GGUF file (needed for the prompt token IDs to mean the same thing).

Memory: weights are dequantized one layer at a time; peak use is roughly
1 GB for TinyLlama.
"""
import struct
import sys

import numpy as np
from gguf.gguf_reader import GGUFReader
from gguf.quants import dequantize

if len(sys.argv) not in (2, 3):
    print("Usage: python reference_forward.py model.gguf [tokenizer.bin]")
    sys.exit(1)

reader = GGUFReader(sys.argv[1])
tensors = {t.name: t for t in reader.tensors}


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def scalar(key, default=None):
    if key not in reader.fields:
        return default
    f = reader.fields[key]
    return f.parts[f.data[0]][0].item()


def string_list(key):
    f = reader.fields[key]
    return [bytes(f.parts[i]).decode("utf-8", errors="replace") for i in f.data]


dim = scalar("llama.embedding_length")
hidden = scalar("llama.feed_forward_length")
n_layers = scalar("llama.block_count")
n_heads = scalar("llama.attention.head_count")
n_kv_heads = scalar("llama.attention.head_count_kv", n_heads)
eps = scalar("llama.attention.layer_norm_rms_epsilon", 1e-5)
rope_base = scalar("llama.rope.freq_base", 10000.0)
head_size = dim // n_heads
rope_dim = scalar("llama.rope.dimension_count", head_size)
kv_mul = n_heads // n_kv_heads


def string_value(key):
    if key not in reader.fields:
        return None
    f = reader.fields[key]
    return bytes(f.parts[f.data[0]]).decode("utf-8")


# RoPE scaling: same rules as gguf_read_config() in C.
scaling_type = string_value("llama.rope.scaling.type")
scaling_factor = scalar("llama.rope.scaling.factor", None)
scale_linear_old = scalar("llama.rope.scale_linear", None)
if scaling_type in (None, "none"):
    if scale_linear_old:
        rope_scale = scale_linear_old
    elif scaling_type is None and scaling_factor:
        rope_scale = scaling_factor
    else:
        rope_scale = 1.0
elif scaling_type == "linear":
    rope_scale = scaling_factor or 1.0
else:
    print(f"RoPE scaling type {scaling_type!r} is not supported by this reference.")
    sys.exit(1)
if "rope_freqs.weight" in [t.name for t in reader.tensors]:
    print("rope_freqs.weight (Llama 3.1 RoPE) is not supported by this reference.")
    sys.exit(1)
vocab_strings = string_list("tokenizer.ggml.tokens")
vocab = len(vocab_strings)

print(f"Reference config: dim={dim} hidden={hidden} layers={n_layers} "
      f"heads={n_heads} kv_heads={n_kv_heads} rope_dim={rope_dim} "
      f"base={rope_base} scale={rope_scale:g} eps={eps:g} vocab={vocab}")


# ---------------------------------------------------------------------------
# Weights (via gguf-py only)
# ---------------------------------------------------------------------------
def weight(name):
    """Whole tensor as float64, shape (rows, cols) or (cols,)."""
    t = tensors[name]
    cols = int(t.shape[0])
    rows = int(t.n_elements) // cols
    if t.tensor_type.name in ("F32", "F16"):
        w = np.asarray(t.data, dtype=np.float32).reshape(rows, cols)
    else:
        raw = np.ascontiguousarray(t.data).reshape(-1).view(np.uint8)
        w = dequantize(raw.reshape(rows, -1), t.tensor_type).reshape(rows, cols)
    w = w.astype(np.float64)
    return w[0] if rows == 1 else w


def weight_rows(name, ids):
    """Selected rows of a tensor as float64 (for the embedding lookup)."""
    t = tensors[name]
    cols = int(t.shape[0])
    rows = int(t.n_elements) // cols
    if t.tensor_type.name in ("F32", "F16"):
        w = np.asarray(t.data, dtype=np.float32).reshape(rows, cols)[ids]
        return w.astype(np.float64)
    raw = np.ascontiguousarray(t.data).reshape(-1).view(np.uint8)
    raw = raw.reshape(rows, -1)[ids]
    return dequantize(np.ascontiguousarray(raw),
                      t.tensor_type).reshape(len(ids), cols).astype(np.float64)


def matmul_chunked(x, name, chunk=4096):
    """x @ W.T for a large W, dequantizing W in row chunks."""
    t = tensors[name]
    cols = int(t.shape[0])
    rows = int(t.n_elements) // cols
    out = np.empty((x.shape[0], rows), dtype=np.float64)
    raw = np.ascontiguousarray(t.data).reshape(-1).view(np.uint8)
    raw = raw.reshape(rows, -1)
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        w = dequantize(np.ascontiguousarray(raw[start:stop]),
                       t.tensor_type).reshape(stop - start, cols)
        out[:, start:stop] = x @ w.astype(np.float64).T
    return out


# ---------------------------------------------------------------------------
# LLaMA maths
# ---------------------------------------------------------------------------
def rmsnorm(x, w):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def rope(x, positions):
    """x: (T, heads, head_size). Rotate adjacent pairs (2i, 2i+1)."""
    i = np.arange(0, rope_dim, 2, dtype=np.float64)
    freq = rope_base ** (-i / rope_dim)                  # (rope_dim/2,)
    angle = positions[:, None] * freq[None, :]           # (T, rope_dim/2)
    cos = np.cos(angle)[:, None, :]
    sin = np.sin(angle)[:, None, :]
    out = x.copy()
    x0 = x[..., 0:rope_dim:2]
    x1 = x[..., 1:rope_dim:2]
    out[..., 0:rope_dim:2] = x0 * cos - x1 * sin
    out[..., 1:rope_dim:2] = x0 * sin + x1 * cos
    return out


with open("build/forward_tokens.bin", "rb") as f:
    (n_tokens,) = struct.unpack("<i", f.read(4))
    token_ids = np.frombuffer(f.read(4 * n_tokens), dtype=np.int32)

T = len(token_ids)
positions = np.arange(T, dtype=np.float64) / rope_scale    # linear RoPE scaling
causal = np.triu(np.full((T, T), -np.inf), k=1)

print("Tokens:", token_ids.tolist())

x = weight_rows("token_embd.weight", token_ids)          # (T, dim)

for l in range(n_layers):
    p = f"blk.{l}."

    h = rmsnorm(x, weight(p + "attn_norm.weight"))
    q = (h @ weight(p + "attn_q.weight").T).reshape(T, n_heads, head_size)
    k = (h @ weight(p + "attn_k.weight").T).reshape(T, n_kv_heads, head_size)
    v = (h @ weight(p + "attn_v.weight").T).reshape(T, n_kv_heads, head_size)

    q = rope(q, positions)
    k = rope(k, positions)

    att_out = np.empty((T, n_heads, head_size))
    for hd in range(n_heads):
        kv = hd // kv_mul
        scores = q[:, hd, :] @ k[:, kv, :].T / np.sqrt(head_size) + causal
        scores -= scores.max(axis=1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=1, keepdims=True)
        att_out[:, hd, :] = probs @ v[:, kv, :]

    x = x + att_out.reshape(T, dim) @ weight(p + "attn_output.weight").T

    h = rmsnorm(x, weight(p + "ffn_norm.weight"))
    gate = h @ weight(p + "ffn_gate.weight").T
    up = h @ weight(p + "ffn_up.weight").T
    act = gate / (1.0 + np.exp(-gate)) * up
    x = x + act @ weight(p + "ffn_down.weight").T

    print(f"\r  reference layer {l + 1}/{n_layers}", end="", flush=True)

print()
x = rmsnorm(x, weight("output_norm.weight"))
out_name = "output.weight" if "output.weight" in tensors else "token_embd.weight"
ref_logits = matmul_chunked(x, out_name)                  # (T, vocab)

# ---------------------------------------------------------------------------
# Compare with C
# ---------------------------------------------------------------------------
c_logits = np.fromfile("build/forward_logits.bin", dtype=np.float32)
c_logits = c_logits.reshape(T, vocab).astype(np.float64)

print(f"\n{'pos':>3s} {'token':>6s} {'max|diff|':>10s} {'ref std':>8s} "
      f"{'rel':>9s} {'top1 C':>7s} {'top1 ref':>8s} {'top5 overlap':>12s}")

worst_rel = 0.0
top1_agree = 0
for pos in range(T):
    diff = np.abs(c_logits[pos] - ref_logits[pos]).max()
    std = ref_logits[pos].std()
    rel = diff / std
    worst_rel = max(worst_rel, rel)
    c_top = np.argsort(-c_logits[pos])[:5]
    r_top = np.argsort(-ref_logits[pos])[:5]
    overlap = len(set(c_top) & set(r_top))
    top1_agree += int(c_top[0] == r_top[0])
    print(f"{pos:3d} {token_ids[pos]:6d} {diff:10.3e} {std:8.4f} {rel:9.2e} "
          f"{c_top[0]:7d} {r_top[0]:8d} {overlap:9d}/5")

print("\nTop 5 after the prompt (reference, GGUF vocabulary):")
last = ref_logits[-1]
probs = np.exp(last - last.max())
probs /= probs.sum()
for idx in np.argsort(-last)[:5]:
    print(f"  {idx:6d}  logit {last[idx]:8.4f}  p={probs[idx]:.4f}  "
          f"{vocab_strings[idx]!r}")

# A float32 engine vs a float64 reference accumulates small rounding
# differences through 22 layers. 1e-3 of the logit spread is far below
# anything that changes predictions, but far above what a real bug causes.
REL_LIMIT = 1e-3
ok = worst_rel < REL_LIMIT and top1_agree == T

print(f"\nWorst max|diff| / std over all positions: {worst_rel:.3e} "
      f"(limit {REL_LIMIT:g})")
print(f"Top-1 agreement: {top1_agree}/{T}")

# ---------------------------------------------------------------------------
# Optional: vocabulary check for tokenizer.bin
# ---------------------------------------------------------------------------
if len(sys.argv) == 3:
    with open(sys.argv[2], "rb") as f:
        tb = f.read()
    off = 4  # max_token_length
    pieces = []
    while off < len(tb) and len(pieces) < vocab:
        _score, length = struct.unpack_from("<fi", tb, off)
        off += 8
        pieces.append(tb[off:off + length].decode("utf-8", errors="replace"))
        off += length
    gguf_pieces = [s.replace("▁", " ") for s in vocab_strings]
    mismatched = [i for i, (a, b) in enumerate(zip(pieces, gguf_pieces)) if a != b]
    print(f"\ntokenizer.bin entries: {len(pieces)}, GGUF tokens: {vocab}, "
          f"mismatches (after '▁' -> ' '): {len(mismatched)}")
    for i in mismatched[:10]:
        print(f"  id {i}: tokenizer.bin {pieces[i]!r}  GGUF {vocab_strings[i]!r}")
    if len(pieces) != vocab or mismatched:
        print("NOTE: tokenizer.bin differs from the GGUF vocabulary.")
    else:
        print("tokenizer.bin vocabulary matches the GGUF vocabulary.")

if ok:
    print("\nFORWARD PASS REFERENCE COMPARISON: PASSED")
    sys.exit(0)

print("\nFORWARD PASS REFERENCE COMPARISON: FAILED")
sys.exit(1)
