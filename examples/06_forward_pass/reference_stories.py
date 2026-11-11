"""
reference_stories.py - independent NumPy forward pass for llama2.c .bin
models (stories15M / 42M / 110M ...), to check the C Stories path.

Run build/llm_logits_dump on the same model first.

Usage:
    python examples/06_forward_pass/reference_stories.py model.bin

The .bin file is read here directly, not through the C code:
  header: 7 int32  dim, hidden_dim, n_layers, n_heads, n_kv_heads,
                   vocab_size (negative = separate classifier), seq_len
  then float32 arrays in this order (matrices row-major, [out][in]):
    token_embedding [vocab][dim]       rms_att  [layers][dim]
    wq [layers][dim][dim]              wk, wv   [layers][kv_dim][dim]
    wo [layers][dim][dim]              rms_ffn  [layers][dim]
    w1 [layers][hidden][dim]           w2       [layers][dim][hidden]
    w3 [layers][hidden][dim]           rms_final [dim]
    freq_cis_real, freq_cis_imag [seq_len][head_size/2]   (unused)
    wcls [vocab][dim]                  only if vocab_size was negative
The maths is the standard LLaMA block (RMSNorm eps 1e-5, RoPE base 10000
on adjacent pairs, grouped-query attention, SwiGLU), in float64, all
positions at once with a causal mask.
"""
import struct
import sys

import numpy as np

if len(sys.argv) != 2:
    print("Usage: python reference_stories.py model.bin")
    sys.exit(1)

with open(sys.argv[1], "rb") as f:
    header = struct.unpack("<7i", f.read(28))

dim, hidden, n_layers, n_heads, n_kv_heads, vocab, seq_len = header
shared = vocab > 0
vocab = abs(vocab)
head_size = dim // n_heads
kv_dim = head_size * n_kv_heads
kv_mul = n_heads // n_kv_heads
eps = 1e-5
rope_base = 10000.0

print(f"Reference config: dim={dim} hidden={hidden} layers={n_layers} "
      f"heads={n_heads} kv_heads={n_kv_heads} vocab={vocab} seq_len={seq_len} "
      f"classifier={'shared' if shared else 'separate'}")

data = np.memmap(sys.argv[1], dtype=np.float32, mode="r", offset=28)
offset = 0


def take(*shape):
    global offset
    n = int(np.prod(shape))
    out = np.asarray(data[offset:offset + n], dtype=np.float64).reshape(shape)
    offset += n
    return out


token_embedding = take(vocab, dim)
rms_att = take(n_layers, dim)
wq = take(n_layers, dim, dim)
wk = take(n_layers, kv_dim, dim)
wv = take(n_layers, kv_dim, dim)
wo = take(n_layers, dim, dim)
rms_ffn = take(n_layers, dim)
w1 = take(n_layers, hidden, dim)
w2 = take(n_layers, dim, hidden)
w3 = take(n_layers, hidden, dim)
rms_final = take(dim)
offset += seq_len * (head_size // 2) * 2          # skip freq_cis_real/imag
wcls = token_embedding if shared else take(vocab, dim)


def rmsnorm(x, w):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w


def rope(x, positions):
    i = np.arange(0, head_size, 2, dtype=np.float64)
    freq = rope_base ** (-i / head_size)
    angle = positions[:, None] * freq[None, :]
    cos = np.cos(angle)[:, None, :]
    sin = np.sin(angle)[:, None, :]
    out = x.copy()
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out[..., 0::2] = x0 * cos - x1 * sin
    out[..., 1::2] = x0 * sin + x1 * cos
    return out


with open("build/forward_tokens.bin", "rb") as f:
    (n_tokens,) = struct.unpack("<i", f.read(4))
    token_ids = np.frombuffer(f.read(4 * n_tokens), dtype=np.int32)

T = len(token_ids)
positions = np.arange(T, dtype=np.float64)
causal = np.triu(np.full((T, T), -np.inf), k=1)
print("Tokens:", token_ids.tolist())

x = token_embedding[token_ids]

for l in range(n_layers):
    h = rmsnorm(x, rms_att[l])
    q = rope((h @ wq[l].T).reshape(T, n_heads, head_size), positions)
    k = rope((h @ wk[l].T).reshape(T, n_kv_heads, head_size), positions)
    v = (h @ wv[l].T).reshape(T, n_kv_heads, head_size)

    att = np.empty((T, n_heads, head_size))
    for hd in range(n_heads):
        kv = hd // kv_mul
        scores = q[:, hd, :] @ k[:, kv, :].T / np.sqrt(head_size) + causal
        scores -= scores.max(axis=1, keepdims=True)
        p = np.exp(scores)
        p /= p.sum(axis=1, keepdims=True)
        att[:, hd, :] = p @ v[:, kv, :]

    x = x + att.reshape(T, dim) @ wo[l].T
    h = rmsnorm(x, rms_ffn[l])
    gate = h @ w1[l].T
    x = x + (gate / (1.0 + np.exp(-gate)) * (h @ w3[l].T)) @ w2[l].T

x = rmsnorm(x, rms_final)
ref = x @ wcls.T

c_logits = np.fromfile("build/forward_logits.bin", dtype=np.float32)
c_logits = c_logits.reshape(T, vocab).astype(np.float64)

print(f"\n{'pos':>3s} {'token':>6s} {'max|diff|':>10s} {'ref std':>8s} "
      f"{'rel':>9s} {'top1 C':>7s} {'top1 ref':>8s}")
worst = 0.0
agree = 0
for pos in range(T):
    diff = np.abs(c_logits[pos] - ref[pos]).max()
    rel = diff / ref[pos].std()
    worst = max(worst, rel)
    c1, r1 = int(np.argmax(c_logits[pos])), int(np.argmax(ref[pos]))
    agree += int(c1 == r1)
    print(f"{pos:3d} {token_ids[pos]:6d} {diff:10.3e} {ref[pos].std():8.4f} "
          f"{rel:9.2e} {c1:7d} {r1:8d}")

REL_LIMIT = 1e-3
print(f"\nWorst max|diff| / std: {worst:.3e} (limit {REL_LIMIT:g})")
print(f"Top-1 agreement: {agree}/{T}")
ok = worst < REL_LIMIT and agree == T
print("\nSTORIES REFERENCE COMPARISON:", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
