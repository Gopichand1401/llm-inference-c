# Algorithms

Notation: `x` is the current activation vector (length `dim`), `W·x` is a
matrix-vector product, `⊙` is element-wise multiplication, and `pos` is the
position of the current token (0-based).

## 1. Token embedding lookup
Token ID `t` selects row `t` of the embedding table: `x = E[t]` (`dim`
floats). For GGUF, that row is dequantized on the fly with
`gguf_weight_dequantize_row()`.

## 2. RMS normalisation
```
rmsnorm(x, w) = w ⊙ x / sqrt( mean(x²) + ε )
```
ε is 1e-5 for llama2.c models and comes from
`llama.attention.layer_norm_rms_epsilon` for GGUF. Unlike LayerNorm there is
no mean subtraction and no bias. It is applied before attention, before the
feed-forward network and once at the end. With the ε, a zero vector gives 0
rather than NaN (tested).

## 3. Rotary positional embedding (RoPE)
Positions are encoded by rotating each pair of dimensions of the query and
key vectors by an angle that grows with the position:
```
for each pair (i, i+1), i = 0, 2, 4, … < rope_dim:
    θ   = (pos / rope_scale) · base^(−i / rope_dim)        base = 10000 by default
    q'ᵢ   = qᵢ·cos θ − qᵢ₊₁·sin θ
    q'ᵢ₊₁ = qᵢ·sin θ + qᵢ₊₁·cos θ                           (same for k)
```
The dot product of a rotated query and a rotated key then depends only on
their *relative* distance. `rope_scale` supports models fine-tuned for a
longer context with linear scaling. Other scaling types are refused at load
time instead of producing wrong output.

## 4. Multi-head attention with grouped-query attention (GQA)
```
q = Wq·x      k = Wk·x      v = Wv·x          (k, v have kv_dim = head_size · n_kv_heads)
apply RoPE to q and k; store k, v at row pos of this layer's KV cache
for each query head h (kv head g = h / (n_heads / n_kv_heads)):
    score[t] = q_h · K_g[t] / sqrt(head_size)        for t = 0 … pos   (causal: never future tokens)
    a        = softmax(score)
    out_h    = Σ_t a[t] · V_g[t]
x = x + Wo · concat(out_h)                           (residual connection)
```
With GQA, several query heads share one key/value head. TinyLlama has 32
query heads and 4 KV heads, which makes the KV cache 8× smaller. The same
code handles ordinary multi-head attention (`n_kv_heads = n_heads`) and
multi-query attention (`n_kv_heads = 1`); the tests cover all three.
**Softmax** subtracts the maximum first, `exp(xᵢ − max) / Σ exp(xⱼ − max)`,
so large scores cannot overflow (tested with logits around 1000).

## 5. Feed-forward network (SwiGLU)
```
h = silu(W1·x) ⊙ (W3·x)          silu(z) = z / (1 + e^(−z))
x = x + W2·h                     (residual connection)
```
`hidden_dim` is usually about 2.7 × `dim`.

## 6. Logits
After the last layer: `x = rmsnorm(x, w_final)`, then `logits = Wcls·x`
(`vocab_size` floats, one score per token). When a model ties its weights,
`Wcls` is the embedding table itself: a positive `vocab_size` in llama2.c
files, or a missing `output.weight` in GGUF.

## 7. Matrix-vector products
`matmul(out, x, W, n, d)` computes `out[r] = Σ_c W[r][c]·x[c]` for a
`d × n` row-major matrix. Rows are independent and are split across
threads with `#pragma omp parallel for`. This is where nearly all the time
goes (see `PERFORMANCE.md`).

### Quantized weights (GGUF)
Weights are stored in blocks. Each block holds a few FP16 scale values and
small integers:

| Type | Weights / block | Bytes / block | Bits / weight | Decoding |
|---|---:|---:|---:|---|
| Q4_0 | 32 | 18 | 4.5 | `w = d · (q − 8)`, q 4-bit |
| Q8_0 | 32 | 34 | 8.5 | `w = d · q`, q int8 |
| Q4_K | 256 | 144 | 4.5 | 8 sub-blocks of 32 with 6-bit scale `s` and min `m`: `w = d·s·q − dmin·m` |
| Q5_K | 256 | 176 | 5.5 | like Q4_K with a fifth bit per weight |
| Q6_K | 256 | 210 | 6.56 | 16 sub-blocks of 16, int8 scales: `w = d·s·(q − 32)`; q split into 4 low + 2 high bits |

`gguf_matvec()` never builds an FP32 copy of a matrix. For each row it
works block by block on the packed integers. In the K-quant formats the
min term `dmin·m·Σxᵢ` depends only on `x`, so the per-sub-block sums of `x`
are computed once per product and reused for every row. There is also a
slower `gguf_matvec_reference()` that decodes to FP32 first and sums in
strict column order. It matches NumPy bit for bit and is used to verify
the fast kernels (`tests/run_gguf_checks.sh`).

## 8. Tokenizer: SentencePiece-style byte-pair encoding
**Encode:**
1. Add a leading space, as SentencePiece does (`"Hello"` → `" Hello"`).
2. Split into UTF-8 characters and look each one up in the vocabulary
   (binary search over the sorted vocabulary). A character that is not in
   the vocabulary becomes its raw bytes, using the `<0xXX>` byte tokens
   (byte fallback), so any input is encodable.
3. Repeatedly merge the adjacent pair whose concatenation is a vocabulary
   token with the **highest score**, leftmost on ties, until no pair merges.
4. Add BOS/EOS if requested. For GGUF chat models, special-token strings
   such as `</s>` or `<|user|>` in the text are emitted directly as their
   IDs.

**Our improvement to step 3:** the reference llama2.c loop rescans every
pair after each merge, which is O(n²) lookups: a 20 KB prompt took 15 s.
`merge_pairs()` keeps the tokens in a doubly linked list and every
mergeable pair in a max-heap ordered by (score, then position). A merge
only creates new pairs with its two neighbours, and entries that went stale
are skipped when popped. This picks exactly the same merge at every step,
so the output is identical (checked on 3 003 inputs against the old code),
in O(n log n): 0.01 s for 20 KB and under 1 s for 200 KB.

**Decode:** concatenate the token strings, turn `<0xXX>` tokens back into
bytes, drop BOS/EOS and drop the leading space after BOS.

## 9. Sampling
| Method | Algorithm |
|---|---|
| Greedy | `argmax(logits)`; ties go to the lowest ID. Deterministic |
| Temperature | `p = softmax(logits / T)` computed in double, then a draw from p by inverse CDF with one uniform random number. T < 1 sharpens, T > 1 flattens |
| Top-k | keep the k largest logits (insertion into a sorted list of k, O(V·k)), then temperature sampling among only those. k = 1 is greedy |

Random numbers come from xorshift32, with the 32-bit state owned by the
caller. A fixed `--seed` gives the same text every time (tested). The
tests also check that the sampled frequencies match the softmax
probabilities (within 2% over 20 000 draws), that top-k never leaves the
top k, and that invalid input (NaN, inf, T ≤ 0, k = 0 or k > V) returns
`-1`.

## 10. Autoregressive generation
```
tokens = encode(prompt, BOS)
for pos = 0, 1, 2, …:
    logits = engine_forward(model, token, pos)       # fills the KV cache at pos
    if pos < len(tokens) − 1:  token = tokens[pos + 1]   # still reading the prompt
    else:                      token = sample(logits); print(decode(token))
    stop on EOS (and BOS for llama2.c story models), at the token limit, or when the context is full
```
In interactive chat (`llm_generate -i`) the KV cache is kept between
turns, so each new turn only runs its own new tokens through the model.
The reply's EOS token is fed back in, so the cache holds exactly the
conversation so far.

## 11. XLSX reading (RAG input)
An `.xlsx` file is a ZIP archive of XML files. `xlsx_reader.c` reads the
ZIP central directory itself, inflates entries with zlib, then parses
`workbook.xml`, its relationships, `sharedStrings.xml`, `styles.xml` (to
recognise date formats) and every sheet. The result is one clean line of
text per cell, ready to be turned into retrieval chunks.
