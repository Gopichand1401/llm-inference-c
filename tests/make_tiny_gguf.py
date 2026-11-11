#!/usr/bin/env python3
"""Build a tiny random-weight llama GGUF (TinyLlama layout, F32 weights, a
byte-level tokenizer) for testing llm_rag without the 638 MB model.
The answers are gibberish; it checks the code paths, not answer quality.
Needs: pip install gguf numpy.   Usage: make_tiny_gguf.py <out.gguf>
"""
import sys, numpy as np, gguf
out = sys.argv[1]
rng = np.random.default_rng(0)
dim, hidden, layers, heads, kv_heads, ctx = 64, 172, 2, 4, 2, 2048
head_dim = dim // heads
tokens, scores, types = ["<unk>", "<s>", "</s>"], [0.0, 0.0, 0.0], [2, 3, 3]
for b in range(256):
    tokens.append("<0x%02X>" % b); scores.append(0.0); types.append(6)
words = ["▁"] + ["▁" + w for w in "the a of in is to and for who what course code room slot instructor tab row file xlsx Name Credits Question rows Spreadsheet".split()]
pieces = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") + ["▁" + c for c in "abcdefghijklmnopqrstuvwxyz"] + ["<|", "|>", "system", "user", "assistant"]
for i, p in enumerate(words + pieces):
    if p in tokens: continue
    tokens.append(p); scores.append(-float(i)); types.append(1)
vocab = len(tokens)
w = gguf.GGUFWriter(out, "llama")
w.add_context_length(ctx); w.add_embedding_length(dim); w.add_block_count(layers)
w.add_feed_forward_length(hidden); w.add_head_count(heads); w.add_head_count_kv(kv_heads)
w.add_rope_dimension_count(head_dim); w.add_layer_norm_rms_eps(1e-5); w.add_rope_freq_base(10000.0)
w.add_tokenizer_model("llama"); w.add_token_list(tokens); w.add_token_scores(scores)
w.add_token_types(types); w.add_bos_token_id(1); w.add_eos_token_id(2)
def t(name, shape, scale=0.08):
    w.add_tensor(name, (rng.standard_normal(shape) * scale).astype(np.float32))
t("token_embd.weight", (vocab, dim), 0.5)
for l in range(layers):
    w.add_tensor(f"blk.{l}.attn_norm.weight", np.ones(dim, np.float32))
    t(f"blk.{l}.attn_q.weight", (dim, dim)); t(f"blk.{l}.attn_k.weight", (dim // heads * kv_heads, dim))
    t(f"blk.{l}.attn_v.weight", (dim // heads * kv_heads, dim)); t(f"blk.{l}.attn_output.weight", (dim, dim))
    w.add_tensor(f"blk.{l}.ffn_norm.weight", np.ones(dim, np.float32))
    t(f"blk.{l}.ffn_gate.weight", (hidden, dim)); t(f"blk.{l}.ffn_up.weight", (hidden, dim)); t(f"blk.{l}.ffn_down.weight", (dim, hidden))
w.add_tensor("output_norm.weight", np.ones(dim, np.float32))
t("output.weight", (vocab, dim), 0.5)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote", out, "vocab", vocab)
