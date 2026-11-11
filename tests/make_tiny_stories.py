#!/usr/bin/env python3
"""Tiny random llama2.c-format model (.bin) for testing completion mode.
Usage: make_tiny_stories.py <out.bin>   (uses the project's tokenizer.bin, vocab 32000)"""
import sys, struct, numpy as np
rng = np.random.default_rng(1)
dim, hidden, layers, heads, kv, vocab, seq = 16, 40, 1, 2, 2, 32000, 256
hs = dim // heads
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("7i", dim, hidden, layers, heads, kv, vocab, seq))
    def w(*shape, scale=0.1): f.write((rng.standard_normal(shape) * scale).astype(np.float32).tobytes())
    w(vocab, dim, scale=0.5)
    f.write(np.ones((layers, dim), np.float32).tobytes())
    w(layers, dim, dim); w(layers, dim, dim); w(layers, dim, dim); w(layers, dim, dim)
    f.write(np.ones((layers, dim), np.float32).tobytes())
    w(layers, hidden, dim); w(layers, dim, hidden); w(layers, hidden, dim)
    f.write(np.ones(dim, np.float32).tobytes())
    w(seq, hs // 2); w(seq, hs // 2)
print("wrote", sys.argv[1])
