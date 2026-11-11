"""
compare_tokenizer_hf.py - compare the C GGUF tokenizer with TinyLlama's
official Hugging Face tokenizer, and show the GGUF scores involved.

Run build/gguf_tokenizer_demo first; it writes build/tokenizer_cases.bin.

Usage:
    pip install tokenizers          # once, inside .venv-gguf-ref
    python examples/07_tokenizer/compare_tokenizer_hf.py model.gguf [tokenizer.bin]

Needs internet access the first time (downloads tokenizer.json, ~2 MB).
"""
import struct
import sys

from gguf.gguf_reader import GGUFReader
from tokenizers import Tokenizer

REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

if len(sys.argv) not in (2, 3):
    print("Usage: python compare_tokenizer_hf.py model.gguf [tokenizer.bin]")
    sys.exit(1)

reader = GGUFReader(sys.argv[1])
tok_field = reader.fields["tokenizer.ggml.tokens"]
score_field = reader.fields["tokenizer.ggml.scores"]
gguf_tokens = [bytes(tok_field.parts[i]).decode("utf-8") for i in tok_field.data]
gguf_scores = [float(score_field.parts[i][0]) for i in score_field.data]

bin_scores = None
if len(sys.argv) == 3:
    data = open(sys.argv[2], "rb").read()
    off, bin_scores = 4, []
    while off < len(data):
        score, length = struct.unpack_from("<fi", data, off)
        off += 8 + length
        bin_scores.append(score)

hf = Tokenizer.from_pretrained(REPO)
print(f"Hugging Face tokenizer: {REPO}, vocab {hf.get_vocab_size()}")

with open("build/tokenizer_cases.bin", "rb") as f:
    data = f.read()

offset = 0
cases = 0
mismatches = 0
involved = set()

while offset < len(data):
    (text_length,) = struct.unpack_from("<I", data, offset)
    offset += 4
    text = data[offset:offset + text_length].decode("utf-8")
    offset += text_length
    (n_ids,) = struct.unpack_from("<I", data, offset)
    offset += 4
    c_ids = list(struct.unpack_from(f"<{n_ids}i", data, offset))
    offset += 4 * n_ids

    hf_ids = hf.encode(text).ids          # adds <s> like the C code
    cases += 1

    if c_ids != hf_ids:
        mismatches += 1
        print(f"\nMISMATCH {text!r}")
        print(f"   C (GGUF): {c_ids}")
        print(f"   HF:       {hf_ids}")
        involved.update(set(c_ids) ^ set(hf_ids))

print(f"\nCases compared: {cases}, differ from Hugging Face: {mismatches}")

if involved:
    print("\nScores of the tokens that differ:")
    print(f"{'id':>6s}  {'piece':16s} {'GGUF score':>14s} {'tokenizer.bin':>14s}")
    for i in sorted(involved):
        b = f"{bin_scores[i]:14.1f}" if bin_scores else ""
        print(f"{i:6d}  {gguf_tokens[i]!r:16s} {gguf_scores[i]:14.1f} {b}")

print("\nHF TOKENIZER COMPARISON:", "MATCH" if mismatches == 0 else "DIFFERENT")
