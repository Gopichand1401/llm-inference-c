"""
compare_tokenizer_spm.py - check the C GGUF tokenizer against SentencePiece.

Run build/gguf_tokenizer_demo first; it writes build/tokenizer_cases.bin.

Usage:
    pip install sentencepiece protobuf     # once, inside .venv-gguf-ref
    python examples/07_tokenizer/compare_tokenizer_spm.py model.gguf

How the reference is built:
  Llama's original tokenizer is a SentencePiece BPE model. The GGUF file
  stores its pieces, scores and piece types, which is everything a
  SentencePiece model contains apart from its normalisation settings. This
  script rebuilds a SentencePiece ModelProto from that metadata, using
  Llama's settings (identity normalisation, dummy "▁" prefix, extra spaces
  kept, byte fallback), and lets Google's official SentencePiece library
  tokenize the same strings as the C code. The IDs must match exactly.

  Two rules are applied exactly as in the C code (gguf_tokenizer.c):
    - if every normal-token score is equal, scores are rebuilt as -id
      (-1e9 for pieces made only of '▁'),
    - special-token strings (<s>, </s>, <unk>, user-defined) in the text
      become their IDs and the text between them is encoded separately.
  SentencePiece itself checks the merge algorithm; compare_tokenizer_hf.py
  checks the final result against TinyLlama's official tokenizer.
"""
import struct
import sys

import sentencepiece as spm
from gguf.gguf_reader import GGUFReader
from sentencepiece import sentencepiece_model_pb2 as pb

if len(sys.argv) != 2:
    print("Usage: python compare_tokenizer_spm.py model.gguf")
    sys.exit(1)

reader = GGUFReader(sys.argv[1])


def field(key):
    return reader.fields.get(key)


def array_values(key):
    f = field(key)
    return [f.parts[i] for i in f.data]


model_name = bytes(field("tokenizer.ggml.model").parts[-1]).decode()
if model_name != "llama":
    raise RuntimeError(f"Tokenizer model is '{model_name}', not 'llama'")

tokens = [bytes(p).decode("utf-8") for p in array_values("tokenizer.ggml.tokens")]
scores = [float(p[0]) for p in array_values("tokenizer.ggml.scores")]
types = [int(p[0]) for p in array_values("tokenizer.ggml.token_type")]


def scalar(key, default):
    f = field(key)
    return int(f.parts[f.data[0]][0]) if f is not None else default


normal_scores = {s for s, t in zip(scores, types) if t == 1}
rebuilt = len(normal_scores) == 1
if rebuilt:
    scores = [-1e9 if set(p) == {"\u2581"} else -float(i)
              for i, p in enumerate(tokens)]

specials = sorted(((p, i) for i, (p, t) in enumerate(zip(tokens, types))
                   if t in (2, 3, 4) and p), key=lambda x: -len(x[0]))

bos = scalar("tokenizer.ggml.bos_token_id", 1)
eos = scalar("tokenizer.ggml.eos_token_id", 2)
unk = scalar("tokenizer.ggml.unknown_token_id", 0)

# ---- Build the SentencePiece model ----------------------------------------
proto = pb.ModelProto()
proto.trainer_spec.model_type = pb.TrainerSpec.BPE
proto.trainer_spec.vocab_size = len(tokens)
proto.trainer_spec.byte_fallback = True
proto.trainer_spec.unk_id = unk
proto.trainer_spec.bos_id = bos
proto.trainer_spec.eos_id = eos
proto.trainer_spec.pad_id = -1
proto.normalizer_spec.name = "identity"
proto.normalizer_spec.add_dummy_prefix = True
proto.normalizer_spec.remove_extra_whitespaces = False
proto.normalizer_spec.escape_whitespaces = True

type_map = {1: pb.ModelProto.SentencePiece.NORMAL,
            2: pb.ModelProto.SentencePiece.UNKNOWN,
            3: pb.ModelProto.SentencePiece.CONTROL,
            4: pb.ModelProto.SentencePiece.USER_DEFINED,
            5: pb.ModelProto.SentencePiece.UNUSED,
            6: pb.ModelProto.SentencePiece.BYTE}

for piece, score, ty in zip(tokens, scores, types):
    sp_piece = proto.pieces.add()
    sp_piece.piece = piece
    sp_piece.score = score
    sp_piece.type = type_map.get(ty, pb.ModelProto.SentencePiece.NORMAL)

sp = spm.SentencePieceProcessor()
sp.LoadFromSerializedProto(proto.SerializeToString())

print(f"SentencePiece reference: {sp.GetPieceSize()} pieces, "
      f"bos={bos} eos={eos} unk={unk}, scores rebuilt: {rebuilt}, "
      f"special tokens: {[p for p, _ in specials]}")


def encode_with_specials(text):
    """Split at the earliest (longest) special string, encode the rest."""
    ids = []
    while True:
        best = None
        for piece, idx in specials:
            pos = text.find(piece)
            if pos >= 0 and (best is None or pos < best[0]):
                best = (pos, piece, idx)
        segment = text if best is None else text[:best[0]]
        if segment:
            ids += sp.EncodeAsIds(segment)
        if best is None:
            return ids
        ids.append(best[2])
        text = text[best[0] + len(best[1]):]

# ---- Compare ---------------------------------------------------------------
with open("build/tokenizer_cases.bin", "rb") as f:
    data = f.read()

offset = 0
cases = 0
mismatches = 0

while offset < len(data):
    (text_length,) = struct.unpack_from("<I", data, offset)
    offset += 4
    text = data[offset:offset + text_length].decode("utf-8")
    offset += text_length
    (n_ids,) = struct.unpack_from("<I", data, offset)
    offset += 4
    c_ids = list(struct.unpack_from(f"<{n_ids}i", data, offset))
    offset += 4 * n_ids

    ref_ids = [bos] + encode_with_specials(text)
    cases += 1

    if c_ids != ref_ids:
        mismatches += 1
        print(f"MISMATCH {text!r}")
        print(f"   C:   {c_ids}")
        print(f"   SPM: {ref_ids}")
        print(f"   SPM pieces: {sp.EncodeAsPieces(text)}")

print(f"\nCases compared: {cases}, mismatches: {mismatches}")

if cases == 0 or mismatches:
    print("\nTOKENIZER SENTENCEPIECE COMPARISON: FAILED")
    sys.exit(1)

print("\nTOKENIZER SENTENCEPIECE COMPARISON: PASSED")
