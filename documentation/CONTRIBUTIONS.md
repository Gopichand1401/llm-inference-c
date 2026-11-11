# Attribution, our contributions and division of work

## 1. Sources we built on

| Source | Licence | What we used |
|---|---|---|
| [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy | MIT | the llama2.c checkpoint format (`.bin` header + FP32 weight layout), the `tokenizer.bin` format, the TinyStories models (stories15M/42M/110M), and the structure of the FP32 forward pass (`rmsnorm` / `softmax` / `matmul` and the layer loop in `transformer.c` follow `run.c`) |
| [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) and the [ggml](https://github.com/ggml-org/ggml) / [llama.cpp](https://github.com/ggml-org/llama.cpp) projects | MIT | the GGUF file layout and metadata keys, and the bit layouts of the Q4_0, Q8_0, Q4_K, Q5_K and Q6_K block formats. Our decoders and fused kernels are our own C implementation of those layouts |
| [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | Apache 2.0 | test model for the GGUF path, and its chat template |
| [SentencePiece](https://github.com/google/sentencepiece) | Apache 2.0 | the BPE algorithm our tokenizer reproduces (encode with a dummy prefix space, score-ordered merges, byte fallback) |
| Okapi BM25 (Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009) and the Lucene variant of idf | — | the ranking formula in `rag_index.c`; the implementation, short-form and typo rules and re-ranking are our own |
| Python reference tools: NumPy, [gguf-py](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py), openpyxl, Hugging Face `tokenizers`, `sentencepiece` | various open source | **test-time only**: independent reference implementations that our C output is compared against. None of them is linked into the library |

## 2. What we added beyond llama2.c

llama2.c is a single `run.c` file that runs FP32 `.bin` models. This
project goes further in the following ways:

1. **A reusable library with one API for two formats.** `engine.h` detects
   llama2.c or GGUF from the file contents and hides the difference. The
   code is split into 20 modules in `src/`, with `include/` headers, a
   static library and separate demo programs.
2. **A complete GGUF loader written from scratch:** header, metadata,
   tensor directory, alignment, LLaMA config keys, linear RoPE scaling and
   tied output weights. Every size is checked against overflow and the
   file length.
3. **Quantized inference without FP32 copies.** Our own decoders for 5
   GGML block types, and fused matrix-vector kernels that compute directly
   on the packed integers (with the reusable per-block input sums for
   K-quants). This is what lets TinyLlama 1.1B run in under 1 GB.
   Everything is verified bit for bit against gguf-py and NumPy.
4. **Grouped-query attention**, **multi-turn chat with KV-cache reuse**
   between turns, and recognition of special tokens for chat templates.
5. **An O(n log n) BPE tokenizer** (heap + linked list) instead of the
   O(n²) rescan. Same output, over 1 000× faster on a 20 KB prompt.
6. **Library-quality error handling**: no `exit()`, no printing, an error
   buffer on every call, and input validation on every public function.
7. **An extensive test suite** in all 7 categories from the brief, run
   under AddressSanitizer, UBSan and Valgrind, plus fuzzing and a
   mutation test (see `TESTING.md`).
8. **Question answering over Excel and CSV files (RAG)**: a self-written
   ZIP + XML reader (only zlib's `inflate` is external), table and header
   detection, BM25 search with typo correction and short forms ("ML" →
   Machine Learning), an exact answerer with follow-up questions, and
   prompts for the model within its context budget. It is checked against
   an independent Python implementation, 131 dialogue checks and a mutation
   test (see `RAG.md`).
9. **PocketLLM**: the whole system as a small C interface (`pocket.h`),
   a shared library and an installable Python package (`pip install .`),
   tested to give exactly the same output as the C programs.
10. **Measurement tooling**: `llm_bench`, `gguf_bench` and
   `run_benchmark.sh` (see `PERFORMANCE.md`).

## 3. Division of work

> **To fill in by the team.** Replace the names and adjust the lists. The
> statement asks for the split between the two students. The GitHub
> history (commits, issues, PRs) should match this table.

| Area | Student 1: *Name* | Student 2: *Name* |
|---|---|---|
| llama2.c loader, `Config` validation, run state | | |
| FP32 forward pass (RMSNorm, RoPE, attention, SwiGLU) | | |
| Tokenizer (llama2.c + GGUF vocabulary, special tokens, O(n log n) merges) | | |
| Samplers (greedy, temperature, top-k) | | |
| GGUF parser and model binding | | |
| Quantization formats and fused matvec kernels | | |
| GGUF forward pass, engine API | | |
| CLI (`llm_generate`, chat mode), showcase | | |
| XLSX reader (RAG step 1) | | |
| RAG chunking, BM25 index, answers, prompts (`rag_*.c`, `llm_rag`) | | |
| RAG tests (Python reference, dialogues, mutation test) | | |
| PocketLLM interface (`pocket.c`) and Python package | | |
| Tests, sanitizers, fuzzing, mutation testing | | |
| Benchmarks and documentation | | |
