# Wild Pointers: LLaMA-style LLM inference in C

A C11 inference engine that loads a small transformer language model from a
file and generates text on the CPU, without training anything. It can also
answer questions about Excel/CSV files (RAG), and can be used from Python. It runs
Karpathy's **llama2.c** checkpoints (FP32) and **GGUF** models with
**quantized** weights (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K), such as
TinyLlama-1.1B-Chat, through one library API.

```bash
make
./build/llm_generate models/stories42M.bin "Once upon a time"          # story completion
./build/llm_generate models/gguf/tinyllama.gguf -i # multi-turn chat
```

## Features

- **The full inference pipeline**, implemented in C: model loading, BPE
  tokenizer, token embedding lookup, transformer layers, multi-head
  attention with **grouped-query attention** and a **KV cache**, SwiGLU
  feed-forward network, RMSNorm, rotary position embeddings, logits,
  sampling and autoregressive generation.
- **Three samplers**: greedy, temperature and top-k, with seeded and
  reproducible random numbers.
- **Two model formats, one API**: `engine_open()` detects llama2.c or GGUF
  from the file contents.
- **Quantized inference without FP32 copies**: fused kernels compute directly
  on the 4- and 6-bit blocks, so TinyLlama 1.1B runs in under 1 GB of RAM.
- **Interactive CLI**: completion, a one-shot chat turn, and multi-turn
  chat that reuses the KV cache between turns.
- **A library, not one big `main.c`**: 20 modules, public headers in
  `include/`, a static library `libllm.a`, separate demo programs.
- **Robust**: every public function validates its input and reports errors
  through a return value plus a message buffer. Corrupt or truncated files
  are refused, never read out of bounds.
- **Tested**: 7 test programs plus scripts covering normal, boundary,
  invalid, empty, large-input, failure and memory cases. They run clean under
  **AddressSanitizer, UBSan and Valgrind**, with fuzzing and a mutation test.
- **Answers from Excel and CSV files (RAG)**: chunking, BM25 search with
  typo and short-form handling, and answers either straight from the rows
  or written by the model from the retrieved rows. The folder is
  re-checked live while the program runs.
- **PocketLLM: a library interface and Python package**: `pocket.h` and
  `libpocketllm.so` expose the model and RAG as a few calls;
  `import pocketllm` uses them from Python.

## Quick start

Requirements: Linux, macOS or **WSL on Windows**; `gcc` (C11, OpenMP);
`make`; `zlib` (`sudo apt install build-essential zlib1g-dev`).

```bash
make                     # library + llm_generate, llm_showcase, llm_bench, llm_demo
make test                # unit tests (needs only models/tokenizer.bin, included)
sh run.sh                # quick start: builds, then chats with models/stories42M.bin
```

### Get a model
Model files are not in the repository. Put them in `models/`:

| Model | Size | Download |
|---|---:|---|
| stories15M / 42M / 110M (llama2.c) | 58 / 159 / 418 MB | [huggingface.co/karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas) |
| TinyLlama-1.1B-Chat Q4_K_M (GGUF) | 669 MB | [huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF), saved as `models/gguf/tinyllama.gguf` |

```bash
wget -P models https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
```

### Run it
```bash
# story completion (llama2.c model; models/tokenizer.bin is found automatically)
./build/llm_generate models/stories42M.bin "Once upon a time"

# choose the sampler
./build/llm_generate models/stories42M.bin --sampler greedy "The dog"
./build/llm_generate models/stories42M.bin --sampler temperature --temperature 0.9 --seed 7 "The dog"
./build/llm_generate models/stories42M.bin --sampler top-k --top-k 20 "The dog"

# chat with a GGUF chat model: one turn, or interactive multi-turn (/reset, /exit)
./build/llm_generate models/gguf/tinyllama.gguf --chat "What is the capital of France?"
./build/llm_generate models/gguf/tinyllama.gguf -i

# a tour of every library feature, and speed/memory measurements
./build/llm_showcase models/stories42M.bin models/tokenizer.bin
./build/llm_bench models/stories42M.bin
make bench               # table for every model in models/
```

### Ask questions about spreadsheets (RAG)
```bash
make rag-tools
./build/llm_rag --retrieve-only                          # answers straight from data/*.xlsx
./build/llm_rag models/gguf/tinyllama.gguf --rag -k 1 --compact -n 80   # the model writes the answer
./build/llm_rag --retrieve-only --docs my_folder         # any folder of .xlsx / .csv
```
```
> Who is the instructor for ML?
Swapnil Hingmire
Source: 04_UG_DS_Aug-Dec-2026.xlsx, tab Core, row 10 (B. Tech. Data Science Core Courses)
Do you want the slot?
```

### From Python
```bash
make lib && export POCKETLLM_LIB=$PWD/build/libpocketllm.so PYTHONPATH=$PWD/python
# or install it: pip install .
```
```python
import pocketllm
model = pocketllm.Model("models/gguf/tinyllama.gguf")
print(model.chat("What is machine learning?", sampler="greedy"))
print(pocketllm.Rag("data").ask("Who is the instructor for ML?").text)
```

### Use it as a library
```c
#include "engine.h"
#include "sampler.h"

LLMOptions opt = { .model_path = "models/stories42M.bin" };
char err[256];
LLMModel *m = engine_open(&opt, err, sizeof err);        /* NULL + err on failure */
float *logits = engine_forward(m, token, pos);           /* owned by m */
int next = sampler_top_k(logits, engine_config(m)->vocab_size, 40, 0.8f, &rng);
engine_close(m);
```
The full program is [`examples/01_apps/minimal.c`](examples/01_apps/minimal.c).
Compile your own with:
`gcc -Iinclude prog.c build/libllm.a -lm -fopenmp`.

## Documentation

| Document | Contents |
|---|---|
| [ARCHITECTURE.md](documentation/ARCHITECTURE.md) | pipeline, module layers, design decisions, **data structures**, how to extend |
| [RAG.md](documentation/RAG.md) | question answering over Excel/CSV: pipeline, BM25, answer rules, tests |
| [POCKETLLM.md](documentation/POCKETLLM.md) | the `pocket.h` library interface and the `pocketllm` Python package |
| [ALGORITHMS.md](documentation/ALGORITHMS.md) | RMSNorm, RoPE, attention/GQA, SwiGLU, quantized matvec, BPE, sampling, generation |
| [API.md](documentation/API.md) | every public function, its errors, and the CLI options |
| [MEMORY.md](documentation/MEMORY.md) | **who allocates / owns / frees**, what happens on failure, memory per model |
| [PERFORMANCE.md](documentation/PERFORMANCE.md) | method, **tokens/second**, optimisations; [results table](documentation/PERFORMANCE_RESULTS.md) |
| [TESTING.md](documentation/TESTING.md) | every test case by category, sanitizer runs, bugs found |
| [CONTRIBUTIONS.md](documentation/CONTRIBUTIONS.md) | attribution, what we added, **division of work** |

## Project layout
```
include/    public headers: engine.h (main API), tokenizer.h, sampler.h,
            transformer.h, llm.h, gguf_*.h, checkpoint.h, xlsx_reader.h,
            rag_*.h (RAG), pocket.h (library interface)
src/        the library (20 modules) → build/libllm.a, build/libpocketllm.so
examples/   01_apps (CLI, showcase, minimal) · 02–07 step-by-step GGUF demos
            · 08_benchmark · 09_rag (llm_rag, rag_inspect, xlsx_dump)
            · 10_library (pocket_chat, pocket_demo)
python/     the pocketllm Python package (setup.py, pyproject.toml)
data/       course timetables (.xlsx) used by the RAG demos and tests
tests/      C unit tests, sanitizer/fuzz/mutation scripts, Python references
models/     tokenizer.bin (model files go here; not committed)
documentation/
```

## Make targets

| Command | What it does |
|---|---|
| `make` | library and main programs |
| `make test [MODEL=…]` | all tests (forward-pass tests need `MODEL`) |
| `make asan` / `make valgrind` | the tests under AddressSanitizer + UBSan / Valgrind |
| `make bench [MODELS="…"]` | speed + memory table → `documentation/PERFORMANCE_RESULTS.md` |
| `make examples` | every example program (checks that they all still compile) |
| `make rag-tools` | `llm_rag`, `rag_inspect`, `xlsx_dump` |
| `make lib` | `build/libpocketllm.so` for Python / other languages |
| `make rag-test`, `make pocket-test` | RAG checks (35) and Python-vs-C checks (16); need `pip install openpyxl gguf numpy` |
| `make gguf-tools` | the GGUF step-by-step demos |
| `make NATIVE=1` | build for this CPU (AVX2 etc.) |
| `make clean` | remove `build/` |

## Limitations

- **CPU only**, one sequence at a time. The prompt is processed token by
  token (no batched prompt evaluation), so prompt speed ≈ generation speed.
- **LLaMA architecture only**: llama2.c checkpoints and GGUF files with
  `general.architecture = llama` (LLaMA 1/2, TinyLlama, and similar). Other
  architectures (Mistral sliding window, Qwen, Phi, Gemma …) are not
  supported.
- **GGUF quant types**: F32, F16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K. Files using
  others (Q2_K, Q3_K, Q5_0/1, IQ*) are refused with a message naming the
  type, so prefer `Q4_K_M` files.
- **Tokenizer**: SentencePiece-style BPE (LLaMA 1/2 vocabularies). The
  tiktoken-style BPE of LLaMA 3 is not implemented.
- **Sampling**: greedy, temperature and top-k. Top-p and repetition
  penalty are not implemented yet.
- **Chat template**: TinyLlama/Zephyr format only.
- **KV cache in FP32**, with a context capped at 4096 by default (`--ctx`).
- **POSIX `mmap`**: native Windows is not supported; use WSL.
- **RAG**: keyword retrieval (BM25), not embeddings, so a question must
  share words, codes or short forms with the rows. `.xlsx` and `.csv` only
  (no encrypted workbooks, ZIP64, `.xls` or PDF). A 1.1B model cannot
  count over many rows, so lists and counts come from `--retrieve-only`.

## Team

**Wild Pointers.** See [CONTRIBUTIONS.md](documentation/CONTRIBUTIONS.md)
for who did what, and for the credits to llama2.c, GGUF/ggml and the
reference tools used in testing.

Licence: MIT (see [LICENSE](LICENSE)).
