# Testing

## Running the tests

```bash
make test                                 # all unit tests; no model file needed
make test MODEL=models/stories42M.bin     # + forward-pass tests on a real checkpoint
make asan  MODEL=models/stories42M.bin    # everything under AddressSanitizer + LeakSanitizer + UBSan
make valgrind MODEL=models/stories42M.bin # everything under Valgrind memcheck
bash tests/run_xlsx_checks.sh             # XLSX reader: openpyxl comparison, fuzzing, mutation test
make rag-test                             # RAG: 35 checks (Python reference, dialogues, mutation, sanitizers)
make pocket-test                          # Python package vs the C programs, also under sanitizers
bash tests/run_model_checks.sh models/gguf/tinyllama.gguf   # GGUF: every stage vs NumPy / gguf-py
bash tests/run_model_matrix.sh models/    # try every model file in a folder
```

Every command prints `PASS`/`FAIL` per check and exits non-zero on any
failure, so they can run in CI. `make test` needs only
`models/tokenizer.bin`, which is in the repository. `test_engine` creates
its own small model files with random weights in a temporary folder, so
the forward pass, KV cache and error paths are all tested without
downloading anything.

## What is tested, by category

The project brief asks for seven categories. Each line is one check, or a
group of checks for the loops.

### Normal cases
| Test | File |
|---|---|
| llama2.c model loads; config, format, KV-cache size, description and tokenizer are correct | `test_engine.c` |
| forward pass gives finite logits; two models given the same token produce bit-identical logits | `test_engine.c` |
| prompt + greedy generation fills the whole context | `test_engine.c` |
| unshared classifier, multi-query attention (1 KV head), plain MHA, GQA | `test_engine.c` |
| tokenizer round-trip `"hello world!"`, BOS/EOS handling | `test_tokenizer.c` |
| greedy / temperature / top-k return valid tokens | `test_sampler.c` |
| RMSNorm, softmax, matmul against hand-computed values | `test_math.c` |
| forward pass on a real checkpoint; logits all finite | `test_forward.c` |
| every cell of 5 workbooks identical to openpyxl (80 004 cells in `big.xlsx`) | `run_xlsx_checks.sh` |
| GGUF: every quant type decodes bit-exactly; matvec and full forward pass match NumPy; TinyLlama answers "Paris" | `run_model_checks.sh` |

### Boundary cases
| Test | File |
|---|---|
| position `seq_len − 1` accepted, `seq_len` refused; token `0` and `vocab − 1` accepted, `vocab` refused | `test_engine.c`, `test_validation.c` |
| `max_context = 1`; `max_context` above the file's own is capped | `test_engine.c` |
| smallest possible model (dim 2, 1 layer, 1 head, context 1) | `test_engine.c` |
| token buffer exactly large enough / one too small; text buffer the same, with a canary byte showing nothing is written past the end | `test_edge_cases.c` |
| single character, single space, runs of spaces, tab/newline, 2/3/4-byte UTF-8, control bytes (byte fallback) | `test_edge_cases.c` |
| vocabulary of 1; ties in greedy; temperature → 0 behaves like greedy; `k = 1` and `k = V`; rng state 0 | `test_edge_cases.c` |
| softmax of ±1000 (no overflow), RMSNorm of a zero vector (no NaN), 1×1 matmul | `test_math.c` |

### Invalid input
| Test | File |
|---|---|
| `NULL` options / path / empty path / negative context / `NULL` or 1-byte error buffer | `test_engine.c` |
| every engine function given a `NULL` model | `test_engine.c` |
| 12 corrupt headers: zero/negative sizes, `dim % heads ≠ 0`, `kv_heads > heads`, `heads % kv_heads ≠ 0`, overflowing sizes; each refused **with a message** | `test_engine.c` |
| random bytes; `GGUF` magic with nothing after it | `test_engine.c` |
| invalid config fields in memory (odd head size, …) | `test_validation.c` |
| tokenizer: `NULL` everywhere, `vocab_size` 0 or too large, absurd lengths, token −1 | `test_edge_cases.c` |
| sampler: `NULL`, size 0, NaN/inf logits, temperature ≤ 0 / NaN / inf, `k` 0 or > V, `NULL` rng | `test_edge_cases.c`, `test_sampler.c` |
| XLSX: `NULL` path, fake ZIP, `NULL` book/row in every accessor | `test_edge_cases.c` |
| XLSX: 200 corrupted + 200 fuzzed workbooks, none may crash (under ASan) | `run_xlsx_checks.sh` |

### Empty input
| Test | File |
|---|---|
| 0-byte model file; header with no weights; 0-byte tokenizer | `test_engine.c` |
| empty prompt → `[BOS]` → forward still works | `test_engine.c` |
| encode `""` with/without BOS, into a 0-capacity buffer; decode 0 tokens | `test_edge_cases.c`, `test_tokenizer.c` |
| 0-byte `.xlsx` | `test_edge_cases.c` |

### Large input
| Test | File |
|---|---|
| 512 positions: full KV cache, all logits finite | `test_engine.c` |
| 20 KB prompt → more tokens than the context; writing past the context is refused | `test_engine.c` |
| 200 KB text round-trips in under 5 s (it took minutes before the O(n log n) tokenizer); 50 000 identical characters | `test_edge_cases.c` |
| 1 048 576-entry logits for all three samplers | `test_edge_cases.c` |
| 129 × 257 matmul against a double-precision reference | `test_math.c` |
| 80 004-cell workbook | `run_xlsx_checks.sh` |

### Failure conditions
| Test | File |
|---|---|
| missing model, a directory instead of a file | `test_engine.c` |
| weights truncated at 4 points (1 float … total − 1) | `test_engine.c` |
| missing / truncated tokenizer; model and tokenizer vocabulary mismatch | `test_engine.c` |
| model still usable after rejected calls | `test_engine.c`, `test_validation.c` |
| **mutation test:** 14 bugs are planted in the XLSX reader one at a time, and the test suite must catch every one | `run_xlsx_checks.sh` |

### Memory management
| Test | File |
|---|---|
| 25 open → forward → close cycles | `test_engine.c` |
| two models open at once keep separate buffers | `test_engine.c` |
| 30 failing opens (header-only, truncated, missing); each must free its partial allocations | `test_engine.c` |
| `free` on `NULL`, on a zeroed struct and twice (tokenizer, engine, XLSX) | `test_edge_cases.c`, `test_engine.c` |
| **all of the above under AddressSanitizer + LeakSanitizer + UBSan** (`make asan`) and **Valgrind** (`make valgrind`): 0 leaks, 0 invalid reads/writes, 0 undefined behaviour | Makefile |

## RAG and library-interface suites

These come with the RAG / PocketLLM work. They need `pip install openpyxl
gguf numpy` and generate tiny random GGUF and llama2.c models themselves, so
they need no download. Details are in [RAG.md](RAG.md#4-tests).

| Suite | Checks | Categories covered |
|---|---|---|
| `run_rag_checks.sh` | 35 checks: C vs an independent Python reference (chunks, terms, BM25 scores byte for byte), 35 retrieval questions, **131 dialogue checks** in 14 conversations, live folder changes (add / edit / delete / broken / lock file), the three `llm_rag` modes under ASan + UBSan, context-budget limits, the right row with `-k 1`, the `--compact` prompt, **mutation test (35 planted bugs, all caught)** | normal, boundary (context budget), invalid (broken files, off-topic questions, typos), empty (nothing found), large (all of `data/`), failure, memory |
| `run_pocket_checks.sh` | 16 checks: the Python package gives exactly the same text as `llm_generate` / `llm_rag` for every sampler and chat format, 67 RAG replies, bad arguments raise errors, an over-long prompt is refused; then all of it again with an ASan + UBSan build of the shared library | normal, invalid, large, failure, memory |

## Results

On the build machine (gcc 13.3, Ubuntu 24.04):

```
make test       PASS  sampler, tokenizer, math (9 cases), engine (78), edge cases (69),
                      validation (15), forward
make asan       PASS  same tests, no sanitizer reports, no leaks
make valgrind   PASS  same tests, 0 errors, 0 definite leaks
run_xlsx_checks PASS  14/14 (incl. 14/14 planted bugs detected)
make rag-test   PASS  35/35 (131/131 dialogue checks, 35/35 mutations caught)
make pocket-test PASS 16/16, and 16/16 again under AddressSanitizer + UBSan
```

## Bugs these tests found and we fixed

| Bug | Found by | Fix |
|---|---|---|
| Encoding was O(n²): 20 KB took 15 s, 200 KB took minutes | "large input" test | heap-based BPE merges, O(n log n), output identical (checked on 3 003 inputs) |
| `tokenizer_encode` failed when the buffer was exactly the size of the result (it staged unmerged characters in the caller's buffer) | "buffer exactly large enough" test | private staging array; the caller's buffer only holds the result |
| XLSX accessors read out of bounds for a bad sheet/row index or a `NULL` book | invalid-input review | bounds checks that return `NULL` / `0` / `""` |
| `xlsx_open(NULL)` passed `NULL` to `fopen()` (undefined behaviour; glibc happened to fail safely) | Valgrind run of the invalid-input tests | path checked first, error message returned |
| A count/list question asked right after an answer ("How many courses are in slot D?") was treated as a follow-up and printed the previous row's slot | trying the RAG demo while merging | count/list questions and plural nouns skip the follow-up rule; a regression conversation was added (the old code fails it) |
| With `--rag -k 1` the model got the CSE sheet's "Check DS sheet" stub for DS3010 instead of the full row, so it said it had no instructor information | running `--rag --show-context` on the real data | duplicates are ordered before the cut to `k` rows; new check in `run_rag_checks.sh` |
| `make test` failed on a fresh clone (looked for `tokenizer.bin` in the wrong folder) | fresh-clone run | tests take the tokenizer path; the default is `models/tokenizer.bin` |
| `test_math.c` declared `matmul` with a different argument order than `transformer.h` | review | includes the real header |
