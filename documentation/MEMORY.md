# Memory management

This document answers the four questions from the project brief for every
public API: **who allocates, who owns, who frees, and what happens when an
operation fails.** The second half gives the memory each model actually
uses.

## 1. General rules

These hold for the whole library. Section 2 lists the few exceptions.

1. **Opaque handles: the library allocates, the caller owns.**
   `engine_open()`, `llm_load()` and `xlsx_open()` return a heap object.
   The caller owns it and must pass it to the matching close/free function
   exactly once. Each of those functions is safe on `NULL`.
2. **Caller-provided structs: the caller allocates, the library fills.**
   `Tokenizer`, `Transformer`, `GGUFModel` and `GGUFTransformer` can live on
   the stack or the heap. They must start zero-initialised (`= {0}`).
   The library allocates their *contents*, and the caller frees them with the
   matching `*_free()`. Calling `*_free()` on a zeroed or already freed
   struct is safe, because each free resets the struct to zero.
3. **Output buffers: the caller allocates.** `tokenizer_encode()`,
   `tokenizer_decode()`, `gguf_weight_dequantize_row()` and `gguf_matvec()`
   write into memory the caller passes, along with its capacity. They never
   write past the given capacity. If the result does not fit, they return
   an error instead of truncating silently.
4. **Returned pointers into a handle are borrowed.** Logits, `Config`,
   `Tokenizer *`, description strings, sheet names and cell text belong to the
   object they came from. They stay valid until that object is closed, and the
   caller must not free them. Logits are also **overwritten by the next
   forward call**, so copy them if you need them later.
5. **Model weights are memory-mapped, not copied.** Both formats `mmap()`
   the model file read-only. The OS loads weight pages on first use and can
   drop them again under memory pressure. The mapping is released by
   `engine_close()` / `free_transformer()` / `gguf_model_free()`.
6. **Errors: nothing leaks and nothing is half-built.** Every function that
   can fail returns `NULL`, `0` or `-1` (see the table) and writes a
   message into the caller's `char *error` buffer when one is given
   (`NULL` or capacity `0` is allowed). Before returning a failure, a
   function frees everything it allocated during that call. The caller is
   left with either a fully valid object or nothing to free. The library
   never calls `exit()` or `abort()` and never prints anything.

All of this is checked automatically. `make asan` runs every C test with
AddressSanitizer + LeakSanitizer + UBSan, and `make valgrind` runs every
test under Valgrind's leak checker. The tests include 25 open/close cycles,
repeated failing opens (missing, truncated and corrupt files) and buffers
that are exactly the right size or one byte short (see `TESTING.md`).

## 2. Ownership by function

### Engine (`engine.h`): the main API

| Function | Allocates | Owner | Freed by | On failure |
|---|---|---|---|---|
| `engine_open(opts, err, cap)` | `LLMModel`, run state, KV cache, tokenizer, file mapping | caller | `engine_close()` | returns `NULL` + message; all partial allocations freed, file unmapped and closed |
| `engine_close(m)` | — | — | — | safe on `NULL` |
| `engine_forward(m, tok, pos)` | nothing (reuses buffers) | **model** (logits) | `engine_close()` | returns `NULL` for a bad token or position; the model stays usable |
| `engine_config(m)` | — | model | — | `NULL` for a `NULL` model |
| `engine_tokenizer(m)` | — | model; do **not** call `tokenizer_free()` on it | `engine_close()` | `NULL` for a `NULL` model |
| `engine_description(m)` | — | model | `engine_close()` | — |

### Tokenizer (`tokenizer.h`)

| Function | Allocates | Owner | Freed by | On failure |
|---|---|---|---|---|
| `tokenizer_load(t, path, vocab, …)` / `tokenizer_load_gguf(t, path, …)` | vocabulary strings, scores, sorted index, special-token list (inside the caller's `Tokenizer`) | caller | `tokenizer_free(t)` | returns `-1` + message; `t` is left zeroed, nothing to free |
| `tokenizer_free(t)` | — | — | — | safe on `NULL`, on a zeroed struct and when called twice |
| `tokenizer_encode(t, text, …, tokens, cap, &n, …)` | a temporary working array, freed before returning | caller (`tokens`) | caller | returns `-1` if the result needs more than `cap` tokens; nothing is written past `cap` |
| `tokenizer_decode(t, ids, n, out, cap, …)` | nothing | caller (`out`) | caller | returns `-1` for an invalid ID or if `out` is too small; never writes past `cap` |

### Sampler (`sampler.h`)

`sampler_greedy()`, `sampler_temperature()` and `sampler_top_k()` read the
caller's logits and return a token ID, or `-1` for invalid input (`NULL`,
size 0, NaN/inf, temperature ≤ 0, `k` out of range). `sampler_top_k()`
allocates a `k`-entry index array internally and frees it before
returning. The RNG state (`uint32_t`) belongs to the caller.

### Low-level llama2.c path (`llm.h`, `transformer.h`)

| Function | Allocates | Owner | Freed by | On failure |
|---|---|---|---|---|
| `llm_load(path, err, cap)` | `Transformer` + everything below | caller | `llm_free_model()` | `NULL` + message; all partial allocations freed |
| `build_transformer(t, path, …)` | file mapping, run state, tensor registry (inside the caller's `Transformer`) | caller | `free_transformer(t)` | returns `0` + message; frees what it built before returning (calling `free_transformer(t)` again is still safe) |
| `malloc_run_state(s, cfg, …)` | activation buffers + KV cache | caller | `free_run_state(s)` | returns `0`; buffers allocated before the failure are freed |
| `forward(t, tok, pos)` | nothing | model (`t->state.logits`) | — | `NULL` for invalid input or an invalid configuration |

### GGUF path (`gguf_model.h`, `gguf_forward.h`, `checkpoint.h`)

| Function | Allocates | Owner | Freed by | On failure |
|---|---|---|---|---|
| `gguf_model_load(m, path, …)` | file mapping, tensor list + names, layer array | caller | `gguf_model_free(m)` | returns `0` + message; `m` is left freed and zeroed |
| `gguf_transformer_init(t, path, …)` | model above + run state + decoded FP32 RMSNorm weights | caller | `gguf_transformer_free(t)` | returns `0` + message; safe to free a partly initialised `t` |
| `gguf_forward(t, tok, pos)` | nothing | model | — | `NULL` for invalid input |
| `gguf_read_tensors(path, &list, &n, …)` | array + one string per name | caller | `gguf_free_tensors(list, n)` | returns `0`, `*list = NULL` |
| `gguf_get_tensor_data(map, …, &ptr, &size)` | nothing; `ptr` points into the caller's mapping | caller's mapping | — | returns `0` if the tensor would lie outside the file |
| `gguf_weight_dequantize_row()` / `gguf_matvec()` | nothing | caller (`out`) | caller | returns `0` for a bad row, `NULL` or an unsupported type |

`GGUFWeight.data` and `GGUFWeight.name` are borrowed pointers into the
mapping and the tensor list. They are valid until `gguf_model_free()`.

### XLSX reader (`xlsx_reader.h`)

| Function | Allocates | Owner | Freed by | On failure |
|---|---|---|---|---|
| `xlsx_open(path, err, cap)` | the whole parsed workbook: sheets, rows, cell strings | caller | `xlsx_close()` | `NULL` + message; everything freed |
| `xlsx_sheet_name()`, `xlsx_row()`, `xlsx_cell()`, `xlsx_row_count()` | — | book (borrowed) | `xlsx_close()` | a `NULL` book or out-of-range index gives `NULL` (name, row), `""` (cell) or `0` (counts), never a crash |

### RAG (`rag_chunk.h`, `rag_index.h`, `rag_answer.h`, `rag_prompt.h`)

| Function | Allocates | Owner | Freed by | On failure |
|---|---|---|---|---|
| `rag_corpus_load(inputs, n, opts, err, cap)` | every file's tables, chunks, fields and strings | caller | `rag_corpus_free()` | `NULL` + message; an unreadable single file is only a warning, and the others still load |
| `rag_corpus_chunk()`, `rag_corpus_file()`, `rag_corpus_warning()` | — | corpus (borrowed) | `rag_corpus_free()` | `NULL` for an out-of-range index |
| `rag_index_build(corpus, opts, err, cap)` | term dictionary, posting lists, lengths | caller; must not outlive the corpus | `rag_index_free()` | `NULL` + message |
| `rag_index_search(ix, q, hits, max)` | nothing | caller (`hits` array) | caller | returns the number of hits written (≤ `max`) |
| `rag_session_new()` | conversation state | caller | `rag_session_free()` | `NULL` when out of memory |
| `rag_prompt_system()`, `rag_prompt_question()`, `rag_render_chunk()` | the prompt text (`malloc`) | **caller** | `free()` | `NULL` (+ message for `rag_prompt_question`) |

### Library interface (`pocket.h`)

| Function | Allocates | Owner | Freed by | On failure |
|---|---|---|---|---|
| `pocket_model_open()` / `pocket_rag_open()` | handle + engine/corpus/index/session | caller | `pocket_model_close()` / `pocket_rag_close()` | `NULL` + message in the caller's buffer |
| `pocket_model_generate/chat()`, `pocket_rag_ask*()`, `pocket_rag_hit_*()` | result text inside the handle | **handle**; valid until the next call on it | the handle | `NULL` / `-1`; the message is in `pocket_*_error()` |

The Python package copies every returned string into a Python string
straight away, so Python code never sees a pointer that a later call could
invalidate. `make pocket-test` runs the whole Python test under
AddressSanitizer + UBSan.

## 3. Memory used by a model

Total memory ≈ **weights (mapped) + KV cache + activations + tokenizer**.

| Part | Formula | Notes |
|---|---|---|
| Weights | model file size | memory-mapped; shared with the OS page cache, not copied. GGUF weights stay quantized (Q4_K ≈ 4.5 bits/weight) and are decoded row by row inside `gguf_matvec()` |
| KV cache | `2 × layers × context × kv_dim × 4` bytes | `kv_dim = dim × n_kv_heads / n_heads`. Shrink it with `LLMOptions.max_context` / `--ctx` |
| Activations | `(4·dim + 2·hidden + 2·kv_dim + n_heads·context + vocab) × 4` bytes | one token's working buffers |
| Tokenizer | ≈ 1–2 MB for 32 000 tokens | strings + scores + sorted index |

Worked out for the supported models (full context, FP32 KV cache):

| Model | Weights | Context | KV cache | Activations | Total ≈ |
|---|---:|---:|---:|---:|---:|
| stories15M (llama2.c FP32) | 58 MB | 256 | 3.4 MB | 0.14 MB | 62 MB |
| stories42M (llama2.c FP32) | 159 MB | 1024 | 32 MB | 0.17 MB | 192 MB |
| stories110M (llama2.c FP32) | 418 MB | 1024 | 72 MB | 0.19 MB | 491 MB |
| TinyLlama-1.1B-Chat Q4_K_M (GGUF) | ≈ 669 MB | 2048 | 88 MB | 0.45 MB | ≈ 760 MB |

The TinyLlama KV cache is small for a 1.1B model because of grouped-query
attention: it has 4 KV heads against 32 query heads, so `kv_dim` is 256
instead of 2048. Without GQA the cache would be 8× larger (704 MB).

If the TinyLlama weights were converted to FP32 up front, they alone would
need about 4.4 GB. Keeping them quantized and computing directly on the
quantized blocks is what lets a 1.1B model run in under 1 GB.

Measured peak RSS (resident memory) for each model is in
`PERFORMANCE_RESULTS.md` (generated by `make bench`). `llm_bench` prints
the same breakdown for any single model file.
