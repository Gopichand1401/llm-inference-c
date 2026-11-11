# API reference

All headers are in `include/`. Link against `build/libllm.a` with
`-lm -fopenmp` (add `-lz` when you use the XLSX reader).

```bash
gcc -Iinclude my_program.c build/libllm.a -lm -fopenmp -o my_program
```

**Conventions used by every module**
- Functions that can fail take `char *error, size_t error_capacity` as
  their last two arguments. On failure they write a one-line message there.
  `NULL` / `0` is allowed if you don't want the message.
- Return values: handles return `NULL` on failure; `int` functions return
  `0`/`-1` (tokenizer) or `1`/`0` (loaders) as documented below.
- Ownership: see [`MEMORY.md`](MEMORY.md) for the full table.

The complete minimal program is
[`examples/01_apps/minimal.c`](../examples/01_apps/minimal.c)
(`make build/minimal`).

---

## `engine.h`: the main API (any model format)

```c
typedef struct {
    const char *model_path;      /* required: .bin (llama2.c) or .gguf      */
    const char *tokenizer_path;  /* NULL = auto (next to model, then ./)    */
    int max_context;             /* 0 = model's own, capped at 4096          */
} LLMOptions;

LLMModel   *engine_open(const LLMOptions *options, char *error, size_t cap);
void        engine_close(LLMModel *model);
float      *engine_forward(LLMModel *model, int token, int pos);
const Config *engine_config(const LLMModel *model);
Tokenizer  *engine_tokenizer(LLMModel *model);
LLMFormat   engine_format(const LLMModel *model);      /* LLM_FORMAT_STORIES / _GGUF */
int         engine_file_context(const LLMModel *model);
size_t      engine_kv_cache_bytes(const LLMModel *model);
const char *engine_description(const LLMModel *model);
int         engine_is_stop_token(const LLMModel *model, int token);
```

| Function | Description | Errors |
|---|---|---|
| `engine_open` | Opens a model. The format is detected from the file contents. Loads the tokenizer (from the GGUF metadata, or `tokenizer.bin` for llama2.c models) and allocates the run state and a KV cache for `max_context` positions | `NULL` + message: missing/unreadable file, bad header, truncated weights, unsupported quant type, missing tokenizer, vocabulary mismatch, negative `max_context`, out of memory |
| `engine_close` | Frees the model, its buffers and its tokenizer, and unmaps the file | safe on `NULL` |
| `engine_forward` | Runs one token at position `pos`. Positions must be fed in order 0, 1, 2, … Returns `vocab_size` logits, valid until the next call | `NULL` if `token ∉ [0, vocab)`, `pos ∉ [0, seq_len)` or the model is `NULL`. The model stays usable |
| `engine_config` | The configuration in use. `seq_len` is the allocated (possibly capped) context | `NULL` for `NULL` |
| `engine_tokenizer` | The model's tokenizer, for `tokenizer_encode` / `decode`. Do not free it | `NULL` for `NULL` |
| `engine_file_context` | The context length stored in the file, before any cap | `0` for `NULL` |
| `engine_kv_cache_bytes` | `2 × layers × seq_len × kv_dim × 4` | `0` for `NULL` |
| `engine_description` | e.g. `"llama2.c FP32, shared classifier"` or `"GGUF, weights Q4_K x135 Q6_K x21 F32 x45"` | — |
| `engine_is_stop_token` | 1 for EOS (and BOS for llama2.c story models) | — |

## `tokenizer.h`: SentencePiece BPE

```c
int  tokenizer_load(Tokenizer *t, const char *path, size_t vocab_size, char *err, size_t cap);
int  tokenizer_load_gguf(Tokenizer *t, const char *gguf_path, char *err, size_t cap);
void tokenizer_free(Tokenizer *t);
int  tokenizer_encode(Tokenizer *t, const char *text, int add_bos, int add_eos,
                      int *tokens, size_t token_capacity, size_t *token_count,
                      char *err, size_t cap);
int  tokenizer_decode(const Tokenizer *t, const int *tokens, size_t token_count,
                      char *output, size_t output_capacity, char *err, size_t cap);
```

| Function | Description | Returns |
|---|---|---|
| `tokenizer_load` | Loads a llama2.c `tokenizer.bin`. `vocab_size` must match the model | `0` OK, `-1` on error (`t` left empty) |
| `tokenizer_load_gguf` | Loads the vocabulary, scores, token types and BOS/EOS IDs from GGUF metadata | `0` / `-1` |
| `tokenizer_free` | Frees the contents and zeroes `*t`. Safe on `NULL`, on a zeroed struct and when called twice | — |
| `tokenizer_encode` | UTF-8 text → token IDs (see `ALGORITHMS.md` §8). The buffer only needs room for the **final** tokens. O(n log n) | `0`, or `-1` for `NULL` arguments or a full buffer (nothing written past `token_capacity`) |
| `tokenizer_decode` | Token IDs → NUL-terminated text. Drops BOS/EOS, and the leading space after BOS | `0`, or `-1` for an invalid ID or a too-small buffer |

## `sampler.h`

```c
int sampler_greedy(const float *logits, size_t vocab_size);
int sampler_temperature(const float *logits, size_t vocab_size, float temperature, uint32_t *rng);
int sampler_top_k(const float *logits, size_t vocab_size, size_t k, float temperature, uint32_t *rng);
```
All three return a token ID, or `-1` for invalid input: `NULL` or empty
logits, a non-finite logit, temperature ≤ 0 or non-finite, `k ∉ [1, V]`, or
a `NULL` rng. `*rng` is the caller's xorshift32 state; seed it with any
value (0 is replaced internally). `sampler_top_k(…, k = 1, …)` is greedy.

## `xlsx_reader.h`: spreadsheet input (RAG)

```c
XlsxBook      *xlsx_open(const char *path, char *err, size_t cap);
void           xlsx_close(XlsxBook *book);
int            xlsx_sheet_count(const XlsxBook *book);
const char    *xlsx_sheet_name(const XlsxBook *book, int sheet);
int            xlsx_sheet_hidden(const XlsxBook *book, int sheet);
int            xlsx_row_count(const XlsxBook *book, int sheet);
const XlsxRow *xlsx_row(const XlsxBook *book, int sheet, int index);
const char    *xlsx_cell(const XlsxRow *row, int col);
```
`xlsx_open` parses the whole workbook. It refuses encrypted files, ZIP64
and old binary `.xls` files with a message. Every cell comes back as one
clean line of text: dates as `YYYY-MM-DD`, numbers as Excel's "General"
format, and vertically merged cells repeat their value. Accessors return
`0` / `NULL` / `""` for a `NULL` book or out-of-range indices. All strings
belong to the book until `xlsx_close()`.

## `rag_*.h`: question answering over Excel / CSV (details: [RAG.md](RAG.md))

```c
RagCorpus *rag_corpus_load(const char *const *inputs, int n, const RagLoadOptions *o, char *err, size_t cap);
void       rag_corpus_free(RagCorpus *c);
int        rag_corpus_chunk_count(const RagCorpus *c);
const RagChunk *rag_corpus_chunk(const RagCorpus *c, int i);
int        rag_corpus_changed(const RagCorpus *c, char *summary, size_t cap);   /* 1: reload needed */

RagIndex  *rag_index_build(const RagCorpus *c, const RagIndexOptions *o, char *err, size_t cap);
void       rag_index_free(RagIndex *ix);
int        rag_index_search(const RagIndex *ix, const char *query, RagHit *hits, int max_hits);
int        rag_index_resolve(const RagIndex *ix, const char *text, RagQueryTerm *out, int max);

RagSession *rag_session_new(void);   void rag_session_free(RagSession *s);
int        rag_answer(RagSession *s, const RagCorpus *c, const RagIndex *ix, const char *input, FILE *out);

char      *rag_prompt_system(const RagPromptOptions *o);                       /* caller frees */
char      *rag_prompt_question(const RagCorpus *c, const RagHit *hits, int n, const char *q,
                               const RagPromptOptions *o, RagCountTokens count, void *user,
                               int *n_used, int *n_tokens, char *err, size_t cap);  /* caller frees */
```
Folders are scanned for `*.xlsx` and `*.csv`. A file that cannot be read is
skipped with a warning (`rag_corpus_warning()`), and the rest still load.
Each header file documents its rules in full.

## `pocket.h`: the library interface (details: [POCKETLLM.md](POCKETLLM.md))

Everything the command-line programs do, as functions that return strings
instead of printing. `make lib` builds `build/libpocketllm.so`, and the
Python package wraps exactly these functions.

| Group | Functions |
|---|---|
| model | `pocket_model_open/close`, `pocket_model_generate`, `pocket_model_chat`, `pocket_model_reset`, `pocket_model_seed`, `pocket_model_stats`, `pocket_model_description`, `pocket_model_error` |
| RAG | `pocket_rag_open/close`, `pocket_rag_ask` (no model), `pocket_rag_ask_model` (RAG with a model), `pocket_rag_search` + `pocket_rag_hit_text/source/score`, `pocket_rag_refresh/reload`, `pocket_rag_last_source/offer/answer`, file and warning accessors |

Conventions: open functions return `NULL` and write the error message;
other functions return `NULL`/`-1` and keep the message for
`pocket_*_error()`. Returned strings belong to the handle and stay valid
until the next call on it. A handle must not be used by two threads at once.

---

## Lower-level APIs

These are for tools that need direct access to one format. Programs
should normally use `engine.h`.

### `llm.h` / `transformer.h`: llama2.c FP32 models
| Function | Description |
|---|---|
| `Transformer *llm_load(path, err, cap)` / `llm_free_model(t)` | load / free a llama2.c checkpoint on the heap |
| `llm_model_config(t)`, `llm_model_file_size(t)` | accessors |
| `int build_transformer(Transformer *t, path, err, cap)` / `free_transformer(t)` | the same, into a caller-provided struct |
| `float *forward(Transformer *t, int token, int pos)` | one forward pass; `NULL` on invalid input |
| `rmsnorm(o, x, w, n)`, `softmax(x, n)`, `matmul(out, x, W, n, d)` | the maths kernels (`out[d] = W[d×n] · x[n]`) |
| `malloc_run_state(s, cfg, err, cap)` / `free_run_state(s)` | activation buffers + KV cache |
| `config_validate(cfg, err, cap)` | checks a `Config` for impossible values |

### `gguf_model.h` / `gguf_forward.h`: GGUF models
| Function | Description |
|---|---|
| `gguf_model_load(m, path, err, cap)` / `gguf_model_free(m)` | map a GGUF file, bind and check every LLaMA tensor |
| `gguf_weight_dequantize_row(w, row, out)` | decode one row to FP32 (`out` holds `w->cols` floats) |
| `gguf_matvec(w, x, out)` | fast fused product on the quantized data (OpenMP) |
| `gguf_matvec_reference(w, x, out)` | slow, bit-exact reference used for verification |
| `gguf_weight_type_supported(type)` | F32, F16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K |
| `gguf_transformer_init(t, path, err, cap)` / `gguf_transformer_free(t)` / `gguf_forward(t, tok, pos)` | GGUF forward pass |

### `checkpoint.h`: GGUF file structure
`gguf_read_header`, `gguf_read_metadata`, `gguf_read_config`,
`gguf_read_layout`, `gguf_read_tensors` (free the list with
`gguf_free_tensors`), `gguf_find_tensor`, `gguf_get_tensor_data` (a bounds-checked
pointer into a mapping), `gguf_tensor_storage_size`, `gguf_ggml_type_name`,
`gguf_ggml_type_info`.

### `gguf_quant.h`: block formats
`gguf_fp16_to_fp32`, the per-type block decoders
`gguf_dequantize_{q4_0,q8_0,q4k,q5k,q6k}_block`, and the generic
`gguf_quant_block_info(type, …)` / `gguf_dequantize_block(type, …)`.

---

## Command-line programs

| Program | Build | Purpose |
|---|---|---|
| `llm_generate` | `make` | **the interactive CLI**: completion, `--chat`, multi-turn `-i` |
| `llm_showcase` | `make` | walks through every library feature (loading, memory, tokenizer, the 3 samplers, reproducibility, speed, error handling) |
| `llm_bench` | `make` | speed and memory for any model |
| `minimal` | `make build/minimal` | the smallest complete program (above) |
| `llm_rag` | `make rag-tools` | **questions about spreadsheets**: `--retrieve-only`, `--rag` with a model, or the model alone |
| `rag_inspect` | `make rag-tools` | shows each RAG stage: files, chunks, terms, resolution, search scores |
| `xlsx_dump` | `make rag-tools` | prints a workbook as text or TSV |
| `pocket_chat`, `pocket_demo` | `make examples` | the `pocket.h` interface from C |
| `gguf_*` demos | `make gguf-tools` | step-by-step GGUF walkthroughs: header, tensors, quantization, matvec, forward pass, tokenizer |

### `llm_generate` options
```
llm_generate <model file> [options] ["prompt"]
  --chat               one chat turn (TinyLlama / Zephyr chat format)
  -i                   interactive: multi-turn chat (/reset, /exit); for
                       llama2.c models every line is a new completion
  -n N                 max new tokens per reply              (default 256)
  --sampler S          greedy | temperature | top-k          (default top-k)
  --temperature T      > 0                                   (default 0.7)
  --top-k K                                                  (default 40)
  --seed N             random seed (default: clock; printed)
  --system TEXT        system message for chat
  --ctx N              KV-cache context (default: model's, max 4096)
  --tokenizer F        tokenizer.bin for llama2.c models
  -v                   print token IDs of every prompt
```
