#ifndef ENGINE_H
#define ENGINE_H

/*
 * engine.h - one interface for every model format the engine supports.
 *
 *   LLMModel *m = engine_open(&options, error, sizeof(error));
 *   float *logits = engine_forward(m, token, pos);
 *   engine_close(m);
 *
 * The format is detected from the file contents, not the file name:
 *   "GGUF" magic      -> GGUF path   (gguf_forward.c, quantized weights,
 *                                      tokenizer from the GGUF metadata)
 *   anything else     -> llama2.c "Stories" path (transformer.c, FP32
 *                                      weights, tokenizer.bin)
 *
 * Both paths share Config, RunState, the KV cache layout, the samplers and
 * the Tokenizer struct, so a program written against this header works
 * with any supported model file.
 */

#include "tokenizer.h"
#include "transformer.h"

#include <stddef.h>

typedef enum {
    LLM_FORMAT_STORIES = 1,   /* llama2.c .bin, FP32 */
    LLM_FORMAT_GGUF = 2       /* GGUF, quantized */
} LLMFormat;

typedef struct {
    const char *model_path;      /* required */

    /*
     * Tokenizer for llama2.c models (ignored for GGUF unless set, in which
     * case it replaces the tokenizer stored in the GGUF file).
     * NULL: look for "tokenizer.bin" next to the model file, then in the
     * current directory.
     */
    const char *tokenizer_path;

    /*
     * Maximum context (positions) to allocate the KV cache for.
     * 0 = the model's own context length, capped at LLM_DEFAULT_MAX_CONTEXT.
     * The KV cache needs 2 * layers * context * kv_dim * 4 bytes.
     */
    int max_context;
} LLMOptions;

#define LLM_DEFAULT_MAX_CONTEXT 4096

typedef struct LLMModel LLMModel;

/* Open a model and its tokenizer. NULL on failure, with a message. */
LLMModel *engine_open(const LLMOptions *options,
                         char *error, size_t error_capacity);

/* Release everything. Safe on NULL. */
void engine_close(LLMModel *model);

/*
 * Run one token at position pos (positions in order 0, 1, 2, ...).
 * Returns vocab_size logits owned by the model, or NULL on error.
 */
float *engine_forward(LLMModel *model, int token, int pos);

/* Configuration in use; seq_len is the allocated (possibly capped) context. */
const Config *engine_config(const LLMModel *model);

/* The model's own context length from the file (before any cap). */
int engine_file_context(const LLMModel *model);

LLMFormat engine_format(const LLMModel *model);
Tokenizer *engine_tokenizer(LLMModel *model);

/* Bytes allocated for the KV cache. */
size_t engine_kv_cache_bytes(const LLMModel *model);

/*
 * One-line description, e.g. "GGUF, weights Q4_K x135 Q6_K x21 F32 x45"
 * or "llama2.c FP32". Valid until engine_close().
 */
const char *engine_description(const LLMModel *model);

/*
 * Token IDs that end generation: EOS for every model; for llama2.c Stories
 * models also BOS, which those models emit between stories.
 */
int engine_is_stop_token(const LLMModel *model, int token);

#endif
