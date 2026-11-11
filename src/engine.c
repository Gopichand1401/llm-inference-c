/*
 * engine.c - format detection and one model interface for llama2.c and
 * GGUF files. See include/engine.h.
 */
#include "engine.h"

#include "gguf_forward.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct LLMModel {
    LLMFormat format;

    Transformer stories;        /* LLM_FORMAT_STORIES */
    GGUFTransformer gguf;       /* LLM_FORMAT_GGUF */

    Tokenizer tokenizer;
    int file_context;           /* context length stored in the file */
    char description[256];
};

static void set_error(char *error, size_t capacity, const char *message)
{
    if (error != NULL && capacity > 0) {
        snprintf(error, capacity, "%s", message);
    }
}

/* Read the first 4 bytes: "GGUF" means a GGUF file. */
static int is_gguf_file(const char *path, int *readable)
{
    unsigned char magic[4];
    FILE *f = fopen(path, "rb");

    *readable = (f != NULL);
    if (f == NULL) {
        return 0;
    }

    size_t n = fread(magic, 1, sizeof(magic), f);
    fclose(f);
    return n == 4 && memcmp(magic, "GGUF", 4) == 0;
}

/*
 * Reallocate the run state (activations + KV cache) for a smaller context.
 * Only the KV cache and the attention buffer depend on seq_len.
 */
static int cap_context(RunState *state, Config *config, int requested,
                       char *error, size_t error_capacity)
{
    int cap = requested;

    if (cap <= 0) {
        cap = config->seq_len < LLM_DEFAULT_MAX_CONTEXT
            ? config->seq_len : LLM_DEFAULT_MAX_CONTEXT;
    }
    if (cap > config->seq_len) {
        cap = config->seq_len;
    }
    if (cap == config->seq_len && state->key_cache != NULL) {
        return 1;
    }

    free_run_state(state);
    config->seq_len = cap;
    return malloc_run_state(state, config, error, error_capacity);
}

/* Directory of `path` + "tokenizer.bin", or just "tokenizer.bin". */
static int tokenizer_next_to(const char *path, char *out, size_t capacity)
{
    const char *slash = strrchr(path, '/');

    if (slash == NULL) {
        return snprintf(out, capacity, "tokenizer.bin") < (int)capacity;
    }
    int dir_length = (int)(slash - path);
    return snprintf(out, capacity, "%.*s/tokenizer.bin", dir_length, path)
           < (int)capacity;
}

static int file_exists(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        return 0;
    }
    fclose(f);
    return 1;
}

static void describe_gguf(LLMModel *m)
{
    const GGUFModel *g = &m->gguf.model;
    unsigned int counts[64] = {0};
    size_t used = 0;

    used += (size_t)snprintf(m->description, sizeof(m->description),
                             "GGUF, weights");

    for (size_t i = 0; i < g->tensor_count; i++) {
        if (g->tensors[i].type < 64) {
            counts[g->tensors[i].type]++;
        }
    }
    for (uint32_t t = 0; t < 64 && used < sizeof(m->description); t++) {
        if (counts[t] != 0) {
            used += (size_t)snprintf(m->description + used,
                                     sizeof(m->description) - used,
                                     " %s x%u", gguf_ggml_type_name(t),
                                     counts[t]);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Open / close                                                        */
/* ------------------------------------------------------------------ */

void engine_close(LLMModel *model)
{
    if (model == NULL) {
        return;
    }
    tokenizer_free(&model->tokenizer);
    if (model->format == LLM_FORMAT_STORIES) {
        free_transformer(&model->stories);
    } else if (model->format == LLM_FORMAT_GGUF) {
        gguf_transformer_free(&model->gguf);
    }
    free(model);
}

LLMModel *engine_open(const LLMOptions *options,
                         char *error, size_t error_capacity)
{
    if (options == NULL || options->model_path == NULL ||
        options->model_path[0] == '\0') {
        set_error(error, error_capacity, "No model path given.");
        return NULL;
    }
    if (options->max_context < 0) {
        set_error(error, error_capacity, "Context size must be positive.");
        return NULL;
    }

    const char *path = options->model_path;
    int readable = 0;
    int gguf = is_gguf_file(path, &readable);

    if (!readable) {
        set_error(error, error_capacity, "Cannot open model file.");
        return NULL;
    }

    LLMModel *m = calloc(1, sizeof(*m));
    if (m == NULL) {
        set_error(error, error_capacity, "Out of memory.");
        return NULL;
    }

    if (gguf) {
        /* ---- GGUF ---- */
        m->format = LLM_FORMAT_GGUF;

        if (!gguf_transformer_init(&m->gguf, path, error, error_capacity)) {
            m->format = 0;
            free(m);
            return NULL;
        }

        Config *c = &m->gguf.model.config;
        m->file_context = c->seq_len;

        if (!cap_context(&m->gguf.state, c, options->max_context,
                         error, error_capacity)) {
            engine_close(m);
            return NULL;
        }

        int rc = options->tokenizer_path != NULL
               ? tokenizer_load(&m->tokenizer, options->tokenizer_path,
                                (size_t)c->vocab_size, error, error_capacity)
               : tokenizer_load_gguf(&m->tokenizer, path,
                                     error, error_capacity);
        if (rc != 0) {
            engine_close(m);
            return NULL;
        }

        describe_gguf(m);

    } else {
        /* ---- llama2.c Stories ---- */
        m->format = LLM_FORMAT_STORIES;

        if (!build_transformer(&m->stories, path, error, error_capacity)) {
            m->format = 0;
            free(m);
            return NULL;
        }

        Config *c = &m->stories.config;
        m->file_context = c->seq_len;

        if (!cap_context(&m->stories.state, c, options->max_context,
                         error, error_capacity)) {
            engine_close(m);
            return NULL;
        }

        char found[1024];
        const char *tok_path = options->tokenizer_path;

        if (tok_path == NULL) {
            if (tokenizer_next_to(path, found, sizeof(found)) &&
                file_exists(found)) {
                tok_path = found;
            } else if (file_exists("tokenizer.bin")) {
                tok_path = "tokenizer.bin";
            } else {
                set_error(error, error_capacity,
                          "llama2.c model needs tokenizer.bin (next to the "
                          "model, in the current directory, or --tokenizer).");
                engine_close(m);
                return NULL;
            }
        }

        if (tokenizer_load(&m->tokenizer, tok_path, (size_t)c->vocab_size,
                           error, error_capacity) != 0) {
            engine_close(m);
            return NULL;
        }

        snprintf(m->description, sizeof(m->description),
                 "llama2.c FP32, %s classifier",
                 m->stories.weights.wcls == m->stories.weights.token_embedding_table
                     ? "shared" : "separate");
    }

    if (error != NULL && error_capacity > 0) {
        error[0] = '\0';
    }
    return m;
}

/* ------------------------------------------------------------------ */
/* Accessors                                                           */
/* ------------------------------------------------------------------ */

float *engine_forward(LLMModel *model, int token, int pos)
{
    if (model == NULL) {
        return NULL;
    }
    if (model->format == LLM_FORMAT_GGUF) {
        return gguf_forward(&model->gguf, token, pos);
    }
    return forward(&model->stories, token, pos);
}

const Config *engine_config(const LLMModel *model)
{
    if (model == NULL) {
        return NULL;
    }
    return model->format == LLM_FORMAT_GGUF ? &model->gguf.model.config
                                            : &model->stories.config;
}

int engine_file_context(const LLMModel *model)
{
    return model == NULL ? 0 : model->file_context;
}

LLMFormat engine_format(const LLMModel *model)
{
    return model == NULL ? 0 : model->format;
}

Tokenizer *engine_tokenizer(LLMModel *model)
{
    return model == NULL ? NULL : &model->tokenizer;
}

size_t engine_kv_cache_bytes(const LLMModel *model)
{
    const Config *c = engine_config(model);
    if (c == NULL) {
        return 0;
    }
    size_t kv_dim = (size_t)(c->dim / c->n_heads * c->n_kv_heads);
    return 2u * (size_t)c->n_layers * (size_t)c->seq_len * kv_dim * sizeof(float);
}

const char *engine_description(const LLMModel *model)
{
    return model == NULL ? "" : model->description;
}

int engine_is_stop_token(const LLMModel *model, int token)
{
    if (model == NULL) {
        return 1;
    }
    if (token == model->tokenizer.eos_id) {
        return 1;
    }
    return model->format == LLM_FORMAT_STORIES &&
           token == model->tokenizer.bos_id;
}
