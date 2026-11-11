/*
 * showcase.c - demonstration program for the wild-pointers LLM library.
 *
 * Walks through every public capability of the library, one section at a time:
 *
 *   1. Model loading        llm_load(), llm_model_config(), llm_model_file_size()
 *   2. Memory usage         weights (mmap), KV cache and run-state buffers
 *   3. Tokenizer            tokenizer_load(), tokenizer_encode(), tokenizer_decode()
 *   4. Sampling methods     sampler_greedy(), sampler_temperature(), sampler_top_k()
 *   5. Reproducibility      same seed -> same text
 *   6. Performance          tokens per second
 *   7. Error handling       invalid input is rejected, not crashed on
 *   8. Cleanup              tokenizer_free(), llm_free_model()
 *
 * Build:  make showcase
 * Run:    ./build/llm_showcase [model.bin] [tokenizer.bin]
 * Default files: models/stories42M.bin and models/tokenizer.bin
 */

#define _POSIX_C_SOURCE 200809L

#include "llm.h"
#include "sampler.h"
#include "tokenizer.h"
#include "transformer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEMO_PROMPT     "Once upon a time"
#define DEMO_NEW_TOKENS 60
#define TEXT_CAPACITY   8192
#define ERROR_CAPACITY  256

typedef enum { MODE_GREEDY, MODE_TEMPERATURE, MODE_TOP_K } SamplerMode;

typedef struct {
    SamplerMode mode;
    float temperature;
    size_t top_k;
    uint32_t seed;
} SamplerSettings;

typedef struct {
    int prompt_tokens;
    int generated_tokens;
    double seconds;
} GenerationStats;

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

static void section(int number, const char *title)
{
    printf("\n==============================================================\n");
    printf(" %d. %s\n", number, title);
    printf("==============================================================\n");
}

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static double to_mb(size_t bytes)
{
    return (double)bytes / (1024.0 * 1024.0);
}

static void check(const char *name, int passed)
{
    printf("  %-52s %s\n", name, passed ? "PASS" : "FAIL");
}

/* Pick the next token with the chosen sampler. Returns -1 on failure. */
static int sample_next(const float *logits, size_t vocab_size,
                       const SamplerSettings *settings, uint32_t *rng)
{
    switch (settings->mode) {
    case MODE_GREEDY:
        return sampler_greedy(logits, vocab_size);
    case MODE_TEMPERATURE:
        return sampler_temperature(logits, vocab_size,
                                   settings->temperature, rng);
    case MODE_TOP_K:
        return sampler_top_k(logits, vocab_size, settings->top_k,
                             settings->temperature, rng);
    }
    return -1;
}

/*
 * Generate text for one prompt into `output` (NUL-terminated).
 * Returns 0 on success, -1 on error (message in `error`).
 * Memory: the token buffer is allocated and freed here; the caller owns
 * `output`; logits returned by forward() belong to the model.
 */
static int generate(Transformer *model, Tokenizer *tokenizer,
                    const char *prompt, int max_new_tokens,
                    const SamplerSettings *settings,
                    char *output, size_t output_capacity,
                    GenerationStats *stats,
                    char *error, size_t error_capacity)
{
    const Config *config = llm_model_config(model);
    size_t vocab_size = (size_t)config->vocab_size;
    size_t token_capacity = strlen(prompt) + 3;
    size_t prompt_len = 0;
    size_t used = 0;
    uint32_t rng = settings->seed;
    int *tokens = malloc(token_capacity * sizeof(*tokens));
    int status = -1;

    memset(stats, 0, sizeof(*stats));
    output[0] = '\0';

    if (tokens == NULL) {
        snprintf(error, error_capacity, "Out of memory.");
        return -1;
    }

    if (tokenizer_encode(tokenizer, prompt, 1, 0, tokens, token_capacity,
                         &prompt_len, error, error_capacity) != 0) {
        goto cleanup;
    }
    if (prompt_len >= (size_t)config->seq_len) {
        snprintf(error, error_capacity, "Prompt longer than the context.");
        goto cleanup;
    }
    if (tokenizer_decode(tokenizer, tokens, prompt_len, output,
                         output_capacity, error, error_capacity) != 0) {
        goto cleanup;
    }
    used = strlen(output);

    int limit = (int)prompt_len + max_new_tokens;
    if (limit > config->seq_len) {
        limit = config->seq_len;
    }

    double start = now_seconds();
    int token = tokens[0];

    for (int pos = 0; pos < limit - 1; ++pos) {
        float *logits = forward(model, token, pos);
        if (logits == NULL) {
            snprintf(error, error_capacity, "forward() failed at %d.", pos);
            goto cleanup;
        }

        int next;
        if ((size_t)pos + 1 < prompt_len) {
            next = tokens[pos + 1];               /* still reading prompt */
        } else {
            next = sample_next(logits, vocab_size, settings, &rng);
            if (next < 0) {
                snprintf(error, error_capacity, "Sampler failed.");
                goto cleanup;
            }
            if (next == TOKENIZER_BOS_ID || next == TOKENIZER_EOS_ID) {
                break;                            /* end of text */
            }

            char piece[256];
            if (tokenizer_decode(tokenizer, &next, 1, piece, sizeof(piece),
                                 error, error_capacity) != 0) {
                goto cleanup;
            }
            size_t len = strlen(piece);
            if (used + len + 1 < output_capacity) {
                memcpy(output + used, piece, len + 1);
                used += len;
            }
            stats->generated_tokens++;
        }
        token = next;
    }

    stats->seconds = now_seconds() - start;
    stats->prompt_tokens = (int)prompt_len;
    status = 0;

cleanup:
    free(tokens);
    return status;
}

/* ------------------------------------------------------------------ */
/* Demo sections                                                       */
/* ------------------------------------------------------------------ */

static void show_model_info(const Transformer *model)
{
    const Config *c = llm_model_config(model);

    printf("  Embedding dimension      %d\n", c->dim);
    printf("  FFN hidden dimension     %d\n", c->hidden_dim);
    printf("  Transformer layers       %d\n", c->n_layers);
    printf("  Attention heads          %d\n", c->n_heads);
    printf("  Key/value heads          %d%s\n", c->n_kv_heads,
           c->n_kv_heads < c->n_heads ? "  (grouped-query attention)" : "");
    printf("  Head size                %d\n", c->dim / c->n_heads);
    printf("  Vocabulary size          %d\n", c->vocab_size);
    printf("  Context length           %d tokens\n", c->seq_len);
}

static void show_memory(const Transformer *model)
{
    const Config *c = llm_model_config(model);
    size_t kv_dim = (size_t)c->dim * (size_t)c->n_kv_heads / (size_t)c->n_heads;
    size_t f = sizeof(float);

    size_t weights  = llm_model_file_size(model);
    size_t kv_cache = 2 * (size_t)c->n_layers * (size_t)c->seq_len * kv_dim * f;
    size_t buffers  = (3 * (size_t)c->dim          /* x, xb, xb2   */
                     + 2 * (size_t)c->hidden_dim   /* hb, hb2      */
                     + (size_t)c->dim + 2 * kv_dim /* q, k, v      */
                     + (size_t)c->n_heads * (size_t)c->seq_len /* att */
                     + (size_t)c->vocab_size) * f; /* logits       */

    printf("  %-26s %10s   %s\n", "Part", "Size", "Owner / freed by");
    printf("  %-26s %8.2f MB   library, mmap (llm_free_model)\n",
           "Weights (file)", to_mb(weights));
    printf("  %-26s %8.2f MB   library, calloc (llm_free_model)\n",
           "KV cache", to_mb(kv_cache));
    printf("  %-26s %8.2f MB   library, calloc (llm_free_model)\n",
           "Activation buffers", to_mb(buffers));
    printf("  %-26s %8.2f MB\n", "Total", to_mb(weights + kv_cache + buffers));
}

static void show_tokenizer(Tokenizer *tokenizer)
{
    const char *text = DEMO_PROMPT;
    int tokens[64];
    size_t count = 0;
    char decoded[256];
    char error[ERROR_CAPACITY];

    if (tokenizer_encode(tokenizer, text, 1, 0, tokens, 64, &count,
                         error, sizeof(error)) != 0) {
        printf("  Encode failed: %s\n", error);
        return;
    }

    printf("  Text      : \"%s\"\n", text);
    printf("  Token IDs : [");
    for (size_t i = 0; i < count; ++i) {
        printf("%d%s", tokens[i], i + 1 < count ? ", " : "");
    }
    printf("]   (%zu tokens, first is BOS)\n", count);

    printf("  Pieces    : ");
    for (size_t i = 1; i < count; ++i) {
        printf("[%s] ", tokenizer->vocab[tokens[i]]);
    }
    printf("\n");

    if (tokenizer_decode(tokenizer, tokens, count, decoded, sizeof(decoded),
                         error, sizeof(error)) == 0) {
        printf("  Decoded   : \"%s\"\n", decoded);
        check("Round trip: decode(encode(text)) == text",
              strcmp(decoded, text) == 0);
    }
}

static void run_sampler(Transformer *model, Tokenizer *tokenizer,
                        const char *label, SamplerSettings settings)
{
    static char text[TEXT_CAPACITY];
    char error[ERROR_CAPACITY] = "";
    GenerationStats stats;

    printf("\n  --- %s ---\n", label);
    if (generate(model, tokenizer, DEMO_PROMPT, DEMO_NEW_TOKENS, &settings,
                 text, sizeof(text), &stats, error, sizeof(error)) != 0) {
        printf("  Error: %s\n", error);
        return;
    }
    printf("  %s\n", text);
    printf("  [%d tokens, %.1f tok/s]\n", stats.generated_tokens,
           stats.seconds > 0 ? (stats.prompt_tokens + stats.generated_tokens)
                               / stats.seconds : 0.0);
}

static void show_reproducibility(Transformer *model, Tokenizer *tokenizer)
{
    static char first[TEXT_CAPACITY];
    static char second[TEXT_CAPACITY];
    static char other[TEXT_CAPACITY];
    char error[ERROR_CAPACITY];
    GenerationStats stats;
    SamplerSettings s = { MODE_TOP_K, 0.8f, 40, 42 };

    generate(model, tokenizer, DEMO_PROMPT, 30, &s, first, sizeof(first),
             &stats, error, sizeof(error));
    generate(model, tokenizer, DEMO_PROMPT, 30, &s, second, sizeof(second),
             &stats, error, sizeof(error));
    s.seed = 7;
    generate(model, tokenizer, DEMO_PROMPT, 30, &s, other, sizeof(other),
             &stats, error, sizeof(error));

    check("Seed 42 twice gives identical text", strcmp(first, second) == 0);
    check("Seed 7 gives different text", strcmp(first, other) != 0);
}

static void show_performance(Transformer *model, Tokenizer *tokenizer)
{
    static char text[TEXT_CAPACITY];
    char error[ERROR_CAPACITY];
    GenerationStats stats;
    SamplerSettings s = { MODE_GREEDY, 1.0f, 0, 0 };
    int new_tokens = 128;

    if (generate(model, tokenizer, DEMO_PROMPT, new_tokens, &s, text,
                 sizeof(text), &stats, error, sizeof(error)) != 0) {
        printf("  Error: %s\n", error);
        return;
    }

    int total = stats.prompt_tokens + stats.generated_tokens;
    printf("  Tokens processed   %d (%d prompt + %d generated)\n",
           total, stats.prompt_tokens, stats.generated_tokens);
    printf("  Time               %.3f s\n", stats.seconds);
    printf("  Speed              %.1f tokens/sec\n",
           stats.seconds > 0 ? total / stats.seconds : 0.0);
    printf("  Latency            %.2f ms/token\n",
           total > 0 ? 1000.0 * stats.seconds / total : 0.0);
}

static void show_error_handling(Transformer *model, Tokenizer *tokenizer)
{
    const Config *c = llm_model_config(model);
    size_t vocab = (size_t)c->vocab_size;
    char error[ERROR_CAPACITY];
    Tokenizer bad = {0};
    int tokens[2];
    size_t count = 0;
    char small[2];
    int good_token = 1;
    uint32_t rng = 1;
    float logits[3] = { 0.1f, 2.0f, -1.0f };

    /* Tokenizer errors are returned, not fatal. */
    int rc = tokenizer_load(&bad, "missing_tokenizer.bin", vocab,
                            error, sizeof(error));
    check("tokenizer_load: missing file -> -1", rc == -1);
    printf("      message: \"%s\"\n", error);
    tokenizer_free(&bad);

    rc = tokenizer_encode(tokenizer, NULL, 1, 0, tokens, 2, &count,
                          error, sizeof(error));
    check("tokenizer_encode: NULL text -> -1", rc == -1);

    rc = tokenizer_encode(tokenizer, "a long sentence here", 1, 0,
                          tokens, 2, &count, error, sizeof(error));
    check("tokenizer_encode: output buffer too small -> -1", rc == -1);
    printf("      message: \"%s\"\n", error);

    rc = tokenizer_decode(tokenizer, &good_token, 1, small, 0,
                          error, sizeof(error));
    check("tokenizer_decode: zero-size output -> -1", rc == -1);

    int bad_id = c->vocab_size + 5;
    rc = tokenizer_decode(tokenizer, &bad_id, 1, small, sizeof(small),
                          error, sizeof(error));
    check("tokenizer_decode: token ID out of range -> -1", rc == -1);

    /* forward() rejects out-of-range input instead of reading past arrays. */
    check("forward: token = -1 -> NULL", forward(model, -1, 0) == NULL);
    check("forward: token = vocab_size -> NULL",
          forward(model, c->vocab_size, 0) == NULL);
    check("forward: pos = seq_len -> NULL",
          forward(model, 1, c->seq_len) == NULL);
    check("forward: model = NULL -> NULL", forward(NULL, 1, 0) == NULL);

    /* Samplers validate their arguments. */
    check("sampler_greedy: NULL logits -> -1",
          sampler_greedy(NULL, 3) == -1);
    check("sampler_greedy: empty vocabulary -> -1",
          sampler_greedy(logits, 0) == -1);
    check("sampler_temperature: temperature 0 -> -1",
          sampler_temperature(logits, 3, 0.0f, &rng) == -1);
    check("sampler_top_k: k = 0 -> -1",
          sampler_top_k(logits, 3, 0, 1.0f, &rng) == -1);
    check("sampler_top_k: k > vocabulary -> -1",
          sampler_top_k(logits, 3, 10, 1.0f, &rng) == -1);
    check("sampler_top_k: k = 1 behaves like greedy",
          sampler_top_k(logits, 3, 1, 1.0f, &rng) == sampler_greedy(logits, 3));
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    const char *model_path = argc > 1 ? argv[1] : "models/stories42M.bin";
    const char *tokenizer_path = argc > 2 ? argv[2] : "models/tokenizer.bin";
    char error[ERROR_CAPACITY] = "";
    Tokenizer tokenizer = {0};
    Transformer *model;

    printf("Wild Pointers - LLM inference library demonstration\n");

    section(1, "Model loading");
    printf("  File: %s\n", model_path);
    model = llm_load(model_path, error, sizeof(error));
    if (model == NULL) {
        fprintf(stderr, "  Error: %s\n", error);
        return EXIT_FAILURE;
    }
    show_model_info(model);

    section(2, "Memory usage");
    show_memory(model);

    section(3, "Tokenizer (BPE)");
    printf("  File: %s\n", tokenizer_path);
    if (tokenizer_load(&tokenizer, tokenizer_path,
                       (size_t)llm_model_config(model)->vocab_size,
                       error, sizeof(error)) != 0) {
        fprintf(stderr, "  Error: %s\n", error);
        llm_free_model(model);
        return EXIT_FAILURE;
    }
    show_tokenizer(&tokenizer);

    section(4, "Sampling methods (same prompt, different samplers)");
    run_sampler(model, &tokenizer, "Greedy (always the most likely token)",
                (SamplerSettings){ MODE_GREEDY, 1.0f, 0, 0 });
    run_sampler(model, &tokenizer, "Temperature 0.8",
                (SamplerSettings){ MODE_TEMPERATURE, 0.8f, 0, 42 });
    run_sampler(model, &tokenizer, "Temperature 1.5 (more random)",
                (SamplerSettings){ MODE_TEMPERATURE, 1.5f, 0, 42 });
    run_sampler(model, &tokenizer, "Top-k (k = 40, temperature 0.8)",
                (SamplerSettings){ MODE_TOP_K, 0.8f, 40, 42 });

    section(5, "Reproducibility (seeded random sampling)");
    show_reproducibility(model, &tokenizer);

    section(6, "Performance");
    show_performance(model, &tokenizer);

    section(7, "Error handling (invalid input is rejected safely)");
    show_error_handling(model, &tokenizer);

    section(8, "Cleanup");
    tokenizer_free(&tokenizer);
    llm_free_model(model);
    printf("  Tokenizer and model freed.\n\nDone.\n");

    return EXIT_SUCCESS;
}
