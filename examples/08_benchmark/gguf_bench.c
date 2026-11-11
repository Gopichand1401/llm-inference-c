/*
 * gguf_bench.c - measure speed and memory of the GGUF inference engine.
 *
 * Usage: gguf_bench <model.gguf> [prompt_tokens] [generated_tokens]
 *        (defaults: 32 prompt tokens, 32 generated tokens)
 *
 * Reports:
 *   1. memory: KV cache, activations, resident set size (RSS) before and
 *      after running, and peak RSS,
 *   2. matrix-vector time for each weight shape/type in layer 0 and for the
 *      vocabulary projection (averaged over repeats),
 *   3. end-to-end tokens/s for prompt processing and for generation.
 *
 * The token IDs are fixed (greedy generation from a fixed prompt), so runs
 * are comparable. A warm-up pass first loads all weight pages from disk, so
 * the numbers measure computation, not disk reads.
 */
#include "gguf_forward.h"
#include "sampler.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Read a "VmRSS:" style line from /proc/self/status, in MB. */
static double proc_status_mb(const char *key)
{
    FILE *f = fopen("/proc/self/status", "r");
    char line[256];
    double mb = -1.0;
    size_t key_length = strlen(key);

    if (f == NULL) {
        return -1.0;
    }
    while (fgets(line, sizeof(line), f) != NULL) {
        if (strncmp(line, key, key_length) == 0) {
            mb = atof(line + key_length) / 1024.0;   /* value is in kB */
            break;
        }
    }
    fclose(f);
    return mb;
}

/* Average time of y = W x over `repeats` runs, in milliseconds. */
static double time_matvec(const GGUFWeight *w, const float *x, float *y,
                          int repeats)
{
    gguf_matvec(w, x, y);                       /* warm-up */
    double start = now_seconds();
    for (int r = 0; r < repeats; r++) {
        gguf_matvec(w, x, y);
    }
    return (now_seconds() - start) * 1000.0 / repeats;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [prompt_tokens] [generated_tokens]\n",
                argv[0]);
        return 1;
    }

    const int n_prompt = argc > 2 ? atoi(argv[2]) : 32;
    const int n_gen = argc > 3 ? atoi(argv[3]) : 32;
    GGUFTransformer t;
    char error[256];

    double rss_start = proc_status_mb("VmRSS:");

    if (n_prompt < 1 || n_gen < 0) {
        fprintf(stderr, "Token counts must be positive.\n");
        return 1;
    }

    double t0 = now_seconds();
    if (!gguf_transformer_init(&t, argv[1], error, sizeof(error))) {
        fprintf(stderr, "Init failed: %s\n", error);
        return 1;
    }
    double load_s = now_seconds() - t0;
    double rss_loaded = proc_status_mb("VmRSS:");

    const Config *c = &t.model.config;
    const size_t kv_dim = (size_t)(c->dim / c->n_heads * c->n_kv_heads);
    const double kv_mb = 2.0 * (double)c->n_layers * (double)c->seq_len *
                         (double)kv_dim * sizeof(float) / (1024.0 * 1024.0);

    printf("=== Setup ===\n");
    printf("Threads (OpenMP): %d\n", omp_get_max_threads());
    printf("Model file: %.1f MB (memory-mapped)\n",
           (double)t.model.file_size / (1024.0 * 1024.0));
    printf("Load time: %.2f s\n", load_s);
    printf("KV cache: %.1f MB (2 x %d layers x %d positions x %zu floats)\n",
           kv_mb, c->n_layers, c->seq_len, kv_dim);

    /* ---- Warm-up: touch every weight page once ---- */
    {
        double w0 = now_seconds();
        float *logits = gguf_forward(&t, 1, 0);
        if (logits == NULL) {
            fprintf(stderr, "Warm-up forward failed.\n");
            gguf_transformer_free(&t);
            return 1;
        }
        printf("Warm-up token (reads weights from disk): %.2f s\n",
               now_seconds() - w0);
    }

    /* ---- Matrix-vector timings ---- */
    {
        const size_t max_n = (size_t)(c->vocab_size > c->hidden_dim
                                      ? c->vocab_size : c->hidden_dim);
        float *x = calloc(max_n, sizeof(float));
        float *y = calloc(max_n, sizeof(float));
        const GGUFLayer *L0 = &t.model.layers[0];
        const GGUFLayer *L2 = &t.model.layers[c->n_layers > 2 ? 2 : 0];
        const GGUFWeight *ws[] = {
            &L0->wq, &L0->wk, &L0->wv, &L2->wv, &L0->wo,
            &L0->ffn_gate, &L0->ffn_up, &L0->ffn_down, &L2->ffn_down,
            &t.model.output
        };

        if (x == NULL || y == NULL) {
            fprintf(stderr, "Out of memory.\n");
            free(x);
            free(y);
            gguf_transformer_free(&t);
            return 1;
        }
        for (size_t i = 0; i < max_n; i++) {
            x[i] = (float)((int)(i % 17) - 8) * 0.01f;
        }

        printf("\n=== Matrix-vector product (y = W x) ===\n");
        printf("%-26s %-5s %6s %6s %10s %10s\n",
               "weight", "type", "rows", "cols", "ms", "GB/s");
        for (size_t i = 0; i < sizeof(ws) / sizeof(ws[0]); i++) {
            const GGUFWeight *w = ws[i];
            double ms = time_matvec(w, x, y, 10);
            double gbs = (double)w->byte_size / (ms * 1e-3) / 1e9;
            printf("%-26s %-5s %6llu %6llu %10.3f %10.2f\n",
                   w->name, gguf_ggml_type_name(w->type),
                   (unsigned long long)w->rows, (unsigned long long)w->cols,
                   ms, gbs);
        }
        free(x);
        free(y);
    }

    /* ---- End-to-end tokens/s ---- */
    {
        int token = 1;
        double p0 = now_seconds();
        float *logits = NULL;

        for (int pos = 0; pos < n_prompt; pos++) {
            /* Fixed, valid token IDs for the prompt. */
            token = (pos == 0) ? 1 : 100 + (pos * 37) % (c->vocab_size - 200);
            logits = gguf_forward(&t, token, pos);
            if (logits == NULL) {
                fprintf(stderr, "Forward failed at %d\n", pos);
                gguf_transformer_free(&t);
                return 1;
            }
        }
        double prompt_s = now_seconds() - p0;

        double g0 = now_seconds();
        int generated = 0;
        for (int i = 0; i < n_gen && n_prompt + i < c->seq_len; i++) {
            token = sampler_greedy(logits, (size_t)c->vocab_size);
            logits = gguf_forward(&t, token, n_prompt + i);
            if (logits == NULL) {
                fprintf(stderr, "Forward failed during generation.\n");
                gguf_transformer_free(&t);
                return 1;
            }
            generated++;
        }
        double gen_s = now_seconds() - g0;

        printf("\n=== End to end ===\n");
        printf("Prompt:     %3d tokens in %6.2f s = %6.2f tokens/s\n",
               n_prompt, prompt_s, n_prompt / prompt_s);
        printf("Generation: %3d tokens in %6.2f s = %6.2f tokens/s "
               "(%.1f ms/token)\n",
               generated, gen_s, generated > 0 ? generated / gen_s : 0.0,
               generated > 0 ? gen_s * 1000.0 / generated : 0.0);
        printf("Last greedy token: %d (compare between versions)\n", token);
    }

    printf("\n=== Memory ===\n");
    printf("RSS at start:        %8.1f MB\n", rss_start);
    printf("RSS after load:      %8.1f MB (weights not yet read)\n", rss_loaded);
    printf("RSS after running:   %8.1f MB (includes mapped weight pages)\n",
           proc_status_mb("VmRSS:"));
    printf("Peak RSS:            %8.1f MB\n", proc_status_mb("VmHWM:"));

    gguf_transformer_free(&t);
    return 0;
}
