/*
 * llm_bench.c - speed and memory of the engine, for ANY supported model
 * (llama2.c .bin or GGUF), through the public engine.h API.
 *
 * Usage: llm_bench <model file> [prompt_tokens] [generated_tokens] [--csv]
 *        (defaults: 32 prompt tokens, 64 generated tokens)
 *        llama2.c models need tokenizer.bin next to the model or in the
 *        current directory (same rule as llm_generate).
 *
 * Measures:
 *   - load time, model file size
 *   - memory: KV cache, activation buffers, resident set size (RSS) after
 *     load, after the run, and peak RSS (Linux /proc/self/status)
 *   - prompt processing speed and generation speed in tokens/s
 *
 * The run is deterministic: a fixed prompt of token IDs and greedy
 * sampling, so numbers from different builds/machines are comparable.
 * A warm-up token first pulls the memory-mapped weights in from disk, so
 * the timings measure computation, not disk reads.
 *
 * Thread count: OpenMP, set with OMP_NUM_THREADS (default: all cores).
 * --csv prints one machine-readable line (used by tests/run_benchmark.sh):
 *   model,format,threads,file_mb,kv_mb,act_mb,rss_load_mb,rss_peak_mb,
 *   load_s,prompt_tok_s,gen_tok_s
 */
#define _POSIX_C_SOURCE 200809L
#include "engine.h"
#include "sampler.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* A /proc/self/status field in MB (VmRSS, VmHWM), or -1 if unavailable. */
static double proc_status_mb(const char *field)
{
    FILE *f = fopen("/proc/self/status", "r");
    if (f == NULL) {
        return -1.0;
    }
    char line[256];
    size_t n = strlen(field);
    double mb = -1.0;
    while (fgets(line, sizeof(line), f) != NULL) {
        if (strncmp(line, field, n) == 0 && line[n] == ':') {
            long kb = strtol(line + n + 1, NULL, 10);
            mb = (double)kb / 1024.0;
            break;
        }
    }
    fclose(f);
    return mb;
}

/* Bytes of the per-token activation buffers (RunState without KV cache). */
static double activation_mb(const Config *c)
{
    size_t dim = (size_t)c->dim, hid = (size_t)c->hidden_dim;
    size_t kv = dim * (size_t)c->n_kv_heads / (size_t)c->n_heads;
    size_t floats = 4 * dim           /* x, xb, xb2, q */
                  + 2 * hid           /* hb, hb2 */
                  + 2 * kv            /* k, v */
                  + (size_t)c->n_heads * (size_t)c->seq_len  /* att */
                  + (size_t)c->vocab_size;                   /* logits */
    return (double)(floats * sizeof(float)) / (1024.0 * 1024.0);
}

static const char *base_name(const char *path)
{
    const char *slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

int main(int argc, char **argv)
{
    const char *path = NULL;
    int prompt_n = 32, gen_n = 64, csv = 0, positional = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--csv") == 0) {
            csv = 1;
        } else if (positional == 0) {
            path = argv[i];
            ++positional;
        } else if (positional == 1) {
            prompt_n = atoi(argv[i]);
            ++positional;
        } else {
            gen_n = atoi(argv[i]);
        }
    }
    if (path == NULL || prompt_n <= 0 || gen_n <= 0) {
        fprintf(stderr, "Usage: %s <model file> [prompt_tokens] "
                        "[generated_tokens] [--csv]\n", argv[0]);
        return 1;
    }

    struct stat st;
    double file_mb = stat(path, &st) == 0 ? (double)st.st_size / 1048576.0 : 0;

    /* Context just large enough for the run (+1 warm-up position). */
    LLMOptions o = {0};
    o.model_path = path;
    o.max_context = prompt_n + gen_n;
    char error[512];

    double t0 = now_s();
    LLMModel *m = engine_open(&o, error, sizeof(error));
    double load_s = now_s() - t0;
    if (m == NULL) {
        fprintf(stderr, "Cannot open %s: %s\n", path, error);
        return 1;
    }
    const Config *c = engine_config(m);
    if (c->seq_len < prompt_n + gen_n) {
        fprintf(stderr, "Model context %d is smaller than %d + %d tokens.\n",
                c->seq_len, prompt_n, gen_n);
        engine_close(m);
        return 1;
    }

    /* Warm-up: touches every weight page once. */
    if (engine_forward(m, 1, 0) == NULL) {
        fprintf(stderr, "Warm-up forward failed.\n");
        engine_close(m);
        return 1;
    }
    double rss_load = proc_status_mb("VmRSS");

    /* Prompt: fixed, valid token IDs (BOS then ordinary tokens). */
    int token = 1;
    double p0 = now_s();
    for (int pos = 0; pos < prompt_n; ++pos) {
        if (engine_forward(m, token, pos) == NULL) {
            fprintf(stderr, "Forward failed at prompt position %d.\n", pos);
            engine_close(m);
            return 1;
        }
        token = 100 + (pos * 97) % (c->vocab_size - 100);
    }
    double prompt_s = now_s() - p0;

    /* Generation: greedy, no early stop, so every run does gen_n tokens. */
    double g0 = now_s();
    for (int i = 0; i < gen_n; ++i) {
        float *logits = engine_forward(m, token, prompt_n + i);
        if (logits == NULL) {
            fprintf(stderr, "Forward failed during generation.\n");
            engine_close(m);
            return 1;
        }
        token = sampler_greedy(logits, (size_t)c->vocab_size);
    }
    double gen_s = now_s() - g0;

    double kv_mb = (double)engine_kv_cache_bytes(m) / 1048576.0;
    double act_mb = activation_mb(c);
    double rss_peak = proc_status_mb("VmHWM");
    int threads = omp_get_max_threads();

    if (csv) {
        printf("%s,%s,%d,%.1f,%.2f,%.2f,%.1f,%.1f,%.2f,%.2f,%.2f\n",
               base_name(path),
               engine_format(m) == LLM_FORMAT_GGUF ? "GGUF" : "llama2.c",
               threads, file_mb, kv_mb, act_mb, rss_load, rss_peak, load_s,
               prompt_n / prompt_s, gen_n / gen_s);
    } else {
        printf("=== Model ===\n");
        printf("File:          %s (%.1f MB, memory-mapped)\n", path, file_mb);
        printf("Format:        %s\n", engine_description(m));
        printf("Architecture:  %d layers, dim %d, %d heads (%d KV), "
               "vocab %d\n", c->n_layers, c->dim, c->n_heads, c->n_kv_heads,
               c->vocab_size);
        printf("Threads:       %d (OMP_NUM_THREADS)\n", threads);
        printf("Load time:     %.3f s\n", load_s);
        printf("\n=== Memory ===\n");
        printf("KV cache:      %.2f MB for %d positions\n", kv_mb, c->seq_len);
        printf("Activations:   %.2f MB\n", act_mb);
        printf("RSS after load + warm-up: %.1f MB\n", rss_load);
        printf("Peak RSS:      %.1f MB\n", rss_peak);
        printf("\n=== Speed ===\n");
        printf("Prompt:     %4d tokens in %7.3f s = %8.2f tokens/s\n",
               prompt_n, prompt_s, prompt_n / prompt_s);
        printf("Generation: %4d tokens in %7.3f s = %8.2f tokens/s "
               "(%.1f ms/token)\n", gen_n, gen_s, gen_n / gen_s,
               1000.0 * gen_s / gen_n);
    }

    engine_close(m);
    return 0;
}
