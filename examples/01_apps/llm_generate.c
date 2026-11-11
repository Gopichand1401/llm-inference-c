/*
 * llm_generate.c - text generation and chat with ANY supported model file:
 * llama2.c .bin (stories15M / 42M / 110M ...) or GGUF (TinyLlama ...).
 * The format is detected from the file contents (see include/engine.h).
 *
 * Usage:
 *   llm_generate <model file> [options] ["prompt"]
 *
 * Modes:
 *   (default)        plain completion: the prompt is continued as text
 *   --chat           one chat turn using TinyLlama's chat format
 *   -i               interactive: multi-turn chat for GGUF chat models
 *                    (/reset, /exit); for llama2.c models every line is a
 *                    new completion prompt
 *
 * Options:
 *   -n N             maximum new tokens per reply          (default 256)
 *   --sampler S      greedy | temperature | top-k          (default top-k)
 *   --temperature T  softmax temperature, > 0              (default 0.7)
 *   --top-k K        candidates kept by top-k              (default 40)
 *   --seed N         random seed (default: from the clock; printed)
 *   --system TEXT    system message for chat modes
 *   --ctx N          context to allocate the KV cache for
 *                    (default: the model's, at most 4096)
 *   --tokenizer F    tokenizer.bin for llama2.c models (default: next to
 *                    the model, then the current directory)
 *   -v               print the token IDs of every prompt
 *
 * Chat format (TinyLlama-1.1B-Chat / Zephyr):
 *   <|system|>\n{system}</s>\n<|user|>\n{message}</s>\n<|assistant|>\n
 * "</s>" is recognised by the tokenizer and becomes the EOS token (ID 2).
 *
 * In interactive mode the KV cache is kept between turns: each new turn
 * only runs the new tokens through the model, and the reply's EOS token is
 * fed back in so the cache holds exactly the tokens of the full
 * conversation.
 */
#include "engine.h"
#include "sampler.h"
#include "tokenizer.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_INPUT 8192

typedef struct {
    LLMModel *model;
    Tokenizer *tok;
    int pos;                /* next free position in the KV cache */
    const char *sampler;
    float temperature;
    int top_k;
    uint32_t rng;
    int max_new;
    int verbose;
} Session;

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void usage(const char *program)
{
    fprintf(stderr,
            "Usage: %s <model file> [-n N] [--sampler greedy|temperature|top-k]\n"
            "          [--temperature T] [--top-k K] [--seed N] [--chat] [-i]\n"
            "          [--system TEXT] [--ctx N] [--tokenizer F] [-v] [\"prompt\"]\n",
            program);
}

/* ------------------------------------------------------------------ */
/* Core steps                                                          */
/* ------------------------------------------------------------------ */

/* Choose the next token from the logits with the configured sampler. */
static int sample(Session *s, const float *logits)
{
    const size_t n = (size_t)engine_config(s->model)->vocab_size;

    if (strcmp(s->sampler, "greedy") == 0) {
        return sampler_greedy(logits, n);
    }
    if (strcmp(s->sampler, "temperature") == 0) {
        return sampler_temperature(logits, n, s->temperature, &s->rng);
    }
    return sampler_top_k(logits, n, (size_t)s->top_k, s->temperature, &s->rng);
}

/*
 * Run tokens through the model at positions pos, pos+1, ...
 * Returns the logits after the last token, or NULL on error / full context.
 */
static float *feed(Session *s, const int *ids, size_t n)
{
    const int seq_len = engine_config(s->model)->seq_len;
    float *logits = NULL;

    if (n == 0 || (size_t)s->pos + n > (size_t)seq_len) {
        fprintf(stderr, "\n[context full: %d of %d positions used; "
                        "use /reset in interactive mode]\n", s->pos, seq_len);
        return NULL;
    }

    for (size_t i = 0; i < n; i++) {
        logits = engine_forward(s->model, ids[i], s->pos);
        if (logits == NULL) {
            fprintf(stderr, "\n[forward pass failed at position %d]\n", s->pos);
            return NULL;
        }
        s->pos++;
    }
    return logits;
}

/* Encode text (optionally with BOS) and feed it. Prints stats if verbose. */
static float *feed_text(Session *s, const char *text, int add_bos,
                        size_t *n_tokens, double *seconds)
{
    char error[256];
    size_t capacity = strlen(text) + 8;
    int *ids = malloc(capacity * sizeof(*ids));
    size_t count = 0;
    float *logits = NULL;

    if (ids == NULL) {
        fprintf(stderr, "Out of memory.\n");
        return NULL;
    }

    if (tokenizer_encode(s->tok, text, add_bos, 0, ids, capacity, &count,
                         error, sizeof(error)) != 0) {
        fprintf(stderr, "Cannot encode prompt: %s\n", error);
        free(ids);
        return NULL;
    }

    if (s->verbose) {
        fprintf(stderr, "[%zu prompt tokens:", count);
        for (size_t i = 0; i < count; i++) {
            fprintf(stderr, " %d", ids[i]);
        }
        fprintf(stderr, "]\n");
    }

    double start = now_seconds();
    logits = feed(s, ids, count);
    *seconds = now_seconds() - start;
    *n_tokens = count;

    free(ids);
    return logits;
}

/*
 * Sample, print and feed tokens until EOS, the token limit, or a full
 * context. Prints the reply as it is produced.
 * strip_first_space: drop the SentencePiece leading space of the first
 * piece (chat replies start right after "<|assistant|>\n").
 */
static int generate(Session *s, float *logits, int strip_first_space,
                    int *generated, int *stopped_by_eos, double *seconds)
{
    const int seq_len = engine_config(s->model)->seq_len;
    double start = now_seconds();

    *generated = 0;
    *stopped_by_eos = 0;

    while (*generated < s->max_new) {
        int next = sample(s, logits);

        if (next < 0) {
            fprintf(stderr, "\n[sampler failed: invalid logits]\n");
            return 0;
        }

        if (engine_is_stop_token(s->model, next)) {
            *stopped_by_eos = 1;
            break;
        }

        char piece[256];
        if (tokenizer_decode(s->tok, &next, 1, piece, sizeof(piece),
                             NULL, 0) == 0) {
            const char *out = piece;
            if (strip_first_space && *generated == 0 && out[0] == ' ') {
                out++;
            }
            fputs(out, stdout);
            fflush(stdout);
        }
        (*generated)++;

        if (s->pos >= seq_len) {
            fprintf(stderr, "\n[context full]\n");
            break;
        }

        logits = feed(s, &next, 1);
        if (logits == NULL) {
            return 0;
        }
    }

    *seconds = now_seconds() - start;
    return 1;
}

static void print_stats(const Session *s, size_t n_prompt, double prompt_s,
                        int n_gen, double gen_s, int eos)
{
    fprintf(stderr,
            "\n[prompt %zu tokens %.2f s (%.2f tok/s) | reply %d tokens "
            "%.2f s (%.2f tok/s)%s | context %d/%d]\n",
            n_prompt, prompt_s, prompt_s > 0 ? (double)n_prompt / prompt_s : 0.0,
            n_gen, gen_s, gen_s > 0 ? (double)n_gen / gen_s : 0.0,
            eos ? " | stopped at end token" : " | stopped at token limit",
            s->pos, engine_config(s->model)->seq_len);
}

/* ------------------------------------------------------------------ */
/* Modes                                                               */
/* ------------------------------------------------------------------ */

/* Build one chat turn. The first turn includes BOS and the system text. */
static char *chat_turn_text(const char *system, const char *message, int first)
{
    size_t size = strlen(message) + (system ? strlen(system) : 0) + 128;
    char *text = malloc(size);

    if (text == NULL) {
        return NULL;
    }

    if (first) {
        if (system != NULL && system[0] != '\0') {
            snprintf(text, size,
                     "<|system|>\n%s</s>\n<|user|>\n%s</s>\n<|assistant|>\n",
                     system, message);
        } else {
            snprintf(text, size, "<|user|>\n%s</s>\n<|assistant|>\n", message);
        }
    } else {
        /* The previous reply's EOS is already in the cache. */
        snprintf(text, size, "\n<|user|>\n%s</s>\n<|assistant|>\n", message);
    }
    return text;
}

/* One chat turn; returns 1 on success. */
static int chat_turn(Session *s, const char *system, const char *message)
{
    const int first = (s->pos == 0);
    char *text = chat_turn_text(system, message, first);
    size_t n_prompt = 0;
    double prompt_s = 0.0, gen_s = 0.0;
    int n_gen = 0, eos = 0;

    if (text == NULL) {
        return 0;
    }

    float *logits = feed_text(s, text, first, &n_prompt, &prompt_s);
    free(text);
    if (logits == NULL) {
        return 0;
    }

    if (!generate(s, logits, 1, &n_gen, &eos, &gen_s)) {
        return 0;
    }

    /* Close the assistant turn with EOS so the next turn follows it. */
    if (s->pos < engine_config(s->model)->seq_len) {
        int eos_id = s->tok->eos_id;
        if (feed(s, &eos_id, 1) == NULL) {
            return 0;
        }
    }

    print_stats(s, n_prompt, prompt_s, n_gen, gen_s, eos);
    return 1;
}

/* Plain completion: print the prompt, then continue it. */
static int completion(Session *s, const char *prompt)
{
    size_t n_prompt = 0;
    double prompt_s = 0.0, gen_s = 0.0;
    int n_gen = 0, eos = 0;

    float *logits = feed_text(s, prompt, 1, &n_prompt, &prompt_s);
    if (logits == NULL) {
        return 0;
    }

    fputs(prompt, stdout);
    fflush(stdout);

    if (!generate(s, logits, 0, &n_gen, &eos, &gen_s)) {
        return 0;
    }
    printf("\n");
    print_stats(s, n_prompt, prompt_s, n_gen, gen_s, eos);
    return 1;
}

static int read_line(char *buffer, size_t capacity)
{
    if (fgets(buffer, (int)capacity, stdin) == NULL) {
        return 0;
    }
    buffer[strcspn(buffer, "\r\n")] = '\0';
    return 1;
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    Session s;
    char error[256];
    const char *prompt = NULL;
    const char *system = NULL;
    int chat = 0, interactive = 0, seed_given = 0;
    LLMOptions options;

    memset(&options, 0, sizeof(options));
    unsigned long seed = 0;
    int status = 1;

    memset(&s, 0, sizeof(s));
    s.sampler = "top-k";
    s.temperature = 0.7f;
    s.top_k = 40;
    s.max_new = 256;

    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    for (int i = 2; i < argc; i++) {
        const char *a = argv[i];
        const int has_value = (i + 1 < argc);

        if (strcmp(a, "-n") == 0 && has_value) {
            s.max_new = atoi(argv[++i]);
        } else if (strcmp(a, "--sampler") == 0 && has_value) {
            s.sampler = argv[++i];
        } else if (strcmp(a, "--temperature") == 0 && has_value) {
            s.temperature = (float)atof(argv[++i]);
        } else if (strcmp(a, "--top-k") == 0 && has_value) {
            s.top_k = atoi(argv[++i]);
        } else if (strcmp(a, "--seed") == 0 && has_value) {
            errno = 0;
            seed = strtoul(argv[++i], NULL, 10);
            seed_given = (errno == 0);
        } else if (strcmp(a, "--ctx") == 0 && has_value) {
            options.max_context = atoi(argv[++i]);
            if (options.max_context <= 0) {
                fprintf(stderr, "--ctx must be positive.\n");
                return 1;
            }
        } else if (strcmp(a, "--tokenizer") == 0 && has_value) {
            options.tokenizer_path = argv[++i];
        } else if (strcmp(a, "--system") == 0 && has_value) {
            system = argv[++i];
        } else if (strcmp(a, "--chat") == 0) {
            chat = 1;
        } else if (strcmp(a, "-i") == 0) {
            interactive = 1;
        } else if (strcmp(a, "-v") == 0) {
            s.verbose = 1;
        } else if (a[0] == '-' && a[1] != '\0') {
            usage(argv[0]);
            return 1;
        } else {
            prompt = a;
        }
    }

    if (strcmp(s.sampler, "greedy") != 0 &&
        strcmp(s.sampler, "temperature") != 0 &&
        strcmp(s.sampler, "top-k") != 0) {
        fprintf(stderr, "Unknown sampler '%s'.\n", s.sampler);
        return 1;
    }
    if (s.max_new <= 0 || s.temperature <= 0.0f || s.top_k <= 0) {
        fprintf(stderr, "-n, --temperature and --top-k must be positive.\n");
        return 1;
    }
    if (!interactive && prompt == NULL) {
        fprintf(stderr, "A prompt is required unless -i is used.\n");
        usage(argv[0]);
        return 1;
    }

    if (!seed_given) {
        seed = (unsigned long)time(NULL);
    }
    s.rng = (uint32_t)seed;
    if (s.rng == 0) {
        s.rng = 1;
    }

    /* ---- Load model and tokenizer (format detected from the file) ---- */
    double t0 = now_seconds();

    options.model_path = argv[1];
    s.model = engine_open(&options, error, sizeof(error));
    if (s.model == NULL) {
        fprintf(stderr, "Cannot load model: %s\n", error);
        return 1;
    }
    s.tok = engine_tokenizer(s.model);

    const Config *c = engine_config(s.model);
    const int is_gguf = engine_format(s.model) == LLM_FORMAT_GGUF;

    if (s.tok->vocab_size != (size_t)c->vocab_size) {
        fprintf(stderr, "Tokenizer has %zu tokens but the model has %d.\n",
                s.tok->vocab_size, c->vocab_size);
        goto cleanup;
    }
    if (s.top_k > c->vocab_size) {
        s.top_k = c->vocab_size;
    }

    fprintf(stderr,
            "[%s | layers %d, dim %d, heads %d/%d kv, vocab %d | RoPE base %g "
            "scale %g | context %d of %d (KV cache %.1f MB) | loaded in %.2f s]\n"
            "[sampler %s, temperature %.2f, top-k %d, seed %lu]\n",
            engine_description(s.model), c->n_layers, c->dim, c->n_heads,
            c->n_kv_heads, c->vocab_size, (double)c->rope_freq_base,
            (double)c->rope_scale, c->seq_len,
            engine_file_context(s.model),
            (double)engine_kv_cache_bytes(s.model) / (1024.0 * 1024.0),
            now_seconds() - t0, s.sampler, (double)s.temperature, s.top_k,
            seed);

    if ((chat || interactive) && !is_gguf) {
        fprintf(stderr, "[llama2.c models are not chat models: %s]\n",
                interactive ? "each line is a new completion prompt"
                            : "--chat ignored, plain completion");
        chat = 0;
    }

    /* ---- Interactive chat ---- */
    if (interactive) {
        char line[MAX_INPUT];

        fprintf(stderr, "Interactive chat. Commands: /reset, /exit\n");
        for (;;) {
            fprintf(stderr, "\n> ");
            fflush(stderr);
            if (!read_line(line, sizeof(line))) {
                break;
            }
            if (line[0] == '\0') {
                continue;
            }
            if (strcmp(line, "/exit") == 0 || strcmp(line, "/quit") == 0) {
                break;
            }
            if (strcmp(line, "/reset") == 0) {
                s.pos = 0;
                fprintf(stderr, "[conversation reset]\n");
                continue;
            }
            printf("\n");
            if (!is_gguf) {
                s.pos = 0;
                if (!completion(&s, line)) {
                    fprintf(stderr, "[completion failed]\n");
                }
                continue;
            }
            if (!chat_turn(&s, system, line)) {
                fprintf(stderr, "[turn failed; use /reset to start over]\n");
            }
        }
        status = 0;
        goto cleanup;
    }

    /* ---- Single chat turn ---- */
    if (chat) {
        status = chat_turn(&s, system, prompt) ? 0 : 1;
        printf("\n");
        goto cleanup;
    }

    /* ---- Plain completion ---- */
    status = completion(&s, prompt) ? 0 : 1;

cleanup:
    engine_close(s.model);
    return status;
}
