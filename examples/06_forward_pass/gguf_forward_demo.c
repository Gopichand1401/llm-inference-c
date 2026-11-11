/*
 * gguf_forward_demo.c - run the GGUF forward pass on a prompt.
 *
 * Usage:
 *   gguf_forward_demo <model.gguf> ["prompt"] [n_generate]
 *   gguf_forward_demo <model.gguf> --ids 1,450,7483 [n_generate]
 *
 * 1. Encodes the prompt (BOS + text) with the tokenizer stored in the GGUF
 *    file itself (tokenizer_load_gguf), or takes token IDs with --ids.
 * 2. Runs gguf_forward() for every prompt position and saves all logits to
 *    build/forward_logits.bin and the token IDs to build/forward_tokens.bin,
 *    for examples/06_forward_pass/reference_forward.py to check.
 * 3. Prints the top 5 next-token predictions after the prompt.
 * 4. Greedy-generates n_generate more tokens (default 16) as a preview.
 */
#include "gguf_forward.h"
#include "tokenizer.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_TOKENS 512

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Print one token as text, or as its ID if there is no tokenizer. */
static void print_piece(const Tokenizer *tok, int id)
{
    if (tok == NULL) {
        printf("[%d]", id);
        return;
    }

    char text[128];
    if (tokenizer_decode(tok, &id, 1, text, sizeof(text), NULL, 0) == 0) {
        printf("%s", text);
    } else {
        printf("[%d]", id);
    }
}

/* Indices of the 5 largest logits, in descending order. */
static void top5(const float *logits, int n, int out[5])
{
    for (int k = 0; k < 5; k++) {
        out[k] = -1;
    }
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < 5; k++) {
            if (out[k] < 0 || logits[i] > logits[out[k]]) {
                for (int j = 4; j > k; j--) {
                    out[j] = out[j - 1];
                }
                out[k] = i;
                break;
            }
        }
    }
}

static int parse_ids(const char *text, int *ids, int capacity)
{
    int count = 0;
    char *end = NULL;

    while (*text != '\0' && count < capacity) {
        long v = strtol(text, &end, 10);
        if (end == text || v < 0 || v > 1000000) {
            return -1;
        }
        ids[count++] = (int)v;
        text = (*end == ',') ? end + 1 : end;
        if (*end != ',' && *end != '\0') {
            return -1;
        }
    }
    return count;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <model.gguf> [\"prompt\"] [n_generate]\n"
                "       %s <model.gguf> --ids 1,450,7483 [n_generate]\n",
                argv[0], argv[0]);
        return 1;
    }

    GGUFTransformer t;
    Tokenizer tokenizer;
    Tokenizer *tok = NULL;
    char error[256];
    int status = 1;
    int tokens[MAX_TOKENS];
    int n_prompt = 0;
    int n_generate = 16;
    FILE *logits_file = NULL;

    memset(&tokenizer, 0, sizeof(tokenizer));

    double t0 = now_seconds();

    if (!gguf_transformer_init(&t, argv[1], error, sizeof(error))) {
        fprintf(stderr, "Init failed: %s\n", error);
        return 1;
    }

    const Config *c = &t.model.config;
    printf("Model loaded in %.2f s (dim=%d layers=%d heads=%d kv_heads=%d "
           "vocab=%d)\n", now_seconds() - t0, c->dim, c->n_layers,
           c->n_heads, c->n_kv_heads, c->vocab_size);

    /* ---- Prompt tokens ---- */
    if (argc > 2 && strcmp(argv[2], "--ids") == 0) {
        if (argc < 4 ||
            (n_prompt = parse_ids(argv[3], tokens, MAX_TOKENS)) <= 0) {
            fprintf(stderr, "Bad --ids list.\n");
            goto cleanup;
        }
        if (argc > 4) {
            n_generate = atoi(argv[4]);
        }
    } else {
        const char *prompt = (argc > 2) ? argv[2] : "The capital of France is";
        if (argc > 3) {
            n_generate = atoi(argv[3]);
        }

        if (tokenizer_load_gguf(&tokenizer, argv[1], error, sizeof(error)) != 0) {
            fprintf(stderr, "Tokenizer: %s\n", error);
            goto cleanup;
        }
        if (tokenizer.vocab_size != (size_t)c->vocab_size) {
            fprintf(stderr, "Tokenizer has %zu tokens but the model has %d.\n",
                    tokenizer.vocab_size, c->vocab_size);
            goto cleanup;
        }
        tok = &tokenizer;

        size_t count = 0;
        if (tokenizer_encode(tok, prompt, 1, 0, tokens, MAX_TOKENS, &count,
                             error, sizeof(error)) != 0) {
            fprintf(stderr, "Encode: %s\n", error);
            goto cleanup;
        }
        n_prompt = (int)count;
        printf("Prompt: \"%s\"\n", prompt);
    }

    if (n_generate < 0) {
        n_generate = 0;
    }
    if (n_prompt + n_generate > c->seq_len) {
        n_generate = c->seq_len - n_prompt;
    }
    if (n_prompt + n_generate > MAX_TOKENS) {
        n_generate = MAX_TOKENS - n_prompt;
    }

    printf("Prompt tokens (%d):", n_prompt);
    for (int i = 0; i < n_prompt; i++) {
        printf(" %d", tokens[i]);
        if (tokens[i] < 0 || tokens[i] >= c->vocab_size) {
            fprintf(stderr, "\nToken %d is outside the vocabulary.\n", tokens[i]);
            goto cleanup;
        }
    }
    printf("\n");
    if (tok != NULL) {
        printf("Pieces:");
        for (int i = 0; i < n_prompt; i++) {
            printf(" |");
            print_piece(tok, tokens[i]);
        }
        printf("|\n");
    }

    /* Save token IDs: int32 count, then int32 IDs. */
    {
        FILE *f = fopen("build/forward_tokens.bin", "wb");
        int32_t n = n_prompt;
        int ok = f != NULL &&
                 fwrite(&n, sizeof(n), 1, f) == 1 &&
                 fwrite(tokens, sizeof(int), (size_t)n_prompt, f)
                     == (size_t)n_prompt;
        if (f != NULL) {
            fclose(f);
        }
        if (!ok) {
            fprintf(stderr, "Cannot write build/forward_tokens.bin\n");
            goto cleanup;
        }
    }

    logits_file = fopen("build/forward_logits.bin", "wb");
    if (logits_file == NULL) {
        fprintf(stderr, "Cannot open build/forward_logits.bin\n");
        goto cleanup;
    }

    /* ---- Prompt forward passes ---- */
    float *logits = NULL;
    double start = now_seconds();

    for (int pos = 0; pos < n_prompt; pos++) {
        logits = gguf_forward(&t, tokens[pos], pos);
        if (logits == NULL) {
            fprintf(stderr, "gguf_forward failed at position %d\n", pos);
            goto cleanup;
        }
        for (int i = 0; i < c->vocab_size; i++) {
            if (!isfinite(logits[i])) {
                fprintf(stderr, "Non-finite logit at pos %d id %d\n", pos, i);
                goto cleanup;
            }
        }
        if (fwrite(logits, sizeof(float), (size_t)c->vocab_size, logits_file)
                != (size_t)c->vocab_size) {
            fprintf(stderr, "Cannot write logits.\n");
            goto cleanup;
        }
    }

    fclose(logits_file);
    logits_file = NULL;

    double prompt_time = now_seconds() - start;
    printf("Prompt processed: %d tokens in %.2f s (%.2f tokens/s)\n",
           n_prompt, prompt_time, n_prompt / prompt_time);

    /* ---- Top-5 next tokens after the prompt ---- */
    {
        int best[5];
        float max_logit = logits[0];
        double sum = 0.0;

        top5(logits, c->vocab_size, best);
        for (int i = 1; i < c->vocab_size; i++) {
            if (logits[i] > max_logit) max_logit = logits[i];
        }
        for (int i = 0; i < c->vocab_size; i++) {
            sum += exp((double)logits[i] - (double)max_logit);
        }

        printf("\nTop 5 next tokens:\n");
        for (int k = 0; k < 5; k++) {
            double prob = exp((double)logits[best[k]] - (double)max_logit) / sum;
            printf("  %6d  logit %8.4f  p=%.4f  '", best[k],
                   (double)logits[best[k]], prob);
            print_piece(tok, best[k]);
            printf("'\n");
        }
    }

    /* ---- Greedy preview ---- */
    if (n_generate > 0) {
        printf("\nGreedy continuation (%d tokens): ", n_generate);
        fflush(stdout);

        int pos = n_prompt;
        int next = 0;
        int best[5];

        top5(logits, c->vocab_size, best);
        next = best[0];
        start = now_seconds();
        int generated = 0;

        for (int g = 0; g < n_generate; g++) {
            if (next == (tok != NULL ? tok->eos_id : TOKENIZER_EOS_ID)) {
                printf("<EOS>");
                break;
            }
            print_piece(tok, next);
            fflush(stdout);
            tokens[pos] = next;

            logits = gguf_forward(&t, next, pos);
            if (logits == NULL) {
                fprintf(stderr, "\ngguf_forward failed at position %d\n", pos);
                goto cleanup;
            }
            pos++;
            generated++;
            top5(logits, c->vocab_size, best);
            next = best[0];
        }

        double gen_time = now_seconds() - start;
        printf("\nGenerated %d tokens in %.2f s (%.2f tokens/s)\n",
               generated, gen_time, generated > 0 ? generated / gen_time : 0.0);
    }

    printf("\nSaved %d positions of logits to build/forward_logits.bin\n",
           n_prompt);
    printf("GGUF forward pass: COMPLETED (run reference_forward.py to verify)\n");
    status = 0;

cleanup:
    if (logits_file != NULL) {
        fclose(logits_file);
    }
    tokenizer_free(&tokenizer);
    gguf_transformer_free(&t);
    return status;
}
