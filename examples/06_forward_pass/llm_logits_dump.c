/*
 * llm_logits_dump.c - run a prompt through ANY supported model (via
 * engine.h) and save the logits of every position.
 *
 * Usage:
 *   llm_logits_dump <model file> ["prompt"]
 *   llm_logits_dump <model file> --ids 1,450,7483
 *   (llama2.c models: tokenizer.bin next to the model or in the current
 *    directory, as for llm_generate)
 *
 * Writes the same files as gguf_forward_demo:
 *   build/forward_tokens.bin   int32 count, then int32 token IDs
 *   build/forward_logits.bin   float32 [positions][vocab]
 * which examples/06_forward_pass/reference_stories.py (llama2.c) and
 * examples/06_forward_pass/reference_forward.py (GGUF) check against NumPy.
 */
#include "engine.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TOKENS 512

static int parse_ids(const char *text, int *ids, int capacity)
{
    int count = 0;

    while (*text != '\0' && count < capacity) {
        char *end = NULL;
        long v = strtol(text, &end, 10);
        if (end == text || v < 0 || v > 1000000 ||
            (*end != ',' && *end != '\0')) {
            return -1;
        }
        ids[count++] = (int)v;
        text = (*end == ',') ? end + 1 : end;
    }
    return count;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model file> [\"prompt\" | --ids 1,2,3]\n",
                argv[0]);
        return 1;
    }

    LLMOptions options;
    char error[256];
    int tokens[MAX_TOKENS];
    int count = 0;
    int status = 1;
    FILE *out = NULL;

    memset(&options, 0, sizeof(options));
    options.model_path = argv[1];
    options.max_context = MAX_TOKENS;

    LLMModel *model = engine_open(&options, error, sizeof(error));
    if (model == NULL) {
        fprintf(stderr, "Cannot load model: %s\n", error);
        return 1;
    }

    const Config *c = engine_config(model);

    if (argc > 3 && strcmp(argv[2], "--ids") == 0) {
        count = parse_ids(argv[3], tokens, MAX_TOKENS);
        if (count <= 0) {
            fprintf(stderr, "Bad --ids list.\n");
            goto cleanup;
        }
    } else {
        const char *prompt = argc > 2 ? argv[2] : "Once upon a time";
        size_t n = 0;
        if (tokenizer_encode(engine_tokenizer(model), prompt, 1, 0, tokens,
                             MAX_TOKENS, &n, error, sizeof(error)) != 0) {
            fprintf(stderr, "Encode: %s\n", error);
            goto cleanup;
        }
        count = (int)n;
    }

    if (count > c->seq_len) {
        fprintf(stderr, "Too many tokens for the context (%d).\n", c->seq_len);
        goto cleanup;
    }

    printf("%s\n", engine_description(model));
    printf("Tokens (%d):", count);
    for (int i = 0; i < count; i++) {
        printf(" %d", tokens[i]);
        if (tokens[i] < 0 || tokens[i] >= c->vocab_size) {
            fprintf(stderr, "\nToken %d is outside the vocabulary.\n", tokens[i]);
            goto cleanup;
        }
    }
    printf("\n");

    {
        FILE *f = fopen("build/forward_tokens.bin", "wb");
        int32_t n = count;
        int ok = f != NULL && fwrite(&n, 4, 1, f) == 1 &&
                 fwrite(tokens, sizeof(int), (size_t)count, f) == (size_t)count;
        if (f != NULL) {
            fclose(f);
        }
        if (!ok) {
            fprintf(stderr, "Cannot write build/forward_tokens.bin\n");
            goto cleanup;
        }
    }

    out = fopen("build/forward_logits.bin", "wb");
    if (out == NULL) {
        fprintf(stderr, "Cannot open build/forward_logits.bin\n");
        goto cleanup;
    }

    for (int pos = 0; pos < count; pos++) {
        float *logits = engine_forward(model, tokens[pos], pos);
        if (logits == NULL) {
            fprintf(stderr, "Forward pass failed at position %d\n", pos);
            goto cleanup;
        }
        for (int i = 0; i < c->vocab_size; i++) {
            if (!isfinite(logits[i])) {
                fprintf(stderr, "Non-finite logit at %d/%d\n", pos, i);
                goto cleanup;
            }
        }
        if (fwrite(logits, sizeof(float), (size_t)c->vocab_size, out)
                != (size_t)c->vocab_size) {
            fprintf(stderr, "Cannot write logits.\n");
            goto cleanup;
        }
    }

    printf("Saved %d positions of logits to build/forward_logits.bin\n", count);
    printf("LOGITS DUMP: COMPLETED\n");
    status = 0;

cleanup:
    if (out != NULL) {
        fclose(out);
    }
    engine_close(model);
    return status;
}
