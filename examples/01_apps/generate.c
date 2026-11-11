

#include "llm.h"
#include "sampler.h"
#include "tokenizer.h"
#include "transformer.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_MAX_NEW_TOKENS 128
#define DEFAULT_TOP_K 40
#define MAX_PROMPT_BYTES 4096
#define PIECE_CAPACITY 64 

static void usage(const char *program)
{
    fprintf(stderr,
            "Usage: %s [-m model.bin] [-t tokenizer.bin] "
            "[-n max_new_tokens] [--sampler greedy|temperature|top-k] "
            "[--temperature T] [--top-k K] [--seed N] [prompt]\n",
            program);
}

/* Parse a positive integer. Returns 0 on success, -1 on bad input. */
static int parse_positive_int(const char *text, int *out)
{
    char *end = NULL;
    long value;

    errno = 0;
    value = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value <= 0 || value > 1000000) {
        return -1;
    }
    *out = (int)value;
    return 0;
}

/* True if the string is empty or only whitespace. */
static int is_blank(const char *text)
{
    for (; *text != '\0'; ++text) {
        if (*text != ' ' && *text != '\t' &&
            *text != '\n' && *text != '\r') {
            return 0;
        }
    }
    return 1;
}


static int generate_one(Transformer *model, Tokenizer *tokenizer,
                        const char *prompt, int max_new_tokens,
                        const char *sampler_mode, float temperature,
                        int top_k, uint32_t *rng_state)
{
    const Config *config = llm_model_config(model);
    char error[256] = "";
    int *tokens = NULL;
    char *prompt_text = NULL;
    size_t prompt_len = 0;
    int status = -1;

    if (is_blank(prompt)) {
        fprintf(stderr, "Error: the prompt is empty.\n");
        return -1;
    }
    if (strlen(prompt) >= MAX_PROMPT_BYTES) {
        fprintf(stderr, "Error: prompt is longer than %d bytes.\n",
                MAX_PROMPT_BYTES - 1);
        return -1;
    }

    /*
     * Worst case: one token per input byte, plus BOS and the dummy
     * leading space the tokenizer adds. +1 for safety.
     */
    size_t token_capacity = strlen(prompt) + 3;
    size_t text_capacity = strlen(prompt) + 64;

    tokens = malloc(token_capacity * sizeof(*tokens));
    prompt_text = malloc(text_capacity);
    if (tokens == NULL || prompt_text == NULL) {
        fprintf(stderr, "Error: out of memory.\n");
        goto cleanup;
    }

    /* Encode: text -> token IDs (BOS first). */
    if (tokenizer_encode(tokenizer, prompt, 1, 0, tokens, token_capacity,
                         &prompt_len, error, sizeof(error)) != 0) {
        fprintf(stderr, "Error: cannot encode prompt: %s\n", error);
        goto cleanup;
    }

    /* Defensive: every ID must be valid for the model's embedding table. */
    for (size_t i = 0; i < prompt_len; ++i) {
        if (tokens[i] < 0 || tokens[i] >= config->vocab_size) {
            fprintf(stderr, "Error: token %zu has invalid ID %d.\n",
                    i, tokens[i]);
            goto cleanup;
        }
    }

    /* The model has seq_len positions (0 .. seq_len-1) in total. */
    if (prompt_len >= (size_t)config->seq_len) {
        fprintf(stderr,
                "Error: prompt is %zu tokens, but the model's context is "
                "%d tokens. Shorten the prompt.\n",
                prompt_len, config->seq_len);
        goto cleanup;
    }

    /* Print the prompt text, decoded from its tokens. Decoding the full
     * array (BOS first) also strips the dummy leading space. */
    if (tokenizer_decode(tokenizer, tokens, prompt_len, prompt_text,
                         text_capacity, error, sizeof(error)) != 0) {
        fprintf(stderr, "Error: cannot decode prompt: %s\n", error);
        goto cleanup;
    }
    printf("\nGenerated text:\n%s", prompt_text);
    fflush(stdout);

    /*
     * One iteration = feed ONE token at position `pos` into forward().
     *   - While pos is still inside the prompt, the model's prediction is
     *     ignored: the next token is simply the next prompt token
     *     (this fills the key/value cache with the prompt).
     *   - Once the prompt is used up, the prediction is sampled and
     *     becomes the next input (autoregressive generation).
     */
    int limit = (int)prompt_len + max_new_tokens;
    if (limit > config->seq_len) {
        limit = config->seq_len; /* positions beyond seq_len don't exist */
    }

    int token = tokens[0];
    int generated = 0;
    int stopped_by_eos = 0;

    for (int pos = 0; pos < limit - 1; ++pos) {
        float *logits = forward(model, token, pos);
        if (logits == NULL) {
            fprintf(stderr, "\nError: forward() failed at position %d.\n",
                    pos);
            goto cleanup;
        }

        int next;
        if ((size_t)pos + 1 < prompt_len) {
            next = tokens[pos + 1]; /* still reading the prompt */
        } else {
            if (strcmp(sampler_mode, "greedy") == 0) {
                next = sampler_greedy(
                    logits,
                    (size_t)config->vocab_size
                );
            } else if (strcmp(sampler_mode, "temperature") == 0) {
                next = sampler_temperature(
                    logits,
                    (size_t)config->vocab_size,
                    temperature,
                    rng_state
                );
            } else if (strcmp(sampler_mode, "top-k") == 0) {
                next = sampler_top_k(
                    logits,
                    (size_t)config->vocab_size,
                    (size_t)top_k,
                    temperature,
                    rng_state
                );
            } else {
                fprintf(stderr, "\nError: unknown sampler mode '%s'.\n",
                        sampler_mode);
                goto cleanup;
            }
            if (next < 0 || next >= config->vocab_size) {
                fprintf(stderr, "\nError: sampler failed (bad logits).\n");
                goto cleanup;
            }
            /* Stop on either special token (see notes). */
            if (next == TOKENIZER_BOS_ID || next == TOKENIZER_EOS_ID) {
                stopped_by_eos = 1;
                break;
            }

            char piece[PIECE_CAPACITY];
            if (tokenizer_decode(tokenizer, &next, 1, piece, sizeof(piece),
                                 error, sizeof(error)) != 0) {
                fprintf(stderr, "\nError: cannot decode token %d: %s\n",
                        next, error);
                goto cleanup;
            }
            fputs(piece, stdout);
            fflush(stdout); /* show each token as it is produced */
            ++generated;
        }
        token = next;
    }

    printf("\n\n[%zu prompt tokens, %d generated%s]\n", prompt_len,
           generated, stopped_by_eos ? ", stopped at end token"
                                     : ", reached length limit");
    status = 0;

cleanup:
    free(prompt_text);
    free(tokens);
    return status;
}


static int read_line(char *buffer, size_t capacity)
{
    if (fgets(buffer, (int)capacity, stdin) == NULL) {
        return 1;
    }
    size_t length = strcspn(buffer, "\r\n");
    if (buffer[length] == '\0') {
        /* No newline found: either a too-long line or EOF without \n. */
        if (length == capacity - 1) {
            int c;
            while ((c = getchar()) != '\n' && c != EOF) {
            }
            return -1;
        }
    }
    buffer[length] = '\0';
    return 0;
}

int main(int argc, char **argv)
{
    const char *model_path = "models/stories42M.bin";
    const char *tokenizer_path = "models/tokenizer.bin";
    const char *prompt = NULL;
    int max_new_tokens = DEFAULT_MAX_NEW_TOKENS;
    const char *sampler_mode = "greedy";
    float temperature = 1.0f;
    int top_k = DEFAULT_TOP_K;
    uint32_t rng_state = UINT32_C(123456789);

    /* Which sampler options did the user type? (to warn if they're unused) */
    int temperature_given = 0;
    int top_k_given = 0;
    int seed_given = 0;

    char error[256] = "";
    char line[MAX_PROMPT_BYTES];

    /* Everything that needs cleanup starts as "nothing yet". */
    Transformer *model = NULL;
    Tokenizer tokenizer = {0};
    int tokenizer_ready = 0;
    int status = 1;

    /* ---- 1. Command-line arguments -------------------------------- */
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0) {
            if (i + 1 >= argc ||
                parse_positive_int(argv[i + 1], &max_new_tokens) != 0) {
                fprintf(stderr, "Error: -n needs a positive integer.\n");
                return 1;
            }
            ++i;
        } else if (strcmp(argv[i], "-m") == 0 ||
                   strcmp(argv[i], "-t") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: %s needs a file path.\n", argv[i]);
                return 1;
            }
            if (argv[i][1] == 'm') {
                model_path = argv[i + 1];
            } else {
                tokenizer_path = argv[i + 1];
            }
            ++i;
        } else if (strcmp(argv[i], "--sampler") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --sampler needs a value.\n");
                return 1;
            }
            sampler_mode = argv[++i];
            if (strcmp(sampler_mode, "greedy") != 0 &&
                strcmp(sampler_mode, "temperature") != 0 &&
                strcmp(sampler_mode, "top-k") != 0) {
                fprintf(stderr, "Error: sampler must be greedy, "
                                "temperature or top-k.\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--temperature") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "Error: --temperature needs a value.\n");
                return 1;
            }
            char *end = NULL;
            temperature = strtof(argv[++i], &end);
            if (end == argv[i] || *end != '\0' ||
                !isfinite(temperature) || temperature <= 0.0f) {
                fprintf(stderr, "Error: temperature must be a positive "
                                "finite number.\n");
                return 1;
            }
            temperature_given = 1;
        } else if (strcmp(argv[i], "--top-k") == 0) {
            if (i + 1 >= argc ||
                parse_positive_int(argv[i + 1], &top_k) != 0) {
                fprintf(stderr, "Error: --top-k needs a positive integer.\n");
                return 1;
            }
            ++i;
            top_k_given = 1;
        } else if (strcmp(argv[i], "--seed") == 0) {
            char *end = NULL;
            unsigned long seed;
            if (i + 1 >= argc || argv[i + 1][0] == '-') {
                fprintf(stderr, "Error: --seed needs a whole number "
                                "from 0 to 4294967295.\n");
                return 1;
            }
            errno = 0;
            seed = strtoul(argv[i + 1], &end, 10);
            if (errno != 0 || end == argv[i + 1] || *end != '\0' ||
                seed > UINT32_MAX) {
                fprintf(stderr, "Error: --seed needs a whole number "
                                "from 0 to 4294967295.\n");
                return 1;
            }
            rng_state = (uint32_t)seed;
            ++i;
            seed_given = 1;
        } else if (argv[i][0] == '-' && argv[i][1] == '-') {
            /* A mistyped option must not silently become the prompt. */
            fprintf(stderr, "Error: unknown option '%s'.\n", argv[i]);
            usage(argv[0]);
            return 1;
        } else if (prompt == NULL) {
            prompt = argv[i];
        } else {
            fprintf(stderr, "Error: unexpected argument '%s'. "
                            "Quote the prompt if it has spaces.\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    /* Options that the chosen sampler ignores: say so instead of hiding it. */
    if (strcmp(sampler_mode, "greedy") == 0 &&
        (temperature_given || top_k_given || seed_given)) {
        fprintf(stderr, "Note: greedy sampling ignores --temperature, "
                        "--top-k and --seed.\n");
    } else if (strcmp(sampler_mode, "temperature") == 0 && top_k_given) {
        fprintf(stderr, "Note: the temperature sampler ignores --top-k.\n");
    }

    /* ---- 2. Load model (mmap + buffers) ONCE ----------------------- */
    /* Note: model.c calls exit() itself on a bad checkpoint. */
    model = llm_load(model_path, error, sizeof(error));
    if (model == NULL) {
        fprintf(stderr, "Error: %s\n", error);
        goto cleanup;
    }
    const Config *config = llm_model_config(model);

    /* ---- 3. Load tokenizer using the MODEL's vocabulary size ------- */
    if (tokenizer_load(&tokenizer, tokenizer_path,
                       (size_t)config->vocab_size,
                       error, sizeof(error)) != 0) {
        fprintf(stderr, "Error: %s (does the tokenizer match the model's "
                        "vocab size of %d?)\n", error, config->vocab_size);
        goto cleanup;
    }
    tokenizer_ready = 1;

    if (strcmp(sampler_mode, "top-k") == 0 &&
        top_k > config->vocab_size) {
        fprintf(stderr, "Error: --top-k %d is larger than the vocabulary "
                        "(%d tokens).\n", top_k, config->vocab_size);
        goto cleanup;
    }

    /* ---- 4. One prompt from the command line, or interactive ------- */
    if (prompt != NULL) {
        if (generate_one(model, &tokenizer, prompt, max_new_tokens,
                         sampler_mode, temperature, top_k,
                         &rng_state) == 0) {
            status = 0;
        }
    } else {
        printf("Interactive mode. Empty line, 'quit' or 'exit' ends it.\n");
        for (;;) {
            printf("\nEnter a prompt: ");
            fflush(stdout);

            int result = read_line(line, sizeof(line));
            if (result == 1 || (result == 0 && (is_blank(line) ||
                strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0))) {
                break;
            }
            if (result < 0) {
                fprintf(stderr, "Error: prompt is longer than %d bytes.\n",
                        MAX_PROMPT_BYTES - 1);
                continue;
            }
            /* An error on one prompt doesn't end the session. */
            (void)generate_one(model, &tokenizer, line, max_new_tokens,
                               sampler_mode, temperature, top_k,
                               &rng_state);
        }
        status = 0;
    }

cleanup:
    /* tokenizer_free(zeroed) and llm_free_model(NULL) are safe. */
    if (tokenizer_ready) {
        tokenizer_free(&tokenizer);
    }
    llm_free_model(model);
    return status;
}