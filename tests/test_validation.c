/*
 * test_validation.c - checks that forward() rejects bad input
 * instead of reading or writing out of bounds.
 *
 * Usage: ./build/test_validation <checkpoint.bin>
 */

#include "transformer.h"

#include <stdio.h>
#include <stdlib.h>

static int failures = 0;

static void expect_null(const char *name, const float *result)
{
    printf("%-40s %s\n", name, result == NULL ? "PASS" : "FAIL");
    if (result != NULL) {
        failures++;
    }
}

static void expect_ok(const char *name, const float *result)
{
    printf("%-40s %s\n", name, result != NULL ? "PASS" : "FAIL");
    if (result == NULL) {
        failures++;
    }
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <checkpoint.bin>\n", argv[0]);
        return EXIT_FAILURE;
    }

    Transformer model = {0};
    model.fd = -1;
    char error[256];

    if (!build_transformer(&model, argv[1], error, sizeof(error))) {
        fprintf(stderr, "FAIL: cannot load %s: %s\n", argv[1], error);
        return EXIT_FAILURE;
    }

    Config saved = model.config;
    int vocab = saved.vocab_size;
    int seq = saved.seq_len;

    printf("=== Normal and boundary cases ===\n");
    expect_ok("first token, pos 0", forward(&model, 0, 0));
    expect_ok("last token, pos 1", forward(&model, vocab - 1, 1));
    expect_ok("last valid position", forward(&model, 1, seq - 1));

    printf("\n=== Invalid input ===\n");
    expect_null("NULL model", forward(NULL, 1, 0));
    expect_null("negative token", forward(&model, -1, 0));
    expect_null("token == vocab_size", forward(&model, vocab, 0));
    expect_null("negative position", forward(&model, 1, -1));
    expect_null("position == seq_len", forward(&model, 1, seq));

    printf("\n=== Invalid configuration (in memory) ===\n");

    /* Odd head_size: keep n_heads, shrink dim so dim/n_heads is odd. */
    model.config.dim = saved.n_heads * 3;
    expect_null("odd head_size", forward(&model, 1, 0));
    model.config = saved;

    /* dim not divisible by n_heads. */
    model.config.dim = saved.dim + 1;
    expect_null("dim % n_heads != 0", forward(&model, 1, 0));
    model.config = saved;

    /* n_heads not divisible by n_kv_heads. */
    model.config.n_kv_heads = saved.n_heads + 1;
    expect_null("n_kv_heads > n_heads", forward(&model, 1, 0));
    model.config = saved;

    model.config.n_layers = 0;
    expect_null("zero layers", forward(&model, 1, 0));
    model.config = saved;

    /* Model still works after all the rejected calls. */
    printf("\n=== Recovery ===\n");
    expect_ok("valid call after invalid ones", forward(&model, 1, 0));

    free_transformer(&model);

    printf("\n%s (%d failure(s))\n",
           failures == 0 ? "ALL VALIDATION TESTS PASSED" : "SOME TESTS FAILED",
           failures);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}