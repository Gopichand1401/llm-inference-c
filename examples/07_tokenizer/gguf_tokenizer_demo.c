/*
 * gguf_tokenizer_demo.c - test the tokenizer loaded from GGUF metadata.
 *
 * Usage: gguf_tokenizer_demo <model.gguf> [tokenizer.bin] ["extra text"]
 *
 * 1. Loads the tokenizer with tokenizer_load_gguf().
 * 2. Encodes a fixed set of test strings (plus an optional extra one),
 *    checks that decoding gives back the original text, and saves the IDs
 *    to build/tokenizer_cases.bin for examples/07_tokenizer/compare_tokenizer_spm.py.
 * 3. If tokenizer.bin is given, also compares with the llama2.c tokenizer.
 */
#include "tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_IDS 1024

static const char *cases[] = {
    "The capital of France is",
    "Hello world!",
    "hello world!",
    "  leading spaces",
    "trailing space ",
    "multiple   inner    spaces",
    "line1\nline2\n\nline4",
    "tab\tseparated\tvalues",
    "Numbers: 1234567890 and 3.14159",
    "Punctuation!?.,;:'\"()[]{}<>",
    "email@example.com https://example.org/path?q=1&r=2",
    "CamelCaseIdentifier snake_case_variable",
    "C code: int main(void) { return 0; }",
    "naive cafe resume / na\xc3\xafve caf\xc3\xa9 r\xc3\xa9sum\xc3\xa9",
    "\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82 \xd0\xbc\xd0\xb8\xd1\x80",
    "\xe0\xa4\xa8\xe0\xa4\xae\xe0\xa4\xb8\xe0\xa5\x8d\xe0\xa4\xa4\xe0\xa5\x87 "
        "\xe0\xa4\xa6\xe0\xa5\x81\xe0\xa4\xa8\xe0\xa4\xbf\xe0\xa4\xaf\xe0\xa4\xbe",
    "\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e\xe3\x81\xae\xe3\x83\x86\xe3\x82\xad"
        "\xe3\x82\xb9\xe3\x83\x88",
    "Emoji: \xf0\x9f\x98\x80\xf0\x9f\x9a\x80",
    "<s> literal special text </s>",
    "<0x41> literal byte text",
    "<|user|>\nWhat is the capital of France?</s>\n<|assistant|>\n",
    "a",
    " ",
    ""
};

static int encode(Tokenizer *t, const char *text, int *ids, size_t *count)
{
    char error[256];
    if (tokenizer_encode(t, text, 1, 0, ids, MAX_IDS, count,
                         error, sizeof(error)) != 0) {
        printf("ENCODE FAILED: %s\n", error);
        return 0;
    }
    return 1;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <model.gguf> [tokenizer.bin] [\"extra text\"]\n",
                argv[0]);
        return 1;
    }

    Tokenizer gguf_tok;
    Tokenizer bin_tok;
    int have_bin = 0;
    char error[256];
    int failures = 0;
    int bin_differences = 0;
    int status = 1;
    FILE *out = NULL;

    memset(&bin_tok, 0, sizeof(bin_tok));

    if (tokenizer_load_gguf(&gguf_tok, argv[1], error, sizeof(error)) != 0) {
        fprintf(stderr, "GGUF tokenizer: %s\n", error);
        return 1;
    }

    int bytes_found = 0;
    for (int b = 0; b < 256; b++) {
        bytes_found += gguf_tok.byte_ids[b] >= 0;
    }

    printf("=== GGUF tokenizer ===\n");
    printf("Vocabulary: %zu tokens, longest piece %zu bytes\n",
           gguf_tok.vocab_size, gguf_tok.max_token_length);
    printf("BOS id %d '%s', EOS id %d '%s'\n",
           gguf_tok.bos_id, gguf_tok.vocab[gguf_tok.bos_id],
           gguf_tok.eos_id, gguf_tok.vocab[gguf_tok.eos_id]);
    printf("Byte-fallback tokens found: %d / 256\n", bytes_found);
    printf("Special tokens recognised in text: %zu (", gguf_tok.n_special);
    for (size_t i = 0; i < gguf_tok.n_special && i < 8; i++) {
        printf("%s%s", i ? " " : "", gguf_tok.vocab[gguf_tok.special_ids[i]]);
    }
    printf("%s)\n", gguf_tok.n_special > 8 ? " ..." : "");
    printf("Merge scores: %s\n\n",
           gguf_tok.scores_reconstructed
               ? "all equal in the file -> rebuilt from token order"
               : "taken from the file");

    if (argc >= 3 && strcmp(argv[2], "-") != 0) {
        if (tokenizer_load(&bin_tok, argv[2], gguf_tok.vocab_size,
                           error, sizeof(error)) != 0) {
            fprintf(stderr, "tokenizer.bin: %s\n", error);
            goto cleanup;
        }
        have_bin = 1;
    }

    out = fopen("build/tokenizer_cases.bin", "wb");
    if (out == NULL) {
        fprintf(stderr, "Cannot open build/tokenizer_cases.bin\n");
        goto cleanup;
    }

    size_t n_cases = sizeof(cases) / sizeof(cases[0]);
    size_t total = n_cases + (argc >= 4 ? 1u : 0u);

    for (size_t k = 0; k < total; k++) {
        const char *text = (k < n_cases) ? cases[k] : argv[3];
        int ids[MAX_IDS];
        size_t count = 0;
        char decoded[8192];

        if (!encode(&gguf_tok, text, ids, &count)) {
            failures++;
            continue;
        }

        /*
         * Decoding drops BOS/EOS, so text containing special-token strings
         * cannot come back unchanged; only check the others.
         */
        int has_special = 0;
        for (size_t s = 0; s < gguf_tok.n_special; s++) {
            if (strstr(text, gguf_tok.vocab[gguf_tok.special_ids[s]]) != NULL) {
                has_special = 1;
            }
        }

        if (!has_special &&
            (tokenizer_decode(&gguf_tok, ids, count, decoded, sizeof(decoded),
                              error, sizeof(error)) != 0 ||
             strcmp(decoded, text) != 0)) {
            printf("ROUND-TRIP FAILED for case %zu\n", k);
            failures++;
        }

        printf("%2zu %3zu ids:", k, count);
        for (size_t i = 0; i < count && i < 12; i++) {
            printf(" %d", ids[i]);
        }
        printf("%s\n", count > 12 ? " ..." : "");

        if (have_bin && strstr(text, "<s>") == NULL &&
            strstr(text, "</s>") == NULL) {
            /* tokenizer.bin has no special-token parsing, so only compare
             * texts without special-token strings. */
            int bin_ids[MAX_IDS];
            size_t bin_count = 0;
            if (encode(&bin_tok, text, bin_ids, &bin_count) &&
                (bin_count != count ||
                 memcmp(bin_ids, ids, count * sizeof(int)) != 0)) {
                printf("   differs from tokenizer.bin (%zu ids there)\n",
                       bin_count);
                bin_differences++;
            }
        }

        /* Record: uint32 text_length, text, uint32 n_ids, int32 ids[]. */
        uint32_t text_length = (uint32_t)strlen(text);
        uint32_t n_ids = (uint32_t)count;
        if (fwrite(&text_length, 4, 1, out) != 1 ||
            fwrite(text, 1, text_length, out) != text_length ||
            fwrite(&n_ids, 4, 1, out) != 1 ||
            fwrite(ids, sizeof(int), count, out) != count) {
            fprintf(stderr, "Cannot write test cases.\n");
            goto cleanup;
        }
    }

    printf("\nCases: %zu, round-trip failures: %d\n", total, failures);
    if (have_bin) {
        printf("Cases where tokenizer.bin gives different IDs: %d "
               "(texts containing <s> or </s> are not compared)\n",
               bin_differences);
    }

    if (failures == 0) {
        printf("GGUF tokenizer C checks: PASSED\n");
        status = 0;
    } else {
        printf("GGUF tokenizer C checks: FAILED\n");
    }

cleanup:
    if (out != NULL) {
        fclose(out);
    }
    tokenizer_free(&gguf_tok);
    tokenizer_free(&bin_tok);
    return status;
}
