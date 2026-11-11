/*
 * test_edge_cases.c - boundary, empty, invalid and large-input tests for
 * the tokenizer (tokenizer.h) and the samplers (sampler.h).
 *
 * Complements test_tokenizer.c / test_sampler.c (normal cases) and
 * test_engine.c (model loading and the forward pass).
 *
 * Usage: test_edge_cases [tokenizer.bin]   (default: models/tokenizer.bin)
 */
#define _POSIX_C_SOURCE 200809L
#include "sampler.h"
#include "tokenizer.h"
#include "xlsx_reader.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int failures = 0;
static int checks = 0;

#define CHECK(cond, msg) do {                                        \
    ++checks;                                                        \
    if (!(cond)) {                                                   \
        ++failures;                                                  \
        fprintf(stderr, "  FAIL %s (line %d)\n", msg, __LINE__);     \
    }                                                                \
} while (0)

static const char *tok_path = "models/tokenizer.bin";

/* encode -> decode must give back the input exactly. */
static int round_trip(Tokenizer *t, const char *text)
{
    size_t cap = strlen(text) * 4 + 8;
    int *ids = malloc(cap * sizeof(int));
    char *out = malloc(strlen(text) * 4 + 64);
    char err[256];
    size_t n = 0;
    int ok = ids != NULL && out != NULL &&
             tokenizer_encode(t, text, 1, 0, ids, cap, &n, err, sizeof(err)) == 0 &&
             tokenizer_decode(t, ids, n, out, strlen(text) * 4 + 64,
                              err, sizeof(err)) == 0 &&
             strcmp(out, text) == 0;
    free(ids);
    free(out);
    return ok;
}

static void write_file(const char *path, const void *data, size_t n)
{
    FILE *f = fopen(path, "wb");
    if (f != NULL) {
        if (n > 0) {
            fwrite(data, 1, n, f);
        }
        fclose(f);
    }
}

static void tokenizer_tests(void)
{
    Tokenizer t = {0};
    char err[256];
    int ids[64];
    size_t n = 0;
    char out[256];

    printf("== tokenizer: invalid input\n");
    CHECK(tokenizer_load(NULL, tok_path, 32000, err, sizeof(err)) == -1,
          "load into NULL tokenizer");
    CHECK(tokenizer_load(&t, NULL, 32000, err, sizeof(err)) == -1,
          "load NULL path");
    CHECK(tokenizer_load(&t, tok_path, 0, err, sizeof(err)) == -1,
          "load with vocab_size 0");
    CHECK(tokenizer_load(&t, tok_path, 32001, err, sizeof(err)) == -1,
          "load with vocab_size larger than the file");
    CHECK(t.vocab == NULL, "failed load leaves the tokenizer empty");

    write_file("/tmp/llm_empty_tok.bin", "", 0);
    err[0] = '\0';
    CHECK(tokenizer_load(&t, "/tmp/llm_empty_tok.bin", 32000, err,
                         sizeof(err)) == -1 && err[0] != '\0',
          "load 0-byte file -> error message");
    {
        /* first 1000 bytes of the real file: truncated in the middle */
        FILE *f = fopen(tok_path, "rb");
        char buf[1000];
        size_t got = f ? fread(buf, 1, sizeof(buf), f) : 0;
        if (f) {
            fclose(f);
        }
        write_file("/tmp/llm_trunc_tok.bin", buf, got);
        CHECK(tokenizer_load(&t, "/tmp/llm_trunc_tok.bin", 32000, err,
                             sizeof(err)) == -1, "load truncated file");
        /* a header that claims a token longer than allowed */
        int32_t bogus[3] = {1 << 30, 0, 1 << 30};
        write_file("/tmp/llm_bogus_tok.bin", bogus, sizeof(bogus));
        CHECK(tokenizer_load(&t, "/tmp/llm_bogus_tok.bin", 32000, err,
                             sizeof(err)) == -1, "load absurd lengths");
    }
    remove("/tmp/llm_empty_tok.bin");
    remove("/tmp/llm_trunc_tok.bin");
    remove("/tmp/llm_bogus_tok.bin");

    tokenizer_free(NULL);
    tokenizer_free(&t);
    tokenizer_free(&t);
    CHECK(1, "free on NULL / empty / twice is safe");

    if (tokenizer_load(&t, tok_path, 32000, err, sizeof(err)) != 0) {
        fprintf(stderr, "  cannot load %s: %s\n", tok_path, err);
        ++failures;
        return;
    }

    CHECK(tokenizer_encode(NULL, "hi", 1, 0, ids, 64, &n, err, sizeof(err)) == -1,
          "encode with NULL tokenizer");
    CHECK(tokenizer_encode(&t, NULL, 1, 0, ids, 64, &n, err, sizeof(err)) == -1,
          "encode NULL text");
    CHECK(tokenizer_encode(&t, "hi", 1, 0, NULL, 64, &n, err, sizeof(err)) == -1,
          "encode into NULL buffer");
    CHECK(tokenizer_encode(&t, "hi", 1, 0, ids, 64, NULL, err, sizeof(err)) == -1,
          "encode with NULL count");
    CHECK(tokenizer_encode(&t, "hi", 1, 0, ids, 64, &n, NULL, 0) == 0,
          "encode with no error buffer");

    int neg[] = {-1};
    CHECK(tokenizer_decode(&t, neg, 1, out, sizeof(out), err, sizeof(err)) == -1,
          "decode token -1");
    CHECK(tokenizer_decode(&t, ids, 1, NULL, 10, err, sizeof(err)) == -1,
          "decode into NULL buffer");
    CHECK(tokenizer_decode(NULL, ids, 1, out, sizeof(out), err, sizeof(err)) == -1,
          "decode with NULL tokenizer");

    printf("== tokenizer: empty input\n");
    CHECK(tokenizer_encode(&t, "", 0, 0, ids, 64, &n, err, sizeof(err)) == 0 &&
          n == 0, "empty text, no BOS -> 0 tokens");
    CHECK(tokenizer_encode(&t, "", 1, 0, ids, 64, &n, err, sizeof(err)) == 0 &&
          n == 1 && ids[0] == TOKENIZER_BOS_ID, "empty text + BOS -> [BOS]");
    CHECK(tokenizer_encode(&t, "", 0, 0, ids, 0, &n, err, sizeof(err)) == 0 &&
          n == 0, "empty text into a 0-capacity buffer");
    CHECK(tokenizer_decode(&t, ids, 0, out, sizeof(out), err, sizeof(err)) == 0 &&
          out[0] == '\0', "decode 0 tokens -> empty string");

    printf("== tokenizer: boundary cases\n");
    CHECK(round_trip(&t, "a"), "single character");
    CHECK(round_trip(&t, " "), "single space");
    CHECK(round_trip(&t, "   leading and trailing   "), "runs of spaces");
    CHECK(round_trip(&t, "tab\there\nnewline"), "tab and newline");
    CHECK(round_trip(&t, "caf\xc3\xa9 \xe4\xb8\xad\xe6\x96\x87 \xf0\x9f\x99\x82"),
          "UTF-8: 2-, 3- and 4-byte characters");
    CHECK(round_trip(&t, "\x01\x02\x7f"), "control bytes via byte fallback");
    {
        int one[] = {32000 - 1};
        CHECK(tokenizer_decode(&t, one, 1, out, sizeof(out), err, sizeof(err)) == 0,
              "decode last token ID");
    }
    /* Output buffer exactly too small by one, then exactly large enough. */
    tokenizer_encode(&t, "hello world", 1, 0, ids, 64, &n, err, sizeof(err));
    size_t need = n;
    CHECK(tokenizer_encode(&t, "hello world", 1, 0, ids, need - 1, &n, err,
                           sizeof(err)) == -1, "token buffer one too small");
    CHECK(tokenizer_encode(&t, "hello world", 1, 0, ids, need, &n, err,
                           sizeof(err)) == 0 && n == need,
          "token buffer exactly large enough");
    tokenizer_decode(&t, ids, n, out, sizeof(out), err, sizeof(err));
    size_t text_need = strlen(out) + 1;
    {
        char canary[64];
        memset(canary, 'Z', sizeof(canary));
        CHECK(tokenizer_decode(&t, ids, n, canary, text_need - 1, err,
                               sizeof(err)) == -1 &&
              canary[text_need - 1] == 'Z',
              "text buffer one too small: refused, nothing written past it");
        CHECK(tokenizer_decode(&t, ids, n, canary, text_need, err,
                               sizeof(err)) == 0 && strcmp(canary, out) == 0,
              "text buffer exactly large enough");
    }

    printf("== tokenizer: large input\n");
    size_t len = 200000;
    char *big = malloc(len + 1);
    if (big != NULL) {
        for (size_t i = 0; i < len; ++i) {
            big[i] = "Lily and Ben went to the park. "[i % 31];
        }
        big[len] = '\0';
        clock_t start = clock();
        CHECK(round_trip(&t, big), "200 KB text round-trips");
        double secs = (double)(clock() - start) / CLOCKS_PER_SEC;
        /* the old O(n^2) merge loop needed minutes for this */
        CHECK(secs < 5.0, "200 KB encode + decode in under 5 s (O(n log n) merges)");
        free(big);
    }
    {
        /* one very long run with no merges possible between pieces */
        char *run = malloc(50001);
        if (run != NULL) {
            memset(run, 'z', 50000);
            run[50000] = '\0';
            CHECK(round_trip(&t, run), "50 000 identical characters");
            free(run);
        }
    }

    tokenizer_free(&t);
}

static void sampler_tests(void)
{
    uint32_t rng = 42;
    float one[1] = {3.0f};
    float flat[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    printf("== sampler: invalid input\n");
    CHECK(sampler_temperature(NULL, 4, 1.0f, &rng) == -1, "temperature: NULL logits");
    CHECK(sampler_temperature(flat, 0, 1.0f, &rng) == -1, "temperature: vocab 0");
    CHECK(sampler_temperature(flat, 4, -1.0f, &rng) == -1, "temperature < 0");
    CHECK(sampler_temperature(flat, 4, NAN, &rng) == -1, "temperature NaN");
    CHECK(sampler_temperature(flat, 4, INFINITY, &rng) == -1, "temperature inf");
    CHECK(sampler_top_k(NULL, 4, 2, 1.0f, &rng) == -1, "top-k: NULL logits");
    CHECK(sampler_top_k(flat, 4, 2, 0.0f, &rng) == -1, "top-k: temperature 0");
    CHECK(sampler_top_k(flat, 4, 2, 1.0f, NULL) == -1, "top-k: NULL rng");
    {
        float nan_logits[3] = {0.0f, NAN, 1.0f};
        float inf_logits[3] = {0.0f, -INFINITY, 1.0f};
        CHECK(sampler_temperature(nan_logits, 3, 1.0f, &rng) == -1,
              "NaN logit rejected");
        CHECK(sampler_top_k(inf_logits, 3, 2, 1.0f, &rng) == -1,
              "infinite logit rejected");
    }

    printf("== sampler: boundary cases\n");
    CHECK(sampler_greedy(one, 1) == 0, "greedy: vocab 1");
    CHECK(sampler_temperature(one, 1, 1.0f, &rng) == 0, "temperature: vocab 1");
    CHECK(sampler_top_k(one, 1, 1, 1.0f, &rng) == 0, "top-k: vocab 1, k 1");
    CHECK(sampler_greedy(flat, 4) == 0, "greedy: ties -> lowest index");
    {
        float l[5] = {0.1f, 5.0f, 0.2f, 4.9f, -3.0f};
        int ok = 1;
        for (int i = 0; i < 200; ++i) {
            ok &= sampler_temperature(l, 5, 1e-4f, &rng) == 1;
        }
        CHECK(ok, "temperature near 0 behaves like greedy");
        ok = 1;
        for (int i = 0; i < 2000; ++i) {
            int s = sampler_top_k(l, 5, 2, 100.0f, &rng);
            ok &= s == 1 || s == 3;
        }
        CHECK(ok, "top-k 2 never leaves the top 2, even at high temperature");
        CHECK(sampler_top_k(l, 5, 5, 1.0f, &rng) >= 0, "top-k with k == vocab");
    }
    {
        uint32_t zero = 0;
        int s = sampler_temperature(flat, 4, 1.0f, &zero);
        CHECK(s >= 0 && s < 4 && zero != 0, "rng state 0 is replaced, not stuck");
    }

    printf("== sampler: distribution\n");
    {
        /* softmax([0, ln 3]) = [0.25, 0.75] */
        float l[2] = {0.0f, logf(3.0f)};
        int count1 = 0, draws = 20000;
        uint32_t r = 7;
        for (int i = 0; i < draws; ++i) {
            count1 += sampler_temperature(l, 2, 1.0f, &r) == 1;
        }
        double p = (double)count1 / draws;
        CHECK(fabs(p - 0.75) < 0.02, "temperature 1 matches softmax (0.75)");
        /* temperature 2 flattens: softmax([0, ln3 / 2]) -> 1.732 / 2.732 */
        count1 = 0;
        for (int i = 0; i < draws; ++i) {
            count1 += sampler_temperature(l, 2, 2.0f, &r) == 1;
        }
        p = (double)count1 / draws;
        CHECK(fabs(p - 0.634) < 0.02, "temperature 2 matches softmax (0.634)");
    }
    {
        uint32_t a = 123, b = 123;
        float l[6] = {1, 2, 3, 4, 5, 6};
        int same = 1;
        for (int i = 0; i < 100; ++i) {
            same &= sampler_top_k(l, 6, 4, 1.0f, &a) ==
                    sampler_top_k(l, 6, 4, 1.0f, &b);
        }
        CHECK(same, "same seed -> same sequence");
    }

    printf("== sampler: large input\n");
    {
        size_t n = 1u << 20;   /* 1M-entry vocabulary */
        float *l = malloc(n * sizeof(float));
        if (l != NULL) {
            for (size_t i = 0; i < n; ++i) {
                l[i] = (float)((i * 2654435761u) % 1000003u) / 1000003.0f;
            }
            l[777777] = 10.0f;
            CHECK(sampler_greedy(l, n) == 777777, "greedy over 1M logits");
            CHECK(sampler_top_k(l, n, 1, 1.0f, &rng) == 777777, "top-1 over 1M");
            int s = sampler_top_k(l, n, 40, 1.0f, &rng);
            CHECK(s >= 0 && (size_t)s < n, "top-40 over 1M");
            s = sampler_temperature(l, n, 0.8f, &rng);
            CHECK(s >= 0 && (size_t)s < n, "temperature over 1M");
            free(l);
        }
    }
}

static void xlsx_tests(void)
{
    char err[256];
    printf("== xlsx reader: invalid / empty input\n");
    err[0] = '\0';
    CHECK(xlsx_open(NULL, err, sizeof(err)) == NULL && err[0] != '\0',
          "open NULL path");
    CHECK(xlsx_open("/tmp/does-not-exist.xlsx", err, sizeof(err)) == NULL,
          "open missing file");
    write_file("/tmp/llm_empty.xlsx", "", 0);
    CHECK(xlsx_open("/tmp/llm_empty.xlsx", err, sizeof(err)) == NULL,
          "open 0-byte file");
    static const char fake_zip[] = "PK\x03\x04 not really a zip";
    write_file("/tmp/llm_junk.xlsx", fake_zip, sizeof(fake_zip) - 1);
    CHECK(xlsx_open("/tmp/llm_junk.xlsx", err, sizeof(err)) == NULL,
          "open a fake ZIP header");
    remove("/tmp/llm_empty.xlsx");
    remove("/tmp/llm_junk.xlsx");

    xlsx_close(NULL);
    CHECK(xlsx_sheet_count(NULL) == 0, "sheet_count(NULL)");
    CHECK(xlsx_sheet_name(NULL, 0) == NULL, "sheet_name(NULL)");
    CHECK(xlsx_row_count(NULL, 0) == 0, "row_count(NULL)");
    CHECK(xlsx_row(NULL, 0, 0) == NULL, "row(NULL)");
    CHECK(xlsx_cell(NULL, 0) != NULL && xlsx_cell(NULL, 0)[0] == '\0',
          "cell(NULL row) -> \"\"");
}

int main(int argc, char **argv)
{
    if (argc > 1) {
        tok_path = argv[1];
    }
    tokenizer_tests();
    sampler_tests();
    xlsx_tests();
    if (failures == 0) {
        printf("PASS: edge-case tests (%d checks)\n", checks);
        return 0;
    }
    printf("FAIL: edge-case tests (%d of %d checks failed)\n", failures, checks);
    return 1;
}
