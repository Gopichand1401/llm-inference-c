/*
 * gguf_matvec_demo.c - test gguf_matvec() on real model weights.
 *
 * Usage: gguf_matvec_demo <model.gguf>
 *
 * For a set of weights covering every type/shape combination in the model,
 * computes y = W x with a fixed pseudo-random x, with both kernels:
 *   gguf_matvec_reference()  exact, column-order summation
 *   gguf_matvec()            fast fused kernels (Phase 8)
 * and:
 *   1. checks both outputs are finite,
 *   2. checks sample rows of the reference bit-exactly against
 *      gguf_weight_dequantize_row() followed by a sequential dot product,
 *   3. reports the largest fast-vs-reference difference,
 *   4. saves x, the reference y (build/matvec_c.bin) and the fast y
 *      (build/matvec_fast.bin) for examples/05_matrix_math/compare_matvec.py,
 *   5. reports the time per product for both kernels.
 */
#include "gguf_model.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Deterministic inputs in [-1, 1): xorshift32. */
static void fill_input(float *x, size_t n, uint32_t seed)
{
    uint32_t s = seed;
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        x[i] = (float)((double)s / 4294967296.0 * 2.0 - 1.0);
    }
}

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int write_all(FILE *f, const void *p, size_t size, size_t n)
{
    return fwrite(p, size, n, f) == n;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    GGUFModel model;
    char error[256];
    int status = 1;
    int failures = 0;

    float *x_dim = NULL, *x_hidden = NULL, *y = NULL, *y_fast = NULL;
    float *row = NULL;
    FILE *out = NULL, *out_fast = NULL;

    if (!gguf_model_load(&model, argv[1], error, sizeof(error))) {
        fprintf(stderr, "Load failed: %s\n", error);
        return 1;
    }

    const size_t dim = (size_t)model.config.dim;
    const size_t hidden = (size_t)model.config.hidden_dim;
    const size_t vocab = (size_t)model.config.vocab_size;
    const size_t max_rows = vocab > hidden ? vocab : hidden;
    const size_t max_cols = hidden > dim ? hidden : dim;

    x_dim = malloc(dim * sizeof(float));
    x_hidden = malloc(hidden * sizeof(float));
    y = malloc(max_rows * sizeof(float));
    y_fast = malloc(max_rows * sizeof(float));
    row = malloc(max_cols * sizeof(float));
    out = fopen("build/matvec_c.bin", "wb");
    out_fast = fopen("build/matvec_fast.bin", "wb");

    if (!x_dim || !x_hidden || !y || !y_fast || !row || !out || !out_fast) {
        fprintf(stderr, "Allocation or file open failed.\n");
        goto cleanup;
    }

    fill_input(x_dim, dim, 12345u);
    fill_input(x_hidden, hidden, 67890u);

    /* Save the inputs so Python uses exactly the same x. */
    {
        FILE *fx = fopen("build/matvec_x_dim.bin", "wb");
        FILE *fh = fopen("build/matvec_x_hidden.bin", "wb");
        int ok = fx && fh &&
                 write_all(fx, x_dim, sizeof(float), dim) &&
                 write_all(fh, x_hidden, sizeof(float), hidden);
        if (fx) fclose(fx);
        if (fh) fclose(fh);
        if (!ok) {
            fprintf(stderr, "Cannot write input vectors.\n");
            goto cleanup;
        }
    }

    /*
     * Weights to test. Layer 0 has Q6_K wv / ffn_down; layer 2 has Q4_K
     * versions of the same roles, so both types are covered for every shape.
     */
    const GGUFLayer *L0 = &model.layers[0];
    const GGUFLayer *L2 = &model.layers[model.config.n_layers > 2 ? 2 : 0];

    const GGUFWeight *tests[] = {
        &L0->wq, &L0->wk, &L0->wv, &L0->wo,
        &L0->ffn_gate, &L0->ffn_up, &L0->ffn_down,
        &L2->wv, &L2->ffn_down,
        &model.output, &model.token_embd, &L0->attn_norm
    };
    const size_t n_tests = sizeof(tests) / sizeof(tests[0]);

    printf("=== Quantized matrix-vector product ===\n");
    printf("%-26s %-5s %6s %5s %9s %9s %10s %12s %s\n",
           "weight", "type", "rows", "cols", "ref(ms)", "fast(ms)",
           "y[0]", "max|f-r|", "self-check");

    for (size_t t = 0; t < n_tests; t++) {
        const GGUFWeight *w = tests[t];
        const float *x = (w->cols == hidden) ? x_hidden : x_dim;

        double start = now_seconds();
        int ok = gguf_matvec_reference(w, x, y);
        double ms = (now_seconds() - start) * 1000.0;

        start = now_seconds();
        int ok_fast = gguf_matvec(w, x, y_fast);
        double ms_fast = (now_seconds() - start) * 1000.0;

        if (!ok || !ok_fast) {
            printf("%-26s matvec FAILED\n", w->name);
            failures++;
            continue;
        }

        /* Finite outputs; largest fast-vs-reference difference. */
        double max_diff = 0.0;
        for (uint64_t r = 0; r < w->rows; r++) {
            if (!isfinite(y[r]) || !isfinite(y_fast[r])) {
                printf("NON-FINITE %s row %" PRIu64 "\n", w->name, r);
                failures++;
                break;
            }
            double diff = fabs((double)y_fast[r] - (double)y[r]);
            if (diff > max_diff) {
                max_diff = diff;
            }
        }

        /* Self-check: decode the row, then a plain sequential dot. */
        const uint64_t sample[3] = {0, w->rows / 2, w->rows - 1};
        int self_ok = 1;
        for (int s = 0; s < 3; s++) {
            const uint64_t r = sample[s];
            float sum = 0.0f;
            gguf_weight_dequantize_row(w, r, row);
            for (uint64_t i = 0; i < w->cols; i++) {
                sum += row[i] * x[i];
            }
            if (sum != y[r]) {
                self_ok = 0;
            }
        }
        if (!self_ok) {
            failures++;
        }

        printf("%-26s %-5s %6" PRIu64 " %5" PRIu64 " %9.3f %9.3f %10.5f "
               "%12.3e %s\n",
               w->name, gguf_ggml_type_name(w->type), w->rows, w->cols,
               ms, ms_fast, (double)y[0], max_diff,
               self_ok ? "ok" : "MISMATCH");

        /* Record: uint32 name_length, name, uint64 rows, uint64 cols,
         * float32[rows] output. */
        const uint32_t name_length = (uint32_t)strlen(w->name);
        if (!write_all(out, &name_length, sizeof(name_length), 1) ||
            !write_all(out, w->name, 1, name_length) ||
            !write_all(out, &w->rows, sizeof(w->rows), 1) ||
            !write_all(out, &w->cols, sizeof(w->cols), 1) ||
            !write_all(out, y, sizeof(float), (size_t)w->rows) ||
            !write_all(out_fast, &name_length, sizeof(name_length), 1) ||
            !write_all(out_fast, w->name, 1, name_length) ||
            !write_all(out_fast, &w->rows, sizeof(w->rows), 1) ||
            !write_all(out_fast, &w->cols, sizeof(w->cols), 1) ||
            !write_all(out_fast, y_fast, sizeof(float), (size_t)w->rows)) {
            fprintf(stderr, "Cannot write results.\n");
            goto cleanup;
        }
    }

    /* Invalid arguments must be rejected. */
    if (gguf_matvec(NULL, x_dim, y) != 0 ||
        gguf_matvec(&model.output, NULL, y) != 0 ||
        gguf_matvec(&model.output, x_dim, NULL) != 0 ||
        gguf_matvec_reference(NULL, x_dim, y) != 0) {
        printf("FAIL: invalid arguments accepted\n");
        failures++;
    }

    printf("\nSaved %zu products to build/matvec_c.bin (reference) and "
           "build/matvec_fast.bin (fast)\n", n_tests);

    if (failures == 0) {
        printf("Quantized matvec C checks: PASSED\n");
        status = 0;
    } else {
        printf("Quantized matvec C checks: FAILED (%d)\n", failures);
    }

cleanup:
    if (out != NULL) {
        fclose(out);
    }
    if (out_fast != NULL) {
        fclose(out_fast);
    }
    free(y_fast);
    free(x_dim);
    free(x_hidden);
    free(y);
    free(row);
    gguf_model_free(&model);
    return status;
}
