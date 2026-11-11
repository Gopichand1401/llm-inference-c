/*
 * gguf_model_demo.c - load a GGUF model and verify every weight binding.
 *
 * Usage: gguf_model_demo <model.gguf>
 *
 * 1. Loads the model with gguf_model_load().
 * 2. Prints the configuration and the type of every weight in every layer.
 * 3. Decodes sample rows (first, middle, last; plus a few token embeddings)
 *    of ALL bound weights with gguf_weight_dequantize_row(), checks they are
 *    finite, and saves them to build/gguf_rows.bin.
 *    examples/04_model_loading/compare_gguf_rows.py then checks each row against gguf-py.
 */
#include "gguf_model.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static FILE *rows_file = NULL;
static float *row_buffer = NULL;
static uint64_t rows_written = 0;
static uint64_t bad_rows = 0;
static uint64_t weights_checked = 0;

/* Decode one row, check it, and append a record to build/gguf_rows.bin:
 *   uint32 name_length, name bytes, uint64 row, uint64 cols, float32[cols] */
static int dump_row(const GGUFWeight *w, uint64_t row)
{
    if (!gguf_weight_dequantize_row(w, row, row_buffer)) {
        printf("DECODE FAILED: %s row %" PRIu64 "\n", w->name, row);
        bad_rows++;
        return 0;
    }

    for (uint64_t i = 0; i < w->cols; i++) {
        if (!isfinite(row_buffer[i])) {
            printf("NON-FINITE: %s row %" PRIu64 " col %" PRIu64 "\n",
                   w->name, row, i);
            bad_rows++;
            return 0;
        }
    }

    const uint32_t name_length = (uint32_t)strlen(w->name);

    if (fwrite(&name_length, sizeof(name_length), 1, rows_file) != 1 ||
        fwrite(w->name, 1, name_length, rows_file) != name_length ||
        fwrite(&row, sizeof(row), 1, rows_file) != 1 ||
        fwrite(&w->cols, sizeof(w->cols), 1, rows_file) != 1 ||
        fwrite(row_buffer, sizeof(float), (size_t)w->cols, rows_file)
            != (size_t)w->cols) {
        printf("WRITE FAILED\n");
        bad_rows++;
        return 0;
    }

    rows_written++;
    return 1;
}

static void check_weight(const GGUFWeight *w)
{
    weights_checked++;

    dump_row(w, 0);

    if (w->rows > 2) {
        dump_row(w, w->rows / 2);
    }

    if (w->rows > 1) {
        dump_row(w, w->rows - 1);
    }
}

static const char *type_name(uint32_t type)
{
    return gguf_ggml_type_name(type);
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

    if (!gguf_model_load(&model, argv[1], error, sizeof(error))) {
        fprintf(stderr, "Load failed: %s\n", error);
        return 1;
    }

    const Config *c = &model.config;

    printf("=== GGUF Model Loaded ===\n");
    printf("File size: %zu bytes, mapped at %p\n",
           model.file_size, model.mapped);
    printf("dim=%d hidden=%d layers=%d heads=%d kv_heads=%d head_size=%d\n",
           c->dim, c->hidden_dim, c->n_layers, c->n_heads, c->n_kv_heads,
           c->dim / c->n_heads);
    printf("vocab=%d ctx=%d rope_dim=%d rope_base=%.1f rope_scale=%g "
           "rms_eps=%g\n",
           c->vocab_size, c->seq_len, c->rope_dim,
           (double)c->rope_freq_base, (double)c->rope_scale,
           (double)c->rms_norm_eps);
    printf("Output projection: %s\n",
           model.output_is_tied ? "tied to token_embd" : "separate output.weight");

    printf("\n=== Global weights ===\n");
    const GGUFWeight *globals[3] = {
        &model.token_embd, &model.output_norm, &model.output
    };
    for (int i = 0; i < 3; i++) {
        const GGUFWeight *w = globals[i];
        printf("%-20s %-5s rows=%-6" PRIu64 " cols=%-5" PRIu64
               " row_bytes=%-6" PRIu64 " file_offset=%td\n",
               w->name, type_name(w->type), w->rows, w->cols, w->row_bytes,
               (const char *)w->data - (const char *)model.mapped);
    }

    printf("\n=== Per-layer weight types ===\n");
    printf("layer  attn_norm  wq    wk    wv    wo    ffn_norm gate  up    down\n");
    for (int l = 0; l < c->n_layers; l++) {
        const GGUFLayer *L = &model.layers[l];
        printf("%5d  %-9s  %-5s %-5s %-5s %-5s %-8s %-5s %-5s %-5s\n",
               l, type_name(L->attn_norm.type), type_name(L->wq.type),
               type_name(L->wk.type), type_name(L->wv.type),
               type_name(L->wo.type), type_name(L->ffn_norm.type),
               type_name(L->ffn_gate.type), type_name(L->ffn_up.type),
               type_name(L->ffn_down.type));
    }

    /* Big enough for the longest row (hidden_dim or dim columns). */
    const size_t max_cols = (size_t)(c->hidden_dim > c->dim
                                     ? c->hidden_dim : c->dim);
    row_buffer = malloc(max_cols * sizeof(float));
    rows_file = fopen("build/gguf_rows.bin", "wb");

    if (row_buffer == NULL || rows_file == NULL) {
        fprintf(stderr, "Cannot allocate buffer or open build/gguf_rows.bin\n");
        goto cleanup;
    }

    /* Row decoding through the loader for every bound weight. */
    check_weight(&model.token_embd);
    check_weight(&model.output_norm);
    if (!model.output_is_tied) {
        check_weight(&model.output);
    }

    /* Embedding rows for a few specific token IDs (1 = BOS in LLaMA). */
    const uint64_t tokens[] = {1, 2, 100, 1000, 29871};
    for (size_t i = 0; i < sizeof(tokens) / sizeof(tokens[0]); i++) {
        if (tokens[i] < model.token_embd.rows) {
            dump_row(&model.token_embd, tokens[i]);
        }
    }

    for (int l = 0; l < c->n_layers; l++) {
        const GGUFLayer *L = &model.layers[l];
        const GGUFWeight *ws[9] = {
            &L->attn_norm, &L->wq, &L->wk, &L->wv, &L->wo,
            &L->ffn_norm, &L->ffn_gate, &L->ffn_up, &L->ffn_down
        };
        for (int i = 0; i < 9; i++) {
            check_weight(ws[i]);
        }
    }

    printf("\n=== Row decoding check ===\n");
    printf("Weights checked: %" PRIu64 "\n", weights_checked);
    printf("Rows decoded and saved: %" PRIu64 "\n", rows_written);
    printf("Bad rows: %" PRIu64 "\n", bad_rows);

    /* Out-of-range row must be rejected. */
    if (gguf_weight_dequantize_row(&model.token_embd,
                                   model.token_embd.rows, row_buffer) != 0) {
        printf("FAIL: out-of-range row was accepted\n");
        bad_rows++;
    }

    if (bad_rows == 0) {
        printf("GGUF model load and row decoding: PASSED\n");
        status = 0;
    } else {
        printf("GGUF model load and row decoding: FAILED\n");
    }

cleanup:
    if (rows_file != NULL) {
        fclose(rows_file);
    }
    free(row_buffer);
    gguf_model_free(&model);
    return status;
}
