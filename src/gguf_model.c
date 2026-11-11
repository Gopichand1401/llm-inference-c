/*
 * gguf_model.c - GGUF LLaMA model loading and per-row weight decoding.
 * See include/gguf_model.h for the memory model.
 */
#include "gguf_model.h"
#include "gguf_quant.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define GGML_TYPE_F32 0
#define GGML_TYPE_F16 1

static void set_error(char *error, size_t capacity, const char *message)
{
    if (error != NULL && capacity > 0) {
        snprintf(error, capacity, "%s", message);
    }
}

int gguf_weight_type_supported(uint32_t type)
{
    /* F32, F16, and every block type gguf_quant.c can decode
     * (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K). */
    return type == GGML_TYPE_F32 ||
           type == GGML_TYPE_F16 ||
           gguf_quant_block_info(type, NULL, NULL);
}

/* ------------------------------------------------------------------ */
/* Binding one tensor                                                  */
/* ------------------------------------------------------------------ */

/*
 * Find `name`, check that its GGUF dimensions are exactly
 * [expected_cols] (1-D, expected_rows == 0) or [expected_cols, expected_rows]
 * (2-D), and fill `weight` with a pointer into the mapping.
 */
static int bind_weight(
    GGUFModel *m,
    const char *name,
    uint64_t expected_cols,
    uint64_t expected_rows,
    GGUFWeight *weight,
    char *error,
    size_t error_capacity
)
{
    const GGUF_TensorInfo *t =
        gguf_find_tensor(m->tensors, m->tensor_count, name);

    if (t == NULL) {
        if (error != NULL && error_capacity > 0) {
            snprintf(error, error_capacity, "Missing tensor: %s", name);
        }
        return 0;
    }

    const uint32_t expected_dims = (expected_rows == 0) ? 1u : 2u;
    const uint64_t rows = (expected_rows == 0) ? 1u : expected_rows;

    if (t->n_dimensions != expected_dims ||
        t->dimensions[0] != expected_cols ||
        (expected_dims == 2 && t->dimensions[1] != expected_rows)) {
        if (error != NULL && error_capacity > 0) {
            snprintf(error, error_capacity,
                     "Tensor %s has an unexpected shape.", name);
        }
        return 0;
    }

    if (!gguf_weight_type_supported(t->type)) {
        if (error != NULL && error_capacity > 0) {
            snprintf(error, error_capacity,
                     "Tensor %s uses unsupported type %s (%u).",
                     name, gguf_ggml_type_name(t->type), t->type);
        }
        return 0;
    }

    const void *data = NULL;
    uint64_t size = 0;

    if (!gguf_get_tensor_data(m->mapped, m->file_size, &m->layout, t,
                              &data, &size)) {
        if (error != NULL && error_capacity > 0) {
            snprintf(error, error_capacity,
                     "Tensor %s lies outside the file.", name);
        }
        return 0;
    }

    /* Bytes per row = (cols / block_size) * bytes_per_block. */
    uint64_t block_size = 0;
    uint64_t type_size = 0;

    if (!gguf_ggml_type_info(t->type, &block_size, &type_size) ||
        expected_cols % block_size != 0) {
        set_error(error, error_capacity, "Invalid row layout.");
        return 0;
    }

    const uint64_t row_bytes = (expected_cols / block_size) * type_size;

    if (row_bytes * rows != size) {
        if (error != NULL && error_capacity > 0) {
            snprintf(error, error_capacity,
                     "Tensor %s: storage size does not match its shape.",
                     name);
        }
        return 0;
    }

    weight->name = t->name;
    weight->data = data;
    weight->type = t->type;
    weight->cols = expected_cols;
    weight->rows = rows;
    weight->row_bytes = row_bytes;
    weight->byte_size = size;

    return 1;
}

static int bind_layer_weight(
    GGUFModel *m,
    int layer,
    const char *suffix,
    uint64_t cols,
    uint64_t rows,
    GGUFWeight *weight,
    char *error,
    size_t error_capacity
)
{
    char name[96];

    snprintf(name, sizeof(name), "blk.%d.%s", layer, suffix);

    return bind_weight(m, name, cols, rows, weight, error, error_capacity);
}

/* ------------------------------------------------------------------ */
/* Load / free                                                         */
/* ------------------------------------------------------------------ */

void gguf_model_free(GGUFModel *model)
{
    if (model == NULL) {
        return;
    }

    free(model->layers);

    if (model->tensors != NULL) {
        for (size_t i = 0; i < model->tensor_count; i++) {
            free(model->tensors[i].name);
        }
        free(model->tensors);
    }

    if (model->mapped != NULL) {
        munmap(model->mapped, model->file_size);
    }

    if (model->fd != -1) {
        close(model->fd);
    }

    memset(model, 0, sizeof(*model));
    model->fd = -1;
}

int gguf_model_load(
    GGUFModel *model,
    const char *path,
    char *error,
    size_t error_capacity
)
{
    struct stat st;

    if (model == NULL || path == NULL) {
        set_error(error, error_capacity, "Invalid model loader arguments.");
        return 0;
    }

    memset(model, 0, sizeof(*model));
    model->fd = -1;

    /* 1. Parse metadata, tensor descriptors and the data-section layout. */
    if (!gguf_read_config(path, &model->config, error, error_capacity) ||
        !gguf_read_tensors(path, &model->tensors, &model->tensor_count,
                           error, error_capacity) ||
        !gguf_read_layout(path, &model->layout, error, error_capacity)) {
        gguf_model_free(model);
        return 0;
    }

    /* 2. Map the whole file read-only. */
    model->fd = open(path, O_RDONLY);

    if (model->fd == -1 || fstat(model->fd, &st) != 0 || st.st_size <= 0) {
        set_error(error, error_capacity, "Cannot open GGUF file for mapping.");
        gguf_model_free(model);
        return 0;
    }

    model->file_size = (size_t)st.st_size;

    void *mapped = mmap(NULL, model->file_size, PROT_READ, MAP_PRIVATE,
                        model->fd, 0);

    if (mapped == MAP_FAILED) {
        set_error(error, error_capacity, "Cannot memory-map GGUF file.");
        gguf_model_free(model);
        return 0;
    }

    model->mapped = mapped;

    /*
     * Llama 3.1-style RoPE stores per-frequency correction factors in a
     * separate tensor. This engine does not apply them, so refuse such
     * models instead of producing subtly wrong positions.
     */
    if (gguf_find_tensor(model->tensors, model->tensor_count,
                         "rope_freqs.weight") != NULL) {
        set_error(error, error_capacity,
                  "Model uses RoPE frequency factors (rope_freqs.weight, "
                  "Llama 3.1 style), which are not supported.");
        gguf_model_free(model);
        return 0;
    }

    /* 3. Derived dimensions. */
    const Config *c = &model->config;
    const uint64_t dim = (uint64_t)c->dim;
    const uint64_t hidden = (uint64_t)c->hidden_dim;
    const uint64_t vocab = (uint64_t)c->vocab_size;
    const uint64_t kv_dim = dim / (uint64_t)c->n_heads * (uint64_t)c->n_kv_heads;

    /* 4. Global tensors. */
    if (!bind_weight(model, "token_embd.weight", dim, vocab,
                     &model->token_embd, error, error_capacity) ||
        !bind_weight(model, "output_norm.weight", dim, 0,
                     &model->output_norm, error, error_capacity)) {
        gguf_model_free(model);
        return 0;
    }

    /* Some models tie the output projection to the embedding table. */
    if (gguf_find_tensor(model->tensors, model->tensor_count,
                         "output.weight") == NULL) {
        model->output = model->token_embd;
        model->output_is_tied = 1;
    } else if (!bind_weight(model, "output.weight", dim, vocab,
                            &model->output, error, error_capacity)) {
        gguf_model_free(model);
        return 0;
    }

    /* 5. Per-layer tensors. */
    model->layers = calloc((size_t)c->n_layers, sizeof(*model->layers));

    if (model->layers == NULL) {
        set_error(error, error_capacity, "Cannot allocate layer table.");
        gguf_model_free(model);
        return 0;
    }

    for (int l = 0; l < c->n_layers; l++) {
        GGUFLayer *L = &model->layers[l];

        if (!bind_layer_weight(model, l, "attn_norm.weight",   dim, 0,
                               &L->attn_norm, error, error_capacity) ||
            !bind_layer_weight(model, l, "attn_q.weight",      dim, dim,
                               &L->wq, error, error_capacity) ||
            !bind_layer_weight(model, l, "attn_k.weight",      dim, kv_dim,
                               &L->wk, error, error_capacity) ||
            !bind_layer_weight(model, l, "attn_v.weight",      dim, kv_dim,
                               &L->wv, error, error_capacity) ||
            !bind_layer_weight(model, l, "attn_output.weight", dim, dim,
                               &L->wo, error, error_capacity) ||
            !bind_layer_weight(model, l, "ffn_norm.weight",    dim, 0,
                               &L->ffn_norm, error, error_capacity) ||
            !bind_layer_weight(model, l, "ffn_gate.weight",    dim, hidden,
                               &L->ffn_gate, error, error_capacity) ||
            !bind_layer_weight(model, l, "ffn_up.weight",      dim, hidden,
                               &L->ffn_up, error, error_capacity) ||
            !bind_layer_weight(model, l, "ffn_down.weight",    hidden, dim,
                               &L->ffn_down, error, error_capacity)) {
            gguf_model_free(model);
            return 0;
        }
    }

    if (error != NULL && error_capacity > 0) {
        error[0] = '\0';
    }

    return 1;
}

/* ------------------------------------------------------------------ */
/* Row decoding                                                        */
/* ------------------------------------------------------------------ */

int gguf_weight_dequantize_row(
    const GGUFWeight *weight,
    uint64_t row,
    float *out
)
{
    if (weight == NULL || weight->data == NULL || out == NULL ||
        row >= weight->rows) {
        return 0;
    }

    const unsigned char *src =
        (const unsigned char *)weight->data + (size_t)(row * weight->row_bytes);
    const size_t cols = (size_t)weight->cols;

    switch (weight->type) {

        case GGML_TYPE_F32:
            memcpy(out, src, cols * sizeof(float));
            return 1;

        case GGML_TYPE_F16:
            for (size_t i = 0; i < cols; i++) {
                uint16_t h;
                memcpy(&h, src + i * 2, sizeof(h));
                out[i] = gguf_fp16_to_fp32(h);
            }
            return 1;

        default: {
            /* Any block-quantized type: decode block by block. */
            size_t block_weights = 0, block_bytes = 0;
            if (!gguf_quant_block_info(weight->type, &block_weights,
                                       &block_bytes)) {
                return 0;
            }
            for (size_t b = 0; b < cols / block_weights; b++) {
                if (!gguf_dequantize_block(weight->type, src + b * block_bytes,
                                           out + b * block_weights)) {
                    return 0;
                }
            }
            return 1;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Matrix-vector product                                               */
/* ------------------------------------------------------------------ */

/*
 * Dot product of one stored row with x. Decodes one block at a time and
 * accumulates in a single float, column by column, so the result is
 * exactly what you get from gguf_weight_dequantize_row() followed by a
 * plain sequential dot product.
 */
static float row_dot(
    const GGUFWeight *w,
    const unsigned char *src,
    const float *x
)
{
    float sum = 0.0f;
    const size_t cols = (size_t)w->cols;

    switch (w->type) {

        case GGML_TYPE_F32:
            for (size_t i = 0; i < cols; i++) {
                float v;
                memcpy(&v, src + i * sizeof(float), sizeof(v));
                sum += v * x[i];
            }
            return sum;

        case GGML_TYPE_F16:
            for (size_t i = 0; i < cols; i++) {
                uint16_t h;
                memcpy(&h, src + i * 2, sizeof(h));
                sum += gguf_fp16_to_fp32(h) * x[i];
            }
            return sum;

        default: {
            /* Block-quantized: decode a block, then add its products in
             * column order. */
            float buf[GGUF_QK_K];          /* the largest block size */
            size_t block_weights = 0, block_bytes = 0;
            gguf_quant_block_info(w->type, &block_weights, &block_bytes);
            for (size_t b = 0; b < cols / block_weights; b++) {
                gguf_dequantize_block(w->type, src + b * block_bytes, buf);
                const float *xb = x + b * block_weights;
                for (size_t i = 0; i < block_weights; i++) {
                    sum += buf[i] * xb[i];
                }
            }
            return sum;
        }
    }
}

int gguf_matvec_reference(
    const GGUFWeight *weight,
    const float *x,
    float *out
)
{
    if (weight == NULL || weight->data == NULL || x == NULL || out == NULL ||
        !gguf_weight_type_supported(weight->type) ||
        weight->rows > (uint64_t)INT64_MAX) {
        return 0;
    }

    const unsigned char *base = weight->data;
    const long long rows = (long long)weight->rows;

    #pragma omp parallel for schedule(static)
    for (long long r = 0; r < rows; r++) {
        out[r] = row_dot(weight,
                         base + (size_t)r * (size_t)weight->row_bytes,
                         x);
    }

    return 1;
}

/* ------------------------------------------------------------------ */
/* Fast fused kernels (Phase 8)                                        */
/* ------------------------------------------------------------------ */

/*
 * Q4_K: every weight in a 32-weight group g is  w = d*sc_g*q - dmin*m_g
 * with q in 0..15. So for one group
 *
 *     sum_i w_i x_i = d*sc_g * sum_i q_i x_i  -  dmin*m_g * sum_i x_i
 *
 * sum_i x_i depends only on x, so it is computed once per matrix-vector
 * product (xsum32[], one value per 32 inputs) instead of once per row.
 * Each group's sum_i q_i x_i is computed as element-wise products followed
 * by a fixed pairwise-sum tree; both steps are plain element-wise loops that
 * GCC vectorises at -O2 without reordering any floating-point operation.
 */
static float q4k_row_dot_fast(const unsigned char *src, const float *x,
                              const float *xsum32, size_t n_blocks)
{
    float total = 0.0f;

    for (size_t b = 0; b < n_blocks; b++) {
        const unsigned char *blk = src + b * GGUF_Q4_K_BLOCK_SIZE;
        uint16_t d_bits, dmin_bits;

        memcpy(&d_bits, blk, 2);
        memcpy(&dmin_bits, blk + 2, 2);

        const float d = gguf_fp16_to_fp32(d_bits);
        const float dmin = gguf_fp16_to_fp32(dmin_bits);
        const uint8_t *scales = blk + 4;
        const uint8_t *qs = blk + 4 + GGUF_Q4_K_SCALE_SIZE;
        const float *xb = x + b * GGUF_QK_K;
        const float *xs = xsum32 + b * 8;

        for (unsigned int j = 0; j < 4; j++) {
            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            gguf_q4k_get_scale_min(scales, 2 * j, &sc_lo, &m_lo);
            gguf_q4k_get_scale_min(scales, 2 * j + 1, &sc_hi, &m_hi);

            const uint8_t *q = qs + 32 * j;
            const float *xa = xb + 64 * j;
            float lo[32], hi[32];

            /* Element-wise products (vectorisable), then a fixed
             * pairwise-sum tree: deterministic and SIMD-friendly. */
            for (unsigned int l = 0; l < 32; l++) {
                lo[l] = (float)(q[l] & 0x0F) * xa[l];
                hi[l] = (float)(q[l] >> 4) * xa[32 + l];
            }
            for (unsigned int w = 16; w > 0; w /= 2) {
                for (unsigned int l = 0; l < w; l++) {
                    lo[l] += lo[l + w];
                    hi[l] += hi[l + w];
                }
            }

            const float s_lo = lo[0];
            const float s_hi = hi[0];

            total += d * (float)sc_lo * s_lo - dmin * (float)m_lo * xs[2 * j];
            total += d * (float)sc_hi * s_hi - dmin * (float)m_hi * xs[2 * j + 1];
        }
    }

    return total;
}

/*
 * Q6_K: w = d*sc_g*q with one signed scale per 16 weights, so
 *     sum_i w_i x_i = sum_g d*sc_g * (sum_{i in g} q_i x_i).
 * The 6-bit quants are unpacked into int8 first (gguf_q6k_unpack_block
 * layout), then each 16-weight group is dotted with 8 lanes of two
 * products each, followed by a fixed pairwise sum.
 */
static float q6k_row_dot_fast(const unsigned char *src, const float *x,
                              size_t n_blocks)
{
    float total = 0.0f;

    for (size_t b = 0; b < n_blocks; b++) {
        const unsigned char *blk = src + b * GGUF_Q6_K_BLOCK_SIZE;
        const uint8_t *ql = blk;
        const uint8_t *qh = blk + GGUF_Q6_K_QL_SIZE;
        const int8_t *sc = (const int8_t *)(blk + GGUF_Q6_K_QL_SIZE +
                                            GGUF_Q6_K_QH_SIZE);
        uint16_t d_bits;
        int8_t q[GGUF_QK_K];

        memcpy(&d_bits, blk + GGUF_Q6_K_BLOCK_SIZE - 2, 2);
        const float d = gguf_fp16_to_fp32(d_bits);

        for (unsigned int half = 0; half < 2; half++) {
            const uint8_t *l4 = ql + 64 * half;
            const uint8_t *h2 = qh + 32 * half;
            int8_t *o = q + 128 * half;

            for (unsigned int l = 0; l < 32; l++) {
                const unsigned int h = h2[l];
                o[l]      = (int8_t)((int)((l4[l]      & 0x0F) | (((h >> 0) & 3) << 4)) - 32);
                o[l + 32] = (int8_t)((int)((l4[l + 32] & 0x0F) | (((h >> 2) & 3) << 4)) - 32);
                o[l + 64] = (int8_t)((int)((l4[l]      >> 4)   | (((h >> 4) & 3) << 4)) - 32);
                o[l + 96] = (int8_t)((int)((l4[l + 32] >> 4)   | (((h >> 6) & 3) << 4)) - 32);
            }
        }

        const float *xb = x + b * GGUF_QK_K;

        for (unsigned int g = 0; g < GGUF_Q6_K_SCALE_COUNT; g++) {
            const int8_t *qg = q + 16 * g;
            const float *xg = xb + 16 * g;
            float acc[8];

            /* 8 lanes x 2 products, then a fixed pairwise sum. */
            for (unsigned int k = 0; k < 8; k++) {
                acc[k] = (float)qg[k] * xg[k] + (float)qg[k + 8] * xg[k + 8];
            }

            const float s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) +
                            ((acc[4] + acc[5]) + (acc[6] + acc[7]));
            total += d * (float)sc[g] * s;
        }
    }

    return total;
}

/*
 * Q5_K: Q4_K with a 5th bit, so the same identity holds:
 *     sum_i w_i x_i = d*sc_g * sum_i q_i x_i  -  dmin*m_g * sum_i x_i
 * with q = low4 | (bit << 4). Block layout: d, dmin, scales[12] (bytes
 * 0..15), qh[32] (16..47), qs[128] (48..175).
 */
static float q5k_row_dot_fast(const unsigned char *src, const float *x,
                              const float *xsum32, size_t n_blocks)
{
    float total = 0.0f;

    for (size_t b = 0; b < n_blocks; b++) {
        const unsigned char *blk = src + b * GGUF_Q5_K_BLOCK_SIZE;
        uint16_t d_bits, dmin_bits;

        memcpy(&d_bits, blk, 2);
        memcpy(&dmin_bits, blk + 2, 2);

        const float d = gguf_fp16_to_fp32(d_bits);
        const float dmin = gguf_fp16_to_fp32(dmin_bits);
        const uint8_t *scales = blk + 4;
        const uint8_t *qh = blk + 4 + GGUF_Q4_K_SCALE_SIZE;
        const uint8_t *qs = qh + GGUF_QK_K / 8;
        const float *xb = x + b * GGUF_QK_K;
        const float *xs = xsum32 + b * 8;

        for (unsigned int j = 0; j < 4; j++) {
            uint8_t sc_lo, m_lo, sc_hi, m_hi;
            gguf_q4k_get_scale_min(scales, 2 * j, &sc_lo, &m_lo);
            gguf_q4k_get_scale_min(scales, 2 * j + 1, &sc_hi, &m_hi);

            const uint8_t *q = qs + 32 * j;
            const float *xa = xb + 64 * j;
            const unsigned int shift_lo = 2 * j;
            const unsigned int shift_hi = 2 * j + 1;
            float lo[32], hi[32];

            for (unsigned int l = 0; l < 32; l++) {
                const unsigned int q_lo = (q[l] & 0x0Fu) |
                                          (((qh[l] >> shift_lo) & 1u) << 4);
                const unsigned int q_hi = (q[l] >> 4) |
                                          (((qh[l] >> shift_hi) & 1u) << 4);
                lo[l] = (float)q_lo * xa[l];
                hi[l] = (float)q_hi * xa[32 + l];
            }
            for (unsigned int w = 16; w > 0; w /= 2) {
                for (unsigned int l = 0; l < w; l++) {
                    lo[l] += lo[l + w];
                    hi[l] += hi[l + w];
                }
            }

            total += d * (float)sc_lo * lo[0] - dmin * (float)m_lo * xs[2 * j];
            total += d * (float)sc_hi * hi[0] - dmin * (float)m_hi * xs[2 * j + 1];
        }
    }

    return total;
}

/*
 * Q4_0: w = d * (q - 8) per 32-weight block, so
 *     sum_i w_i x_i = d * sum_i (q_i - 8) x_i.
 * Low nibbles are weights 0..15 of the block, high nibbles 16..31.
 */
static float q4_0_row_dot_fast(const unsigned char *src, const float *x,
                               size_t n_blocks)
{
    float total = 0.0f;

    for (size_t b = 0; b < n_blocks; b++) {
        const unsigned char *blk = src + b * GGUF_Q4_0_BLOCK_SIZE;
        const uint8_t *qs = blk + 2;
        const float *xb = x + b * GGUF_QK4_0;
        uint16_t d_bits;
        float t[32];

        memcpy(&d_bits, blk, 2);

        for (unsigned int l = 0; l < 16; l++) {
            t[l] = (float)((int)(qs[l] & 0x0F) - 8) * xb[l];
            t[l + 16] = (float)((int)(qs[l] >> 4) - 8) * xb[l + 16];
        }
        for (unsigned int w = 16; w > 0; w /= 2) {
            for (unsigned int l = 0; l < w; l++) {
                t[l] += t[l + w];
            }
        }

        total += gguf_fp16_to_fp32(d_bits) * t[0];
    }

    return total;
}

/* Q8_0: w = d * q per 32-weight block, q signed 8-bit. */
static float q8_0_row_dot_fast(const unsigned char *src, const float *x,
                               size_t n_blocks)
{
    float total = 0.0f;

    for (size_t b = 0; b < n_blocks; b++) {
        const unsigned char *blk = src + b * GGUF_Q8_0_BLOCK_SIZE;
        const int8_t *qs = (const int8_t *)(blk + 2);
        const float *xb = x + b * GGUF_QK8_0;
        uint16_t d_bits;
        float t[32];

        memcpy(&d_bits, blk, 2);

        for (unsigned int l = 0; l < 32; l++) {
            t[l] = (float)qs[l] * xb[l];
        }
        for (unsigned int w = 16; w > 0; w /= 2) {
            for (unsigned int l = 0; l < w; l++) {
                t[l] += t[l + w];
            }
        }

        total += gguf_fp16_to_fp32(d_bits) * t[0];
    }

    return total;
}

int gguf_matvec(
    const GGUFWeight *weight,
    const float *x,
    float *out
)
{
    if (weight == NULL || weight->data == NULL || x == NULL || out == NULL ||
        !gguf_weight_type_supported(weight->type) ||
        weight->rows > (uint64_t)INT64_MAX) {
        return 0;
    }

    const uint32_t type = weight->type;
    const unsigned char *base = weight->data;
    const long long rows = (long long)weight->rows;
    const size_t row_bytes = (size_t)weight->row_bytes;
    size_t block_weights = 0;

    /* F32 / F16 weights use the exact reference kernel. */
    if (!gguf_quant_block_info(type, &block_weights, NULL)) {
        return gguf_matvec_reference(weight, x, out);
    }

    const size_t n_blocks = (size_t)weight->cols / block_weights;

    if (type == GGUF_GGML_TYPE_Q6_K) {
        #pragma omp parallel for schedule(static)
        for (long long r = 0; r < rows; r++) {
            out[r] = q6k_row_dot_fast(base + (size_t)r * row_bytes, x, n_blocks);
        }
        return 1;
    }

    if (type == GGUF_GGML_TYPE_Q4_0) {
        #pragma omp parallel for schedule(static)
        for (long long r = 0; r < rows; r++) {
            out[r] = q4_0_row_dot_fast(base + (size_t)r * row_bytes, x, n_blocks);
        }
        return 1;
    }

    if (type == GGUF_GGML_TYPE_Q8_0) {
        #pragma omp parallel for schedule(static)
        for (long long r = 0; r < rows; r++) {
            out[r] = q8_0_row_dot_fast(base + (size_t)r * row_bytes, x, n_blocks);
        }
        return 1;
    }

    /* Q4_K / Q5_K: sums of x over each 32-input group, shared by every row. */
    float *xsum32 = malloc(n_blocks * 8 * sizeof(float));
    if (xsum32 == NULL) {
        return gguf_matvec_reference(weight, x, out);
    }
    for (size_t g = 0; g < n_blocks * 8; g++) {
        float s = 0.0f;
        for (size_t i = 0; i < 32; i++) {
            s += x[g * 32 + i];
        }
        xsum32[g] = s;
    }

    if (type == GGUF_GGML_TYPE_Q5_K) {
        #pragma omp parallel for schedule(static)
        for (long long r = 0; r < rows; r++) {
            out[r] = q5k_row_dot_fast(base + (size_t)r * row_bytes, x, xsum32,
                                      n_blocks);
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (long long r = 0; r < rows; r++) {
            out[r] = q4k_row_dot_fast(base + (size_t)r * row_bytes, x, xsum32,
                                      n_blocks);
        }
    }

    free(xsum32);
    return 1;
}
