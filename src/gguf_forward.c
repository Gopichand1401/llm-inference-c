/*
 * gguf_forward.c - LLaMA forward pass using quantized GGUF weights.
 *
 * Shapes for TinyLlama (dim 2048, 32 query heads, 4 KV heads):
 *   head_size = dim / n_heads            = 64
 *   kv_dim    = head_size * n_kv_heads   = 256
 *   kv_mul    = n_heads / n_kv_heads     = 8 query heads share one KV head
 */
#include "gguf_forward.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void set_error(char *error, size_t capacity, const char *message)
{
    if (error != NULL && capacity > 0) {
        snprintf(error, capacity, "%s", message);
    }
}

/* RMSNorm: o = weight * x / sqrt(mean(x^2) + eps). o may alias x. */
static void rmsnorm_eps(float *o, const float *x, const float *weight,
                        int size, float eps)
{
    float ss = 0.0f;

    for (int i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }

    ss /= (float)size;
    ss += eps;
    ss = 1.0f / sqrtf(ss);

    for (int i = 0; i < size; i++) {
        o[i] = weight[i] * (ss * x[i]);
    }
}

/*
 * RoPE on one head, rotating adjacent pairs (v[i], v[i+1]) by the angle
 * (pos / scale) * base^(-i / rope_dim), i = 0, 2, 4, ...
 * This is GGML's "normal" RoPE, which is what GGUF llama models use
 * (the converter reorders the Q/K weights so that adjacent pairs rotate).
 * scale is the linear RoPE scaling factor (1 for unscaled models, for
 * which pos / 1.0f is exactly pos, so results are unchanged).
 */
static void rope_head(float *vec, int rope_dim, int pos, float base,
                      float scale)
{
    const float position = (float)pos / scale;

    for (int i = 0; i < rope_dim; i += 2) {
        const float freq = 1.0f / powf(base, (float)i / (float)rope_dim);
        const float angle = position * freq;
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float v0 = vec[i];
        const float v1 = vec[i + 1];

        vec[i]     = v0 * c - v1 * s;
        vec[i + 1] = v0 * s + v1 * c;
    }
}

/* ------------------------------------------------------------------ */
/* Setup / teardown                                                    */
/* ------------------------------------------------------------------ */

void gguf_transformer_free(GGUFTransformer *t)
{
    if (t == NULL) {
        return;
    }

    free_run_state(&t->state);
    free(t->norms);
    gguf_model_free(&t->model);

    memset(t, 0, sizeof(*t));
    t->model.fd = -1;
}

int gguf_transformer_init(
    GGUFTransformer *t,
    const char *path,
    char *error,
    size_t error_capacity
)
{
    if (t == NULL || path == NULL) {
        set_error(error, error_capacity, "Invalid arguments.");
        return 0;
    }

    memset(t, 0, sizeof(*t));
    t->model.fd = -1;

    if (!gguf_model_load(&t->model, path, error, error_capacity)) {
        return 0;
    }

    const Config *c = &t->model.config;

    /* The forward pass below needs RoPE to fit inside each head. */
    if (c->rope_dim <= 0 || c->rope_dim % 2 != 0 ||
        c->rope_dim > c->dim / c->n_heads ||
        !(c->rope_scale > 0.0f)) {
        set_error(error, error_capacity, "Unsupported RoPE dimension.");
        gguf_transformer_free(t);
        return 0;
    }

    /* Decode all RMSNorm weights once (2 per layer + final). */
    const size_t dim = (size_t)c->dim;
    const size_t n_norms = 2u * (size_t)c->n_layers + 1u;

    t->norms = malloc(n_norms * dim * sizeof(float));

    if (t->norms == NULL) {
        set_error(error, error_capacity, "Cannot allocate norm weights.");
        gguf_transformer_free(t);
        return 0;
    }

    for (int l = 0; l < c->n_layers; l++) {
        if (!gguf_weight_dequantize_row(&t->model.layers[l].attn_norm, 0,
                                        t->norms + (2u * (size_t)l) * dim) ||
            !gguf_weight_dequantize_row(&t->model.layers[l].ffn_norm, 0,
                                        t->norms + (2u * (size_t)l + 1u) * dim)) {
            set_error(error, error_capacity, "Cannot decode norm weights.");
            gguf_transformer_free(t);
            return 0;
        }
    }

    if (!gguf_weight_dequantize_row(&t->model.output_norm, 0,
                                    t->norms + (n_norms - 1u) * dim)) {
        set_error(error, error_capacity, "Cannot decode output norm.");
        gguf_transformer_free(t);
        return 0;
    }

    /* Activations and KV cache: the same RunState the Stories path uses. */
    if (!malloc_run_state(&t->state, c, error, error_capacity)) {
        gguf_transformer_free(t);
        return 0;
    }

    return 1;
}

/* ------------------------------------------------------------------ */
/* Forward pass                                                        */
/* ------------------------------------------------------------------ */

float *gguf_forward(GGUFTransformer *t, int token, int pos)
{
    if (t == NULL || t->norms == NULL || t->model.layers == NULL ||
        t->state.x == NULL) {
        return NULL;
    }

    const GGUFModel *m = &t->model;
    const Config *p = &m->config;
    RunState *s = &t->state;

    if (token < 0 || token >= p->vocab_size ||
        pos < 0 || pos >= p->seq_len) {
        return NULL;
    }

    const int dim = p->dim;
    const int hidden_dim = p->hidden_dim;
    const int head_size = dim / p->n_heads;
    const int kv_dim = head_size * p->n_kv_heads;
    const int kv_mul = p->n_heads / p->n_kv_heads;
    const size_t seq_len = (size_t)p->seq_len;
    const float att_scale = 1.0f / sqrtf((float)head_size);
    const float eps = p->rms_norm_eps;

    /* 1. Token embedding: decode row `token` of token_embd into x. */
    if (!gguf_weight_dequantize_row(&m->token_embd, (uint64_t)token, s->x)) {
        return NULL;
    }

    /* 2. Transformer blocks. */
    for (int layer = 0; layer < p->n_layers; layer++) {
        const GGUFLayer *L = &m->layers[layer];
        const size_t l = (size_t)layer;
        const float *attn_norm = t->norms + (2u * l) * (size_t)dim;
        const float *ffn_norm = t->norms + (2u * l + 1u) * (size_t)dim;

        /* KV cache layout: [layer][position][kv_dim]. */
        const size_t layer_cache = l * seq_len * (size_t)kv_dim;

        /* 2a. Attention RMSNorm. */
        rmsnorm_eps(s->xb, s->x, attn_norm, dim, eps);

        /* 2b. Q (dim), K (kv_dim), V (kv_dim) projections. */
        if (!gguf_matvec(&L->wq, s->xb, s->q) ||
            !gguf_matvec(&L->wk, s->xb, s->k) ||
            !gguf_matvec(&L->wv, s->xb, s->v)) {
            return NULL;
        }

        /* 2c. RoPE on every query head and every key head. */
        for (int h = 0; h < p->n_heads; h++) {
            rope_head(s->q + (size_t)h * (size_t)head_size,
                      p->rope_dim, pos, p->rope_freq_base, p->rope_scale);
        }
        for (int h = 0; h < p->n_kv_heads; h++) {
            rope_head(s->k + (size_t)h * (size_t)head_size,
                      p->rope_dim, pos, p->rope_freq_base, p->rope_scale);
        }

        /* 2d. Append rotated K and raw V to the cache. */
        float *k_cache_pos = s->key_cache + layer_cache +
                             (size_t)pos * (size_t)kv_dim;
        float *v_cache_pos = s->value_cache + layer_cache +
                             (size_t)pos * (size_t)kv_dim;

        memcpy(k_cache_pos, s->k, (size_t)kv_dim * sizeof(float));
        memcpy(v_cache_pos, s->v, (size_t)kv_dim * sizeof(float));

        /*
         * 2e. Grouped-query attention. Query head h reads KV head
         * h / kv_mul, so heads 0..7 share KV head 0, 8..15 share KV head 1,
         * and so on. Each head writes its output into its own slice of xb.
         */
        for (int h = 0; h < p->n_heads; h++) {
            const float *q_head = s->q + (size_t)h * (size_t)head_size;
            float *att_head = s->att + (size_t)h * seq_len;
            const size_t kv_head_off = (size_t)(h / kv_mul) * (size_t)head_size;

            for (int tpos = 0; tpos <= pos; tpos++) {
                const float *k = s->key_cache + layer_cache +
                                 (size_t)tpos * (size_t)kv_dim + kv_head_off;
                float score = 0.0f;

                for (int i = 0; i < head_size; i++) {
                    score += q_head[i] * k[i];
                }

                att_head[tpos] = score * att_scale;
            }

            softmax(att_head, pos + 1);

            float *head_out = s->xb + (size_t)h * (size_t)head_size;

            for (int i = 0; i < head_size; i++) {
                head_out[i] = 0.0f;
            }

            for (int tpos = 0; tpos <= pos; tpos++) {
                const float *v = s->value_cache + layer_cache +
                                 (size_t)tpos * (size_t)kv_dim + kv_head_off;
                const float a = att_head[tpos];

                for (int i = 0; i < head_size; i++) {
                    head_out[i] += a * v[i];
                }
            }
        }

        /* 2f. Output projection and residual connection. */
        if (!gguf_matvec(&L->wo, s->xb, s->xb2)) {
            return NULL;
        }

        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb2[i];
        }

        /* 2g. FFN RMSNorm. */
        rmsnorm_eps(s->xb, s->x, ffn_norm, dim, eps);

        /* 2h. SwiGLU: down( silu(gate x) * (up x) ). */
        if (!gguf_matvec(&L->ffn_gate, s->xb, s->hb) ||
            !gguf_matvec(&L->ffn_up, s->xb, s->hb2)) {
            return NULL;
        }

        for (int i = 0; i < hidden_dim; i++) {
            const float g = s->hb[i];
            s->hb[i] = (g / (1.0f + expf(-g))) * s->hb2[i];
        }

        if (!gguf_matvec(&L->ffn_down, s->hb, s->xb)) {
            return NULL;
        }

        /* 2i. FFN residual connection. */
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb[i];
        }
    }

    /* 3. Final RMSNorm (in place). */
    rmsnorm_eps(s->x, s->x,
                t->norms + (2u * (size_t)p->n_layers) * (size_t)dim,
                dim, eps);

    /* 4. Vocabulary projection: logits = output * x. */
    if (!gguf_matvec(&m->output, s->x, s->logits)) {
        return NULL;
    }

    return s->logits;
}
