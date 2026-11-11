/*
 * transformer.c - math kernels and the transformer forward pass.
 *
 * Attribution:
 *   rmsnorm(), softmax(), matmul() and the overall structure of forward()
 *   are adapted from Andrej Karpathy's llama2.c (run.c), MIT License,
 *   https://github.com/karpathy/llama2.c
 *
 * Changes from the reference:
 *   - Full input validation: forward() returns NULL instead of reading or
 *     writing out of bounds (bad token/pos, odd head_size, bad head counts,
 *     missing buffers).
 *   - All offsets are computed in size_t to avoid int overflow on large
 *     models (including the row offset inside matmul()).
 *   - K is copied into the cache explicitly; s->k / s->v keep pointing to
 *     their own buffers (llama2.c re-points s->k into the cache).
 *   - RoPE is applied per head; read-only inputs are const-qualified.
 */
 
#include "transformer.h"
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
 
/* ------------------------------------------------------------------ */
/* Math kernels                                                        */
/* ------------------------------------------------------------------ */
 
/*
 * RMSNorm: o = weight * x / sqrt(mean(x^2) + eps)
 * o and x may point to the same buffer.
 */
void rmsnorm(float *o, const float *x, const float *weight, int size)
{
    if (size <= 0) {
        return;
    }
 
    float ss = 0.0f;
 
    for (int i = 0; i < size; i++) {
        ss += x[i] * x[i];
    }
 
    ss /= (float)size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
 
    for (int i = 0; i < size; i++) {
        o[i] = weight[i] * (ss * x[i]);
    }
}

static void rmsnorm_with_eps(
    float *o,
    const float *x,
    const float *weight,
    int size,
    float eps
)
{
    if (size <= 0 || eps <= 0.0f) {
        return;
    }

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
 * Softmax in place, numerically stable (subtracts the max first).
 */
void softmax(float *x, int size)
{
    if (size <= 0) {
        return;
    }
 
    float max_val = x[0];
 
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
 
    float sum = 0.0f;
 
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
 
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}
 
/*
 * Matrix-vector product: xout = W * x
 *
 * x:    input vector, length n
 * w:    row-major matrix, d rows x n columns
 * xout: output vector, length d (must not alias x)
 */
void matmul(float *xout, const float *x, const float *w, int n, int d)
{
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        const float *row = w + (size_t)i * (size_t)n;
        float val = 0.0f;

        for (int j = 0; j < n; j++) {
            val += row[j] * x[j];
        }

        xout[i] = val;
    }
}
 
/* ------------------------------------------------------------------ */
/* Helpers for forward()                                               */
/* ------------------------------------------------------------------ */
 
/* Check that the configuration is one forward() can run safely. */
static int config_is_valid(const Config *p)
{
    if (p->dim <= 0 || p->hidden_dim <= 0 ||
        p->n_layers <= 0 || p->n_heads <= 0 ||
        p->n_kv_heads <= 0 || p->vocab_size <= 0 ||
        p->seq_len <= 0) {
        return 0;
    }
 
    /* Every head must have the same size. */
    if (p->dim % p->n_heads != 0) {
        return 0;
    }
 
    /* RoPE rotates pairs of values, so head_size must be even. */
    if ((p->dim / p->n_heads) % 2 != 0) {
        return 0;
    }
 
    /* Grouped-query attention: each KV head serves an equal group. */
    if (p->n_kv_heads > p->n_heads ||
        p->n_heads % p->n_kv_heads != 0) {
        return 0;
    }

    if (p->rope_dim <= 0 ||
        p->rope_dim > p->dim / p->n_heads ||
        p->rope_dim % 2 != 0 ||
        p->rope_freq_base <= 0.0f ||
        p->rms_norm_eps <= 0.0f) {
        return 0;
    }
 
    return 1;
}
 
/* Check that every weight pointer and runtime buffer exists. */
static int buffers_are_valid(const TransformerWeights *w,
                             const RunState *s)
{
    return w->token_embedding_table != NULL &&
           w->rms_att_weight != NULL &&
           w->rms_ffn_weight != NULL &&
           w->wq != NULL && w->wk != NULL &&
           w->wv != NULL && w->wo != NULL &&
           w->w1 != NULL && w->w2 != NULL && w->w3 != NULL &&
           w->rms_final_weight != NULL &&
           s->x != NULL && s->xb != NULL && s->xb2 != NULL &&
           s->hb != NULL && s->hb2 != NULL &&
           s->q != NULL && s->k != NULL && s->v != NULL &&
           s->att != NULL && s->logits != NULL &&
           s->key_cache != NULL && s->value_cache != NULL;
}
 
/*
 * Rotary positional embedding for one head (in place).
 * head_size is guaranteed even by config_is_valid().
 */
static void rope_head(
    float *vec,
    int rope_dim,
    int pos,
    float rope_freq_base
)
{
    for (int i = 0; i < rope_dim; i += 2) {
        const float freq =
            1.0f / powf(
                rope_freq_base,
                (float)i / (float)rope_dim
            );

        const float angle = (float)pos * freq;
        const float c = cosf(angle);
        const float sn = sinf(angle);

        const float v0 = vec[i];
        const float v1 = vec[i + 1];

        vec[i]     = v0 * c - v1 * sn;
        vec[i + 1] = v0 * sn + v1 * c;
    }
}
 
/* ------------------------------------------------------------------ */
/* Forward pass                                                        */
/* ------------------------------------------------------------------ */
 
/*
 * Run one token through the model.
 *
 * token: token ID, 0 <= token < vocab_size
 * pos:   position in the sequence, 0 <= pos < seq_len
 *
 * Returns a pointer to vocab_size logits, or NULL on invalid input.
 * The returned buffer is owned by t->state; do not free it. It is
 * overwritten by the next call to forward().
 */
float *forward(Transformer *t, int token, int pos)
{
    if (t == NULL) {
        return NULL;
    }
 
    const Config *p = &t->config;
    const TransformerWeights *w = &t->weights;
    RunState *s = &t->state;
 
    if (!config_is_valid(p) || !buffers_are_valid(w, s)) {
        return NULL;
    }
 
    if (token < 0 || token >= p->vocab_size ||
        pos < 0 || pos >= p->seq_len) {
        return NULL;
    }
 
    const int dim = p->dim;
    const int hidden_dim = p->hidden_dim;
    const int head_size = dim / p->n_heads;
    const int kv_dim = head_size * p->n_kv_heads;   /* <= dim, fits in int */
    const int kv_mul = p->n_heads / p->n_kv_heads;  /* query heads per KV head */
    const size_t seq_len = (size_t)p->seq_len;
    const float att_scale = 1.0f / sqrtf((float)head_size);
 
    /* 1. Token embedding lookup. */
    const float *embedding =
        w->token_embedding_table + (size_t)token * (size_t)dim;
 
    for (int i = 0; i < dim; i++) {
        s->x[i] = embedding[i];
    }
 
    /* 2. Transformer layers. */
    for (int layer = 0; layer < p->n_layers; layer++) {
        const size_t l = (size_t)layer;
 
        const float *att_weight = w->rms_att_weight + l * (size_t)dim;
        const float *ffn_weight = w->rms_ffn_weight + l * (size_t)dim;
 
        const float *wq = w->wq + l * (size_t)dim * (size_t)dim;
        const float *wk = w->wk + l * (size_t)kv_dim * (size_t)dim;
        const float *wv = w->wv + l * (size_t)kv_dim * (size_t)dim;
        const float *wo = w->wo + l * (size_t)dim * (size_t)dim;
 
        const float *w1 = w->w1 + l * (size_t)hidden_dim * (size_t)dim;
        const float *w2 = w->w2 + l * (size_t)dim * (size_t)hidden_dim;
        const float *w3 = w->w3 + l * (size_t)hidden_dim * (size_t)dim;
 
        /* Start of this layer's KV cache: [layer][pos][kv_dim] */
        const size_t layer_cache = l * seq_len * (size_t)kv_dim;
 
        /* 2a. Attention RMSNorm. */
        rmsnorm_with_eps(s->xb, s->x, att_weight, dim,p->rms_norm_eps); 
        /* 2b. Q, K, V projections. */
        matmul(s->q, s->xb, wq, dim, dim);
        matmul(s->k, s->xb, wk, dim, kv_dim);
        matmul(s->v, s->xb, wv, dim, kv_dim);
 
        /* 2c. RoPE on every query head and every key head. */
        for (int h = 0; h < p->n_heads; h++) {
            rope_head(
                s->q + (size_t)h * (size_t)head_size,
                p->rope_dim,
                pos,
                p->rope_freq_base
            );
        }

        for (int h = 0; h < p->n_kv_heads; h++) {
            rope_head(
                s->k + (size_t)h * (size_t)head_size,
                p->rope_dim,
                pos,
                p->rope_freq_base
            );
        }
 
        /* 2d. Store rotated K and raw V in the cache at this position. */
        float *k_cache_pos =
            s->key_cache + layer_cache + (size_t)pos * (size_t)kv_dim;
        float *v_cache_pos =
            s->value_cache + layer_cache + (size_t)pos * (size_t)kv_dim;
 
        for (int i = 0; i < kv_dim; i++) {
            k_cache_pos[i] = s->k[i];
            v_cache_pos[i] = s->v[i];
        }
 
        /* 2e. Causal multi-head attention over positions 0..pos. */
        for (int h = 0; h < p->n_heads; h++) {
            const float *q_head = s->q + (size_t)h * (size_t)head_size;
            float *att_head = s->att + (size_t)h * seq_len;
            const size_t kv_head_off =
                (size_t)(h / kv_mul) * (size_t)head_size;
 
            /* Scaled dot-product scores. */
            for (int tpos = 0; tpos <= pos; tpos++) {
                const float *k = s->key_cache + layer_cache +
                                 (size_t)tpos * (size_t)kv_dim + kv_head_off;
                float score = 0.0f;
 
                for (int i = 0; i < head_size; i++) {
                    score += q_head[i] * k[i];
                }
 
                att_head[tpos] = score * att_scale;
            }
 
            /* Scores -> probabilities. */
            softmax(att_head, pos + 1);
 
            /* Weighted sum of values into this head's slice of xb. */
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
 
        /* 2f. Output projection + residual. */
        matmul(s->xb2, s->xb, wo, dim, dim);
 
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb2[i];
        }
 
        /* 2g. FFN RMSNorm. */
        rmsnorm_with_eps(s->xb, s->x, ffn_weight, dim,p->rms_norm_eps);
 
        /* 2h. SwiGLU feed-forward: w2( silu(w1 x) * (w3 x) ). */
        matmul(s->hb, s->xb, w1, dim, hidden_dim);
        matmul(s->hb2, s->xb, w3, dim, hidden_dim);
 
        for (int i = 0; i < hidden_dim; i++) {
            const float val = s->hb[i];
            const float silu = val / (1.0f + expf(-val));
            s->hb[i] = silu * s->hb2[i];
        }
 
        matmul(s->xb, s->hb, w2, hidden_dim, dim);
 
        /* 2i. FFN residual. */
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb[i];
        }
    }
 
    /* 3. Final RMSNorm (in place is safe). */
    rmsnorm_with_eps(s->x, s->x, w->rms_final_weight, dim,p->rms_norm_eps);
 
    /* 4. Classifier: logits = Wcls * x. Shared weights fall back to embeddings. */
    const float *classifier =
        w->wcls != NULL ? w->wcls : w->token_embedding_table;
 
    matmul(s->logits, s->x, classifier, dim, p->vocab_size);
 
    return s->logits;
}

