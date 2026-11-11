#ifndef GGUF_FORWARD_H
#define GGUF_FORWARD_H

/*
 * gguf_forward.h - LLaMA forward pass over a GGUF model with quantized
 * weights.
 *
 * This is the GGUF counterpart of forward() in transformer.c. The maths is
 * the same; the difference is where weights come from:
 *   - every projection uses gguf_matvec() on the stored Q4_K / Q6_K data,
 *   - the token embedding row is decoded with gguf_weight_dequantize_row(),
 *   - RMSNorm weights are decoded once into a small owned FP32 buffer.
 *
 * The Stories path (Transformer / forward()) is untouched.
 */

#include "gguf_model.h"
#include "transformer.h"

typedef struct {
    GGUFModel model;     /* mmap()ed weights (see gguf_model.h) */
    RunState state;      /* activations + KV cache, from malloc_run_state() */

    /*
     * RMSNorm weights decoded to FP32, owned here:
     *   attn_norm[l] = norms + (2*l)     * dim
     *   ffn_norm[l]  = norms + (2*l + 1) * dim
     *   output_norm  = norms + (2*n_layers) * dim
     */
    float *norms;
} GGUFTransformer;

/* Load the model and allocate run-time buffers. Returns 1 on success. */
int gguf_transformer_init(
    GGUFTransformer *t,
    const char *path,
    char *error,
    size_t error_capacity
);

/* Release everything. Safe on a zeroed or partly initialised object. */
void gguf_transformer_free(GGUFTransformer *t);

/*
 * Run one token at position pos (0 <= pos < seq_len) and return
 * vocab_size logits, owned by t->state (overwritten by the next call).
 * Positions must be fed in order 0, 1, 2, ... because each call appends
 * this position's keys and values to the KV cache.
 * Returns NULL on invalid input.
 */
float *gguf_forward(GGUFTransformer *t, int token, int pos);

#endif
