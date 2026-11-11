#ifndef GGUF_MODEL_H
#define GGUF_MODEL_H

/*
 * gguf_model.h - load a LLaMA GGUF file and bind its tensors to named weights.
 *
 * Memory model
 * ------------
 * The whole GGUF file is mmap()ed once, read-only. No weight is copied:
 * every GGUFWeight.data pointer points straight into that mapping, and the
 * operating system pages weight bytes in from disk the first time they are
 * touched. The pointers stay valid until gguf_model_free() unmaps the file.
 *
 * Quantized weights stay quantized in memory. A row is converted to FP32
 * only when gguf_weight_dequantize_row() is called, into a buffer the caller
 * owns. (Converting all of TinyLlama to FP32 up front would need ~4.4 GB.)
 */

#include "checkpoint.h"
#include "transformer.h"

#include <stddef.h>
#include <stdint.h>

/*
 * One weight tensor, viewed as a matrix of `rows` rows x `cols` columns.
 *
 * GGUF dimension order is ne0 first, and ne0 is contiguous in memory, so:
 *   cols = ne0  (input size of a projection; length of one stored row)
 *   rows = ne1  (output size of a projection; 1 for a vector)
 * Row r starts at  (const uint8_t *)data + r * row_bytes.
 *
 * For a projection y = W x with x of length cols, y[r] = dot(row r, x).
 * This is the same layout as matmul(xout, x, w, n = cols, d = rows).
 */
typedef struct {
    const char *name;     /* points at the name owned by GGUFModel.tensors */
    const void *data;     /* points into the mmap()ed file */
    uint32_t type;        /* GGML type ID: 0 F32, 1 F16, 2 Q4_0, 8 Q8_0,
                             12 Q4_K, 13 Q5_K, 14 Q6_K */
    uint64_t cols;        /* ne0 */
    uint64_t rows;        /* ne1, or 1 for 1-D tensors */
    uint64_t row_bytes;   /* bytes per stored row */
    uint64_t byte_size;   /* rows * row_bytes */
} GGUFWeight;

/*
 * The nine weights of one transformer block.
 * Names in brackets are the llama2.c / TransformerWeights equivalents.
 */
typedef struct {
    GGUFWeight attn_norm;    /* blk.N.attn_norm.weight   [rms_att_weight] */
    GGUFWeight wq;           /* blk.N.attn_q.weight      [wq]  dim -> dim */
    GGUFWeight wk;           /* blk.N.attn_k.weight      [wk]  dim -> kv_dim */
    GGUFWeight wv;           /* blk.N.attn_v.weight      [wv]  dim -> kv_dim */
    GGUFWeight wo;           /* blk.N.attn_output.weight [wo]  dim -> dim */
    GGUFWeight ffn_norm;     /* blk.N.ffn_norm.weight    [rms_ffn_weight] */
    GGUFWeight ffn_gate;     /* blk.N.ffn_gate.weight    [w1]  dim -> hidden */
    GGUFWeight ffn_up;       /* blk.N.ffn_up.weight      [w3]  dim -> hidden */
    GGUFWeight ffn_down;     /* blk.N.ffn_down.weight    [w2]  hidden -> dim */
} GGUFLayer;

typedef struct {
    Config config;

    /* Parsed GGUF structure (names are owned here). */
    GGUFLayout layout;
    GGUF_TensorInfo *tensors;
    size_t tensor_count;

    /* The mapping that every GGUFWeight.data points into. */
    int fd;
    void *mapped;
    size_t file_size;

    /* Global weights. */
    GGUFWeight token_embd;   /* [token_embedding_table] rows = vocab */
    GGUFWeight output_norm;  /* [rms_final_weight] */
    GGUFWeight output;       /* [wcls] rows = vocab */
    int output_is_tied;      /* 1 if output.weight was absent and token_embd is reused */

    /* Array of config.n_layers blocks (heap-allocated). */
    GGUFLayer *layers;
} GGUFModel;

/*
 * Load and validate a LLaMA GGUF model. Every required tensor must exist,
 * have the shape implied by the configuration, lie inside the file, and use
 * a type this engine can decode (F32, F16, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K).
 * Returns 1 on success. On failure returns 0, writes a message to error,
 * and leaves the model in a freed, zeroed state.
 */
int gguf_model_load(
    GGUFModel *model,
    const char *path,
    char *error,
    size_t error_capacity
);

/* Unmap the file and free everything. Safe to call on a zeroed model. */
void gguf_model_free(GGUFModel *model);

/* 1 if this engine can decode the GGML type, else 0. */
int gguf_weight_type_supported(uint32_t type);

/*
 * Decode one row of a weight into `out`, which must hold weight->cols
 * floats. Works for every supported type. Returns 1 on success, 0 on bad
 * arguments, an out-of-range row, or an unsupported type.
 */
int gguf_weight_dequantize_row(
    const GGUFWeight *weight,
    uint64_t row,
    float *out
);

/*
 * Matrix-vector product with a weight kept in its stored format:
 *
 *     out[r] = sum_{c=0}^{cols-1} W[r][c] * x[c]      for r = 0..rows-1
 *
 * x must hold weight->cols floats and out weight->rows floats; out must not
 * alias x. No FP32 copy of the matrix is ever made. Rows are independent
 * and are split across OpenMP threads; the result does not depend on the
 * number of threads.
 *
 * Two implementations, same maths, different floating-point summation
 * order:
 *
 * gguf_matvec_reference(): decodes each block to FP32, then sums
 *   W[r][c] * x[c] strictly in column order. Bit-exactly reproducible
 *   with NumPy (examples/05_matrix_math/compare_matvec.py). Used to verify the fast path.
 *
 * gguf_matvec(): the fast path used by the forward pass. Q4_0, Q8_0,
 *   Q4_K, Q5_K and Q6_K are computed from the quantized form directly
 *   (fused kernels, see gguf_model.c); F32 / F16 use the reference kernel. Results differ
 *   from the reference only by float32 rounding (~1e-7 relative).
 *
 * Both return 1 on success, 0 on bad arguments or an unsupported type.
 */
int gguf_matvec(
    const GGUFWeight *weight,
    const float *x,
    float *out
);

int gguf_matvec_reference(
    const GGUFWeight *weight,
    const float *x,
    float *out
);

#endif
