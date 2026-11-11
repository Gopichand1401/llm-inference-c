#ifndef GGUF_QUANT_H
#define GGUF_QUANT_H

#include <stddef.h>
#include <stdint.h>

/*
 * GGML "K-quant" formats group weights into super-blocks of 256 values.
 * Every block type below must match the official GGML C struct byte-for-byte,
 * because we read the blocks straight out of the memory-mapped GGUF file.
 */
#define GGUF_QK_K 256

/* GGML tensor type IDs (enum ggml_type) used by this file. */
#define GGUF_GGML_TYPE_F32  0
#define GGUF_GGML_TYPE_F16  1
#define GGUF_GGML_TYPE_Q4_0 2
#define GGUF_GGML_TYPE_Q8_0 8
#define GGUF_GGML_TYPE_Q4_K 12
#define GGUF_GGML_TYPE_Q5_K 13
#define GGUF_GGML_TYPE_Q6_K 14

/* Convert one IEEE-754 half-precision value to single precision. */
float gguf_fp16_to_fp32(uint16_t value);

/* ===================================================================== */
/* Q4_K                                                                  */
/* ===================================================================== */

#define GGUF_Q4_K_SCALE_SIZE 12
#define GGUF_Q4_K_BLOCK_SIZE 144

/*
 * Q4_K quantization block (matches GGML block_q4_K).
 * Each block stores 256 quantized weights.
 */
typedef struct {
    uint16_t d;       /* FP16 super-block scale */
    uint16_t dmin;    /* FP16 super-block minimum scale */

    uint8_t scales[GGUF_Q4_K_SCALE_SIZE];

    uint8_t qs[GGUF_QK_K / 2];
} GGUFBlockQ4K;

int gguf_q4k_get_scale_min(
    const uint8_t scales[GGUF_Q4_K_SCALE_SIZE],
    unsigned int index,
    uint8_t *scale,
    uint8_t *min
);

int gguf_dequantize_q4k_block(
    const GGUFBlockQ4K *block,
    float output[GGUF_QK_K]
);

/* Verify exact memory layout. */
_Static_assert(
    sizeof(GGUFBlockQ4K) == GGUF_Q4_K_BLOCK_SIZE,
    "Incorrect Q4_K block size"
);

/* ===================================================================== */
/* Q6_K                                                                  */
/* ===================================================================== */

#define GGUF_Q6_K_QL_SIZE     (GGUF_QK_K / 2)   /* 128: low 4 bits  */
#define GGUF_Q6_K_QH_SIZE     (GGUF_QK_K / 4)   /*  64: high 2 bits */
#define GGUF_Q6_K_SCALE_COUNT (GGUF_QK_K / 16)  /*  16: one per 16 weights */
#define GGUF_Q6_K_BLOCK_SIZE  210

/*
 * Q6_K quantization block (matches GGML block_q6_K exactly):
 *
 *   uint8_t ql[128];   low 4 bits of every 6-bit quant (two per byte)
 *   uint8_t qh[64];    high 2 bits of every 6-bit quant (four per byte)
 *   int8_t  scales[16];signed 8-bit scale for each group of 16 weights
 *   ggml_half d;       FP16 super-block scale (stored LAST, unlike Q4_K)
 *
 * Reconstruction:  weight[i] = d * scales[i / 16] * (q[i] - 32)
 * where q[i] is the unsigned 6-bit value 0..63, so (q - 32) is -32..31.
 * Q6_K has no "min" term: it is a symmetric format.
 */
typedef struct {
    uint8_t ql[GGUF_Q6_K_QL_SIZE];
    uint8_t qh[GGUF_Q6_K_QH_SIZE];
    int8_t scales[GGUF_Q6_K_SCALE_COUNT];
    uint16_t d;
} GGUFBlockQ6K;

/*
 * Unpack the 256 signed 6-bit quants of one block, in output order.
 * Every value written is in the range -32..31.
 * Returns 1 on success, 0 on NULL arguments.
 */
int gguf_q6k_unpack_block(
    const GGUFBlockQ6K *block,
    int8_t quants[GGUF_QK_K]
);

/*
 * Reconstruct 256 FP32 weights from one Q6_K block.
 * Returns 1 on success, 0 on NULL arguments.
 */
int gguf_dequantize_q6k_block(
    const GGUFBlockQ6K *block,
    float output[GGUF_QK_K]
);

/* 128 + 64 + 16 + 2 = 210 bytes, with no compiler padding. */
_Static_assert(
    sizeof(GGUFBlockQ6K) == GGUF_Q6_K_BLOCK_SIZE,
    "Incorrect Q6_K block size"
);

/* ===================================================================== */
/* Q4_0  (32 weights per block)                                          */
/* ===================================================================== */

#define GGUF_QK4_0 32
#define GGUF_Q4_0_BLOCK_SIZE 18

/*
 * Q4_0 block (matches GGML block_q4_0):
 *   ggml_half d;        FP16 scale
 *   uint8_t   qs[16];   32 x 4-bit quants: low nibbles = weights 0..15,
 *                       high nibbles = weights 16..31
 * weight[i] = d * (q[i] - 8)          (q in 0..15, so -8..7)
 */
typedef struct {
    uint16_t d;
    uint8_t qs[GGUF_QK4_0 / 2];
} GGUFBlockQ4_0;

int gguf_dequantize_q4_0_block(const GGUFBlockQ4_0 *block,
                               float output[GGUF_QK4_0]);

_Static_assert(sizeof(GGUFBlockQ4_0) == GGUF_Q4_0_BLOCK_SIZE,
               "Incorrect Q4_0 block size");

/* ===================================================================== */
/* Q8_0  (32 weights per block)                                          */
/* ===================================================================== */

#define GGUF_QK8_0 32
#define GGUF_Q8_0_BLOCK_SIZE 34

/*
 * Q8_0 block (matches GGML block_q8_0):
 *   ggml_half d;        FP16 scale
 *   int8_t    qs[32];   signed 8-bit quants
 * weight[i] = d * q[i]
 */
typedef struct {
    uint16_t d;
    int8_t qs[GGUF_QK8_0];
} GGUFBlockQ8_0;

int gguf_dequantize_q8_0_block(const GGUFBlockQ8_0 *block,
                               float output[GGUF_QK8_0]);

_Static_assert(sizeof(GGUFBlockQ8_0) == GGUF_Q8_0_BLOCK_SIZE,
               "Incorrect Q8_0 block size");

/* ===================================================================== */
/* Q5_K  (256 weights per block)                                         */
/* ===================================================================== */

#define GGUF_Q5_K_BLOCK_SIZE 176

/*
 * Q5_K block (matches GGML block_q5_K): Q4_K plus one extra bit per weight.
 *   ggml_half d, dmin;     super-block scale and min scale
 *   uint8_t scales[12];    8 packed 6-bit (scale, min) pairs, as Q4_K
 *   uint8_t qh[32];        5th bit: bit g of qh[l] belongs to weight
 *                          32*g + l
 *   uint8_t qs[128];       low 4 bits, laid out as Q4_K
 * weight = d * sc_g * q - dmin * m_g,  q = low4 | (bit << 4)  (0..31)
 */
typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[GGUF_Q4_K_SCALE_SIZE];
    uint8_t qh[GGUF_QK_K / 8];
    uint8_t qs[GGUF_QK_K / 2];
} GGUFBlockQ5K;

int gguf_dequantize_q5k_block(const GGUFBlockQ5K *block,
                              float output[GGUF_QK_K]);

_Static_assert(sizeof(GGUFBlockQ5K) == GGUF_Q5_K_BLOCK_SIZE,
               "Incorrect Q5_K block size");

/* ===================================================================== */
/* Generic per-type access                                               */
/* ===================================================================== */

/*
 * Block geometry of a quantized type this engine can decode:
 * Q4_0, Q8_0 (32 weights), Q4_K, Q5_K, Q6_K (256 weights).
 * Returns 0 for any other type (including F32 / F16, which are not
 * block-quantized).
 */
int gguf_quant_block_info(uint32_t type, size_t *block_weights,
                          size_t *block_bytes);

/*
 * Decode one block of `type` from raw bytes (any alignment) into output,
 * which must hold block_weights floats. Returns 1 on success.
 */
int gguf_dequantize_block(uint32_t type, const void *block_bytes,
                          float *output);

#endif
