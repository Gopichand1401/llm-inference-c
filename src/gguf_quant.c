#include "gguf_quant.h"

#include <stdint.h>
#include <string.h>

float gguf_fp16_to_fp32(uint16_t value)
{
    uint32_t sign = ((uint32_t)value & 0x8000u) << 16;
    uint32_t exponent = ((uint32_t)value >> 10) & 0x1Fu;
    uint32_t mantissa = (uint32_t)value & 0x03FFu;

    uint32_t result;

    if (exponent == 0) {

        if (mantissa == 0) {
            /* Positive or negative zero. */
            result = sign;
        } else {
            /* Handle FP16 subnormal numbers. */
            int32_t e = -14;

            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                e--;
            }

            mantissa &= 0x03FFu;

            uint32_t exp32 = (uint32_t)(e + 127);

            result = sign |
                     (exp32 << 23) |
                     (mantissa << 13);
        }

    } else if (exponent == 31) {

        /* Infinity or NaN. */
        result = sign |
                 0x7F800000u |
                 (mantissa << 13);

    } else {

        /* Normal FP16 number. */
        uint32_t exp32 = exponent - 15 + 127;

        result = sign |
                 (exp32 << 23) |
                 (mantissa << 13);
    }

    float output;

    memcpy(&output, &result, sizeof(output));

    return output;
}

/* ===================================================================== */
/* Q4_K                                                                  */
/* ===================================================================== */

int gguf_q4k_get_scale_min(
        const uint8_t scales[GGUF_Q4_K_SCALE_SIZE],
        unsigned int index,
        uint8_t *scale,
        uint8_t *min
    ){
        if (scales == NULL ||
            scale == NULL ||
            min == NULL ||
            index >= 8) {
            return 0;
        }

        if (index < 4) {
            *scale = scales[index] & 0x3F;
            *min = scales[index + 4] & 0x3F;
        } else {
            *scale = (scales[index + 4] & 0x0F) |
                    ((scales[index - 4] >> 6) << 4);

            *min = (scales[index + 4] >> 4) |
                ((scales[index] >> 6) << 4);
        }

        return 1;
}

int gguf_dequantize_q4k_block(
    const GGUFBlockQ4K *block,
    float output[GGUF_QK_K]
)
{
    if (block == NULL || output == NULL) {
        return 0;
    }

    /* Convert FP16 scale values into FP32. */
    const float d = gguf_fp16_to_fp32(block->d);
    const float dmin = gguf_fp16_to_fp32(block->dmin);

    unsigned int scale_index = 0;

    /* Process 256 weights in groups of 64. */
    for (unsigned int j = 0; j < GGUF_QK_K; j += 64) {

        uint8_t scale1, min1;
        uint8_t scale2, min2;

        /* Decode scale and minimum for first 32 weights. */
        if (!gguf_q4k_get_scale_min(
                block->scales,
                scale_index,
                &scale1,
                &min1)) {
            return 0;
        }

        /* Decode scale and minimum for next 32 weights. */
        if (!gguf_q4k_get_scale_min(
                block->scales,
                scale_index + 1,
                &scale2,
                &min2)) {
            return 0;
        }

        const float d1 = d * (float)scale1;
        const float m1 = dmin * (float)min1;

        const float d2 = d * (float)scale2;
        const float m2 = dmin * (float)min2;

        const uint8_t *q = block->qs + (j / 64) * 32;

        /* Decode lower 4 bits. */
        for (unsigned int l = 0; l < 32; l++) {

            uint8_t quant = q[l] & 0x0F;

            output[j + l] =
                d1 * (float)quant - m1;
        }

        /* Decode upper 4 bits. */
        for (unsigned int l = 0; l < 32; l++) {

            uint8_t quant = q[l] >> 4;

            output[j + 32 + l] =
                d2 * (float)quant - m2;
        }

        scale_index += 2;
    }

    return 1;
}

/* ===================================================================== */
/* Q6_K                                                                  */
/* ===================================================================== */

/*
 * Bit layout (from GGML dequantize_row_q6_K).
 *
 * The 256 weights are processed as two halves of 128. For half h
 * (h = 0 or 1), ql advances by 64 bytes and qh by 32 bytes:
 *
 *     ql_h = ql + 64*h       qh_h = qh + 32*h      base = 128*h
 *
 * Inside a half, for l = 0..31, one qh byte supplies the high 2 bits of
 * four different weights, and two ql bytes supply their low 4 bits:
 *
 *   weight base+l      : low = ql_h[l]      & 0xF, high = (qh_h[l] >> 0) & 3
 *   weight base+l+32   : low = ql_h[l + 32] & 0xF, high = (qh_h[l] >> 2) & 3
 *   weight base+l+64   : low = ql_h[l]      >> 4,  high = (qh_h[l] >> 4) & 3
 *   weight base+l+96   : low = ql_h[l + 32] >> 4,  high = (qh_h[l] >> 6) & 3
 *
 *   q = (low | (high << 4)) - 32        (range -32..31)
 */
int gguf_q6k_unpack_block(
    const GGUFBlockQ6K *block,
    int8_t quants[GGUF_QK_K]
)
{
    if (block == NULL || quants == NULL) {
        return 0;
    }

    for (unsigned int half = 0; half < 2; half++) {

        const uint8_t *ql = block->ql + 64u * half;
        const uint8_t *qh = block->qh + 32u * half;
        int8_t *q = quants + 128u * half;

        for (unsigned int l = 0; l < 32; l++) {

            const unsigned int h = qh[l];

            const int q1 = (int)((ql[l]      & 0x0Fu) | (((h >> 0) & 3u) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0x0Fu) | (((h >> 2) & 3u) << 4)) - 32;
            const int q3 = (int)((ql[l]      >> 4)    | (((h >> 4) & 3u) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4)    | (((h >> 6) & 3u) << 4)) - 32;

            q[l]      = (int8_t)q1;
            q[l + 32] = (int8_t)q2;
            q[l + 64] = (int8_t)q3;
            q[l + 96] = (int8_t)q4;
        }
    }

    return 1;
}

int gguf_dequantize_q6k_block(
    const GGUFBlockQ6K *block,
    float output[GGUF_QK_K]
)
{
    int8_t quants[GGUF_QK_K];

    if (block == NULL || output == NULL) {
        return 0;
    }

    if (!gguf_q6k_unpack_block(block, quants)) {
        return 0;
    }

    const float d = gguf_fp16_to_fp32(block->d);

    /*
     * Each group of 16 consecutive weights shares one signed scale.
     * Evaluate (d * scale) first and then multiply by q, the same order
     * GGML and gguf-py use, so results match bit-for-bit.
     */
    for (unsigned int group = 0; group < GGUF_Q6_K_SCALE_COUNT; group++) {

        const float group_scale = d * (float)block->scales[group];

        for (unsigned int k = 0; k < 16; k++) {
            const unsigned int i = group * 16u + k;
            output[i] = group_scale * (float)quants[i];
        }
    }

    return 1;
}

/* ===================================================================== */
/* Q4_0                                                                  */
/* ===================================================================== */

int gguf_dequantize_q4_0_block(const GGUFBlockQ4_0 *block,
                               float output[GGUF_QK4_0])
{
    if (block == NULL || output == NULL) {
        return 0;
    }

    const float d = gguf_fp16_to_fp32(block->d);

    for (unsigned int j = 0; j < GGUF_QK4_0 / 2; j++) {
        const int lo = (int)(block->qs[j] & 0x0F) - 8;
        const int hi = (int)(block->qs[j] >> 4) - 8;
        output[j] = d * (float)lo;
        output[j + GGUF_QK4_0 / 2] = d * (float)hi;
    }

    return 1;
}

/* ===================================================================== */
/* Q8_0                                                                  */
/* ===================================================================== */

int gguf_dequantize_q8_0_block(const GGUFBlockQ8_0 *block,
                               float output[GGUF_QK8_0])
{
    if (block == NULL || output == NULL) {
        return 0;
    }

    const float d = gguf_fp16_to_fp32(block->d);

    for (unsigned int j = 0; j < GGUF_QK8_0; j++) {
        output[j] = (float)block->qs[j] * d;
    }

    return 1;
}

/* ===================================================================== */
/* Q5_K                                                                  */
/* ===================================================================== */

/*
 * Same structure as Q4_K (dequantize_row_q5_K in GGML): 4 chunks of 64
 * weights; chunk j uses qs[32j .. 32j+31] (low nibbles for its first 32
 * weights, high nibbles for the next 32) and bits 2j / 2j+1 of qh[0..31]
 * as the 5th bit.
 */
int gguf_dequantize_q5k_block(const GGUFBlockQ5K *block,
                              float output[GGUF_QK_K])
{
    if (block == NULL || output == NULL) {
        return 0;
    }

    const float d = gguf_fp16_to_fp32(block->d);
    const float dmin = gguf_fp16_to_fp32(block->dmin);

    for (unsigned int j = 0; j < 4; j++) {
        uint8_t sc1, m1, sc2, m2;

        if (!gguf_q4k_get_scale_min(block->scales, 2 * j, &sc1, &m1) ||
            !gguf_q4k_get_scale_min(block->scales, 2 * j + 1, &sc2, &m2)) {
            return 0;
        }

        const float d1 = d * (float)sc1;
        const float min1 = dmin * (float)m1;
        const float d2 = d * (float)sc2;
        const float min2 = dmin * (float)m2;
        const uint8_t *ql = block->qs + 32 * j;
        const uint8_t bit1 = (uint8_t)(1u << (2 * j));
        const uint8_t bit2 = (uint8_t)(1u << (2 * j + 1));
        float *y = output + 64 * j;

        for (unsigned int l = 0; l < 32; l++) {
            const unsigned int q1 = (ql[l] & 0x0Fu) +
                                    ((block->qh[l] & bit1) ? 16u : 0u);
            const unsigned int q2 = (ql[l] >> 4) +
                                    ((block->qh[l] & bit2) ? 16u : 0u);
            y[l] = d1 * (float)q1 - min1;
            y[l + 32] = d2 * (float)q2 - min2;
        }
    }

    return 1;
}

/* ===================================================================== */
/* Generic per-type access                                               */
/* ===================================================================== */

int gguf_quant_block_info(uint32_t type, size_t *block_weights,
                          size_t *block_bytes)
{
    size_t weights = 0, bytes = 0;

    switch (type) {
        case GGUF_GGML_TYPE_Q4_0: weights = GGUF_QK4_0; bytes = GGUF_Q4_0_BLOCK_SIZE; break;
        case GGUF_GGML_TYPE_Q8_0: weights = GGUF_QK8_0; bytes = GGUF_Q8_0_BLOCK_SIZE; break;
        case GGUF_GGML_TYPE_Q4_K: weights = GGUF_QK_K;  bytes = GGUF_Q4_K_BLOCK_SIZE; break;
        case GGUF_GGML_TYPE_Q5_K: weights = GGUF_QK_K;  bytes = GGUF_Q5_K_BLOCK_SIZE; break;
        case GGUF_GGML_TYPE_Q6_K: weights = GGUF_QK_K;  bytes = GGUF_Q6_K_BLOCK_SIZE; break;
        default: return 0;
    }

    if (block_weights != NULL) *block_weights = weights;
    if (block_bytes != NULL) *block_bytes = bytes;
    return 1;
}

/* Copy the bytes into a properly aligned struct, then decode. */
int gguf_dequantize_block(uint32_t type, const void *block_bytes,
                          float *output)
{
    if (block_bytes == NULL || output == NULL) {
        return 0;
    }

    switch (type) {
        case GGUF_GGML_TYPE_Q4_0: {
            GGUFBlockQ4_0 b;
            memcpy(&b, block_bytes, sizeof(b));
            return gguf_dequantize_q4_0_block(&b, output);
        }
        case GGUF_GGML_TYPE_Q8_0: {
            GGUFBlockQ8_0 b;
            memcpy(&b, block_bytes, sizeof(b));
            return gguf_dequantize_q8_0_block(&b, output);
        }
        case GGUF_GGML_TYPE_Q4_K: {
            GGUFBlockQ4K b;
            memcpy(&b, block_bytes, sizeof(b));
            return gguf_dequantize_q4k_block(&b, output);
        }
        case GGUF_GGML_TYPE_Q5_K: {
            GGUFBlockQ5K b;
            memcpy(&b, block_bytes, sizeof(b));
            return gguf_dequantize_q5k_block(&b, output);
        }
        case GGUF_GGML_TYPE_Q6_K: {
            GGUFBlockQ6K b;
            memcpy(&b, block_bytes, sizeof(b));
            return gguf_dequantize_q6k_block(&b, output);
        }
        default:
            return 0;
    }
}
