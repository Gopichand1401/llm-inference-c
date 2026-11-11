/*
 * gguf_q6k_demo.c - synthetic tests for Q6_K unpacking and dequantization.
 *
 * Every expected value below is computed by hand from the GGML Q6_K bit
 * layout, so this test does not depend on a model file.
 */
#include "gguf_quant.h"

#include <stdio.h>
#include <string.h>

static int failures = 0;

static void check_float(const char *test, unsigned int index,
                        float actual, float expected)
{
    if (actual != expected) {
        printf("FAIL %-28s index %3u: got %.8f expected %.8f\n",
               test, index, (double)actual, (double)expected);
        failures++;
    }
}

/* Fill a block with ql = qh = 0, scales = 1, d = 1.0. All weights = -32. */
static void reset_block(GGUFBlockQ6K *b)
{
    memset(b, 0, sizeof(*b));

    for (unsigned int i = 0; i < GGUF_Q6_K_SCALE_COUNT; i++) {
        b->scales[i] = 1;
    }

    b->d = 0x3C00; /* 1.0 */
}

int main(void)
{
    GGUFBlockQ6K b;
    float out[GGUF_QK_K];
    int8_t q[GGUF_QK_K];

    printf("=== Q6_K synthetic tests ===\n");
    printf("sizeof(GGUFBlockQ6K) = %zu (expected 210)\n", sizeof(GGUFBlockQ6K));

    /* Test 1: all-zero bits -> every quant is 0 - 32 = -32. */
    reset_block(&b);
    gguf_dequantize_q6k_block(&b, out);
    for (unsigned int i = 0; i < GGUF_QK_K; i++) {
        check_float("all zero bits", i, out[i], -32.0f);
    }

    /* Test 2: all bits set -> every quant is 63 - 32 = 31. */
    reset_block(&b);
    memset(b.ql, 0xFF, sizeof(b.ql));
    memset(b.qh, 0xFF, sizeof(b.qh));
    gguf_dequantize_q6k_block(&b, out);
    for (unsigned int i = 0; i < GGUF_QK_K; i++) {
        check_float("all bits set", i, out[i], 31.0f);
    }

    /* Test 3: negative int8 scale (-1) flips the sign: -32 * -1 = 32. */
    reset_block(&b);
    memset(b.scales, 0xFF, sizeof(b.scales)); /* int8 -1 */
    gguf_dequantize_q6k_block(&b, out);
    for (unsigned int i = 0; i < GGUF_QK_K; i++) {
        check_float("negative scale", i, out[i], 32.0f);
    }

    /* Test 4: each scale group k uses scales[k] = k + 1 -> -32 * (i/16 + 1). */
    reset_block(&b);
    for (unsigned int i = 0; i < GGUF_Q6_K_SCALE_COUNT; i++) {
        b.scales[i] = (int8_t)(i + 1);
    }
    gguf_dequantize_q6k_block(&b, out);
    for (unsigned int i = 0; i < GGUF_QK_K; i++) {
        check_float("scale per 16 weights", i, out[i],
                    -32.0f * (float)(i / 16 + 1));
    }

    /* Test 5: FP16 super-scale d = 0.5 (0x3800) -> -16 everywhere. */
    reset_block(&b);
    b.d = 0x3800;
    gguf_dequantize_q6k_block(&b, out);
    for (unsigned int i = 0; i < GGUF_QK_K; i++) {
        check_float("d = 0.5", i, out[i], -16.0f);
    }

    /*
     * Test 6: bit placement. Set one field at a time and check that exactly
     * one output weight changes, at the position GGML defines.
     */
    struct {
        const char *name;
        unsigned int byte_index;  /* index into ql (is_qh=0) or qh (is_qh=1) */
        int is_qh;
        uint8_t value;
        unsigned int changed;     /* output index that must change */
        float expected;           /* new value at that index */
    } cases[] = {
        /* ql[l] low nibble -> weight l */
        { "ql[0] low  -> w0",     0, 0, 0x0F,   0, 15.0f - 32.0f },
        /* ql[l] high nibble -> weight l+64 */
        { "ql[0] high -> w64",    0, 0, 0xF0,  64, 15.0f - 32.0f },
        /* ql[l+32] low nibble -> weight l+32 */
        { "ql[32] low -> w32",   32, 0, 0x0F,  32, 15.0f - 32.0f },
        /* ql[l+32] high nibble -> weight l+96 */
        { "ql[32] high-> w96",   32, 0, 0xF0,  96, 15.0f - 32.0f },
        /* qh[l] bits 0-1 -> weight l (adds 16 per unit: 3 << 4 = 48) */
        { "qh[0] b0-1 -> w0",     0, 1, 0x03,   0, 48.0f - 32.0f },
        /* qh[l] bits 2-3 -> weight l+32 */
        { "qh[0] b2-3 -> w32",    0, 1, 0x0C,  32, 48.0f - 32.0f },
        /* qh[l] bits 4-5 -> weight l+64 */
        { "qh[0] b4-5 -> w64",    0, 1, 0x30,  64, 48.0f - 32.0f },
        /* qh[l] bits 6-7 -> weight l+96 */
        { "qh[0] b6-7 -> w96",    0, 1, 0xC0,  96, 48.0f - 32.0f },
        /* qh[l] value 1 in bits 6-7 -> +16 */
        { "qh[5] b6=1 -> w101",   5, 1, 0x40, 101, 16.0f - 32.0f },
        /* second half: ql[64 + l] low nibble -> weight 128 + l */
        { "ql[64] low -> w128",  64, 0, 0x0F, 128, 15.0f - 32.0f },
        /* second half: ql[64+32+l] high nibble -> weight 128+96+l */
        { "ql[127] hi -> w255", 127, 0, 0xF0, 255, 15.0f - 32.0f },
        /* second half: qh[32 + l] bits 4-5 -> weight 128 + 64 + l */
        { "qh[33] b4-5-> w193",  33, 1, 0x30, 193, 48.0f - 32.0f },
        /* mid nibble value 5 -> 5 - 32 */
        { "ql[10] low=5 -> w10", 10, 0, 0x05,  10,  5.0f - 32.0f },
    };

    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        reset_block(&b);

        if (cases[c].is_qh) {
            b.qh[cases[c].byte_index] = cases[c].value;
        } else {
            b.ql[cases[c].byte_index] = cases[c].value;
        }

        gguf_dequantize_q6k_block(&b, out);

        for (unsigned int i = 0; i < GGUF_QK_K; i++) {
            float expected = (i == cases[c].changed)
                           ? cases[c].expected
                           : -32.0f;
            check_float(cases[c].name, i, out[i], expected);
        }
    }

    /* Test 7: unpack range. With every byte pattern 0xA5 all quants lie in -32..31. */
    reset_block(&b);
    memset(b.ql, 0xA5, sizeof(b.ql));
    memset(b.qh, 0x5A, sizeof(b.qh));
    gguf_q6k_unpack_block(&b, q);
    for (unsigned int i = 0; i < GGUF_QK_K; i++) {
        if (q[i] < -32 || q[i] > 31) {
            printf("FAIL unpack range index %u: %d\n", i, q[i]);
            failures++;
        }
    }

    /* Test 8: NULL arguments are rejected. */
    if (gguf_dequantize_q6k_block(NULL, out) != 0 ||
        gguf_dequantize_q6k_block(&b, NULL) != 0 ||
        gguf_q6k_unpack_block(NULL, q) != 0 ||
        gguf_q6k_unpack_block(&b, NULL) != 0) {
        printf("FAIL NULL argument handling\n");
        failures++;
    }

    if (sizeof(GGUFBlockQ6K) != 210) {
        failures++;
    }

    if (failures == 0) {
        printf("Q6_K synthetic tests: PASSED\n");
        return 0;
    }

    printf("Q6_K synthetic tests: FAILED (%d failures)\n", failures);
    return 1;
}
