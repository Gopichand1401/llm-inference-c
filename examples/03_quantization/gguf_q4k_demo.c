#include "gguf_quant.h"

#include <stdio.h>

int main(void)
{
    printf("Q4_K block size: %zu bytes\n",
           sizeof(GGUFBlockQ4K));

    printf("\n=== FP16 to FP32 Tests ===\n");

    uint16_t test_values[] = {
        0x3C00,  /* 1.0 */
        0x4000,  /* 2.0 */
        0xC000,  /* -2.0 */
        0x3800,  /* 0.5 */
        0x0000   /* 0.0 */
    };

    for (size_t i = 0;
         i < sizeof(test_values) / sizeof(test_values[0]);
         i++) {

        float result = gguf_fp16_to_fp32(test_values[i]);

        printf("FP16: 0x%04X -> FP32: %.9g\n",
               test_values[i], result);
    };

    printf("\n=== Q4_K Scale/Min Decoding ===\n");

    uint8_t packed_scales[12] = {
        0xC1, 0x82, 0x43, 0x04,
        0x85, 0x46, 0xA7, 0xE8,
        0xB9, 0x6A, 0xDB, 0xFC
    };

    for (unsigned int i = 0; i < 8; i++) {
        uint8_t scale;
        uint8_t min;

        if (!gguf_q4k_get_scale_min(
                packed_scales, i, &scale, &min)) {
            return 1;
        }

        printf("Group %u: Scale = %u, Min = %u\n",
            i, scale, min);
    }

    printf("\n=== Q4_K Dequantization Test ===\n");

    GGUFBlockQ4K block = {0};

    /* d = 1.0 */
    block.d = 0x3C00;

    /* dmin = 0.0 */
    block.dmin = 0x0000;

    /* First group scale = 1, second group scale = 2. */
    block.scales[0] = 1;
    block.scales[1] = 2;

    /* Every byte contains low = 1 and high = 2. */
    for (unsigned int i = 0; i < 32; i++) {
        block.qs[i] = 0x21;
    }

    float output[GGUF_QK_K];

    if (!gguf_dequantize_q4k_block(&block, output)) {
        printf("Dequantization FAILED\n");
        return 1;
    }
    
    /*
     * Note: this synthetic test deliberately does NOT write
     * build/q4k_c_first256.bin. That file belongs to gguf_q4k_real_demo and
     * is compared against the real TinyLlama block by
     * compare_q4k_reference.py; overwriting it with synthetic data would make
     * that comparison meaningless.
     */

    printf("Weight 0   : %.2f\n", output[0]);
    printf("Weight 32  : %.2f\n", output[32]);
    printf("Weight 64  : %.2f\n", output[64]);
    printf("Weight 128 : %.2f\n", output[128]);

    printf("Q4_K dequantization: SUCCESS\n");
    
    return 0;
}