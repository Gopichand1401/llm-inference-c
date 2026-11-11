/*
 * gguf_quant_check.c - decode raw quantized blocks with the C decoders.
 *
 * Usage: gguf_quant_check <ggml type id> <blocks.raw> <output.f32>
 *
 * Reads consecutive blocks of the given type from blocks.raw, decodes each
 * with gguf_dequantize_block(), and writes the float32 results to
 * output.f32. Used by examples/03_quantization/compare_quant_random.py, which generates
 * random blocks and compares the output with gguf-py bit for bit.
 */
#include "gguf_quant.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <type id> <blocks.raw> <output.f32>\n",
                argv[0]);
        return 1;
    }

    const uint32_t type = (uint32_t)strtoul(argv[1], NULL, 10);
    size_t block_weights = 0, block_bytes = 0;

    if (!gguf_quant_block_info(type, &block_weights, &block_bytes)) {
        fprintf(stderr, "Unsupported type %u\n", type);
        return 1;
    }

    FILE *in = fopen(argv[2], "rb");
    FILE *out = fopen(argv[3], "wb");
    unsigned char *raw = malloc(block_bytes);
    float *values = malloc(block_weights * sizeof(float));
    size_t blocks = 0;
    int status = 1;

    if (in == NULL || out == NULL || raw == NULL || values == NULL) {
        fprintf(stderr, "Cannot open files or allocate buffers.\n");
        goto cleanup;
    }

    while (fread(raw, 1, block_bytes, in) == block_bytes) {
        if (!gguf_dequantize_block(type, raw, values) ||
            fwrite(values, sizeof(float), block_weights, out) != block_weights) {
            fprintf(stderr, "Decode or write failed at block %zu\n", blocks);
            goto cleanup;
        }
        blocks++;
    }

    printf("type %u: decoded %zu blocks (%zu weights each)\n",
           type, blocks, block_weights);
    status = 0;

cleanup:
    if (in != NULL) fclose(in);
    if (out != NULL) fclose(out);
    free(raw);
    free(values);
    return status;
}
