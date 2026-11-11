/*
 * gguf_q6k_real_demo.c - decode a real Q6_K tensor from a GGUF file.
 *
 * Default tensor: output.weight (TinyLlama vocabulary projection).
 * Usage: gguf_q6k_real_demo <model.gguf> [tensor_name]
 *
 * Writes three files for examples/03_quantization/compare_q6k_reference.py:
 *   build/q6k_c_first256.bin   first block, 256 float32
 *   build/q6k_c_last256.bin    last block, 256 float32
 *   build/q6k_c_row_stats.bin  for every row: sum, sum of squares (float64)
 *
 * The row statistics cover every block of the tensor, so the Python script
 * can check the whole tensor numerically, not only the first block.
 */
#include "checkpoint.h"
#include "gguf_quant.h"

#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static int write_floats(const char *path, const float *values, size_t count)
{
    FILE *file = fopen(path, "wb");

    if (file == NULL) {
        perror(path);
        return 0;
    }

    size_t written = fwrite(values, sizeof(float), count, file);
    int close_result = fclose(file);

    return written == count && close_result == 0;
}

int main(int argc, char **argv)
{
    if (argc != 2 && argc != 3) {
        fprintf(stderr, "Usage: %s <model.gguf> [tensor_name]\n", argv[0]);
        return 1;
    }

    const char *tensor_name = (argc == 3) ? argv[2] : "output.weight";

    int status = 1;
    int fd = -1;
    size_t file_size = 0;
    void *mapped_data = MAP_FAILED;
    struct stat st;

    GGUF_TensorInfo *tensors = NULL;
    size_t tensor_count = 0;
    GGUFLayout layout = {0};
    char error[256] = {0};

    const void *tensor_data = NULL;
    uint64_t tensor_size = 0;

    FILE *stats_file = NULL;

    /* 1. Map the whole GGUF file read-only. Nothing is copied yet. */
    fd = open(argv[1], O_RDONLY);
    if (fd == -1) {
        perror("open");
        goto cleanup;
    }

    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "Invalid GGUF file size.\n");
        goto cleanup;
    }

    file_size = (size_t)st.st_size;

    mapped_data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped_data == MAP_FAILED) {
        perror("mmap");
        goto cleanup;
    }

    /* 2. Read tensor descriptors and the data-section layout. */
    if (!gguf_read_tensors(argv[1], &tensors, &tensor_count,
                           error, sizeof(error))) {
        fprintf(stderr, "Tensor reading failed: %s\n", error);
        goto cleanup;
    }

    if (!gguf_read_layout(argv[1], &layout, error, sizeof(error))) {
        fprintf(stderr, "Layout reading failed: %s\n", error);
        goto cleanup;
    }

    /* 3. Find and describe the tensor. */
    const GGUF_TensorInfo *tensor =
        gguf_find_tensor(tensors, tensor_count, tensor_name);

    if (tensor == NULL) {
        fprintf(stderr, "Tensor '%s' not found.\n", tensor_name);
        goto cleanup;
    }

    uint64_t file_offset = 0;
    if (!gguf_tensor_file_offset(&layout, tensor, &file_offset)) {
        fprintf(stderr, "Cannot compute tensor file offset.\n");
        goto cleanup;
    }

    printf("=== Real Q6_K Tensor ===\n");
    printf("Name: %s\n", tensor->name);
    printf("GGML type: %u\n", tensor->type);
    printf("Dimensions (GGUF order, ne0 first):");
    for (uint32_t i = 0; i < tensor->n_dimensions; i++) {
        printf(" %" PRIu64, tensor->dimensions[i]);
    }
    printf("\n");
    printf("Data section offset: %" PRIu64 "\n", layout.data_offset);
    printf("Tensor relative offset: %" PRIu64 "\n", tensor->offset);
    printf("Tensor absolute file offset: %" PRIu64 "\n", file_offset);
    printf("Absolute offset %% alignment(%u): %" PRIu64 "\n",
           layout.alignment, file_offset % layout.alignment);

    if (tensor->type != GGUF_GGML_TYPE_Q6_K) {
        fprintf(stderr, "Tensor is type %u, not Q6_K (14).\n", tensor->type);
        goto cleanup;
    }

    if (tensor->n_dimensions < 1 ||
        tensor->dimensions[0] == 0 ||
        tensor->dimensions[0] % GGUF_QK_K != 0) {
        fprintf(stderr, "Row length is not a multiple of 256.\n");
        goto cleanup;
    }

    /* 4. Get a pointer into the mapping (bounds-checked). */
    if (!gguf_get_tensor_data(mapped_data, file_size, &layout, tensor,
                              &tensor_data, &tensor_size)) {
        fprintf(stderr, "Cannot retrieve tensor data.\n");
        goto cleanup;
    }

    /*
     * ne0 (dimensions[0]) is the contiguous row length.
     * All remaining dimensions multiply into the row count.
     */
    const uint64_t row_length = tensor->dimensions[0];
    uint64_t rows = 1;
    for (uint32_t i = 1; i < tensor->n_dimensions; i++) {
        rows *= tensor->dimensions[i];
    }

    const uint64_t blocks_per_row = row_length / GGUF_QK_K;
    const uint64_t block_count = tensor_size / GGUF_Q6_K_BLOCK_SIZE;

    printf("Storage size: %" PRIu64 " bytes\n", tensor_size);
    printf("Block size: %zu bytes\n", sizeof(GGUFBlockQ6K));
    printf("Rows: %" PRIu64 ", row length: %" PRIu64
           ", blocks per row: %" PRIu64 "\n",
           rows, row_length, blocks_per_row);
    printf("Total blocks: %" PRIu64 "\n", block_count);

    if (tensor_size % GGUF_Q6_K_BLOCK_SIZE != 0 ||
        block_count != rows * blocks_per_row) {
        fprintf(stderr, "Storage size does not match the tensor shape.\n");
        goto cleanup;
    }

    const unsigned char *raw = tensor_data;

    /* 5. First block. memcpy avoids any alignment assumptions. */
    GGUFBlockQ6K block;
    float output[GGUF_QK_K];

    memcpy(&block, raw, sizeof(block));

    printf("\n=== First Q6_K Block ===\n");
    printf("d: %.8f (raw 0x%04X)\n",
           (double)gguf_fp16_to_fp32(block.d), (unsigned)block.d);
    printf("scales:");
    for (unsigned int i = 0; i < GGUF_Q6_K_SCALE_COUNT; i++) {
        printf(" %d", block.scales[i]);
    }
    printf("\n");

    if (!gguf_dequantize_q6k_block(&block, output)) {
        fprintf(stderr, "Dequantization failed.\n");
        goto cleanup;
    }

    printf("\n=== First 16 Dequantized Weights ===\n");
    for (unsigned int i = 0; i < 16; i++) {
        printf("Weight[%u] = %.8f\n", i, (double)output[i]);
    }

    if (!write_floats("build/q6k_c_first256.bin", output, GGUF_QK_K)) {
        fprintf(stderr, "Failed to write build/q6k_c_first256.bin\n");
        goto cleanup;
    }

    /* 6. Last block. */
    memcpy(&block,
           raw + (size_t)(block_count - 1) * GGUF_Q6_K_BLOCK_SIZE,
           sizeof(block));

    if (!gguf_dequantize_q6k_block(&block, output) ||
        !write_floats("build/q6k_c_last256.bin", output, GGUF_QK_K)) {
        fprintf(stderr, "Failed to decode or write the last block.\n");
        goto cleanup;
    }

    printf("\nSaved first and last block for reference comparison.\n");

    /* 7. Full tensor: finiteness, min/max, and per-row statistics. */
    printf("\n=== Full Q6_K Tensor Validation ===\n");

    stats_file = fopen("build/q6k_c_row_stats.bin", "wb");
    if (stats_file == NULL) {
        perror("build/q6k_c_row_stats.bin");
        goto cleanup;
    }

    uint64_t total_weights = 0;
    uint64_t invalid_blocks = 0;
    float global_min = INFINITY;
    float global_max = -INFINITY;

    for (uint64_t r = 0; r < rows; r++) {

        double row_stats[2] = {0.0, 0.0}; /* sum, sum of squares */

        for (uint64_t b = 0; b < blocks_per_row; b++) {

            const uint64_t index = r * blocks_per_row + b;

            memcpy(&block,
                   raw + (size_t)index * GGUF_Q6_K_BLOCK_SIZE,
                   sizeof(block));

            if (!gguf_dequantize_q6k_block(&block, output)) {
                invalid_blocks++;
                continue;
            }

            int block_valid = 1;

            for (unsigned int j = 0; j < GGUF_QK_K; j++) {
                const float value = output[j];

                if (!isfinite(value)) {
                    block_valid = 0;
                    break;
                }

                if (value < global_min) global_min = value;
                if (value > global_max) global_max = value;

                row_stats[0] += (double)value;
                row_stats[1] += (double)value * (double)value;
            }

            if (block_valid) {
                total_weights += GGUF_QK_K;
            } else {
                invalid_blocks++;
            }
        }

        if (fwrite(row_stats, sizeof(double), 2, stats_file) != 2) {
            fprintf(stderr, "Failed to write row statistics.\n");
            goto cleanup;
        }
    }

    if (fclose(stats_file) != 0) {
        stats_file = NULL;
        fprintf(stderr, "Failed to close row statistics file.\n");
        goto cleanup;
    }
    stats_file = NULL;

    printf("Total blocks: %" PRIu64 "\n", block_count);
    printf("Expected weights: %" PRIu64 "\n", rows * row_length);
    printf("Total decoded weights: %" PRIu64 "\n", total_weights);
    printf("Invalid blocks: %" PRIu64 "\n", invalid_blocks);
    printf("Global minimum: %.8f\n", (double)global_min);
    printf("Global maximum: %.8f\n", (double)global_max);
    printf("Saved %" PRIu64 " row statistics to build/q6k_c_row_stats.bin\n",
           rows);

    if (invalid_blocks == 0 && total_weights == rows * row_length) {
        printf("Full Q6_K tensor validation: PASSED\n");
        status = 0;
    } else {
        printf("Full Q6_K tensor validation: FAILED\n");
    }

cleanup:
    if (stats_file != NULL) {
        fclose(stats_file);
    }

    if (tensors != NULL) {
        for (size_t i = 0; i < tensor_count; i++) {
            free(tensors[i].name);
        }
        free(tensors);
    }

    if (mapped_data != MAP_FAILED) {
        munmap(mapped_data, file_size);
    }

    if (fd != -1) {
        close(fd);
    }

    return status;
}
