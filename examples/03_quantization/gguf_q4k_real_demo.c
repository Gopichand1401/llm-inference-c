
#include "checkpoint.h"
#include "gguf_quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <math.h>
#include <string.h>

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

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

    /* 1. Open GGUF file. */
    fd = open(argv[1], O_RDONLY);

    if (fd == -1) {
        perror("open");
        goto cleanup;
    }

    /* 2. Get file size. */
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        fprintf(stderr, "Invalid GGUF file size.\n");
        goto cleanup;
    }

    file_size = (size_t)st.st_size;

    /* 3. Memory map the file. */
    mapped_data = mmap(
        NULL,
        file_size,
        PROT_READ,
        MAP_PRIVATE,
        fd,
        0
    );

    if (mapped_data == MAP_FAILED) {
        perror("mmap");
        goto cleanup;
    }

    /* 4. Read tensor descriptors. */
    if (!gguf_read_tensors(
            argv[1],
            &tensors,
            &tensor_count,
            error,
            sizeof(error))) {
        fprintf(stderr, "Tensor reading failed: %s\n", error);
        goto cleanup;
    }

    /* 5. Read GGUF layout. */
    if (!gguf_read_layout(
            argv[1],
            &layout,
            error,
            sizeof(error))) {
        fprintf(stderr, "Layout reading failed: %s\n", error);
        goto cleanup;
    }

    /* 6. Find embedding tensor. */
    const GGUF_TensorInfo *embedding =
        gguf_find_tensor(
            tensors,
            tensor_count,
            "token_embd.weight"
        );

    if (embedding == NULL) {
        fprintf(stderr, "Embedding tensor not found.\n");
        goto cleanup;
    }

    if (embedding->type != 12) {
        fprintf(stderr, "Embedding tensor is not Q4_K.\n");
        goto cleanup;
    }

    /* 7. Retrieve actual tensor data. */
    if (!gguf_get_tensor_data(
            mapped_data,
            file_size,
            &layout,
            embedding,
            &tensor_data,
            &tensor_size)) {
        fprintf(stderr, "Cannot retrieve embedding data.\n");
        goto cleanup;
    }

    printf("=== Real TinyLlama Q4_K Tensor ===\n");
    printf("Name: %s\n", embedding->name);
    printf("GGML type: %u\n", embedding->type);
    printf("Dimensions: ");

    for (uint32_t i = 0; i < embedding->n_dimensions; i++) {
        printf("%" PRIu64 " ", embedding->dimensions[i]);
    }

    printf("\nStorage size: %" PRIu64 " bytes\n", tensor_size);

    /* 8. Validate tensor storage. */
    if (tensor_size < sizeof(GGUFBlockQ4K) ||
        tensor_size % GGUF_Q4_K_BLOCK_SIZE != 0) {
        fprintf(stderr, "Invalid Q4_K tensor storage size.\n");
        goto cleanup;
    }

    /* 9. Access the first Q4_K block safely. */
    GGUFBlockQ4K block;

    memcpy(&block, tensor_data, sizeof(block));

    printf("\n=== First Q4_K Block ===\n");
    printf("Block size: %zu bytes\n", sizeof(block));

    printf("d: %.8f\n",
           gguf_fp16_to_fp32(block.d));

    printf("dmin: %.8f\n",
           gguf_fp16_to_fp32(block.dmin));

    /* 10. Dequantize first 256 weights. */
    float output[GGUF_QK_K];

    if (!gguf_dequantize_q4k_block(&block, output)) {
        fprintf(stderr, "Dequantization failed.\n");
        goto cleanup;
    }

    /* 11. Save C output for official Python comparison. */
    FILE *reference_file =
        fopen("build/q4k_c_first256.bin", "wb");

    if (reference_file == NULL) {
        perror("Cannot create reference file");
        goto cleanup;
    }

    size_t written = fwrite(
        output,
        sizeof(float),
        GGUF_QK_K,
        reference_file
    );

    int close_result = fclose(reference_file);

    if (written != GGUF_QK_K || close_result != 0) {
        fprintf(stderr, "Failed to write C reference weights.\n");
        goto cleanup;
    }

    printf("\nSaved 256 C weights for reference comparison.\n");

    /* 12. Check reconstructed values. */
    float min_value = output[0];
    float max_value = output[0];

    for (unsigned int i = 0; i < GGUF_QK_K; i++) {

        if (!isfinite(output[i])) {
            fprintf(stderr, "Invalid FP32 value at index %u.\n", i);
            goto cleanup;
        }

        if (output[i] < min_value)
            min_value = output[i];

        if (output[i] > max_value)
            max_value = output[i];
    }

    printf("\n=== First 16 Dequantized Weights ===\n");

    for (unsigned int i = 0; i < 16; i++) {
        printf("Weight[%u] = %.8f\n", i, output[i]);
    }

    printf("\nMinimum weight: %.8f\n", min_value);
    printf("Maximum weight: %.8f\n", max_value);

    printf("\nReal Q4_K dequantization: SUCCESS\n");

    /* 13. Validate the complete Q4_K tensor. */
    printf("\n=== Full Q4_K Tensor Validation ===\n");

    uint64_t block_count =
        tensor_size / GGUF_Q4_K_BLOCK_SIZE;

    uint64_t total_weights = 0;

    float global_min = INFINITY;
    float global_max = -INFINITY;

    int invalid_blocks = 0;

    float block_output[GGUF_QK_K];

    const unsigned char *raw_data = tensor_data;

    for (uint64_t i = 0; i < block_count; i++) {

        GGUFBlockQ4K current_block;

        memcpy(
            &current_block,
            raw_data + i * GGUF_Q4_K_BLOCK_SIZE,
            sizeof(current_block)
        );

        if (!gguf_dequantize_q4k_block(
                &current_block,
                block_output)) {
            invalid_blocks++;
            continue;
        }

        int block_valid = 1;

        for (unsigned int j = 0; j < GGUF_QK_K; j++) {

            float value = block_output[j];

            if (!isfinite(value)) {
                invalid_blocks++;
                block_valid = 0;
                break;
            }

            if (value < global_min)
                global_min = value;

            if (value > global_max)
                global_max = value;
        }

        if (block_valid) {
            total_weights += GGUF_QK_K;
        }
    }

    printf("Total blocks: %" PRIu64 "\n", block_count);
    printf("Total decoded weights: %" PRIu64 "\n", total_weights);
    printf("Invalid blocks: %d\n", invalid_blocks);
    printf("Global minimum: %.8f\n", global_min);
    printf("Global maximum: %.8f\n", global_max);

    if (invalid_blocks == 0 &&
        total_weights == block_count * GGUF_QK_K) {

        printf("Full Q4_K tensor validation: PASSED\n");
        status = 0;

    } else {
        printf("Full Q4_K tensor validation: FAILED\n");
        goto cleanup;
    }

cleanup:

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
