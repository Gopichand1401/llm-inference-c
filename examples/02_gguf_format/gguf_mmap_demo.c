
#include "checkpoint.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    const char *path = argv[1];
    int fd = -1;
    void *mapped_data = MAP_FAILED;
    struct stat st;

    GGUF_TensorInfo *tensors = NULL;
    size_t tensor_count = 0;
    GGUFLayout layout = {0};
    char error[256] = {0};

    /* 1. Open the GGUF checkpoint. */
    fd = open(path, O_RDONLY);

    if (fd == -1) {
        perror("Cannot open GGUF file");
        return 1;
    }

    /* 2. Get the actual file size. */
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        perror("Cannot inspect GGUF file");
        close(fd);
        return 1;
    }

    size_t file_size = (size_t)st.st_size;

    printf("GGUF file size: %zu bytes\n", file_size);

    /* 3. Map the complete file into virtual memory. */
    mapped_data = mmap(
        NULL,
        file_size,
        PROT_READ,
        MAP_PRIVATE,
        fd,
        0
    );

    if (mapped_data == MAP_FAILED) {
        perror("mmap failed");
        close(fd);
        return 1;
    }

    printf("Memory mapping: SUCCESS\n");
    printf("Mapped address: %p\n", mapped_data);

    /* 4. Read GGUF tensor descriptors. */
    if (!gguf_read_tensors(
            path,
            &tensors,
            &tensor_count,
            error,
            sizeof(error))) {
        fprintf(stderr, "Tensor reading failed: %s\n", error);
        goto cleanup;
    }

    /* 5. Determine where tensor data begins. */
    if (!gguf_read_layout(
            path,
            &layout,
            error,
            sizeof(error))) {
        fprintf(stderr, "Layout reading failed: %s\n", error);
        goto cleanup;
    }

    printf("\nTensor count: %zu\n", tensor_count);
    printf("Alignment: %u bytes\n", layout.alignment);
    printf("Tensor data offset: %" PRIu64 "\n",
           layout.data_offset);

    /* 6. Locate and inspect the first tensor. */
    if (tensor_count == 0) {
        fprintf(stderr, "No tensors found.\n");
        goto cleanup;
    }

    GGUF_TensorInfo *tensor = &tensors[0];

    uint64_t file_offset;
    uint64_t storage_size;

    if (!gguf_tensor_file_offset(
            &layout, tensor, &file_offset) ||
        !gguf_tensor_storage_size(tensor, &storage_size)) {
        fprintf(stderr, "Cannot calculate tensor location.\n");
        goto cleanup;
    }

    /* Check that the tensor lies completely inside the mapped file. */
    if (file_offset > file_size ||
        storage_size > file_size - file_offset) {
        fprintf(stderr, "Tensor is outside the mapped file.\n");
        goto cleanup;
    }

    /* 7. Get the tensor's address using our reusable function. */
    const void *tensor_data = NULL;
    uint64_t tensor_size = 0;

    if (!gguf_get_tensor_data(
            mapped_data,
            file_size,
            &layout,
            tensor,
            &tensor_data,
            &tensor_size)) {
        fprintf(stderr, "Failed to access tensor data.\n");
        goto cleanup;
    }

    printf("\n=== First Tensor ===\n");
    printf("Name: %s\n", tensor->name);
    printf("GGML type: %u\n", tensor->type);
    printf("File offset: %" PRIu64 "\n", file_offset);
    printf("Storage size: %" PRIu64 " bytes\n", tensor_size);
    printf("Mapped tensor address: %p\n", tensor_data);

    printf("First 16 bytes: ");
    const unsigned char *bytes = tensor_data;

    for (int i = 0; i < 16 && (uint64_t)i < tensor_size; i++) {
        printf("%02X ", bytes[i]);
    }
    printf("\n");

    const char *target_name = "token_embd.weight";

    const GGUF_TensorInfo *target =
        gguf_find_tensor(tensors, tensor_count, target_name);

    if (target == NULL) {
        printf("Tensor not found: %s\n", target_name);
    } else {
        printf("\n=== Tensor Lookup ===\n");
        printf("Name: %s\n", target->name);
        printf("GGML type: %u\n", target->type);
        printf("Dimensions: ");

        for (uint32_t i = 0; i < target->n_dimensions; i++) {
            printf("%" PRIu64 " ", target->dimensions[i]);
        }

        printf("\n");
    }

    

    const void *embedding_data = NULL;
    uint64_t embedding_size = 0;

    if (target != NULL &&
        gguf_get_tensor_data(
            mapped_data,
            file_size,
            &layout,
            target,
            &embedding_data,
            &embedding_size)) {

        printf("\n=== Embedding Tensor Data ===\n");
        printf("Tensor address: %p\n", embedding_data);
        printf("Storage size: %" PRIu64 " bytes\n", embedding_size);

        printf("First 16 bytes: ");

        const unsigned char *bytes = embedding_data;

        for (int i = 0; i < 16; i++) {
            printf("%02X ", bytes[i]);
        }

        printf("\n");
    } else {
        printf("Could not retrieve embedding tensor data.\n");
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

    return 0;
}
