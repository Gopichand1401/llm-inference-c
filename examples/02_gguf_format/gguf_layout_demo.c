#include "checkpoint.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <checkpoint.gguf>\n", argv[0]);
        return 1;
    }

    GGUFLayout layout;
    GGUF_TensorInfo *tensors = NULL;
    size_t tensor_count = 0;
    char error[256] = {0};

    if (!gguf_read_layout(argv[1], &layout, error, sizeof(error))) {
        fprintf(stderr, "Layout error: %s\n", error);
        return 1;
    }

    if (!gguf_read_tensors(
            argv[1], &tensors, &tensor_count,
            error, sizeof(error))) {
        fprintf(stderr, "Tensor error: %s\n", error);
        return 1;
    }

    printf("Data section starts at: %llu\n",
           (unsigned long long)layout.data_offset);

    printf("\n=== Actual Tensor File Positions ===\n");

    for (size_t i = 0; i < tensor_count; i++) {
        uint64_t file_offset;

        if (!gguf_tensor_file_offset(
                &layout, &tensors[i], &file_offset)) {
            fprintf(stderr, "Offset overflow: %s\n", tensors[i].name);
            continue;
        }

        if (i < 10) {
            printf("%-30s Relative: %12llu  Actual: %12llu\n",
                   tensors[i].name,
                   (unsigned long long)tensors[i].offset,
                   (unsigned long long)file_offset);
        }
    }

    for (size_t i = 0; i < tensor_count; i++) {
        free(tensors[i].name);
    }

    free(tensors);

    return 0;
}