#include "checkpoint.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <checkpoint.gguf>\n", argv[0]);
        return 1;
    }

    GGUFLayout layout;
    GGUF_TensorInfo *tensors = NULL;
    size_t count = 0;
    char error[256] = {0};

    struct stat st;

    if (stat(argv[1], &st) != 0) {
        perror("stat");
        return 1;
    }

    uint64_t file_size = (uint64_t)st.st_size;

    if (!gguf_read_layout(argv[1], &layout, error, sizeof(error)) ||
        !gguf_read_tensors(argv[1], &tensors, &count,
                           error, sizeof(error))) {
        fprintf(stderr, "Error: %s\n", error);
        return 1;
    }

    size_t valid = 0;
    size_t invalid = 0;

    printf("File size: %llu bytes\n", (unsigned long long)file_size);
    printf("Tensor count: %zu\n\n", count);

    for (size_t i = 0; i < count; i++) {
        uint64_t position;
        uint64_t size;

        if (!gguf_tensor_file_offset(&layout, &tensors[i], &position) ||
            !gguf_tensor_storage_size(&tensors[i], &size) ||
            position > file_size ||
            size > file_size - position) {

            printf("INVALID: %s\n", tensors[i].name);
            invalid++;
            continue;
        }

        valid++;

        if (i < 10) {
            printf("%-30s Size: %llu bytes\n",
                   tensors[i].name,
                   (unsigned long long)size);
        }
    }

    printf("\n=== Validation Summary ===\n");
    printf("Valid tensors: %zu\n", valid);
    printf("Invalid tensors: %zu\n", invalid);

    for (size_t i = 0; i < count; i++) {
        free(tensors[i].name);
    }

    free(tensors);

    return invalid == 0 ? 0 : 1;
}