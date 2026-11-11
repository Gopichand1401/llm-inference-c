#include "checkpoint.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <checkpoint.gguf>\n", argv[0]);
        return 1;
    }

    GGUF_TensorInfo *tensors = NULL;
    size_t tensor_count = 0;

    char error[256] = {0};

    if (!gguf_read_tensors(
            argv[1],
            &tensors,
            &tensor_count,
            error,
            sizeof(error))) {

        fprintf(stderr, "Tensor reading failed: %s\n", error);
        return 1;
    }

    printf("\nTotal tensors: %zu\n", tensor_count);

    printf("\n=== First 10 Tensors ===\n");

    for (size_t i = 0;
         i < tensor_count;
         i++) {

        printf("\nTensor: %s\n", tensors[i].name);

        printf("Dimensions: ");

        for (uint32_t d = 0;
             d < tensors[i].n_dimensions;
             d++) {

            printf("%llu",
                   (unsigned long long)tensors[i].dimensions[d]);

            if (d + 1 < tensors[i].n_dimensions)
                printf(" x ");
        }

        printf("\nType ID: %u\n", tensors[i].type);
        printf("Offset: %llu\n",
               (unsigned long long)tensors[i].offset);
    }

    for (size_t i = 0; i < tensor_count; i++) {
        free(tensors[i].name);
    }

    free(tensors);

    return 0;
}