
#include "transformer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <checkpoint.bin>\n", argv[0]);
        return EXIT_FAILURE;
    }

    Transformer model = {0};
    model.fd = -1;

    char error[256];

    if (!build_transformer(&model, argv[1], error, sizeof(error))) {
        fprintf(stderr, "FAIL: cannot load %s: %s\n", argv[1], error);
        return EXIT_FAILURE;
    }

    float *logits = forward(&model, 1, 0);

    if (logits == NULL) {
        fprintf(stderr, "FAIL: forward() returned NULL.\n");
        free_transformer(&model);
        return EXIT_FAILURE;
    }

    int vocab_size = model.config.vocab_size;

    for (int i = 0; i < vocab_size; i++) {
        if (!isfinite(logits[i])) {
            fprintf(stderr, "FAIL: non-finite logit at index %d\n", i);
            free_transformer(&model);
            return EXIT_FAILURE;
        }
    }

    FILE *file = fopen("custom_logits.bin", "wb");
    if (file == NULL) {
        perror("Could not create custom_logits.bin");
        free_transformer(&model);
        return EXIT_FAILURE;
    }

    size_t count = fwrite(logits, sizeof(float),
                          (size_t)vocab_size, file);

    fclose(file);
    free_transformer(&model);

    if (count != (size_t)vocab_size) {
        fprintf(stderr, "FAIL: incomplete logits output.\n");
        return EXIT_FAILURE;
    }

    printf("Custom logits saved: %zu values\n", count);
    return EXIT_SUCCESS;
}
