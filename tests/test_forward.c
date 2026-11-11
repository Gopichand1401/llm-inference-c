
#include "transformer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <checkpoint.bin>\n", argv[0]);
        return EXIT_FAILURE;
    }

    Transformer model = {0};
    model.fd = -1;

    printf("Loading checkpoint: %s\n", argv[1]);

    char error[256];

    if (!build_transformer(&model, argv[1], error, sizeof(error))) {
        fprintf(stderr, "FAIL: cannot load %s: %s\n", argv[1], error);
        return EXIT_FAILURE;
    }

    printf("Model loaded successfully.\n");
    printf("Dimension: %d\n", model.config.dim);
    printf("Vocabulary size: %d\n", model.config.vocab_size);

    /* Test one token at position zero. */
    int token = 1;
    int pos = 0;

    printf("\nRunning forward(token=%d, pos=%d)...\n", token, pos);

    float *logits = forward(&model, token, pos);

    if (logits == NULL) {
        fprintf(stderr, "FAIL: forward() returned NULL.\n");
        free_transformer(&model);
        return EXIT_FAILURE;
    }

    int vocab_size = model.config.vocab_size;
    int non_finite = 0;
    int best_token = 0;

    for (int i = 0; i < vocab_size; i++) {
        if (!isfinite(logits[i])) {
            non_finite++;
        }

        if (i > 0 && logits[i] > logits[best_token]) {
            best_token = i;
        }
    }

    printf("\nForward-pass results:\n");
    printf("Logits checked: %d\n", vocab_size);
    printf("Non-finite logits: %d\n", non_finite);
    printf("Highest-logit token ID: %d\n", best_token);
    printf("Highest logit value: %.6f\n", logits[best_token]);

    if (non_finite == 0) {
        printf("\nPASS: all logits are finite.\n");
    } else {
        printf("\nFAIL: some logits are NaN or infinite.\n");
    }

    free_transformer(&model);
    printf("Model memory released.\n");

    return non_finite == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
