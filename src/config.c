
#include "transformer.h"

#include <stdio.h>

int config_validate(
    const Config *config,
    char *error,
    size_t error_capacity
)
{
    if (config == NULL) {
        goto invalid;
    }

    if (config->dim <= 0 ||
        config->hidden_dim <= 0 ||
        config->n_layers <= 0 ||
        config->n_heads <= 0 ||
        config->n_kv_heads <= 0 ||
        config->vocab_size <= 0 ||
        config->seq_len <= 0) {
        goto invalid;
    }

    if (config->n_kv_heads > config->n_heads ||
        config->n_heads % config->n_kv_heads != 0) {
        goto invalid;
    }

    if (config->dim % config->n_heads != 0) {
        goto invalid;
    }

    int head_size = config->dim / config->n_heads;

    if (head_size % 2 != 0) {
        goto invalid;
    }

    if (error != NULL && error_capacity > 0) {
        error[0] = '\0';
    }

    return 1;

invalid:
    if (error != NULL && error_capacity > 0) {
        snprintf(error, error_capacity,
                 "Invalid LLaMA model configuration.");
    }

    return 0;
}
    