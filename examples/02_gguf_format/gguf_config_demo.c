#include "checkpoint.h"

#include <stdio.h>

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    Config config = {0};
    char error[256] = {0};

    if (!gguf_read_config(
            argv[1],
            &config,
            error,
            sizeof(error))) {
        fprintf(stderr, "Error: %s\n", error);
        return 1;
    }

    printf("=== GGUF Model Configuration ===\n");
    printf("Embedding dimension : %d\n", config.dim);
    printf("Hidden dimension    : %d\n", config.hidden_dim);
    printf("Layers              : %d\n", config.n_layers);
    printf("Attention heads     : %d\n", config.n_heads);
    printf("KV heads            : %d\n", config.n_kv_heads);
    printf("Vocabulary size     : %d\n", config.vocab_size);
    printf("Context length      : %d\n", config.seq_len);

    printf("\n=== RoPE and RMSNorm ===\n");
    printf("RoPE dimension      : %d\n", config.rope_dim);
    printf("RoPE frequency base : %.2f\n", config.rope_freq_base);
    printf("RMSNorm epsilon     : %.10g\n", config.rms_norm_eps);

    return 0;
}