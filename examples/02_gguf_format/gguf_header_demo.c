
#include "checkpoint.h"

#include <stdio.h>

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s model.gguf\n", argv[0]);
        return 1;
    }

    GGUFHeader header;
    char error[256];

    if (!gguf_read_header(
            argv[1],
            &header,
            error,
            sizeof(error))) {
        fprintf(stderr, "GGUF error: %s\n", error);
        return 1;
    }

    if (!gguf_read_metadata(
            argv[1], error, sizeof(error))) {
        fprintf(stderr, "Metadata error: %s\n", error);
        return 1;
    }

    printf("GGUF Header Successfully Read!\n");
    printf("Version: %u\n", header.version);
    printf("Tensor count: %llu\n",
           (unsigned long long)header.tensor_count);
    printf("Metadata count: %llu\n",
           (unsigned long long)header.metadata_count);
    
    Config config;

    if (!gguf_read_config(argv[1], &config, error, sizeof(error))) {
        fprintf(stderr, "Config error: %s\n", error);
        return 1;
    }

    printf("\n=== Extracted Model Configuration ===\n");
    printf("Dimension: %d\n", config.dim);
    printf("Hidden dimension: %d\n", config.hidden_dim);
    printf("Layers: %d\n", config.n_layers);
    printf("Attention heads: %d\n", config.n_heads);
    printf("KV heads: %d\n", config.n_kv_heads);
    printf("Vocabulary: %d\n", config.vocab_size);
    printf("Context length: %d\n", config.seq_len);
    return 0;
}
