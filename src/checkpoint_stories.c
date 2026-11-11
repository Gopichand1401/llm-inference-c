#include "checkpoint.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

/* Header layout stored in legacy Stories checkpoint files. */
typedef struct {
    int32_t dim;
    int32_t hidden_dim;
    int32_t n_layers;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t vocab_size;
    int32_t seq_len;
} StoriesCheckpointHeader;

/* Ensure compatibility with the existing Stories binary layout. */
_Static_assert(sizeof(int) == sizeof(int32_t),
               "Stories format requires 32-bit int.");


/* Assign pointers to the Stories checkpoint weight tensors. */
static void memory_map_weights(
    TransformerWeights *w,
    Config *p,
    float *ptr,
    int shared_weights
)
{
    size_t dim = (size_t)p->dim;
    size_t hidden = (size_t)p->hidden_dim;
    size_t layers = (size_t)p->n_layers;
    size_t vocab = (size_t)p->vocab_size;
    size_t head_size = dim / (size_t)p->n_heads;
    size_t kv_dim = dim * (size_t)p->n_kv_heads
                    / (size_t)p->n_heads;

    w->token_embedding_table = ptr;
    ptr += vocab * dim;

    w->rms_att_weight = ptr;
    ptr += layers * dim;

    w->wq = ptr;
    ptr += layers * dim * dim;

    w->wk = ptr;
    ptr += layers * dim * kv_dim;

    w->wv = ptr;
    ptr += layers * dim * kv_dim;

    w->wo = ptr;
    ptr += layers * dim * dim;

    w->rms_ffn_weight = ptr;
    ptr += layers * dim;

    w->w1 = ptr;
    ptr += layers * dim * hidden;

    w->w2 = ptr;
    ptr += layers * hidden * dim;

    w->w3 = ptr;
    ptr += layers * dim * hidden;

    w->rms_final_weight = ptr;
    ptr += dim;

    /* Skip legacy RoPE frequency data. */
    ptr += (size_t)p->seq_len * (head_size / 2);
    ptr += (size_t)p->seq_len * (head_size / 2);

    w->wcls = shared_weights
        ? w->token_embedding_table
        : ptr;
}

/* Set an error message safely. */
static void set_error(
    char *error,
    size_t error_capacity,
    const char *message
)
{
    if (error != NULL && error_capacity > 0) {
        snprintf(error, error_capacity, "%s", message);
    }
}

/* Safely multiply two size values. */
static int checked_mul_size(
    size_t a,
    size_t b,
    size_t *result
)
{
    if (result == NULL || (b != 0 && a > SIZE_MAX / b)) {
        return 0;
    }

    *result = a * b;
    return 1;
}

/* Safely add a tensor's element count. */
static int add_weight_count(
    size_t *total,
    size_t a,
    size_t b
)
{
    size_t count;

    if (!checked_mul_size(a, b, &count) ||
        count > SIZE_MAX - *total) {
        return 0;
    }

    *total += count;
    return 1;
}

/* Validate the expected Stories checkpoint size. */
static int validate_checkpoint_size(
    const Config *p,
    size_t file_size,
    int shared_weights
)
{
    size_t total = 0;

    size_t dim = (size_t)p->dim;
    size_t hidden = (size_t)p->hidden_dim;
    size_t layers = (size_t)p->n_layers;
    size_t vocab = (size_t)p->vocab_size;
    size_t seq_len = (size_t)p->seq_len;

    size_t head_size = dim / (size_t)p->n_heads;
    size_t kv_dim;
    size_t layers_dim;
    size_t layers_hidden;

    if (!checked_mul_size(
            head_size,
            (size_t)p->n_kv_heads,
            &kv_dim) ||
        !checked_mul_size(
            layers,
            dim,
            &layers_dim) ||
        !checked_mul_size(
            layers,
            hidden,
            &layers_hidden)) {
        return 0;
    }

    if (!add_weight_count(&total, vocab, dim) ||
        !add_weight_count(&total, layers, dim) ||
        !add_weight_count(&total, layers_dim, dim) ||
        !add_weight_count(&total, layers_dim, kv_dim) ||
        !add_weight_count(&total, layers_dim, kv_dim) ||
        !add_weight_count(&total, layers_dim, dim) ||
        !add_weight_count(&total, layers, dim) ||
        !add_weight_count(&total, layers_dim, hidden) ||
        !add_weight_count(&total, layers_hidden, dim) ||
        !add_weight_count(&total, layers_dim, hidden) ||
        !add_weight_count(&total, 1, dim)) {
        return 0;
    }

    if (!add_weight_count(&total, seq_len, head_size / 2) ||
        !add_weight_count(&total, seq_len, head_size / 2)) {
        return 0;
    }

    if (!shared_weights &&
        !add_weight_count(&total, vocab, dim)) {
        return 0;
    }

    if (total > (SIZE_MAX - sizeof(StoriesCheckpointHeader))
            / sizeof(float)) {
    return 0;
    }

    size_t expected =
        sizeof(StoriesCheckpointHeader) + total * sizeof(float);

    return expected <= file_size;
}


/* Load and memory-map a Stories-format checkpoint. */
int stories_checkpoint_load(
    const char *checkpoint,
    Config *config,
    TransformerWeights *weights,
    int *fd,
    float **data,
    ssize_t *file_size,
    char *error,
    size_t error_capacity
)
{
    FILE *file;
    struct stat st;
    int shared_weights;
    StoriesCheckpointHeader header;

    if (checkpoint == NULL || config == NULL ||
        weights == NULL || fd == NULL ||
        data == NULL || file_size == NULL) {
        set_error(error, error_capacity,
                  "Invalid checkpoint loader arguments.");
        return 0;
    }

    *fd = -1;
    *data = NULL;
    *file_size = 0;

    file = fopen(checkpoint, "rb");

    if (file == NULL) {
        set_error(error, error_capacity,
                  "Cannot open checkpoint.");
        return 0;
    }

    if (fread(&header, sizeof(header), 1, file) != 1) {
    fclose(file);
    set_error(error, error_capacity,
              "Cannot read Stories checkpoint header.");
    return 0;
    }

    /* Convert the Stories header into our runtime configuration. */
    config->dim = header.dim;
    config->hidden_dim = header.hidden_dim;
    config->n_layers = header.n_layers;
    config->n_heads = header.n_heads;
    config->n_kv_heads = header.n_kv_heads;
    config->vocab_size = header.vocab_size;
    config->seq_len = header.seq_len;

    /*
     * RoPE/RMSNorm defaults. rope_dim depends on n_heads, so it is set only
     * after config_validate() has confirmed n_heads > 0 (a corrupt header
     * with n_heads == 0 would otherwise divide by zero here).
     */
    config->rope_dim = 0;
    config->rope_freq_base = 10000.0f;
    config->rms_norm_eps = 1e-5f;
    config->rope_scale = 1.0f;       /* llama2.c files never scale RoPE */

    if (fclose(file) != 0) {
        set_error(error, error_capacity,
                  "Cannot close checkpoint file.");
        return 0;
    }

    shared_weights = config->vocab_size > 0;

    if (config->vocab_size == 0 ||
        config->vocab_size == INT_MIN) {
        set_error(error, error_capacity,
                  "Invalid vocabulary size.");
        return 0;
    }

    if (config->vocab_size < 0) {
        config->vocab_size = -config->vocab_size;
    }

    if (!config_validate(config, error, error_capacity)) {
        return 0;
    }

    config->rope_dim = config->dim / config->n_heads;

    *fd = open(checkpoint, O_RDONLY);

    if (*fd == -1) {
        set_error(error, error_capacity,
                  "Cannot open checkpoint for mapping.");
        return 0;
    }

    if (fstat(*fd, &st) != 0 ||
        st.st_size < (off_t)sizeof(StoriesCheckpointHeader) ||
        st.st_size > (off_t)SSIZE_MAX) {
        close(*fd);
        *fd = -1;

        set_error(error, error_capacity,
                  "Cannot inspect checkpoint file size.");
        return 0;
    }

    *file_size = (ssize_t)st.st_size;

    if (!validate_checkpoint_size(
            config, (size_t)*file_size, shared_weights)) {
        close(*fd);
        *fd = -1;
        *file_size = 0;

        set_error(error, error_capacity,
                  "Checkpoint is too small or has an invalid weight layout.");
        return 0;
    }

    *data = mmap(
        NULL,
        (size_t)*file_size,
        PROT_READ,
        MAP_PRIVATE,
        *fd,
        0
    );

    if (*data == MAP_FAILED) {
        *data = NULL;

        close(*fd);
        *fd = -1;
        *file_size = 0;

        set_error(error, error_capacity,
                  "Cannot memory-map checkpoint.");
        return 0;
    }

    float *weights_ptr =
        (float *)((char *)*data + sizeof(StoriesCheckpointHeader));

    memory_map_weights(
        weights,
        config,
        weights_ptr,
        shared_weights
    );

    return 1;
}