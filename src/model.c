
#include "llm.h"

#include "checkpoint.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stddef.h>

static void set_error(char *error, size_t capacity,const char *message){
    if (error != NULL && capacity > 0) {
        snprintf(error, capacity, "%s", message);
    }
}

static int checked_mul_size(size_t a, size_t b, size_t *result)
{
    if (a != 0 && b > SIZE_MAX / a) {
        return 0;
    }

    *result = a * b;
    return 1;
}

int malloc_run_state(RunState *s,const Config *p,char *error,size_t error_capacity){
    size_t kv_dim;
    size_t cache_count;
    size_t attention_count;

    if (s == NULL || p == NULL) {
        set_error(error, error_capacity,
                  "Invalid runtime allocation arguments.");
        return 0;
    }

    memset(s, 0, sizeof(*s));

    kv_dim = (size_t)p->dim *
             (size_t)p->n_kv_heads /
             (size_t)p->n_heads;

    if (!checked_mul_size(
            (size_t)p->n_layers,
            (size_t)p->seq_len,
            &cache_count) ||
        !checked_mul_size(
            cache_count,
            kv_dim,
            &cache_count) ||
        !checked_mul_size(
            (size_t)p->n_heads,
            (size_t)p->seq_len,
            &attention_count)) {

        set_error(error, error_capacity,
                  "Runtime buffer size calculation overflow.");
        return 0;
    }

    s->x = calloc((size_t)p->dim, sizeof(float));
    s->xb = calloc((size_t)p->dim, sizeof(float));
    s->xb2 = calloc((size_t)p->dim, sizeof(float));

    s->hb = calloc((size_t)p->hidden_dim, sizeof(float));
    s->hb2 = calloc((size_t)p->hidden_dim, sizeof(float));

    s->q = calloc((size_t)p->dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));

    s->key_cache = calloc(cache_count, sizeof(float));
    s->value_cache = calloc(cache_count, sizeof(float));

    s->att = calloc(attention_count, sizeof(float));
    s->logits = calloc((size_t)p->vocab_size, sizeof(float));

    if (!s->x || !s->xb || !s->xb2 ||
        !s->hb || !s->hb2 ||
        !s->q || !s->k || !s->v ||
        !s->key_cache || !s->value_cache ||
        !s->att || !s->logits) {

        free_run_state(s);

        set_error(error, error_capacity,
                  "Failed to allocate inference buffers.");

        return 0;
    }

    return 1;
}

void free_run_state(RunState *s)
{
    if (s == NULL) {
        return;
    }

    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);

    memset(s, 0, sizeof(*s));
}

// void memory_map_weights(TransformerWeights *w, Config *p,
//                         float *ptr, int shared_weights)
// {
//     size_t dim = (size_t)p->dim;
//     size_t hidden = (size_t)p->hidden_dim;
//     size_t layers = (size_t)p->n_layers;
//     size_t vocab = (size_t)p->vocab_size;
//     size_t head_size = dim / (size_t)p->n_heads;
//     size_t kv_dim = dim * (size_t)p->n_kv_heads
//                     / (size_t)p->n_heads;

//     w->token_embedding_table = ptr;
//     ptr += vocab * dim;

//     w->rms_att_weight = ptr;
//     ptr += layers * dim;

//     w->wq = ptr;
//     ptr += layers * dim * dim;

//     w->wk = ptr;
//     ptr += layers * dim * kv_dim;

//     w->wv = ptr;
//     ptr += layers * dim * kv_dim;

//     w->wo = ptr;
//     ptr += layers * dim * dim;

//     w->rms_ffn_weight = ptr;
//     ptr += layers * dim;

//     w->w1 = ptr;
//     ptr += layers * dim * hidden;

//     w->w2 = ptr;
//     ptr += layers * hidden * dim;

//     w->w3 = ptr;
//     ptr += layers * dim * hidden;

//     w->rms_final_weight = ptr;
//     ptr += dim;

//     /* Skip legacy RoPE frequency data in the checkpoint. */
//     ptr += (size_t)p->seq_len * (head_size / 2);
//     ptr += (size_t)p->seq_len * (head_size / 2);

//     w->wcls = shared_weights
//         ? w->token_embedding_table
//         : ptr;
// }

/* Initialize one FP32 tensor descriptor. */
// static int initialize_tensor(
//     Tensor *tensor,
//     const char *name,
//     void *data,
//     size_t ndim,
//     const size_t *shape
// )
// {
//     size_t elements = 1;
//     size_t byte_size;

//     if (tensor == NULL || data == NULL ||
//         ndim == 0 || ndim > TENSOR_MAX_DIMS) {
//         return 0;
//     }

//     for (size_t i = 0; i < ndim; i++) {
//         if (shape[i] == 0 ||
//             !checked_mul_size(elements, shape[i], &elements)) {
//             return 0;
//         }
//     }

//     if (!checked_mul_size(elements, sizeof(float), &byte_size)) {
//         return 0;
//     }

//     tensor->name = name;
//     tensor->data = data;
//     tensor->dtype = TENSOR_F32;
//     tensor->storage = TENSOR_STORAGE_MAPPED;
//     tensor->ndim = ndim;
//     tensor->byte_size = byte_size;

//     for (size_t i = 0; i < ndim; i++) {
//         tensor->shape[i] = shape[i];
//     }

//     return 1;
// }


/* Register all model weights as generic tensors. */
static int register_model_tensors(
    Transformer *t,
    char *error,
    size_t error_capacity
)
{
    size_t dim = (size_t)t->config.dim;
    size_t hidden = (size_t)t->config.hidden_dim;
    size_t layers = (size_t)t->config.n_layers;
    size_t vocab = (size_t)t->config.vocab_size;

    size_t head_size = dim / (size_t)t->config.n_heads;
    size_t kv_dim = head_size * (size_t)t->config.n_kv_heads;

    size_t shapes[12][3] = {
        {vocab, dim, 0},
        {layers, dim, 0},
        {layers, dim, dim},
        {layers, kv_dim, dim},
        {layers, kv_dim, dim},
        {layers, dim, dim},
        {layers, dim, 0},
        {layers, hidden, dim},
        {layers, dim, hidden},
        {layers, hidden, dim},
        {dim, 0, 0},
        {vocab, dim, 0}
    };

    const char *names[12] = {
        "token_embedding",
        "rms_att_weight",
        "wq",
        "wk",
        "wv",
        "wo",
        "rms_ffn_weight",
        "w1",
        "w2",
        "w3",
        "rms_final_weight",
        "classifier"
    };

    void *data[12] = {
        t->weights.token_embedding_table,
        t->weights.rms_att_weight,
        t->weights.wq,
        t->weights.wk,
        t->weights.wv,
        t->weights.wo,
        t->weights.rms_ffn_weight,
        t->weights.w1,
        t->weights.w2,
        t->weights.w3,
        t->weights.rms_final_weight,
        t->weights.wcls
    };

    size_t dimensions[12] = {
        2, 2, 3, 3, 3, 3,
        2, 3, 3, 3, 1, 2
    };

    for (size_t i = 0; i < 12; i++) {

        if (!tensor_registry_add(
                &t->tensors,
                &t->num_tensors,
                &t->tensor_capacity,
                names[i],
                data[i],
                TENSOR_F32,
                TENSOR_STORAGE_MAPPED,
                dimensions[i],
                shapes[i])) {

            free(t->tensors);
            t->tensors = NULL;
            t->num_tensors = 0;
            t->tensor_capacity = 0;

            set_error(
                error,
                error_capacity,
                "Failed to register model tensor."
            );

            return 0;
        }
    }

    return 1;
}


/* Safely add a * b to the expected number of floats. */

// static int add_weight_count(size_t *total,
//                            size_t a, size_t b)
// {
//     if (a != 0 && b > SIZE_MAX / a) {
//         return 0;
//     }

//     size_t count = a * b;

//     if (count > SIZE_MAX - *total) {
//         return 0;
//     }

//     *total += count;
//     return 1;
// }
// /* Validate the checkpoint's expected weight layout. */
// static int validate_checkpoint_size(
//     const Config *p,
//     size_t file_size,
//     int shared_weights
// )
// {
//     size_t total = 0;

//     size_t dim = (size_t)p->dim;
//     size_t hidden = (size_t)p->hidden_dim;
//     size_t layers = (size_t)p->n_layers;
//     size_t vocab = (size_t)p->vocab_size;
//     size_t seq_len = (size_t)p->seq_len;

//     size_t head_size = dim / (size_t)p->n_heads;
//     size_t kv_dim;
//     size_t layers_dim;
//     size_t layers_hidden;

//     /* Safely calculate derived dimensions. */
//     if (!checked_mul_size(
//             head_size,
//             (size_t)p->n_kv_heads,
//             &kv_dim) ||
//         !checked_mul_size(
//             layers,
//             dim,
//             &layers_dim) ||
//         !checked_mul_size(
//             layers,
//             hidden,
//             &layers_hidden)) {
//         return 0;
//     }

//     /* Validate all model weight tensors. */
//     if (!add_weight_count(&total, vocab, dim) ||
//         !add_weight_count(&total, layers, dim) ||
//         !add_weight_count(&total, layers_dim, dim) ||
//         !add_weight_count(&total, layers_dim, kv_dim) ||
//         !add_weight_count(&total, layers_dim, kv_dim) ||
//         !add_weight_count(&total, layers_dim, dim) ||
//         !add_weight_count(&total, layers, dim) ||
//         !add_weight_count(&total, layers_dim, hidden) ||
//         !add_weight_count(&total, layers_hidden, dim) ||
//         !add_weight_count(&total, layers_dim, hidden) ||
//         !add_weight_count(&total, 1, dim)) {
//         return 0;
//     }

//     /* Two legacy RoPE frequency arrays. */
//     if (!add_weight_count(&total, seq_len, head_size / 2) ||
//         !add_weight_count(&total, seq_len, head_size / 2)) {
//         return 0;
//     }

//     /* Unshared classifier weights. */
//     if (!shared_weights &&
//         !add_weight_count(&total, vocab, dim)) {
//         return 0;
//     }

//     /* Safely convert float count into bytes. */
//     if (total > (SIZE_MAX - sizeof(Config)) / sizeof(float)) {
//         return 0;
//     }

//     size_t expected =
//         sizeof(Config) + total * sizeof(float);

//     return expected <= file_size;
// }

// int read_checkpoint(const char *checkpoint,Config *config,
//     TransformerWeights *weights,
//     int *fd,
//     float **data,
//     ssize_t *file_size,
//     char *error,
//     size_t error_capacity
// )
// {
//     FILE *file;
//     struct stat st;
//     int shared_weights;

//     *fd = -1;
//     *data = NULL;
//     *file_size = 0;

//     /* Read the configuration header. */
//     file = fopen(checkpoint, "rb");

//     if (file == NULL) {
//         set_error(error, error_capacity,
//                   "Cannot open checkpoint.");
//         return 0;
//     }

//     if (fread(config, sizeof(*config), 1, file) != 1) {
//         fclose(file);
//         set_error(error, error_capacity,
//                   "Cannot read checkpoint configuration.");
//         return 0;
//     }

//     if (fclose(file) != 0) {
//         set_error(error, error_capacity,
//                   "Cannot close checkpoint file.");
//         return 0;
//     }

//     /* Negative vocabulary size means unshared classifier weights. */
//     shared_weights = config->vocab_size > 0;

//     if (config->vocab_size == 0 ||
//         config->vocab_size == INT_MIN) {
//         set_error(error, error_capacity,
//                   "Invalid vocabulary size.");
//         return 0;
//     }

//     if (config->vocab_size < 0) {
//         config->vocab_size = -config->vocab_size;
//     }

//     /* Validate dimensions. */
//     if (config->dim <= 0 ||
//         config->hidden_dim <= 0 ||
//         config->n_layers <= 0 ||
//         config->n_heads <= 0 ||
//         config->n_kv_heads <= 0 ||
//         config->seq_len <= 0 ||
//         config->dim % config->n_heads != 0 ||
//         (config->dim / config->n_heads) % 2 != 0 ||
//         config->n_heads % config->n_kv_heads != 0) {

//         set_error(error, error_capacity,
//                   "Invalid checkpoint dimensions.");
//         return 0;
//     }

//     /* Open the checkpoint for memory mapping. */
//     *fd = open(checkpoint, O_RDONLY);

//     if (*fd == -1) {
//         set_error(error, error_capacity,
//                   "Cannot open checkpoint for mapping.");
//         return 0;
//     }

//     if (fstat(*fd, &st) != 0 ||
//         st.st_size < (off_t)sizeof(Config)) {

//         close(*fd);
//         *fd = -1;

//         set_error(error, error_capacity,
//                   "Cannot inspect checkpoint file size.");
//         return 0;
//     }

//     *file_size = (ssize_t)st.st_size;

//     /* Validate the expected weight layout. */
//     if (!validate_checkpoint_size(
//             config, (size_t)*file_size, shared_weights)) {

//         close(*fd);
//         *fd = -1;
//         *file_size = 0;

//         set_error(error, error_capacity,
//                   "Checkpoint is too small or has an invalid weight layout.");
//         return 0;
//     }

//     /* Map the checkpoint into memory. */
//     *data = mmap(NULL, (size_t)*file_size,
//                  PROT_READ, MAP_PRIVATE, *fd, 0);

//     if (*data == MAP_FAILED) {
//         *data = NULL;

//         close(*fd);
//         *fd = -1;
//         *file_size = 0;

//         set_error(error, error_capacity,
//                   "Cannot memory-map checkpoint.");
//         return 0;
//     }

//     /* Assign pointers to the checkpoint's weight tensors. */
//     float *weights_ptr =
//         (float *)((char *)*data + sizeof(Config));

//     // memory_map_weights(
//     //     weights,
//     //     config,
//     //     weights_ptr,
//     //     shared_weights
//     // );

//     return 1;
// }

int build_transformer(
    Transformer *t,
    const char *checkpoint_path,
    char *error,
    size_t error_capacity
)
{
    memset(t, 0, sizeof(*t));
    t->fd = -1;

    if (!checkpoint_load(
            checkpoint_path,
            CHECKPOINT_FORMAT_AUTO,
            &t->config,
            &t->weights,
            &t->fd,
            &t->data,
            &t->file_size,
            error,
            error_capacity)) {
        free_transformer(t);
        return 0;
    }

    if (!register_model_tensors(t, error, error_capacity)) {
        free_transformer(t);
        return 0;
    }

    if (!malloc_run_state(&t->state,&t->config,error,error_capacity)) {
        free_transformer(t);
        return 0;
    }

    return 1;
}

static void free_gguf_metadata(Transformer *t)
{
    if (t == NULL) {
        return;
    }

    if (t->gguf_tensors != NULL) {
        for (size_t i = 0; i < t->gguf_tensor_count; i++) {
            free(t->gguf_tensors[i].name);
        }

        free(t->gguf_tensors);
    }

    t->gguf_tensors = NULL;
    t->gguf_tensor_count = 0;

    t->gguf_alignment = 0;
    t->gguf_data_offset = 0;
}

void free_transformer(Transformer *t)
{
    if (t == NULL) {
        return;
    }

    free_run_state(&t->state);
    free_gguf_metadata(t);
    free(t->tensors);
    t->tensors = NULL;
    t->num_tensors = 0;

    if (t->data != NULL && t->file_size > 0) {
        munmap(t->data, (size_t)t->file_size);
    }

    if (t->fd != -1) {
        close(t->fd);
    }

    memset(t, 0, sizeof(*t));
    t->fd = -1;
}



Transformer *llm_load(const char *model_path,
                      char *error,
                      size_t error_capacity)
{
    Transformer *model;

    if (error != NULL && error_capacity > 0) {
        error[0] = '\0';
    }

    if (model_path == NULL || model_path[0] == '\0') {
        if (error != NULL && error_capacity > 0) {
            snprintf(error, error_capacity, "Invalid model path.");
        }
        return NULL;
    }

    model = calloc(1, sizeof(*model));
    if (model == NULL) {
        if (error != NULL && error_capacity > 0) {
            snprintf(error, error_capacity, "Memory allocation failed.");
        }
        return NULL;
    }

    model->fd = -1;
    model->data = NULL;

    if (!build_transformer(model,model_path,error,error_capacity)){
        free_transformer(model);
        free(model);
        return NULL;
    }

return model;
}

const Config *llm_model_config(const Transformer *model)
{
    return model == NULL ? NULL : &model->config;
}

size_t llm_model_file_size(const Transformer *model)
{
    if (model == NULL || model->file_size < 0) {
        return 0;
    }

    return (size_t)model->file_size;
}

void llm_free_model(Transformer *model)
{
    if (model == NULL) {
        return;
    }

    free_transformer(model);
    free(model);
}
