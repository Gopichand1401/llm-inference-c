#ifndef TRANSFORMER_H
#define TRANSFORMER_H
#define TENSOR_MAX_DIMS 4
 
#include <sys/types.h>
#include <stddef.h>
#include <stdint.h>
 
/* Model architecture configuration */

/* Supported tensor data types. */
typedef enum {
    TENSOR_F32,
    TENSOR_F16,
    TENSOR_BF16,
    TENSOR_I8,
    TENSOR_U8
} TensorDataType;

/* Where the tensor's memory is stored. */
typedef enum {
    TENSOR_STORAGE_NONE,
    TENSOR_STORAGE_MAPPED,
    TENSOR_STORAGE_OWNED
} TensorStorage;

/* Generic tensor representation. */
typedef struct {
    const char *name;
    void *data;
    TensorDataType dtype;
    uint32_t ggml_type;  /* Original GGUF/GGML type ID */
    TensorStorage storage;
    size_t ndim;
    size_t shape[TENSOR_MAX_DIMS];
    size_t byte_size;
} Tensor;

/* Initialize a tensor descriptor. */
int tensor_init(
    Tensor *tensor,
    const char *name,
    void *data,
    TensorDataType dtype,
    TensorStorage storage,
    size_t ndim,
    const size_t *shape
);

/* Find a tensor by its name. */
Tensor *tensor_find(
    Tensor *tensors,
    size_t num_tensors,
    const char *name
);

/* Model architecture configuration */
typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;

    /* LLaMA-specific parameters */
    int rope_dim;
    float rope_freq_base;
    float rms_norm_eps;

    /*
     * Linear RoPE scaling factor (1 = no scaling). Models fine-tuned for a
     * longer context with "linear" scaling divide every position by this
     * factor before computing the rotation angle. Set from the GGUF keys
     * llama.rope.scaling.type/factor (or the older llama.rope.scale_linear);
     * any other scaling type is refused at load time.
     */
    float rope_scale;
} Config;

/* Validate common LLaMA-style model configuration. */
int config_validate(
    const Config *config,
    char *error,
    size_t error_capacity
);

 
/* Learned model parameters */
typedef struct {
    float *token_embedding_table;
 
    float *rms_att_weight;
    float *rms_ffn_weight;
 
    float *wq;
    float *wk;
    float *wv;
    float *wo;
 
    float *w1;
    float *w2;
    float *w3;
 
    float *rms_final_weight;
    float *wcls;
} TransformerWeights;
 
/* Intermediate activations and key/value cache */
typedef struct {
    float *x;
    float *xb;
    float *xb2;
 
    float *hb;
    float *hb2;
 
    float *q;
    float *k;
    float *v;
 
    float *att;
    float *logits;
 
    float *key_cache;
    float *value_cache;
} RunState;
 
/* Complete transformer model */
typedef struct Transformer {
    Config config;
    TransformerWeights weights;
    RunState state;

    /* Generic tensor metadata */
    Tensor *tensors;
    size_t num_tensors;
    size_t tensor_capacity;

    /* GGUF tensor metadata */
    struct GGUF_TensorInfo *gguf_tensors;
    size_t gguf_tensor_count;

    uint32_t gguf_alignment;
    uint64_t gguf_data_offset;

    /* Checkpoint memory mapping */
    int fd;
    float *data;
    ssize_t file_size;

} Transformer;

/* Dynamically add a tensor to the registry. */
int tensor_registry_add(
    Tensor **tensors,
    size_t *num_tensors,
    size_t *capacity,
    const char *name,
    void *data,
    TensorDataType dtype,
    TensorStorage storage,
    size_t ndim,
    const size_t *shape
);
 
/* Model initialization and cleanup */
int malloc_run_state(
    RunState *s,
    const Config *p,
    char *error,
    size_t error_capacity
);
void free_run_state(RunState *s);
 
// void memory_map_weights(
//     TransformerWeights *w,
//     Config *p,
//     float *ptr,
//     int shared_weights
// );
 
int read_checkpoint(
    const char *checkpoint,
    Config *config,
    TransformerWeights *weights,
    int *fd,
    float **data,
    ssize_t *file_size,
    char *error,
    size_t error_capacity
);

int build_transformer(
    Transformer *t,
    const char *checkpoint_path,
    char *error,
    size_t error_capacity
);

void free_transformer(Transformer *t);
/* Transformer mathematical operations */
void rmsnorm(float *o, const float *x, const float *weight, int size);
void softmax(float *x, int size);
void matmul(float *xout, const float *x, const float *w, int n, int d);
 
/* Transformer forward pass */
float *forward(Transformer *t, int token, int pos);
#endif
 
