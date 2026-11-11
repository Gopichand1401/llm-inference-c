#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include "transformer.h"
#include <stdint.h>
#include <stddef.h>
#include <sys/types.h>

/* Supported checkpoint formats. */
typedef enum {
    CHECKPOINT_FORMAT_AUTO,
    CHECKPOINT_FORMAT_STORIES,
    CHECKPOINT_FORMAT_GGUF,
    CHECKPOINT_FORMAT_SAFETENSORS
} CheckpointFormat;

typedef struct {
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;
} GGUFHeader;

typedef struct {
    uint32_t alignment;
    uint64_t data_offset;
} GGUFLayout;

typedef struct GGUF_TensorInfo {
    char *name;
    uint32_t n_dimensions;
    uint64_t dimensions[4];
    uint32_t type;
    uint64_t offset;
} GGUF_TensorInfo;

/* Generic checkpoint loading interface. */
int checkpoint_load(
    const char *checkpoint,
    CheckpointFormat format,
    Config *config,
    TransformerWeights *weights,
    int *fd,
    float **data,
    ssize_t *file_size,
    char *error,
    size_t error_capacity
);

/* Stories-specific checkpoint reader. */
int stories_checkpoint_load(
    const char *checkpoint,
    Config *config,
    TransformerWeights *weights,
    int *fd,
    float **data,
    ssize_t *file_size,
    char *error,
    size_t error_capacity
);

int gguf_read_header(
    const char *checkpoint,
    GGUFHeader *header,
    char *error,
    size_t error_capacity
);

int gguf_read_metadata(
    const char *checkpoint,
    char *error,
    size_t error_capacity
);

int gguf_read_config(
    const char *checkpoint,
    Config *config,
    char *error,
    size_t error_capacity
);

int gguf_read_tensors(
    const char *checkpoint,
    GGUF_TensorInfo **tensors,
    size_t *tensor_count,
    char *error,
    size_t error_capacity
);

/*
 * Free a tensor list returned by gguf_read_tensors(): every name, then the
 * array. Safe on NULL. (The caller owns the list; this is the matching free.)
 */
void gguf_free_tensors(GGUF_TensorInfo *tensors, size_t tensor_count);

int gguf_read_layout(
    const char *checkpoint,
    GGUFLayout *layout,
    char *error,
    size_t error_capacity
);

int gguf_tensor_file_offset(
    const GGUFLayout *layout,
    const GGUF_TensorInfo *tensor,
    uint64_t *file_offset
);

/*
 * GGML tensor type helpers.
 * gguf_ggml_type_name() returns "UNKNOWN" for unlisted IDs.
 * gguf_ggml_type_info() returns 0 for unlisted IDs.
 */
const char *gguf_ggml_type_name(uint32_t type);

int gguf_ggml_type_info(
    uint32_t type,
    uint64_t *block_size,
    uint64_t *type_size
);

int gguf_tensor_storage_size(
    const GGUF_TensorInfo *tensor,
    uint64_t *storage_size
);

int gguf_get_tensor_data(
    const void *mapped_data,
    size_t file_size,
    const GGUFLayout *layout,
    const GGUF_TensorInfo *tensor,
    const void **tensor_data,
    uint64_t *tensor_size
);

const GGUF_TensorInfo *gguf_find_tensor(
    const GGUF_TensorInfo *tensors,
    size_t tensor_count,
    const char *name
);

#endif

