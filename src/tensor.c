#include "transformer.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* Return the size of one tensor element. */
static size_t tensor_dtype_size(TensorDataType dtype)
{
    switch (dtype) {
        case TENSOR_F32:  return 4;
        case TENSOR_F16:  return 2;
        case TENSOR_BF16: return 2;
        case TENSOR_I8:   return 1;
        case TENSOR_U8:   return 1;
        default:          return 0;
    }
}

/* Initialize a tensor descriptor. */
int tensor_init(
    Tensor *tensor,
    const char *name,
    void *data,
    TensorDataType dtype,
    TensorStorage storage,
    size_t ndim,
    const size_t *shape
)
{
    size_t elements = 1;
    size_t element_size;

    if (tensor == NULL || name == NULL || data == NULL ||
        shape == NULL || ndim == 0 || ndim > TENSOR_MAX_DIMS) {
        return 0;
    }

    element_size = tensor_dtype_size(dtype);

    if (element_size == 0) {
        return 0;
    }

    for (size_t i = 0; i < ndim; i++) {
        if (shape[i] == 0 ||
            elements > SIZE_MAX / shape[i]) {
            return 0;
        }

        elements *= shape[i];
    }

    if (elements > SIZE_MAX / element_size) {
        return 0;
    }

    tensor->name = name;
    tensor->data = data;
    tensor->dtype = dtype;
    tensor->ggml_type = UINT32_MAX;
    tensor->storage = storage;
    tensor->ndim = ndim;
    tensor->byte_size = elements * element_size;

    for (size_t i = 0; i < TENSOR_MAX_DIMS; i++) {
        tensor->shape[i] = (i < ndim) ? shape[i] : 0;
    }

    return 1;
}

/* Find a tensor by its name. */
Tensor *tensor_find(
    Tensor *tensors,
    size_t num_tensors,
    const char *name
)
{
    if (tensors == NULL || name == NULL) {
        return NULL;
    }

    for (size_t i = 0; i < num_tensors; i++) {
        if (tensors[i].name != NULL &&
            strcmp(tensors[i].name, name) == 0) {
            return &tensors[i];
        }
    }

    return NULL;
}

/* Add a tensor to a dynamically growing registry. */
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
)
{
    Tensor new_tensor;

    if (tensors == NULL || num_tensors == NULL ||
        capacity == NULL) {
        return 0;
    }

    /* Prevent duplicate tensor names. */
    if (tensor_find(*tensors, *num_tensors, name) != NULL) {
        return 0;
    }

    /* Validate and initialize before modifying the registry. */
    if (!tensor_init(
            &new_tensor,
            name,
            data,
            dtype,
            storage,
            ndim,
            shape)) {
        return 0;
    }

    /* Grow the registry when it becomes full. */
    if (*num_tensors == *capacity) {
        size_t new_capacity = (*capacity == 0)
                            ? 8
                            : *capacity * 2;

        if (new_capacity < *capacity ||
            new_capacity > SIZE_MAX / sizeof(Tensor)) {
            return 0;
        }

        Tensor *new_tensors = realloc(
            *tensors,
            new_capacity * sizeof(Tensor)
        );

        if (new_tensors == NULL) {
            return 0;
        }

        *tensors = new_tensors;
        *capacity = new_capacity;
    }

    /* Insert the new tensor. */
    (*tensors)[*num_tensors] = new_tensor;
    (*num_tensors)++;

    return 1;
}