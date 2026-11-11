
#include "checkpoint.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>

/* =========================================================
   LITTLE-ENDIAN READERS
   ========================================================= */

static uint32_t read_u32_le(const unsigned char *b)
{
    return (uint32_t)b[0]
         | ((uint32_t)b[1] << 8)
         | ((uint32_t)b[2] << 16)
         | ((uint32_t)b[3] << 24);
}

static uint64_t read_u64_le(const unsigned char *b)
{
    uint64_t value = 0;

    for (int i = 0; i < 8; i++) {
        value |= (uint64_t)b[i] << (i * 8);
    }

    return value;
}

/* =========================================================
   ERROR HANDLING
   ========================================================= */

static void set_error(
    char *error,
    size_t capacity,
    const char *message
)
{
    if (error != NULL && capacity > 0) {
        snprintf(error, capacity, "%s", message);
    }
}

/* =========================================================
   GGUF HEADER READER
   ========================================================= */

int gguf_read_header(
    const char *checkpoint,
    GGUFHeader *header,
    char *error,
    size_t error_capacity
)
{
    if (checkpoint == NULL || header == NULL) {
        set_error(error, error_capacity, "Invalid arguments.");
        return 0;
    }

    FILE *file = fopen(checkpoint, "rb");

    if (file == NULL) {
        set_error(error, error_capacity,
                  "Cannot open GGUF file.");
        return 0;
    }

    unsigned char buffer[8];
    unsigned char magic[4];

    /* Read and verify magic number. */
    if (fread(magic, 1, 4, file) != 4 ||
        memcmp(magic, "GGUF", 4) != 0) {
        fclose(file);
        set_error(error, error_capacity,
                  "Invalid GGUF magic number.");
        return 0;
    }

    /* Read version. */
    if (fread(buffer, 1, 4, file) != 4) {
        fclose(file);
        set_error(error, error_capacity,
                  "Cannot read GGUF version.");
        return 0;
    }

    header->version = read_u32_le(buffer);

    /*
     * GGUF v1 stored counts and string lengths as 32-bit values; v2 and v3
     * use 64-bit values, which is what every reader in this file expects.
     * Reject v1 instead of misreading it.
     */
    if (header->version == 1) {
        fclose(file);
        set_error(error, error_capacity,
                  "GGUF version 1 is not supported (use a v2/v3 file).");
        return 0;
    }

    if (header->version < 2 || header->version > 3) {
        fclose(file);
        set_error(error, error_capacity,
                  "Unsupported GGUF version.");
        return 0;
    }

    /* Read tensor count. */
    if (fread(buffer, 1, 8, file) != 8) {
        fclose(file);
        set_error(error, error_capacity,
                  "Cannot read tensor count.");
        return 0;
    }

    header->tensor_count = read_u64_le(buffer);

    /* Read metadata count. */
    if (fread(buffer, 1, 8, file) != 8) {
        fclose(file);
        set_error(error, error_capacity,
                  "Cannot read metadata count.");
        return 0;
    }

    header->metadata_count = read_u64_le(buffer);

    fclose(file);

    if (error != NULL && error_capacity > 0) {
        error[0] = '\0';
    }

    return 1;
}

/* =========================================================
   GGUF VALUE HELPERS
   ========================================================= */

/* Return the size of a fixed-width GGUF value. */
static size_t gguf_type_size(uint32_t type)
{
    switch (type) {
        case 0:
        case 1:
        case 7:
            return 1;

        case 2:
        case 3:
            return 2;

        case 4:
        case 5:
        case 6:
            return 4;

        case 10:
        case 11:
        case 12:
            return 8;

        default:
            return 0;
    }
}

/* Return a readable name for a GGUF type. */
static const char *gguf_type_name(uint32_t type)
{
    static const char *names[] = {
        "UINT8", "INT8", "UINT16", "INT16",
        "UINT32", "INT32", "FLOAT32", "BOOL",
        "STRING", "ARRAY", "UINT64", "INT64",
        "FLOAT64"
    };

    if (type >= 13) {
        return "UNKNOWN";
    }

    return names[type];
}

/* Read a length-prefixed GGUF string. */
static int read_gguf_string(FILE *file, char **result)
{
    unsigned char buffer[8];

    if (fread(buffer, 1, 8, file) != 8) {
        return 0;
    }

    uint64_t length = read_u64_le(buffer);

    if (length > SIZE_MAX - 1 ||
        length > 1024 * 1024) {
        return 0;
    }

    char *value = malloc((size_t)length + 1);

    if (value == NULL) {
        return 0;
    }

    if (fread(value, 1, (size_t)length, file) != length) {
        free(value);
        return 0;
    }

    value[length] = '\0';
    *result = value;

    return 1;
}

/* Read a UINT32 value. */
static int read_gguf_u32(FILE *file, uint32_t *value)
{
    unsigned char buffer[4];

    if (fread(buffer, 1, 4, file) != 4) {
        return 0;
    }

    *value = read_u32_le(buffer);

    return 1;
}

static int read_gguf_f32(FILE *file, float *value)
{
    uint32_t bits;

    if (value == NULL || !read_gguf_u32(file, &bits)) {
        return 0;
    }

    memcpy(value, &bits, sizeof(bits));

    return 1;
}

/* Skip a GGUF string. */
static int skip_gguf_string(FILE *file)
{
    char *value = NULL;

    if (!read_gguf_string(file, &value)) {
        return 0;
    }

    free(value);

    return 1;
}

/* Skip one scalar metadata value. */
static int skip_gguf_value(FILE *file, uint32_t type)
{
    if (type == 8) {
        return skip_gguf_string(file);
    }

    size_t size = gguf_type_size(type);

    if (size == 0) {
        return 0;
    }

    unsigned char buffer[8];

    return fread(buffer, 1, size, file) == size;
}

/* Skip an entire metadata array. */
static int skip_gguf_array(
    FILE *file,
    uint32_t *element_type,
    uint64_t *element_count
)
{
    unsigned char buffer[8];

    if (fread(buffer, 1, 4, file) != 4) {
        return 0;
    }

    *element_type = read_u32_le(buffer);

    if (*element_type > 12 || *element_type == 9) {
        return 0;
    }

    if (fread(buffer, 1, 8, file) != 8) {
        return 0;
    }

    *element_count = read_u64_le(buffer);

    for (uint64_t i = 0; i < *element_count; i++) {
        if (!skip_gguf_value(file, *element_type)) {
            return 0;
        }
    }

    return 1;
}

/* Skip a metadata value, including arrays. */
static int skip_gguf_metadata_value(
    FILE *file,
    uint32_t type
)
{
    if (type == 9) {
        uint32_t element_type;
        uint64_t element_count;

        return skip_gguf_array(
            file, &element_type, &element_count);
    }

    return skip_gguf_value(file, type);
}

/* =========================================================
   GGUF METADATA DISPLAY
   ========================================================= */

int gguf_read_metadata(
    const char *checkpoint,
    char *error,
    size_t error_capacity
)
{
    GGUFHeader header;

    if (!gguf_read_header(
            checkpoint, &header, error, error_capacity)) {
        return 0;
    }

    FILE *file = fopen(checkpoint, "rb");

    if (file == NULL) {
        set_error(error, error_capacity,
                  "Cannot open GGUF file.");
        return 0;
    }

    if (fseek(file, 24, SEEK_SET) != 0) {
        fclose(file);
        set_error(error, error_capacity,
                  "Cannot seek to GGUF metadata.");
        return 0;
    }

    printf("\n=== GGUF Metadata ===\n");

    for (uint64_t i = 0; i < header.metadata_count; i++) {

        char *key = NULL;
        unsigned char buffer[4];

        if (!read_gguf_string(file, &key)) {
            goto metadata_error;
        }

        if (fread(buffer, 1, 4, file) != 4) {
            free(key);
            goto metadata_error;
        }

        uint32_t type = read_u32_le(buffer);

        printf("%s : %s", key, gguf_type_name(type));

        free(key);

        if (type == 9) {

            uint32_t element_type;
            uint64_t count;

            if (!skip_gguf_array(
                    file, &element_type, &count)) {
                goto metadata_error;
            }

            printf(" [element=%s, count=%llu]\n",
                   gguf_type_name(element_type),
                   (unsigned long long)count);
        }
        else {

            printf("\n");

            if (type > 12 ||
                !skip_gguf_value(file, type)) {
                goto metadata_error;
            }
        }
    }

    fclose(file);

    if (error != NULL && error_capacity > 0) {
        error[0] = '\0';
    }

    return 1;

metadata_error:
    fclose(file);

    set_error(error, error_capacity,
              "Invalid or truncated GGUF metadata.");

    return 0;
}

/* =========================================================
   GGUF CONFIGURATION EXTRACTION
   ========================================================= */

int gguf_read_config(
    const char *checkpoint,
    Config *config,
    char *error,
    size_t error_capacity
)
{
    if (checkpoint == NULL || config == NULL) {
        set_error(error, error_capacity, "Invalid arguments.");
        return 0;
    }

    GGUFHeader header;

    if (!gguf_read_header(
            checkpoint, &header, error, error_capacity)) {
        return 0;
    }

    FILE *file = fopen(checkpoint, "rb");

    if (file == NULL) {
        set_error(error, error_capacity,
                  "Cannot open GGUF file.");
        return 0;
    }

    if (fseek(file, 24, SEEK_SET) != 0) {
        fclose(file);
        set_error(error, error_capacity,
                  "Cannot seek to GGUF metadata.");
        return 0;
    }

    Config candidate = {0};

    unsigned int seen = 0;

    /* RoPE scaling metadata (see Config.rope_scale). */
    char scaling_type[32] = "";
    float scaling_factor = 0.0f;
    float scale_linear_old = 0.0f;

    enum {
        SEEN_ARCH     = 1u << 0,
        SEEN_DIM      = 1u << 1,
        SEEN_HIDDEN   = 1u << 2,
        SEEN_LAYERS   = 1u << 3,
        SEEN_HEADS    = 1u << 4,
        SEEN_KV_HEADS = 1u << 5,
        SEEN_CONTEXT  = 1u << 6,
        SEEN_VOCAB    = 1u << 7
    };

    for (uint64_t i = 0; i < header.metadata_count; i++) {

        char *key = NULL;
        unsigned char buffer[4];
        uint32_t type;

        if (!read_gguf_string(file, &key)) {
            goto config_error;
        }

        if (fread(buffer, 1, 4, file) != 4) {
            free(key);
            goto config_error;
        }

        type = read_u32_le(buffer);

        /* Architecture */
        if (strcmp(key, "general.architecture") == 0) {

            char *architecture = NULL;

            if (type != 8 ||
                !read_gguf_string(file, &architecture)) {
                free(key);
                goto config_error;
            }

            if (strcmp(architecture, "llama") != 0) {
                free(architecture);
                free(key);
                fclose(file);

                set_error(error, error_capacity,
                          "Only llama architecture is supported.");

                return 0;
            }

            free(architecture);

            seen |= SEEN_ARCH;
        }

        /* Tokenizer vocabulary */
        else if (strcmp(key, "tokenizer.ggml.tokens") == 0) {

            uint32_t element_type;
            uint64_t count;

            if (type != 9 ||
                !skip_gguf_array(
                    file, &element_type, &count)) {
                free(key);
                goto config_error;
            }

            if (element_type != 8 ||
                count == 0 ||
                count > INT_MAX) {
                free(key);
                goto config_error;
            }

            candidate.vocab_size = (int)count;

            seen |= SEEN_VOCAB;
        }
                /* RoPE dimension count */
        else if (strcmp(key, "llama.rope.dimension_count") == 0) {
            uint32_t value;

            if (type != 4 ||
                !read_gguf_u32(file, &value) ||
                value == 0 ||
                value > INT_MAX) {
                free(key);
                goto config_error;
            }

            candidate.rope_dim = (int)value;
        }

        /* RoPE frequency base */
        else if (strcmp(key, "llama.rope.freq_base") == 0) {
            float value;

            if (type != 6 ||
                !read_gguf_f32(file, &value) ||
                !isfinite(value) ||
                value <= 0.0f) {
                free(key);
                goto config_error;
            }

            candidate.rope_freq_base = value;
        }

        /* RoPE scaling type: "none", "linear", "yarn", ... */
        else if (strcmp(key, "llama.rope.scaling.type") == 0) {
            char *value = NULL;

            if (type != 8 || !read_gguf_string(file, &value)) {
                free(key);
                goto config_error;
            }

            snprintf(scaling_type, sizeof(scaling_type), "%s", value);
            free(value);
        }

        /* RoPE scaling factor (newer key) and linear scale (older key) */
        else if (strcmp(key, "llama.rope.scaling.factor") == 0 ||
                 strcmp(key, "llama.rope.scale_linear") == 0) {
            float value;

            if (type != 6 ||
                !read_gguf_f32(file, &value) ||
                !isfinite(value) ||
                value <= 0.0f) {
                free(key);
                goto config_error;
            }

            if (strcmp(key, "llama.rope.scaling.factor") == 0) {
                scaling_factor = value;
            } else {
                scale_linear_old = value;
            }
        }

        /* RMSNorm epsilon */
        else if (strcmp(key, "llama.attention.layer_norm_rms_epsilon") == 0) {
            float value;

            if (type != 6 ||
                !read_gguf_f32(file, &value) ||
                !isfinite(value) ||
                value <= 0.0f) {
                free(key);
                goto config_error;
            }

            candidate.rms_norm_eps = value;
        }

        /* Other required configuration fields */
        else {

            int *destination = NULL;
            unsigned int flag = 0;

            if (strcmp(key, "llama.embedding_length") == 0) {
                destination = &candidate.dim;
                flag = SEEN_DIM;
            }
            else if (strcmp(key, "llama.feed_forward_length") == 0) {
                destination = &candidate.hidden_dim;
                flag = SEEN_HIDDEN;
            }
            else if (strcmp(key, "llama.block_count") == 0) {
                destination = &candidate.n_layers;
                flag = SEEN_LAYERS;
            }
            else if (strcmp(key, "llama.attention.head_count") == 0) {
                destination = &candidate.n_heads;
                flag = SEEN_HEADS;
            }
            else if (strcmp(key, "llama.attention.head_count_kv") == 0) {
                destination = &candidate.n_kv_heads;
                flag = SEEN_KV_HEADS;
            }
            else if (strcmp(key, "llama.context_length") == 0) {
                destination = &candidate.seq_len;
                flag = SEEN_CONTEXT;
            }

            if (destination != NULL) {

                uint32_t value;

                if (type != 4 ||
                    !read_gguf_u32(file, &value) ||
                    value == 0 ||
                    value > INT_MAX) {
                    free(key);
                    goto config_error;
                }

                *destination = (int)value;

                seen |= flag;
            }

            else if (!skip_gguf_metadata_value(file, type)) {
                free(key);
                goto config_error;
            }
        }

        free(key);
    }

    fclose(file);

    if (seen != 0xFFu) {
        set_error(error, error_capacity,
                  "Required LLaMA metadata is missing.");
        return 0;
    }

    /* Apply defaults when optional metadata is absent. */
    if (candidate.rope_dim == 0) {
        candidate.rope_dim = candidate.dim / candidate.n_heads;
    }

    if (candidate.rope_freq_base == 0.0f) {
        candidate.rope_freq_base = 10000.0f;
    }

    if (candidate.rms_norm_eps == 0.0f) {
        candidate.rms_norm_eps = 1e-5f;
    }

    /*
     * RoPE scaling. Only linear scaling is implemented (positions divided
     * by the factor); every other type changes the rotation in ways this
     * engine does not compute, so such models are refused rather than run
     * with silently wrong positions.
     */
    candidate.rope_scale = 1.0f;

    if (scaling_type[0] == '\0' || strcmp(scaling_type, "none") == 0) {
        if (scale_linear_old > 0.0f) {
            candidate.rope_scale = scale_linear_old;
        } else if (scaling_type[0] == '\0' && scaling_factor > 0.0f) {
            /* A factor with no type: GGML treats it as linear. */
            candidate.rope_scale = scaling_factor;
        }
    } else if (strcmp(scaling_type, "linear") == 0) {
        candidate.rope_scale = scaling_factor > 0.0f ? scaling_factor : 1.0f;
    } else {
        if (error != NULL && error_capacity > 0) {
            snprintf(error, error_capacity,
                     "RoPE scaling type '%s' is not supported "
                     "(only 'none' and 'linear').", scaling_type);
        }
        return 0;
    }

    if (!config_validate(&candidate, error, error_capacity)) {
        return 0;
    }

    *config = candidate;

    if (error != NULL && error_capacity > 0) {
        error[0] = '\0';
    }

    return 1;

config_error:
    fclose(file);

    set_error(error, error_capacity,
              "Invalid or unsupported GGUF metadata.");

    return 0;
}

/* =========================================================
   GGUF TENSOR INFORMATION READER
   ========================================================= */

void gguf_free_tensors(GGUF_TensorInfo *tensors, size_t tensor_count)
{
    if (tensors == NULL) {
        return;
    }
    for (size_t i = 0; i < tensor_count; ++i) {
        free(tensors[i].name);
    }
    free(tensors);
}

int gguf_read_tensors(
    const char *checkpoint,
    GGUF_TensorInfo **tensors,
    size_t *tensor_count,
    char *error,
    size_t error_capacity
)
{
    if (checkpoint == NULL || tensors == NULL ||
        tensor_count == NULL) {
        set_error(error, error_capacity, "Invalid arguments.");
        return 0;
    }

    *tensors = NULL;
    *tensor_count = 0;

    GGUFHeader header;

    if (!gguf_read_header(
            checkpoint, &header, error, error_capacity)) {
        return 0;
    }

    if (header.tensor_count > SIZE_MAX / sizeof(GGUF_TensorInfo)) {
        set_error(error, error_capacity, "Too many GGUF tensors.");
        return 0;
    }

    FILE *file = fopen(checkpoint, "rb");

    if (file == NULL) {
        set_error(error, error_capacity, "Cannot open GGUF file.");
        return 0;
    }

    /* Metadata starts immediately after the 24-byte header. */
    if (fseek(file, 24, SEEK_SET) != 0) {
        fclose(file);
        set_error(error, error_capacity,
                  "Cannot seek to GGUF metadata.");
        return 0;
    }

    /* Skip all metadata to reach the tensor information section. */
    for (uint64_t i = 0; i < header.metadata_count; i++) {
        char *key = NULL;
        uint32_t type;

        if (!read_gguf_string(file, &key)) {
            goto tensor_error;
        }

        if (!read_gguf_u32(file, &type)) {
            free(key);
            goto tensor_error;
        }

        free(key);

        if (!skip_gguf_metadata_value(file, type)) {
            goto tensor_error;
        }
    }

    size_t count = (size_t)header.tensor_count;

    GGUF_TensorInfo *result = calloc(count, sizeof(*result));

    if (result == NULL && count != 0) {
        fclose(file);
        set_error(error, error_capacity,
                  "Cannot allocate tensor information.");
        return 0;
    }

    /* Read every tensor descriptor. */
    for (size_t i = 0; i < count; i++) {
        unsigned char buffer[8];

        /* Tensor name */
        if (!read_gguf_string(file, &result[i].name)) {
            goto tensor_array_error;
        }

        /* Number of dimensions */
        if (!read_gguf_u32(file, &result[i].n_dimensions)) {
            goto tensor_array_error;
        }

        if (result[i].n_dimensions == 0 ||
            result[i].n_dimensions > 4) {
            goto tensor_array_error;
        }

        /* Tensor dimensions */
        for (uint32_t d = 0;
             d < result[i].n_dimensions; d++) {

            if (fread(buffer, 1, 8, file) != 8) {
                goto tensor_array_error;
            }

            result[i].dimensions[d] = read_u64_le(buffer);
        }

        /* GGML tensor data type */
        if (!read_gguf_u32(file, &result[i].type)) {
            goto tensor_array_error;
        }

        /* Offset relative to the tensor data section */
        if (fread(buffer, 1, 8, file) != 8) {
            goto tensor_array_error;
        }

        result[i].offset = read_u64_le(buffer);
    }

    fclose(file);

    *tensors = result;
    *tensor_count = count;

    if (error != NULL && error_capacity > 0) {
        error[0] = '\0';
    }

    return 1;

tensor_array_error:
    for (size_t i = 0; i < count; i++) {
        free(result[i].name);
    }

    free(result);
    fclose(file);

    set_error(error, error_capacity,
              "Invalid or truncated GGUF tensor information.");

    return 0;

tensor_error:
    fclose(file);

    set_error(error, error_capacity,
              "Invalid or truncated GGUF metadata.");

    return 0;
}

/* =========================================================
   GGUF DATA SECTION LOCATION
   ========================================================= */

int gguf_read_layout(
    const char *checkpoint,
    GGUFLayout *layout,
    char *error,
    size_t error_capacity
)
{
    if (checkpoint == NULL || layout == NULL) {
        set_error(error, error_capacity, "Invalid arguments.");
        return 0;
    }

    GGUFHeader header;

    if (!gguf_read_header(
            checkpoint, &header, error, error_capacity)) {
        return 0;
    }

    FILE *file = fopen(checkpoint, "rb");

    if (file == NULL) {
        set_error(error, error_capacity, "Cannot open GGUF file.");
        return 0;
    }

    layout->alignment = 32; /* GGUF default */

    if (fseek(file, 24, SEEK_SET) != 0) {
        goto layout_error;
    }

    /* Read metadata and find general.alignment. */
    for (uint64_t i = 0; i < header.metadata_count; i++) {
        char *key = NULL;
        uint32_t type;

        if (!read_gguf_string(file, &key)) {
            goto layout_error;
        }

        if (!read_gguf_u32(file, &type)) {
            free(key);
            goto layout_error;
        }

        if (strcmp(key, "general.alignment") == 0) {
            uint32_t alignment;

            if (type != 4 ||
                !read_gguf_u32(file, &alignment) ||
                alignment == 0 ||
                (alignment & (alignment - 1)) != 0) {
                free(key);
                goto layout_error;
            }

            layout->alignment = alignment;
        }
        else if (!skip_gguf_metadata_value(file, type)) {
            free(key);
            goto layout_error;
        }

        free(key);
    }

    /* Skip all tensor descriptors. */
    for (uint64_t i = 0; i < header.tensor_count; i++) {
        char *name = NULL;
        uint32_t dimensions;
        uint32_t type;
        unsigned char buffer[8];

        if (!read_gguf_string(file, &name)) {
            goto layout_error;
        }

        free(name);

        if (!read_gguf_u32(file, &dimensions) ||
            dimensions == 0 || dimensions > 4) {
            goto layout_error;
        }

        for (uint32_t d = 0; d < dimensions; d++) {
            if (fread(buffer, 1, 8, file) != 8) {
                goto layout_error;
            }
        }

        if (!read_gguf_u32(file, &type) ||
            type > 40) {
            goto layout_error;
        }

        if (fread(buffer, 1, 8, file) != 8) {
            goto layout_error;
        }
    }

    /* Position immediately after the tensor descriptors. */
    long position = ftell(file);

    if (position < 0) {
        goto layout_error;
    }

    uint64_t pos = (uint64_t)position;
    uint64_t alignment = layout->alignment;

    /* Align the tensor data section. */
    uint64_t remainder = pos % alignment;
    uint64_t padding = remainder == 0 ? 0 : alignment - remainder;

    if (pos > UINT64_MAX - padding) {
        goto layout_error;
    }

    layout->data_offset = pos + padding;

    fclose(file);

    if (error != NULL && error_capacity > 0) {
        error[0] = '\0';
    }

    return 1;

layout_error:
    fclose(file);

    set_error(error, error_capacity,
              "Cannot determine GGUF tensor data location.");

    return 0;
}

int gguf_tensor_file_offset(
    const GGUFLayout *layout,
    const GGUF_TensorInfo *tensor,
    uint64_t *file_offset
)
{
    if (layout == NULL || tensor == NULL || file_offset == NULL) {
        return 0;
    }

    if (tensor->offset > UINT64_MAX - layout->data_offset) {
        return 0;
    }

    *file_offset = layout->data_offset + tensor->offset;

    return 1;
}

/* =========================================================
   GGUF TENSOR STORAGE SIZE
   ========================================================= */


/*
 * GGML tensor types (enum ggml_type), checked against the official
 * gguf-py 0.19 table GGML_QUANT_SIZES.
 *   block_size = elements per block
 *   type_size  = bytes per block
 * IDs 4 and 5 (old Q4_2/Q4_3) were removed from GGML and are not listed.
 */
typedef struct {
    uint32_t id;
    const char *name;
    uint64_t block_size;
    uint64_t type_size;
} GGMLTypeInfo;

static const GGMLTypeInfo ggml_types[] = {
    {  0, "F32",       1,   4 },
    {  1, "F16",       1,   2 },
    {  2, "Q4_0",     32,  18 },
    {  3, "Q4_1",     32,  20 },
    {  6, "Q5_0",     32,  22 },
    {  7, "Q5_1",     32,  24 },
    {  8, "Q8_0",     32,  34 },
    {  9, "Q8_1",     32,  40 },
    { 10, "Q2_K",    256,  84 },
    { 11, "Q3_K",    256, 110 },
    { 12, "Q4_K",    256, 144 },
    { 13, "Q5_K",    256, 176 },
    { 14, "Q6_K",    256, 210 },
    { 15, "Q8_K",    256, 292 },
    { 16, "IQ2_XXS", 256,  66 },
    { 17, "IQ2_XS",  256,  74 },
    { 18, "IQ3_XXS", 256,  98 },
    { 19, "IQ1_S",   256,  50 },
    { 20, "IQ4_NL",   32,  18 },
    { 21, "IQ3_S",   256, 110 },
    { 22, "IQ2_S",   256,  82 },
    { 23, "IQ4_XS",  256, 136 },
    { 24, "I8",        1,   1 },
    { 25, "I16",       1,   2 },
    { 26, "I32",       1,   4 },
    { 27, "I64",       1,   8 },
    { 28, "F64",       1,   8 },
    { 29, "IQ1_M",   256,  56 },
    { 30, "BF16",      1,   2 }
};

static const GGMLTypeInfo *find_ggml_type(uint32_t type)
{
    for (size_t i = 0; i < sizeof(ggml_types) / sizeof(ggml_types[0]); i++) {
        if (ggml_types[i].id == type) {
            return &ggml_types[i];
        }
    }

    return NULL;
}

const char *gguf_ggml_type_name(uint32_t type)
{
    const GGMLTypeInfo *info = find_ggml_type(type);

    return info != NULL ? info->name : "UNKNOWN";
}

int gguf_ggml_type_info(
    uint32_t type,
    uint64_t *block_size,
    uint64_t *type_size
)
{
    const GGMLTypeInfo *info = find_ggml_type(type);

    if (info == NULL || block_size == NULL || type_size == NULL) {
        return 0;
    }

    *block_size = info->block_size;
    *type_size = info->type_size;

    return 1;
}

int gguf_tensor_storage_size(
    const GGUF_TensorInfo *tensor,
    uint64_t *storage_size
)
{
    if (tensor == NULL || storage_size == NULL ||
        tensor->n_dimensions == 0 ||
        tensor->n_dimensions > 4) {
        return 0;
    }

    uint64_t block_size = 1;
    uint64_t type_size = 0;

    if (!gguf_ggml_type_info(tensor->type, &block_size, &type_size)) {
        return 0;
    }

    uint64_t elements = 1;

    for (uint32_t i = 0; i < tensor->n_dimensions; i++) {

        uint64_t dimension = tensor->dimensions[i];

        if (dimension == 0 ||
            elements > UINT64_MAX / dimension) {
            return 0;
        }

        elements *= dimension;
    }

    /*
     * Quantized tensors must have complete blocks
     * along their first dimension.
     */
    if (tensor->dimensions[0] % block_size != 0 ||
        elements % block_size != 0) {
        return 0;
    }

    uint64_t blocks = elements / block_size;

    if (blocks > UINT64_MAX / type_size) {
        return 0;
    }

    *storage_size = blocks * type_size;

    return 1;
}

int gguf_get_tensor_data(
    const void *mapped_data,
    size_t file_size,
    const GGUFLayout *layout,
    const GGUF_TensorInfo *tensor,
    const void **tensor_data,
    uint64_t *tensor_size
)
{
    if (mapped_data == NULL ||
        layout == NULL ||
        tensor == NULL ||
        tensor_data == NULL ||
        tensor_size == NULL) {
        return 0;
    }

    *tensor_data = NULL;
    *tensor_size = 0;

    uint64_t file_offset;
    uint64_t storage_size;

    /* Calculate the tensor's file offset. */
    if (!gguf_tensor_file_offset(
            layout, tensor, &file_offset)) {
        return 0;
    }

    /* Calculate its storage size. */
    if (!gguf_tensor_storage_size(
            tensor, &storage_size)) {
        return 0;
    }

    /* Validate that the tensor is inside the mapped file. */
    if (file_offset > (uint64_t)file_size ||
        storage_size > (uint64_t)file_size - file_offset) {
        return 0;
    }

    /* Get the tensor's address inside mapped memory. */
    *tensor_data =
        (const unsigned char *)mapped_data + (size_t)file_offset;

    *tensor_size = storage_size;

    return 1;
}

const GGUF_TensorInfo *gguf_find_tensor(
    const GGUF_TensorInfo *tensors,
    size_t tensor_count,
    const char *name
)
{
    if (tensors == NULL || name == NULL) {
        return NULL;
    }

    for (size_t i = 0; i < tensor_count; i++) {
        if (tensors[i].name != NULL &&
            strcmp(tensors[i].name, name) == 0) {
            return &tensors[i];
        }
    }

    return NULL;
}