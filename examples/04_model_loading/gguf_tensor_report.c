/*
 * gguf_tensor_report.c - list every tensor in a GGUF file and check it.
 *
 * Usage: gguf_tensor_report <model.gguf>
 *
 * For each tensor it reports: name, GGUF dimensions, GGML type, storage size,
 * the LLaMA role implied by its name, whether its shape matches the model
 * configuration, and whether this engine can decode its type yet.
 *
 * It also checks the file as a whole:
 *   - every tensor's absolute offset is aligned and inside the file
 *   - no two tensors overlap
 *   - every layer has all nine LLaMA tensors, plus the three global ones
 *
 * A CSV copy is written to build/gguf_tensor_report.csv so that
 * examples/04_model_loading/compare_tensor_report.py can cross-check it against gguf-py.
 */
#include "checkpoint.h"
#include "transformer.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* ------------------------------------------------------------------ */
/* LLaMA tensor roles                                                  */
/* ------------------------------------------------------------------ */

typedef enum {
    ROLE_UNKNOWN = 0,
    ROLE_TOKEN_EMBD,
    ROLE_OUTPUT_NORM,
    ROLE_OUTPUT,
    ROLE_ATTN_NORM,
    ROLE_ATTN_Q,
    ROLE_ATTN_K,
    ROLE_ATTN_V,
    ROLE_ATTN_OUTPUT,
    ROLE_FFN_NORM,
    ROLE_FFN_GATE,
    ROLE_FFN_UP,
    ROLE_FFN_DOWN,
    ROLE_COUNT
} TensorRole;

static const char *role_names[ROLE_COUNT] = {
    "unknown",
    "token embeddings",
    "final RMSNorm",
    "vocab projection",
    "attention RMSNorm",
    "query proj (wq)",
    "key proj (wk)",
    "value proj (wv)",
    "attn output (wo)",
    "FFN RMSNorm",
    "FFN gate (w1)",
    "FFN up (w3)",
    "FFN down (w2)"
};

/* Per-layer suffixes, in the same order as ROLE_ATTN_NORM..ROLE_FFN_DOWN. */
static const char *layer_suffixes[] = {
    "attn_norm.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight"
};

#define LAYER_ROLE_COUNT 9

/*
 * Work out a tensor's role from its name.
 * Per-layer names look like "blk.<layer>.<suffix>".
 * Sets *layer to the layer index, or -1 for global tensors.
 */
static TensorRole classify(const char *name, long *layer)
{
    *layer = -1;

    if (strcmp(name, "token_embd.weight") == 0)  return ROLE_TOKEN_EMBD;
    if (strcmp(name, "output_norm.weight") == 0) return ROLE_OUTPUT_NORM;
    if (strcmp(name, "output.weight") == 0)      return ROLE_OUTPUT;

    if (strncmp(name, "blk.", 4) != 0) {
        return ROLE_UNKNOWN;
    }

    char *end = NULL;
    long index = strtol(name + 4, &end, 10);

    if (end == name + 4 || *end != '.' || index < 0) {
        return ROLE_UNKNOWN;
    }

    for (int i = 0; i < LAYER_ROLE_COUNT; i++) {
        if (strcmp(end + 1, layer_suffixes[i]) == 0) {
            *layer = index;
            return (TensorRole)(ROLE_ATTN_NORM + i);
        }
    }

    return ROLE_UNKNOWN;
}

/*
 * Expected GGUF dimensions for a role. GGUF lists ne0 first, and ne0 is the
 * contiguous (input) dimension of a weight matrix, so a matrix that maps
 * n inputs to d outputs is stored as [n, d] = d rows of n values.
 */
static int expected_shape(TensorRole role, const Config *c,
                          uint32_t *n_dims, uint64_t dims[2])
{
    const uint64_t dim = (uint64_t)c->dim;
    const uint64_t hidden = (uint64_t)c->hidden_dim;
    const uint64_t vocab = (uint64_t)c->vocab_size;
    const uint64_t head_size = dim / (uint64_t)c->n_heads;
    const uint64_t kv_dim = head_size * (uint64_t)c->n_kv_heads;

    switch (role) {
        case ROLE_OUTPUT_NORM:
        case ROLE_ATTN_NORM:
        case ROLE_FFN_NORM:
            *n_dims = 1; dims[0] = dim; dims[1] = 0; return 1;

        case ROLE_TOKEN_EMBD:
        case ROLE_OUTPUT:
            *n_dims = 2; dims[0] = dim; dims[1] = vocab; return 1;

        case ROLE_ATTN_Q:
        case ROLE_ATTN_OUTPUT:
            *n_dims = 2; dims[0] = dim; dims[1] = dim; return 1;

        case ROLE_ATTN_K:
        case ROLE_ATTN_V:
            *n_dims = 2; dims[0] = dim; dims[1] = kv_dim; return 1;

        case ROLE_FFN_GATE:
        case ROLE_FFN_UP:
            *n_dims = 2; dims[0] = dim; dims[1] = hidden; return 1;

        case ROLE_FFN_DOWN:
            *n_dims = 2; dims[0] = hidden; dims[1] = dim; return 1;

        default:
            return 0;
    }
}

/* What this engine can decode today (src/gguf_quant.c, gguf_model.c). */
static const char *decode_status(uint32_t type)
{
    switch (type) {
        case 0:  /* F32: used as-is */
        case 1:  /* F16 */
        case 2:  /* Q4_0 */
        case 8:  /* Q8_0 */
        case 12: /* Q4_K */
        case 13: /* Q5_K */
        case 14: /* Q6_K */
            return "yes";
        default:
            return "NO";
    }
}

/* ------------------------------------------------------------------ */
/* Overlap check helpers                                               */
/* ------------------------------------------------------------------ */

typedef struct {
    uint64_t start;
    uint64_t end;
    size_t index;
} Extent;

static int compare_extents(const void *a, const void *b)
{
    const Extent *x = a;
    const Extent *y = b;

    if (x->start < y->start) return -1;
    if (x->start > y->start) return 1;
    return 0;
}

static void format_dims(const GGUF_TensorInfo *t, char *out, size_t cap)
{
    size_t used = 0;

    out[0] = '\0';

    for (uint32_t d = 0; d < t->n_dimensions && used < cap; d++) {
        int n = snprintf(out + used, cap - used, "%s%" PRIu64,
                         d == 0 ? "" : " x ", t->dimensions[d]);
        if (n < 0) {
            return;
        }
        used += (size_t)n;
    }
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    const char *path = argv[1];
    char error[256] = {0};
    int status = 1;

    GGUF_TensorInfo *tensors = NULL;
    size_t count = 0;
    GGUFLayout layout = {0};
    Config config = {0};
    struct stat st;

    Extent *extents = NULL;
    unsigned char *seen = NULL;
    FILE *csv = NULL;

    if (stat(path, &st) != 0 || st.st_size <= 0) {
        perror(path);
        return 1;
    }

    const uint64_t file_size = (uint64_t)st.st_size;

    if (!gguf_read_config(path, &config, error, sizeof(error))) {
        fprintf(stderr, "Config: %s\n", error);
        goto cleanup;
    }

    if (!gguf_read_tensors(path, &tensors, &count, error, sizeof(error))) {
        fprintf(stderr, "Tensors: %s\n", error);
        goto cleanup;
    }

    if (!gguf_read_layout(path, &layout, error, sizeof(error))) {
        fprintf(stderr, "Layout: %s\n", error);
        goto cleanup;
    }

    printf("=== GGUF Tensor Report ===\n");
    printf("File: %s (%" PRIu64 " bytes)\n", path, file_size);
    printf("Config: dim=%d hidden=%d layers=%d heads=%d kv_heads=%d "
           "vocab=%d ctx=%d\n",
           config.dim, config.hidden_dim, config.n_layers, config.n_heads,
           config.n_kv_heads, config.vocab_size, config.seq_len);
    printf("Tensors: %zu  alignment: %u  data offset: %" PRIu64 "\n\n",
           count, layout.alignment, layout.data_offset);

    csv = fopen("build/gguf_tensor_report.csv", "w");
    if (csv == NULL) {
        perror("build/gguf_tensor_report.csv");
        goto cleanup;
    }

    fprintf(csv, "index,name,type_id,type_name,n_dims,ne0,ne1,ne2,ne3,"
                 "storage_bytes,abs_offset,role,layer,shape_ok,decode\n");

    extents = calloc(count == 0 ? 1 : count, sizeof(*extents));
    seen = calloc((size_t)config.n_layers * LAYER_ROLE_COUNT + 3, 1);

    if (extents == NULL || seen == NULL) {
        fprintf(stderr, "Out of memory.\n");
        goto cleanup;
    }

    uint64_t type_counts[64] = {0};
    uint64_t type_bytes[64] = {0};
    uint64_t total_bytes = 0;
    int bad_size = 0, bad_shape = 0, unknown_roles = 0;
    int bad_layer = 0, duplicates = 0, undecodable = 0;

    printf("%4s  %-28s %-6s %-14s %12s  %-18s %-5s %s\n",
           "#", "name", "type", "dims(ne0 x ne1)", "bytes",
           "role", "shape", "decode");

    for (size_t i = 0; i < count; i++) {
        const GGUF_TensorInfo *t = &tensors[i];
        uint64_t size = 0, abs_offset = 0;
        char dims[64];

        int size_ok = gguf_tensor_storage_size(t, &size) &&
                      gguf_tensor_file_offset(&layout, t, &abs_offset);

        if (!size_ok) {
            bad_size++;
        }

        long layer = -1;
        TensorRole role = classify(t->name, &layer);

        /* Shape check against the configuration. */
        const char *shape_text = "?";
        uint32_t exp_n = 0;
        uint64_t exp_dims[2] = {0, 0};

        if (role == ROLE_UNKNOWN) {
            unknown_roles++;
        } else if (layer >= config.n_layers) {
            bad_layer++;
            shape_text = "LAYER";
        } else if (expected_shape(role, &config, &exp_n, exp_dims)) {
            int ok = t->n_dimensions == exp_n;
            for (uint32_t d = 0; ok && d < exp_n; d++) {
                ok = t->dimensions[d] == exp_dims[d];
            }
            shape_text = ok ? "ok" : "BAD";
            if (!ok) {
                bad_shape++;
            }

            /* Record coverage: slot per (layer, role) or per global role. */
            size_t slot;
            if (layer >= 0) {
                slot = (size_t)layer * LAYER_ROLE_COUNT +
                       (size_t)(role - ROLE_ATTN_NORM);
            } else {
                slot = (size_t)config.n_layers * LAYER_ROLE_COUNT +
                       (size_t)(role - ROLE_TOKEN_EMBD);
            }
            if (seen[slot]) {
                duplicates++;
            }
            seen[slot] = 1;
        }

        const char *decode = decode_status(t->type);
        if (strcmp(decode, "yes") != 0) {
            undecodable++;
        }

        if (t->type < 64) {
            type_counts[t->type]++;
            type_bytes[t->type] += size;
        }
        total_bytes += size;

        extents[i].start = abs_offset;
        extents[i].end = abs_offset + size;
        extents[i].index = i;

        format_dims(t, dims, sizeof(dims));

        printf("%4zu  %-28s %-6s %-14s %12" PRIu64 "  %-18s %-5s %s\n",
               i, t->name, gguf_ggml_type_name(t->type), dims, size,
               role_names[role], shape_text, decode);

        fprintf(csv, "%zu,%s,%u,%s,%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64
                     ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%s,%ld,%s,%s\n",
                i, t->name, t->type, gguf_ggml_type_name(t->type),
                t->n_dimensions,
                t->dimensions[0],
                t->n_dimensions > 1 ? t->dimensions[1] : 1,
                t->n_dimensions > 2 ? t->dimensions[2] : 1,
                t->n_dimensions > 3 ? t->dimensions[3] : 1,
                size, abs_offset, role_names[role], layer,
                shape_text, decode);
    }

    /* ---- File-level checks ---- */
    int misaligned = 0, out_of_file = 0, overlaps = 0;

    qsort(extents, count, sizeof(*extents), compare_extents);

    for (size_t i = 0; i < count; i++) {
        if (extents[i].start % layout.alignment != 0) {
            misaligned++;
        }
        if (extents[i].end > file_size || extents[i].end < extents[i].start) {
            out_of_file++;
        }
        if (i + 1 < count && extents[i].end > extents[i + 1].start) {
            overlaps++;
            printf("OVERLAP: %s and %s\n",
                   tensors[extents[i].index].name,
                   tensors[extents[i + 1].index].name);
        }
    }

    int missing = 0;
    for (int l = 0; l < config.n_layers; l++) {
        for (int r = 0; r < LAYER_ROLE_COUNT; r++) {
            if (!seen[(size_t)l * LAYER_ROLE_COUNT + (size_t)r]) {
                printf("MISSING: blk.%d.%s\n", l, layer_suffixes[r]);
                missing++;
            }
        }
    }

    const char *global_names[3] = {
        "token_embd.weight", "output_norm.weight", "output.weight"
    };
    for (int g = 0; g < 3; g++) {
        if (!seen[(size_t)config.n_layers * LAYER_ROLE_COUNT + (size_t)g]) {
            /* output.weight may be absent when it is tied to token_embd. */
            printf("MISSING: %s%s\n", global_names[g],
                   g == 2 ? " (would mean tied embeddings)" : "");
            missing++;
        }
    }

    /* ---- Summary ---- */
    printf("\n=== Types used ===\n");
    for (uint32_t ty = 0; ty < 64; ty++) {
        if (type_counts[ty] != 0) {
            printf("  %-6s (id %2u): %3" PRIu64 " tensors, %12" PRIu64
                   " bytes, decode: %s\n",
                   gguf_ggml_type_name(ty), ty, type_counts[ty],
                   type_bytes[ty], decode_status(ty));
        }
    }

    printf("\n=== Checks ===\n");
    printf("Total tensor bytes:        %" PRIu64 "\n", total_bytes);
    printf("Expected tensor count:     %d (9 x %d layers + 3)\n",
           LAYER_ROLE_COUNT * config.n_layers + 3, config.n_layers);
    printf("Unknown storage size:      %d\n", bad_size);
    printf("Misaligned offsets:        %d\n", misaligned);
    printf("Outside file:              %d\n", out_of_file);
    printf("Overlapping tensors:       %d\n", overlaps);
    printf("Unrecognised names:        %d\n", unknown_roles);
    printf("Layer index out of range:  %d\n", bad_layer);
    printf("Duplicate roles:           %d\n", duplicates);
    printf("Missing tensors:           %d\n", missing);
    printf("Shape mismatches:          %d\n", bad_shape);
    printf("Types we cannot decode:    %d tensors\n", undecodable);
    printf("Usable in inference now:   0 / %zu "
           "(GGUF model loading is Phase 3)\n", count);
    printf("CSV written to build/gguf_tensor_report.csv\n");

    if (bad_size == 0 && misaligned == 0 && out_of_file == 0 &&
        overlaps == 0 && unknown_roles == 0 && bad_layer == 0 &&
        duplicates == 0 && missing == 0 && bad_shape == 0) {
        printf("\nStructure check: PASSED\n");
        if (undecodable == 0) {
            printf("All tensor types are decodable: YES\n");
        } else {
            printf("All tensor types are decodable: NO (see 'Types used')\n");
        }
        status = 0;
    } else {
        printf("\nStructure check: FAILED\n");
    }

cleanup:
    if (csv != NULL) {
        fclose(csv);
    }
    free(extents);
    free(seen);

    if (tensors != NULL) {
        for (size_t i = 0; i < count; i++) {
            free(tensors[i].name);
        }
        free(tensors);
    }

    return status;
}
