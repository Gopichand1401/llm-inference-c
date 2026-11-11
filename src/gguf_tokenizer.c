/*
 * gguf_tokenizer.c - build a Tokenizer from GGUF tokenizer metadata.
 *
 * GGUF "llama" tokenizers are SentencePiece BPE vocabularies stored as:
 *   tokenizer.ggml.model         string   "llama"
 *   tokenizer.ggml.tokens        string[] pieces; spaces written as U+2581 '▁'
 *   tokenizer.ggml.scores        float[]  merge priority (higher merges first)
 *   tokenizer.ggml.token_type    int32[]  1 normal, 2 unknown, 3 control,
 *                                         4 user-defined, 5 unused, 6 byte
 *   tokenizer.ggml.bos_token_id  uint32
 *   tokenizer.ggml.eos_token_id  uint32
 *
 * The result is the same Tokenizer struct the llama2.c loader produces, so
 * tokenizer_encode() / tokenizer_decode() work unchanged:
 *   - '▁' is converted to ' ' in every piece (as llama2.c's exporter did),
 *   - byte tokens "<0xXX>" fill tokenizer->byte_ids[],
 *   - control / unknown / unused / byte tokens get score -INFINITY, so the
 *     merge loop can never produce them from ordinary text,
 *   - unknown / control / user-defined tokens are listed in special_ids, so
 *     "<s>" or "</s>" written in the text becomes that token (as in the
 *     Hugging Face tokenizer),
 *   - if all normal-token scores are equal, the scores are rebuilt (below).
 *
 * Score reconstruction
 * --------------------
 * Some GGUF converters store every score as 0.0 (TinyLlama-1.1B-Chat Q4_K_M
 * is one). With equal scores the merge loop has no priorities and produces
 * different tokens from the original tokenizer. A SentencePiece BPE
 * vocabulary is stored in merge-priority order, and Llama's original
 * tokenizer.model uses exactly
 *     score(id) = -(id - 259)       for normal tokens
 *     score     = -1e9              for pieces made only of spaces ("▁▁" ...)
 * (checked against all 31741 normal tokens of the Llama-2 vocabulary).
 * When the file's scores carry no information we rebuild them that way:
 * score = -id, or -1e9 for all-space pieces. Only the ordering matters.
 *
 * The metadata is read through a bounds-checked cursor over a read-only
 * mmap() of the file; the file is unmapped before returning.
 */
#include "tokenizer.h"

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    GGUF_T_UINT8 = 0, GGUF_T_INT8 = 1, GGUF_T_UINT16 = 2, GGUF_T_INT16 = 3,
    GGUF_T_UINT32 = 4, GGUF_T_INT32 = 5, GGUF_T_FLOAT32 = 6, GGUF_T_BOOL = 7,
    GGUF_T_STRING = 8, GGUF_T_ARRAY = 9, GGUF_T_UINT64 = 10, GGUF_T_INT64 = 11,
    GGUF_T_FLOAT64 = 12
};

enum {
    TOKEN_NORMAL = 1, TOKEN_UNKNOWN = 2, TOKEN_CONTROL = 3,
    TOKEN_USER_DEFINED = 4, TOKEN_UNUSED = 5, TOKEN_BYTE = 6
};

/* ------------------------------------------------------------------ */
/* Bounds-checked cursor                                               */
/* ------------------------------------------------------------------ */

typedef struct {
    const unsigned char *data;
    size_t size;
    size_t pos;
    int failed;
} Cursor;

static int need(Cursor *c, size_t n)
{
    if (c->failed || n > c->size - c->pos) {
        c->failed = 1;
        return 0;
    }
    return 1;
}

static uint32_t read_u32(Cursor *c)
{
    if (!need(c, 4)) return 0;
    const unsigned char *b = c->data + c->pos;
    c->pos += 4;
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
           ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

static uint64_t read_u64(Cursor *c)
{
    uint64_t lo = read_u32(c);
    uint64_t hi = read_u32(c);
    return lo | (hi << 32);
}

static float read_f32(Cursor *c)
{
    uint32_t bits = read_u32(c);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/* A GGUF string: uint64 length + bytes (not NUL-terminated). */
static const char *read_str(Cursor *c, uint64_t *length)
{
    *length = read_u64(c);
    if (*length > SIZE_MAX || !need(c, (size_t)*length)) {
        return NULL;
    }
    const char *s = (const char *)(c->data + c->pos);
    c->pos += (size_t)*length;
    return s;
}

static size_t scalar_size(uint32_t type)
{
    switch (type) {
        case GGUF_T_UINT8: case GGUF_T_INT8: case GGUF_T_BOOL: return 1;
        case GGUF_T_UINT16: case GGUF_T_INT16: return 2;
        case GGUF_T_UINT32: case GGUF_T_INT32: case GGUF_T_FLOAT32: return 4;
        case GGUF_T_UINT64: case GGUF_T_INT64: case GGUF_T_FLOAT64: return 8;
        default: return 0;
    }
}

static void skip_value(Cursor *c, uint32_t type)
{
    if (type == GGUF_T_STRING) {
        uint64_t length;
        (void)read_str(c, &length);
    } else if (type == GGUF_T_ARRAY) {
        uint32_t element_type = read_u32(c);
        uint64_t count = read_u64(c);
        if (element_type == GGUF_T_ARRAY) {
            c->failed = 1;
            return;
        }
        if (element_type == GGUF_T_STRING) {
            for (uint64_t i = 0; i < count && !c->failed; i++) {
                skip_value(c, GGUF_T_STRING);
            }
        } else {
            size_t size = scalar_size(element_type);
            if (size == 0 || count > (c->size - c->pos) / size) {
                c->failed = 1;
                return;
            }
            c->pos += (size_t)count * size;
        }
    } else {
        size_t size = scalar_size(type);
        if (size == 0 || !need(c, size)) {
            c->failed = 1;
            return;
        }
        c->pos += size;
    }
}

/* Read an integer scalar of any integer type as int64. */
static int read_int_value(Cursor *c, uint32_t type, int64_t *out)
{
    switch (type) {
        case GGUF_T_UINT32: *out = (int64_t)read_u32(c); return !c->failed;
        case GGUF_T_INT32:  *out = (int64_t)(int32_t)read_u32(c); return !c->failed;
        case GGUF_T_UINT64: *out = (int64_t)read_u64(c); return !c->failed;
        case GGUF_T_INT64:  *out = (int64_t)read_u64(c); return !c->failed;
        default: skip_value(c, type); return 0;
    }
}

static int key_is(const char *key, uint64_t length, const char *name)
{
    return length == strlen(name) && memcmp(key, name, (size_t)length) == 0;
}

/* Copy a piece, converting every U+2581 (E2 96 81) into a plain space. */
static char *copy_piece(const char *s, uint64_t length)
{
    char *out = malloc((size_t)length + 1);
    size_t n = 0;

    if (out == NULL) {
        return NULL;
    }

    for (uint64_t i = 0; i < length; i++) {
        if (i + 2 < length &&
            (unsigned char)s[i] == 0xE2 &&
            (unsigned char)s[i + 1] == 0x96 &&
            (unsigned char)s[i + 2] == 0x81) {
            out[n++] = ' ';
            i += 2;
        } else {
            out[n++] = s[i];
        }
    }
    out[n] = '\0';
    return out;
}

static void set_error(char *error, size_t capacity, const char *message)
{
    if (error != NULL && capacity > 0) {
        (void)snprintf(error, capacity, "%s", message);
    }
}

static int compare_indices(const void *left, const void *right)
{
    const TokenizerIndex *a = left;
    const TokenizerIndex *b = right;
    return strcmp(a->text, b->text);
}

/* ------------------------------------------------------------------ */
/* Loader                                                              */
/* ------------------------------------------------------------------ */

int tokenizer_load_gguf(Tokenizer *tokenizer, const char *gguf_path,
                        char *error, size_t error_capacity)
{
    int fd = -1;
    void *mapped = MAP_FAILED;
    size_t file_size = 0;
    struct stat st;
    int32_t *types = NULL;
    size_t n_types = 0;
    size_t n_scores = 0;
    int have_model = 0;
    int64_t bos = 1, eos = 2;
    const char *message = "Invalid or truncated GGUF tokenizer metadata.";

    if (tokenizer == NULL || gguf_path == NULL) {
        set_error(error, error_capacity, "Invalid tokenizer arguments.");
        return -1;
    }

    memset(tokenizer, 0, sizeof(*tokenizer));

    fd = open(gguf_path, O_RDONLY);
    if (fd == -1 || fstat(fd, &st) != 0 || st.st_size <= 0) {
        message = "Cannot open GGUF file.";
        goto fail;
    }

    file_size = (size_t)st.st_size;
    mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        message = "Cannot memory-map GGUF file.";
        goto fail;
    }

    Cursor c = { mapped, file_size, 0, 0 };

    /* Header: magic, version, tensor count, metadata count. */
    if (!need(&c, 4) || memcmp(c.data, "GGUF", 4) != 0) {
        message = "Not a GGUF file.";
        goto fail;
    }
    c.pos = 4;

    uint32_t version = read_u32(&c);
    if (version < 2 || version > 3) {
        message = "Unsupported GGUF version.";
        goto fail;
    }
    (void)read_u64(&c);                 /* tensor count (unused here) */
    uint64_t n_kv = read_u64(&c);

    for (uint64_t k = 0; k < n_kv && !c.failed; k++) {
        uint64_t key_length;
        const char *key = read_str(&c, &key_length);
        uint32_t type = read_u32(&c);

        if (key == NULL || c.failed) {
            break;
        }

        if (key_is(key, key_length, "tokenizer.ggml.model")) {
            uint64_t length;
            const char *model = (type == GGUF_T_STRING)
                              ? read_str(&c, &length) : NULL;
            if (model == NULL || length != 5 || memcmp(model, "llama", 5) != 0) {
                message = "Only SentencePiece ('llama') GGUF tokenizers are supported.";
                goto fail;
            }
            have_model = 1;

        } else if (key_is(key, key_length, "tokenizer.ggml.tokens")) {
            if (type != GGUF_T_ARRAY || read_u32(&c) != GGUF_T_STRING) {
                goto fail;
            }
            uint64_t count = read_u64(&c);
            if (count < 1 || count > (uint64_t)INT32_MAX ||
                count > (c.size - c.pos) / 8) {
                goto fail;
            }

            tokenizer->vocab_size = (size_t)count;
            tokenizer->vocab = calloc((size_t)count, sizeof(*tokenizer->vocab));
            if (tokenizer->vocab == NULL) {
                message = "Memory allocation failed.";
                goto fail;
            }

            for (uint64_t i = 0; i < count; i++) {
                uint64_t length;
                const char *piece = read_str(&c, &length);
                if (piece == NULL || memchr(piece, '\0', (size_t)length) != NULL) {
                    goto fail;
                }
                tokenizer->vocab[i] = copy_piece(piece, length);
                if (tokenizer->vocab[i] == NULL) {
                    message = "Memory allocation failed.";
                    goto fail;
                }
                size_t n = strlen(tokenizer->vocab[i]);
                if (n > tokenizer->max_token_length) {
                    tokenizer->max_token_length = n;
                }
            }

        } else if (key_is(key, key_length, "tokenizer.ggml.scores")) {
            if (type != GGUF_T_ARRAY || read_u32(&c) != GGUF_T_FLOAT32) {
                goto fail;
            }
            uint64_t count = read_u64(&c);
            if (count > (c.size - c.pos) / 4) {
                goto fail;
            }
            free(tokenizer->scores);
            tokenizer->scores = malloc((size_t)(count ? count : 1) * sizeof(float));
            if (tokenizer->scores == NULL) {
                message = "Memory allocation failed.";
                goto fail;
            }
            for (uint64_t i = 0; i < count; i++) {
                tokenizer->scores[i] = read_f32(&c);
            }
            n_scores = (size_t)count;

        } else if (key_is(key, key_length, "tokenizer.ggml.token_type")) {
            if (type != GGUF_T_ARRAY || read_u32(&c) != GGUF_T_INT32) {
                goto fail;
            }
            uint64_t count = read_u64(&c);
            if (count > (c.size - c.pos) / 4) {
                goto fail;
            }
            free(types);
            types = malloc((size_t)(count ? count : 1) * sizeof(*types));
            if (types == NULL) {
                message = "Memory allocation failed.";
                goto fail;
            }
            for (uint64_t i = 0; i < count; i++) {
                types[i] = (int32_t)read_u32(&c);
            }
            n_types = (size_t)count;

        } else if (key_is(key, key_length, "tokenizer.ggml.bos_token_id")) {
            if (!read_int_value(&c, type, &bos)) goto fail;

        } else if (key_is(key, key_length, "tokenizer.ggml.eos_token_id")) {
            if (!read_int_value(&c, type, &eos)) goto fail;

        } else {
            skip_value(&c, type);
        }
    }

    if (c.failed) {
        goto fail;
    }

    if (!have_model || tokenizer->vocab == NULL || tokenizer->scores == NULL) {
        message = "GGUF file has no SentencePiece tokenizer metadata.";
        goto fail;
    }

    const size_t n = tokenizer->vocab_size;

    if (n_scores != n || (types != NULL && n_types != n)) {
        message = "Tokenizer arrays have different lengths.";
        goto fail;
    }

    if (bos < 0 || (size_t)bos >= n || eos < 0 || (size_t)eos >= n) {
        message = "BOS/EOS token ID is outside the vocabulary.";
        goto fail;
    }
    tokenizer->bos_id = (int)bos;
    tokenizer->eos_id = (int)eos;

    /* Do the normal-token scores carry any information? */
    {
        float lo = INFINITY;
        float hi = -INFINITY;
        for (size_t i = 0; i < n; i++) {
            const int32_t ty = (types != NULL) ? types[i] : TOKEN_NORMAL;
            if (ty == TOKEN_NORMAL) {
                if (tokenizer->scores[i] < lo) lo = tokenizer->scores[i];
                if (tokenizer->scores[i] > hi) hi = tokenizer->scores[i];
            }
        }

        if (lo == hi) {
            for (size_t i = 0; i < n; i++) {
                const char *p = tokenizer->vocab[i];
                int all_spaces = p[0] != '\0';
                for (const char *q = p; *q != '\0'; q++) {
                    if (*q != ' ') {
                        all_spaces = 0;
                        break;
                    }
                }
                tokenizer->scores[i] = all_spaces ? -1e9f : -(float)i;
            }
            tokenizer->scores_reconstructed = 1;
        }
    }

    /* Byte tokens, special tokens and merge exclusion. */
    for (int b = 0; b < 256; b++) {
        tokenizer->byte_ids[b] = -1;
    }

    tokenizer->special_ids = malloc(n * sizeof(*tokenizer->special_ids));
    if (tokenizer->special_ids == NULL) {
        message = "Memory allocation failed.";
        goto fail;
    }

    for (size_t i = 0; i < n; i++) {
        const char *p = tokenizer->vocab[i];
        const int32_t ty = (types != NULL) ? types[i] : TOKEN_NORMAL;

        if (!isfinite(tokenizer->scores[i])) {
            message = "Tokenizer score is not finite.";
            goto fail;
        }

        int is_byte_name = strlen(p) == 6 && p[0] == '<' && p[1] == '0' &&
                           p[2] == 'x' && p[5] == '>';

        if (is_byte_name && (types == NULL || ty == TOKEN_BYTE)) {
            char hex[3] = { p[3], p[4], '\0' };
            char *end = NULL;
            unsigned long value = strtoul(hex, &end, 16);
            if (end == hex + 2 && value < 256) {
                tokenizer->byte_ids[value] = (int)i;
            }
        }

        if (ty == TOKEN_UNKNOWN || ty == TOKEN_CONTROL ||
            ty == TOKEN_UNUSED || ty == TOKEN_BYTE ||
            (types == NULL && is_byte_name)) {
            tokenizer->scores[i] = -INFINITY;
        }

        if ((ty == TOKEN_UNKNOWN || ty == TOKEN_CONTROL ||
             ty == TOKEN_USER_DEFINED) && p[0] != '\0') {
            tokenizer->special_ids[tokenizer->n_special++] = (int)i;
        }
    }

    tokenizer->sorted_vocab = malloc(n * sizeof(*tokenizer->sorted_vocab));
    if (tokenizer->sorted_vocab == NULL) {
        message = "Memory allocation failed.";
        goto fail;
    }
    for (size_t i = 0; i < n; i++) {
        tokenizer->sorted_vocab[i].text = tokenizer->vocab[i];
        tokenizer->sorted_vocab[i].id = (int)i;
    }
    qsort(tokenizer->sorted_vocab, n, sizeof(*tokenizer->sorted_vocab),
          compare_indices);

    for (size_t i = 0; i < 256; i++) {
        tokenizer->byte_pieces[i * 2] = (unsigned char)i;
        tokenizer->byte_pieces[i * 2 + 1] = '\0';
    }

    free(types);
    munmap(mapped, file_size);
    close(fd);

    if (error != NULL && error_capacity > 0) {
        error[0] = '\0';
    }
    return 0;

fail:
    set_error(error, error_capacity, message);
    free(types);
    tokenizer_free(tokenizer);
    if (mapped != MAP_FAILED) {
        munmap(mapped, file_size);
    }
    if (fd != -1) {
        close(fd);
    }
    return -1;
}
