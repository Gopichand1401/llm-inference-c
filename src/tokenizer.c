#include "tokenizer.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static int lookup(const Tokenizer *tokenizer, const char *text)
{
    TokenizerIndex key;
    TokenizerIndex *found;

    key.text = (char *)text;
    key.id = -1;
    found = bsearch(&key, tokenizer->sorted_vocab,
                    tokenizer->vocab_size, sizeof(*tokenizer->sorted_vocab),
                    compare_indices);
    return found == NULL ? -1 : found->id;
}

void tokenizer_free(Tokenizer *tokenizer)
{
    if (tokenizer == NULL) {
        return;
    }

    if (tokenizer->vocab != NULL) {
        for (size_t i = 0; i < tokenizer->vocab_size; ++i) {
            free(tokenizer->vocab[i]);
        }
    }
    free(tokenizer->vocab);
    free(tokenizer->scores);
    free(tokenizer->sorted_vocab);
    free(tokenizer->special_ids);
    memset(tokenizer, 0, sizeof(*tokenizer));
}

int tokenizer_load(Tokenizer *tokenizer, const char *path,
                   size_t vocab_size, char *error, size_t error_capacity)
{
    FILE *file = NULL;
    uint32_t max_length_u32;
    size_t loaded = 0;

    if (tokenizer == NULL || path == NULL || path[0] == '\0' ||
        vocab_size < 259 || vocab_size > (size_t)INT32_MAX) {
        set_error(error, error_capacity, "Invalid tokenizer arguments.");
        return -1;
    }

    memset(tokenizer, 0, sizeof(*tokenizer));
    if (error != NULL && error_capacity > 0) {
        error[0] = '\0';
    }

    file = fopen(path, "rb");
    if (file == NULL) {
        set_error(error, error_capacity, "Cannot open tokenizer file.");
        return -1;
    }

    if (fread(&max_length_u32, sizeof(max_length_u32), 1, file) != 1 ||
        max_length_u32 == 0 || max_length_u32 > 1048576U) {
        set_error(error, error_capacity, "Invalid tokenizer header.");
        (void)fclose(file);
        return -1;
    }

    tokenizer->vocab_size = vocab_size;
    tokenizer->max_token_length = (size_t)max_length_u32;
    tokenizer->vocab = calloc(vocab_size, sizeof(*tokenizer->vocab));
    tokenizer->scores = calloc(vocab_size, sizeof(*tokenizer->scores));
    if (tokenizer->vocab == NULL || tokenizer->scores == NULL) {
        set_error(error, error_capacity, "Memory allocation failed.");
        (void)fclose(file);
        tokenizer_free(tokenizer);
        return -1;
    }

    for (size_t i = 0; i < vocab_size; ++i) {
        uint32_t length;
        char *piece;

        if (fread(&tokenizer->scores[i], sizeof(float), 1, file) != 1 ||
            !isfinite(tokenizer->scores[i]) ||
            fread(&length, sizeof(length), 1, file) != 1 ||
            length == 0 || length > max_length_u32) {
            set_error(error, error_capacity,
                      "Tokenizer file is truncated or contains invalid entries.");
            (void)fclose(file);
            tokenizer_free(tokenizer);
            return -1;
        }

        piece = malloc((size_t)length + 1);
        if (piece == NULL) {
            set_error(error, error_capacity, "Memory allocation failed.");
            (void)fclose(file);
            tokenizer_free(tokenizer);
            return -1;
        }
        if (fread(piece, 1, length, file) != (size_t)length) {
            free(piece);
            set_error(error, error_capacity, "Tokenizer file is truncated.");
            (void)fclose(file);
            tokenizer_free(tokenizer);
            return -1;
        }
        piece[length] = '\0';
        if (memchr(piece, '\0', length) != NULL) {
            free(piece);
            set_error(error, error_capacity,
                      "Tokenizer entry contains an embedded NUL byte.");
            (void)fclose(file);
            tokenizer_free(tokenizer);
            return -1;
        }
        tokenizer->vocab[i] = piece;
        ++loaded;
    }

    if (fgetc(file) != EOF) {
        set_error(error, error_capacity,
                  "Tokenizer file has unexpected trailing data.");
        (void)fclose(file);
        tokenizer_free(tokenizer);
        return -1;
    }
    if (ferror(file) != 0) {
        set_error(error, error_capacity, "Error reading tokenizer file.");
        (void)fclose(file);
        tokenizer_free(tokenizer);
        return -1;
    }
    if (fclose(file) != 0) {
        set_error(error, error_capacity, "Cannot close tokenizer file.");
        tokenizer_free(tokenizer);
        return -1;
    }

    tokenizer->sorted_vocab =
        malloc(vocab_size * sizeof(*tokenizer->sorted_vocab));
    if (tokenizer->sorted_vocab == NULL) {
        set_error(error, error_capacity, "Memory allocation failed.");
        tokenizer_free(tokenizer);
        return -1;
    }

    for (size_t i = 0; i < vocab_size; ++i) {
        tokenizer->sorted_vocab[i].text = tokenizer->vocab[i];
        tokenizer->sorted_vocab[i].id = (int)i;
    }
    qsort(tokenizer->sorted_vocab, vocab_size,
          sizeof(*tokenizer->sorted_vocab), compare_indices);

    for (size_t i = 0; i < 256; ++i) {
        tokenizer->byte_pieces[i * 2] = (unsigned char)i;
        tokenizer->byte_pieces[i * 2 + 1] = '\0';
    }

    /* These IDs are fixed by the exported llama2.c tokenizer format. */
    if (strcmp(tokenizer->vocab[TOKENIZER_BOS_ID], "\n<s>\n") != 0 ||
        strcmp(tokenizer->vocab[TOKENIZER_EOS_ID], "\n</s>\n") != 0) {
        set_error(error, error_capacity,
                  "Tokenizer special-token IDs do not match expected format.");
        tokenizer_free(tokenizer);
        return -1;
    }

    tokenizer->bos_id = TOKENIZER_BOS_ID;
    tokenizer->eos_id = TOKENIZER_EOS_ID;

    /* llama2.c files store byte tokens <0x00>..<0xFF> at IDs 3..258. */
    for (int b = 0; b < 256; ++b) {
        tokenizer->byte_ids[b] = b + 3;
    }

    (void)loaded;
    return 0;
}

static int append_token(int *tokens, size_t capacity, size_t *count, int id)
{
    if (*count >= capacity) {
        return -1;
    }
    tokens[(*count)++] = id;
    return 0;
}

typedef struct {
    float score;
    size_t left;      /* index of the left token (list position order) */
    size_t right;
    int left_id;      /* IDs when the entry was made; stale if changed */
    int right_id;
    int merged_id;
} MergeCandidate;

/* 1 if a should be merged before b. */
static int merge_before(const MergeCandidate *a, const MergeCandidate *b)
{
    if (a->score != b->score) {
        return a->score > b->score;
    }
    return a->left < b->left;
}

static void heap_push(MergeCandidate *heap, size_t *size, MergeCandidate item)
{
    size_t i = (*size)++;
    while (i > 0) {
        size_t parent = (i - 1) / 2;
        if (!merge_before(&item, &heap[parent])) {
            break;
        }
        heap[i] = heap[parent];
        i = parent;
    }
    heap[i] = item;
}

static MergeCandidate heap_pop(MergeCandidate *heap, size_t *size)
{
    MergeCandidate top = heap[0];
    MergeCandidate last = heap[--*size];
    size_t i = 0;
    for (;;) {
        size_t child = 2 * i + 1;
        if (child >= *size) {
            break;
        }
        if (child + 1 < *size && merge_before(&heap[child + 1], &heap[child])) {
            ++child;
        }
        if (!merge_before(&heap[child], &last)) {
            break;
        }
        heap[i] = heap[child];
        i = child;
    }
    if (*size > 0) {
        heap[i] = last;
    }
    return top;
}

/* Push the pair (left, right) if its concatenation is a vocabulary piece. */
static void push_pair(const Tokenizer *tokenizer, const int *ids,
                      size_t left, size_t right,
                      MergeCandidate *heap, size_t *heap_size,
                      char *buffer, size_t buffer_capacity)
{
    const char *l = tokenizer->vocab[ids[left]];
    const char *r = tokenizer->vocab[ids[right]];
    size_t left_len = strlen(l);
    size_t right_len = strlen(r);

    if (left_len > buffer_capacity - right_len - 1) {
        return;
    }
    memcpy(buffer, l, left_len);
    memcpy(buffer + left_len, r, right_len + 1);

    int merged = lookup(tokenizer, buffer);
    if (merged < 0) {
        return;
    }
    MergeCandidate c;
    c.score = tokenizer->scores[merged];
    c.left = left;
    c.right = right;
    c.left_id = ids[left];
    c.right_id = ids[right];
    c.merged_id = merged;
    heap_push(heap, heap_size, c);
}

/*
 * BPE merge over ids[0..n-1] (see encode_segment). On return the merged
 * tokens are compacted to the front of ids and *count = first + their number.
 */
static int merge_pairs(const Tokenizer *tokenizer, int *ids, size_t n,
                       size_t *count, size_t first,
                       char *buffer, size_t buffer_capacity,
                       char *error, size_t error_capacity)
{
    if (n < 2) {
        *count = first + n;
        return 0;
    }

    /* Every merge adds at most 2 new pairs, so 3n entries always suffice. */
    if (n > SIZE_MAX / (3 * sizeof(MergeCandidate))) {
        set_error(error, error_capacity, "Text too long to encode.");
        return -1;
    }
    size_t *next = malloc(n * sizeof(*next));
    size_t *prev = malloc(n * sizeof(*prev));
    unsigned char *alive = malloc(n);
    MergeCandidate *heap = malloc(3 * n * sizeof(*heap));
    if (next == NULL || prev == NULL || alive == NULL || heap == NULL) {
        free(next);
        free(prev);
        free(alive);
        free(heap);
        set_error(error, error_capacity, "Out of memory while encoding.");
        return -1;
    }

    const size_t none = SIZE_MAX;
    size_t heap_size = 0;
    for (size_t i = 0; i < n; ++i) {
        prev[i] = i == 0 ? none : i - 1;
        next[i] = i + 1 < n ? i + 1 : none;
        alive[i] = 1;
    }
    for (size_t i = 0; i + 1 < n; ++i) {
        push_pair(tokenizer, ids, i, i + 1, heap, &heap_size,
                  buffer, buffer_capacity);
    }

    while (heap_size > 0) {
        MergeCandidate c = heap_pop(heap, &heap_size);
        if (!alive[c.left] || !alive[c.right] || next[c.left] != c.right ||
            ids[c.left] != c.left_id || ids[c.right] != c.right_id) {
            continue;   /* stale: one side has been merged since */
        }

        /* Merge right into left. */
        ids[c.left] = c.merged_id;
        alive[c.right] = 0;
        next[c.left] = next[c.right];
        if (next[c.right] != none) {
            prev[next[c.right]] = c.left;
        }

        if (prev[c.left] != none) {
            push_pair(tokenizer, ids, prev[c.left], c.left, heap, &heap_size,
                      buffer, buffer_capacity);
        }
        if (next[c.left] != none) {
            push_pair(tokenizer, ids, c.left, next[c.left], heap, &heap_size,
                      buffer, buffer_capacity);
        }
    }

    size_t out = 0;
    for (size_t i = 0; i < n; ++i) {
        if (alive[i]) {
            ids[out++] = ids[i];
        }
    }
    *count = first + out;

    free(next);
    free(prev);
    free(alive);
    free(heap);
    return 0;
}

/*
 * Encode one plain-text segment (no special tokens inside) and append its
 * tokens at tokens[*count...]. SentencePiece style:
 *   1. a dummy leading space for a non-empty segment,
 *   2. one token per UTF-8 character (byte-fallback tokens when a character
 *      is not in the vocabulary),
 *   3. repeatedly merge the adjacent pair whose merged piece has the highest
 *      score (leftmost on ties) until no pair can be merged.
 * Merges only look at this segment's tokens, never across a special token.
 */
static int encode_segment(Tokenizer *tokenizer,
                          const char *text, size_t text_length,
                          int *tokens, size_t token_capacity, size_t *count,
                          char *buffer, size_t buffer_capacity,
                          char *error, size_t error_capacity)
{
    if (text_length == 0) {
        return 0;
    }

    /* SentencePiece uses a dummy leading space for non-empty input. */
    int space_id = lookup(tokenizer, " ");
    if (space_id < 0) {
        set_error(error, error_capacity, "Tokenizer vocabulary has no space token.");
        return -1;
    }

    /*
     * Unmerged tokens are staged in a private array (at most one per input
     * byte, plus the space), so the caller's buffer only has to hold the
     * final, merged result.
     */
    if (text_length > SIZE_MAX / sizeof(int) - 1) {
        set_error(error, error_capacity, "Text too long to encode.");
        return -1;
    }
    int *work = malloc((text_length + 1) * sizeof(*work));
    if (work == NULL) {
        set_error(error, error_capacity, "Out of memory while encoding.");
        return -1;
    }
    size_t n = 0;
    work[n++] = space_id;

    size_t i = 0;
    while (i < text_length) {
        size_t start = i;
        unsigned char lead = (unsigned char)text[i];
        size_t char_length = 1;

        if ((lead & 0xE0U) == 0xC0U) {
            char_length = 2;
        } else if ((lead & 0xF0U) == 0xE0U) {
            char_length = 3;
        } else if ((lead & 0xF8U) == 0xF0U) {
            char_length = 4;
        }
        if (char_length > text_length - i) {
            char_length = 1;
        } else {
            for (size_t j = 1; j < char_length; ++j) {
                if (((unsigned char)text[i + j] & 0xC0U) != 0x80U) {
                    char_length = 1;
                    break;
                }
            }
        }

        memcpy(buffer, text + start, char_length);
        buffer[char_length] = '\0';
        int id = lookup(tokenizer, buffer);
        if (id >= 0) {
            work[n++] = id;
        } else {
            for (size_t j = 0; j < char_length; ++j) {
                int byte_id =
                    tokenizer->byte_ids[(unsigned char)text[start + j]];
                if (byte_id < 0 || (size_t)byte_id >= tokenizer->vocab_size) {
                    free(work);
                    set_error(error, error_capacity,
                              "Byte fallback is unavailable for this text.");
                    return -1;
                }
                work[n++] = byte_id;
            }
        }
        i += char_length;
    }

    /*
     * Merge the adjacent pair with the highest score (leftmost on ties)
     * until no merge remains.
     *
     * The straightforward way rescans every pair after each merge, which is
     * O(n^2) lookups: a 20 KB prompt took ~15 s. merge_pairs() keeps the
     * tokens in a doubly linked list and every mergeable pair in a max-heap
     * ordered by (score desc, position asc), so each step picks the same
     * pair as the full rescan would and the output is identical, in
     * O(n log n).
     */
    size_t merged = 0;
    if (merge_pairs(tokenizer, work, n, &merged, 0, buffer, buffer_capacity,
                    error, error_capacity) != 0) {
        free(work);
        return -1;
    }
    for (size_t k = 0; k < merged; ++k) {
        if (append_token(tokens, token_capacity, count, work[k]) != 0) {
            free(work);
            set_error(error, error_capacity, "Token output buffer is too small.");
            return -1;
        }
    }
    free(work);
    return 0;
}

/*
 * Find the earliest special-token string in text (longest one on ties).
 * Returns its position in *match_pos, its length in *match_length and its
 * token ID; returns -1 if the text contains no special token.
 */
static int find_special(const Tokenizer *tokenizer, const char *text,
                        size_t *match_pos, size_t *match_length)
{
    int best_id = -1;
    size_t best_pos = 0;
    size_t best_length = 0;

    for (size_t k = 0; k < tokenizer->n_special; ++k) {
        const int id = tokenizer->special_ids[k];
        const char *piece = tokenizer->vocab[id];
        const size_t length = strlen(piece);

        if (length == 0) {
            continue;
        }

        const char *found = strstr(text, piece);
        if (found == NULL) {
            continue;
        }

        size_t pos = (size_t)(found - text);
        if (best_id < 0 || pos < best_pos ||
            (pos == best_pos && length > best_length)) {
            best_id = id;
            best_pos = pos;
            best_length = length;
        }
    }

    *match_pos = best_pos;
    *match_length = best_length;
    return best_id;
}

int tokenizer_encode(Tokenizer *tokenizer, const char *text,
                     int add_bos, int add_eos,
                     int *tokens, size_t token_capacity,
                     size_t *token_count,
                     char *error, size_t error_capacity)
{
    char *buffer = NULL;
    size_t buffer_capacity;
    size_t count = 0;
    size_t text_length;

    if (token_count != NULL) {
        *token_count = 0;
    }
    if (tokenizer == NULL || tokenizer->vocab == NULL ||
        tokenizer->sorted_vocab == NULL || text == NULL ||
        tokens == NULL || token_count == NULL) {
        set_error(error, error_capacity, "Invalid tokenizer encode arguments.");
        return -1;
    }
    if (error != NULL && error_capacity > 0) {
        error[0] = '\0';
    }

    text_length = strlen(text);
    if (text_length > SIZE_MAX - 3) {
        set_error(error, error_capacity, "Input text is too large.");
        return -1;
    }
    if (add_bos != 0 &&
        append_token(tokens, token_capacity, &count, tokenizer->bos_id) != 0) {
        set_error(error, error_capacity, "Token output buffer is too small.");
        return -1;
    }

    if (tokenizer->max_token_length > (SIZE_MAX - 8) / 2) {
        set_error(error, error_capacity, "Tokenizer maximum token length is invalid.");
        return -1;
    }
    buffer_capacity = tokenizer->max_token_length * 2 + 8;
    buffer = malloc(buffer_capacity);
    if (buffer == NULL) {
        set_error(error, error_capacity, "Memory allocation failed.");
        return -1;
    }

    /*
     * Split the text at special-token strings (e.g. "</s>" in a chat
     * template). Each special string becomes its token ID directly, and
     * the text between them is encoded as independent segments, which is
     * what the Hugging Face Llama tokenizer does. Tokenizers without a
     * special-token list (llama2.c tokenizer.bin) encode the whole text
     * as one segment, exactly as before.
     */
    size_t pos = 0;
    for (;;) {
        size_t match_pos = 0;
        size_t match_length = 0;
        int special = find_special(tokenizer, text + pos,
                                   &match_pos, &match_length);
        size_t segment_length = (special >= 0) ? match_pos : text_length - pos;

        if (encode_segment(tokenizer, text + pos, segment_length,
                           tokens, token_capacity, &count,
                           buffer, buffer_capacity,
                           error, error_capacity) != 0) {
            free(buffer);
            return -1;
        }

        if (special < 0) {
            break;
        }

        if (append_token(tokens, token_capacity, &count, special) != 0) {
            free(buffer);
            set_error(error, error_capacity, "Token output buffer is too small.");
            return -1;
        }
        pos += match_pos + match_length;
    }

    if (add_eos != 0 &&
        append_token(tokens, token_capacity, &count, tokenizer->eos_id) != 0) {
        free(buffer);
        set_error(error, error_capacity, "Token output buffer is too small.");
        return -1;
    }

    free(buffer);
    *token_count = count;
    return 0;
}

static const char *decode_piece(const Tokenizer *tokenizer, int id,
                               unsigned char *byte_value)
{
    const char *piece;
    unsigned int value;

    if (id < 0 || (size_t)id >= tokenizer->vocab_size) {
        return NULL;
    }
    piece = tokenizer->vocab[id];
    if (strlen(piece) == 6 && piece[0] == '<' && piece[1] == '0' &&
        piece[2] == 'x' && piece[5] == '>') {
        char hex[3] = {piece[3], piece[4], '\0'};
        char *end = NULL;
        unsigned long parsed = strtoul(hex, &end, 16);
        if (end != hex && *end == '\0' && parsed <= 255UL) {
            value = (unsigned int)parsed;
            *byte_value = (unsigned char)value;
            return (const char *)byte_value;
        }
    }
    return piece;
}

int tokenizer_decode(const Tokenizer *tokenizer,
                     const int *tokens, size_t token_count,
                     char *output, size_t output_capacity,
                     char *error, size_t error_capacity)
{
    size_t used = 0;
    int previous = -1;

    if (output != NULL && output_capacity > 0) {
        output[0] = '\0';
    }
    if (tokenizer == NULL || tokenizer->vocab == NULL ||
        (tokens == NULL && token_count != 0) ||
        output == NULL || output_capacity == 0) {
        set_error(error, error_capacity, "Invalid tokenizer decode arguments.");
        return -1;
    }
    if (error != NULL && error_capacity > 0) {
        error[0] = '\0';
    }

    for (size_t i = 0; i < token_count; ++i) {
        int id = tokens[i];
        unsigned char byte_value = 0;
        const char *piece;
        size_t length;

        if (id == tokenizer->bos_id || id == tokenizer->eos_id) {
            previous = id;
            continue;
        }
        piece = decode_piece(tokenizer, id, &byte_value);
        if (piece == NULL) {
            set_error(error, error_capacity, "Token ID is outside the vocabulary.");
            return -1;
        }
        if (previous == tokenizer->bos_id && piece[0] == ' ') {
            ++piece;
        }
        length = (piece == (const char *)&byte_value) ? 1 : strlen(piece);
        if (length > output_capacity - used - 1) {
            set_error(error, error_capacity, "Decoded output buffer is too small.");
            return -1;
        }
        memcpy(output + used, piece, length);
        used += length;
        output[used] = '\0';
        previous = id;
    }
    return 0;
}
