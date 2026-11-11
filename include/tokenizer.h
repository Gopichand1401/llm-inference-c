#ifndef WILD_POINTERS_TOKENIZER_H
#define WILD_POINTERS_TOKENIZER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char *text;
    int id;
} TokenizerIndex;

typedef struct {
    char **vocab;
    float *scores;
    TokenizerIndex *sorted_vocab;
    size_t vocab_size;
    size_t max_token_length;
    unsigned char byte_pieces[512];

    /* Special token IDs (1 / 2 for llama2.c files, from metadata for GGUF). */
    int bos_id;
    int eos_id;

    /* Token ID for each raw byte value (byte fallback), or -1 if none. */
    int byte_ids[256];

    /*
     * Special tokens recognised inside input text (e.g. "<s>", "</s>").
     * Empty for llama2.c tokenizer.bin files; filled from the GGUF token
     * types (unknown / control / user-defined) by tokenizer_load_gguf().
     */
    int *special_ids;
    size_t n_special;

    /* 1 if the GGUF file had no usable merge scores and they were rebuilt
     * from token order (see tokenizer_load_gguf). */
    int scores_reconstructed;
} Tokenizer;

/*
 * Load a Karpathy llama2.c-format tokenizer.bin.
 * vocab_size must match the model checkpoint's vocabulary size.
 * Returns 0 on success and -1 on failure; errors are written to error when
 * an error buffer is supplied.
 */
int tokenizer_load(Tokenizer *tokenizer, const char *path,
                   size_t vocab_size, char *error, size_t error_capacity);

/*
 * Load a SentencePiece ("llama") tokenizer from GGUF metadata:
 * tokenizer.ggml.tokens / scores / token_type and the BOS/EOS IDs.
 * The vocabulary size is whatever the file contains.
 * Control, unknown, unused and byte tokens are never produced by merges.
 * Special tokens (unknown / control / user-defined, e.g. "<s>", "</s>")
 * written inside the text are recognised and emitted as their IDs, like the
 * Hugging Face Llama tokenizer.
 * If every score in the file is the same (some converters write all zeros),
 * the scores are rebuilt from token order, see gguf_tokenizer.c.
 * Returns 0 on success and -1 on failure.
 */
int tokenizer_load_gguf(Tokenizer *tokenizer, const char *gguf_path,
                        char *error, size_t error_capacity);

/* Release all memory owned by the tokenizer. Safe on a zero-initialized object. */
void tokenizer_free(Tokenizer *tokenizer);

/*
 * Encode UTF-8 text using the tokenizer's BPE vocabulary.
 * Special-token strings listed in tokenizer->special_ids are emitted as
 * their IDs; the text between them is encoded segment by segment.
 * add_bos/add_eos add tokenizer->bos_id / tokenizer->eos_id.
 * Returns 0 on success, -1 on invalid input, insufficient capacity, or failure.
 */
int tokenizer_encode(Tokenizer *tokenizer, const char *text,
                     int add_bos, int add_eos,
                     int *tokens, size_t token_capacity,
                     size_t *token_count,
                     char *error, size_t error_capacity);

/*
 * Decode token IDs into UTF-8/text bytes.
 * BOS and EOS tokens are omitted. The output is always NUL-terminated on
 * success. Returns 0 on success, -1 on invalid token/capacity/error.
 */
int tokenizer_decode(const Tokenizer *tokenizer,
                     const int *tokens, size_t token_count,
                     char *output, size_t output_capacity,
                     char *error, size_t error_capacity);

/* Special token IDs fixed by the llama2.c tokenizer.bin format.
 * (GGUF tokenizers store theirs in tokenizer->bos_id / eos_id.) */
enum {
    TOKENIZER_BOS_ID = 1,
    TOKENIZER_EOS_ID = 2
};

#ifdef __cplusplus
}
#endif

#endif
