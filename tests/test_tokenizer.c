#include "tokenizer.h"

#include <stdio.h>
#include <string.h>

#define CHECK(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL: %s (line %d)\n", message, __LINE__); \
        tokenizer_free(&tokenizer); \
        return 1; \
    } \
} while (0)

/* Usage: test_tokenizer [tokenizer.bin]   (default: models/tokenizer.bin) */
int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "models/tokenizer.bin";
    Tokenizer tokenizer = {0};
    char error[256];
    int tokens[512];
    size_t count = 0;
    char decoded[2048];
    const char *sample = "hello world!";

    CHECK(tokenizer_load(&tokenizer, path, 32000,
                         error, sizeof(error)) == 0,
          "load tokenizer.bin (pass its path as the first argument)");
    CHECK(tokenizer.vocab_size == 32000, "vocabulary size");
    CHECK(tokenizer.max_token_length == 27, "maximum token length");

    CHECK(tokenizer_encode(&tokenizer, sample, 1, 0, tokens, 512,
                           &count, error, sizeof(error)) == 0,
          "encode sample text");
    CHECK(count > 1 && tokens[0] == TOKENIZER_BOS_ID,
          "BOS token is first");
    CHECK(tokenizer_decode(&tokenizer, tokens, count, decoded,
                           sizeof(decoded), error, sizeof(error)) == 0,
          "decode sample text");
    CHECK(strcmp(decoded, sample) == 0, "round-trip sample text");

    CHECK(tokenizer_encode(&tokenizer, "", 1, 1, tokens, 512,
                           &count, error, sizeof(error)) == 0,
          "encode empty text with special tokens");
    CHECK(count == 2 && tokens[0] == TOKENIZER_BOS_ID &&
          tokens[1] == TOKENIZER_EOS_ID, "empty text special tokens");

    CHECK(tokenizer_encode(&tokenizer, sample, 0, 0, tokens, 1,
                           &count, error, sizeof(error)) == -1,
          "reject insufficient token buffer");

    {
        int invalid[] = {32000};
        CHECK(tokenizer_decode(&tokenizer, invalid, 1, decoded,
                               sizeof(decoded), error, sizeof(error)) == -1,
              "reject out-of-range token ID");
    }

    tokenizer_free(&tokenizer);
    CHECK(tokenizer.vocab == NULL && tokenizer.vocab_size == 0,
          "free resets tokenizer");

    CHECK(tokenizer_load(&tokenizer, "does-not-exist.tokenizer", 32000,
                         error, sizeof(error)) == -1,
          "reject missing tokenizer file");

    puts("PASS: tokenizer tests");
    return 0;
}
