/*
 * minimal.c - the smallest complete program using the library:
 * load any model, encode a prompt, generate with top-k sampling.
 *
 * Build: make build/minimal
 * Run:   ./build/minimal models/stories42M.bin "Once upon a time"
 */
#include "engine.h"
#include "sampler.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model> \"prompt\"\n", argv[0]);
        return 1;
    }

    char error[256];
    LLMOptions options = { .model_path = argv[1] };   /* tokenizer found automatically */
    LLMModel *model = engine_open(&options, error, sizeof(error));
    if (model == NULL) {
        fprintf(stderr, "error: %s\n", error);
        return 1;
    }

    Tokenizer *tok = engine_tokenizer(model);          /* owned by the model */
    const Config *cfg = engine_config(model);
    int prompt[512];
    size_t n = 0;
    if (tokenizer_encode(tok, argv[2], 1, 0, prompt, 512, &n,
                         error, sizeof(error)) != 0) {
        fprintf(stderr, "error: %s\n", error);
        engine_close(model);
        return 1;
    }

    uint32_t rng = 1234;                               /* fixed seed: same text every run */
    int token = prompt[0];
    for (int pos = 0; pos < cfg->seq_len && pos < 200; ++pos) {
        float *logits = engine_forward(model, token, pos);   /* owned by the model */
        if (logits == NULL) {
            break;
        }
        int next = (size_t)pos + 1 < n
                 ? prompt[pos + 1]                                 /* still in the prompt */
                 : sampler_top_k(logits, (size_t)cfg->vocab_size,   /* generate */
                                 40, 0.8f, &rng);
        if (next < 0 || ((size_t)pos + 1 >= n && engine_is_stop_token(model, next))) {
            break;
        }
        char piece[256];
        if (tokenizer_decode(tok, &next, 1, piece, sizeof(piece), NULL, 0) == 0) {
            /* the first piece carries SentencePiece's leading space */
            fputs(pos == 0 && piece[0] == ' ' ? piece + 1 : piece, stdout);
            fflush(stdout);
        }
        token = next;
    }
    putchar('\n');

    engine_close(model);                               /* frees everything above */
    return 0;
}
