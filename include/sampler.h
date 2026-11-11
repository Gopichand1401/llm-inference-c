#ifndef WILD_POINTERS_SAMPLER_H
#define WILD_POINTERS_SAMPLER_H

#include <stddef.h>
#include <stdint.h>

/* Returns the index of the largest logit, or -1 for invalid input. */
int sampler_greedy(const float *logits, size_t vocab_size);

/*
 * Sample from softmax(logits / temperature).
 * rng_state must point to a mutable 32-bit state.
 * Returns a token ID, or -1 for invalid input.
 */
int sampler_temperature(const float *logits, size_t vocab_size,
                        float temperature, uint32_t *rng_state);

/*
 * Sample from the top-k logits after temperature scaling.
 * k must be in [1, vocab_size]. k == 1 behaves like greedy sampling.
 * Returns a token ID, or -1 for invalid input.
 */
int sampler_top_k(const float *logits, size_t vocab_size, size_t k,
                  float temperature, uint32_t *rng_state);

#endif
