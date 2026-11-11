#include "sampler.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static int valid_logits(const float *logits, size_t n)
{
    if (logits == NULL || n == 0 || n > (size_t)INT32_MAX) {
        return 0;
    }
    for (size_t i = 0; i < n; ++i) {
        if (!isfinite(logits[i])) {
            return 0;
        }
    }
    return 1;
}

static uint32_t next_random(uint32_t *state)
{
    /* xorshift32; zero is not a valid state. */
    uint32_t x = *state;
    if (x == 0) {
        x = UINT32_C(0xA341316C);
    }
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static double random_unit(uint32_t *state)
{
    return (double)next_random(state) / 4294967296.0;
}

int sampler_greedy(const float *logits, size_t vocab_size)
{
    if (!valid_logits(logits, vocab_size)) {
        return -1;
    }

    size_t best = 0;
    for (size_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return (int)best;
}

int sampler_temperature(const float *logits, size_t vocab_size,
                        float temperature, uint32_t *rng_state)
{
    if (!valid_logits(logits, vocab_size) ||
        !isfinite(temperature) || temperature <= 0.0f ||
        rng_state == NULL) {
        return -1;
    }

    float max_logit = logits[0];
    for (size_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    double total = 0.0;
    for (size_t i = 0; i < vocab_size; ++i) {
        total += exp(((double)logits[i] - (double)max_logit) /
                     (double)temperature);
    }
    if (!isfinite(total) || total <= 0.0) {
        return -1;
    }

    double target = random_unit(rng_state) * total;
    double cumulative = 0.0;
    for (size_t i = 0; i < vocab_size; ++i) {
        cumulative += exp(((double)logits[i] - (double)max_logit) /
                          (double)temperature);
        if (target < cumulative) {
            return (int)i;
        }
    }
    return (int)(vocab_size - 1); /* floating-point rounding fallback */
}

int sampler_top_k(const float *logits, size_t vocab_size, size_t k,
                  float temperature, uint32_t *rng_state)
{
    if (!valid_logits(logits, vocab_size) || k == 0 || k > vocab_size ||
        !isfinite(temperature) || temperature <= 0.0f ||
        rng_state == NULL) {
        return -1;
    }
    if (k == 1) {
        return sampler_greedy(logits, vocab_size);
    }

    size_t *indices = malloc(k * sizeof(*indices));
    if (indices == NULL) {
        return -1;
    }

    /* Keep the k highest logit indices in descending order. */
    size_t count = 0;
    for (size_t i = 0; i < vocab_size; ++i) {
        size_t insert_at = 0;
        while (insert_at < count && logits[indices[insert_at]] >= logits[i]) {
            ++insert_at;
        }
        if (insert_at >= k) {
            continue;
        }
        size_t new_count = count < k ? count + 1 : count;
        for (size_t j = new_count; j > insert_at + 1; --j) {
            indices[j - 1] = indices[j - 2];
        }
        indices[insert_at] = i;
        count = new_count;
    }

    float max_logit = logits[indices[0]];
    double total = 0.0;
    for (size_t i = 0; i < count; ++i) {
        total += exp(((double)logits[indices[i]] - (double)max_logit) /
                     (double)temperature);
    }

    double target = random_unit(rng_state) * total;
    double cumulative = 0.0;
    int selected = (int)indices[count - 1];
    for (size_t i = 0; i < count; ++i) {
        cumulative += exp(((double)logits[indices[i]] - (double)max_logit) /
                          (double)temperature);
        if (target < cumulative) {
            selected = (int)indices[i];
            break;
        }
    }

    free(indices);
    return selected;
}
