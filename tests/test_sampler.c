#include "sampler.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

int main(void)
{
    const float logits[] = {0.2f, 1.5f, -0.7f, 0.9f};
    const size_t n = sizeof(logits) / sizeof(logits[0]);
    uint32_t rng = UINT32_C(123456789);

    assert(sampler_greedy(logits, n) == 1);
    assert(sampler_greedy(NULL, n) == -1);
    assert(sampler_greedy(logits, 0) == -1);

    int sampled = sampler_temperature(logits, n, 1.0f, &rng);
    assert(sampled >= 0 && (size_t)sampled < n);
    assert(sampler_temperature(logits, n, 0.0f, &rng) == -1);
    assert(sampler_temperature(logits, n, 1.0f, NULL) == -1);

    /* With k=1, top-k must always select the maximum-logit token. */
    assert(sampler_top_k(logits, n, 1, 0.8f, &rng) == 1);

    sampled = sampler_top_k(logits, n, 2, 0.8f, &rng);
    assert(sampled == 1 || sampled == 3);
    assert(sampler_top_k(logits, n, 0, 1.0f, &rng) == -1);
    assert(sampler_top_k(logits, n, n + 1, 1.0f, &rng) == -1);

    const float bad_logits[] = {0.0f, NAN, 2.0f};
    assert(sampler_greedy(bad_logits, 3) == -1);

    puts("PASS: sampler tests");
    return 0;
}
