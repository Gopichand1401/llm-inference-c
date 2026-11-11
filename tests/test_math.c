
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#include "transformer.h"   /* rmsnorm, softmax, matmul */

#define TOL 1e-4f

static bool nearly_equal(float actual, float expected)
{
    return isfinite(actual) && fabsf(actual - expected) < TOL;
}

static int check_array(
    const char *name,
    const float *actual,
    const float *expected,
    int size
)
{
    bool passed = true;

    printf("\nTesting %s:\n", name);

    for (int i = 0; i < size; i++) {
        printf("  [%d] actual = %.6f, expected = %.6f\n",
               i, actual[i], expected[i]);

        if (!nearly_equal(actual[i], expected[i])) {
            passed = false;
        }
    }

    printf("Result: %s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}

/* Test 1: RMS normalization */
static int test_rmsnorm(void)
{
    float x[] = {3.0f, 4.0f};
    float weight[] = {1.0f, 1.0f};
    float output[2];

    /*
     * RMS = sqrt((3^2 + 4^2) / 2)
     *     = sqrt(12.5)
     *
     * output[i] = x[i] / sqrt(12.5 + 1e-5)
     *             * weight[i]
     */
    float rms = sqrtf((9.0f + 16.0f) / 2.0f + 1e-5f);

    float expected[] = {
        3.0f / rms,
        4.0f / rms
    };

    rmsnorm(output, x, weight, 2);

    return check_array("RMSNorm", output, expected, 2);
}

/* Test 2: Softmax */
static int test_softmax(void)
{
    float x[] = {0.0f, logf(3.0f)};
    float expected[] = {0.25f, 0.75f};

    softmax(x, 2);

    int failures = check_array("Softmax", x, expected, 2);

    float sum = x[0] + x[1];
    printf("  Probability sum = %.6f (expected 1.000000)\n", sum);

    if (!nearly_equal(sum, 1.0f)) {
        failures++;
    }

    if (x[0] < 0.0f || x[1] < 0.0f ||
        x[0] > 1.0f || x[1] > 1.0f) {
        failures++;
    }

    return failures == 0 ? 0 : 1;
}

/* Test 3: Matrix multiplication */
static int test_matmul(void)
{
    /*
     * Input vector:
     * x = [1, 2, 3]
     *
     * Weight matrix (2 rows, 3 columns):
     * W = [1, 2, 3]
     *     [4, 5, 6]
     *
     * Expected output:
     * y[0] = 1*1 + 2*2 + 3*3 = 14
     * y[1] = 1*4 + 2*5 + 3*6 = 32
     */
    float x[] = {1.0f, 2.0f, 3.0f};
    float w[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    float output[2] = {0.0f, 0.0f};
    float expected[] = {14.0f, 32.0f};

    matmul(output, x, w, 3, 2);

    return check_array("Matrix multiplication", output, expected, 2);
}

/* Test 4: numerical edge cases */
static int test_edge_cases(void)
{
    int failures = 0;

    /* softmax must subtract the max: exp(1000) would overflow to inf. */
    float big[3] = {1000.0f, 1000.0f, 999.0f};
    softmax(big, 3);
    float e = expf(-1.0f), sum = 2.0f + e;
    float big_expected[3] = {1.0f / sum, 1.0f / sum, e / sum};
    failures += check_array("Softmax of large logits (no overflow)",
                            big, big_expected, 3);

    float neg[2] = {-1000.0f, -1001.0f};
    softmax(neg, 2);
    float neg_expected[2] = {1.0f / (1.0f + e), e / (1.0f + e)};
    failures += check_array("Softmax of very negative logits",
                            neg, neg_expected, 2);

    float single[1] = {42.0f};
    softmax(single, 1);
    float one[1] = {1.0f};
    failures += check_array("Softmax of one value", single, one, 1);

    /* rmsnorm of a zero vector: the epsilon keeps it finite (0, not NaN). */
    float zeros[4] = {0}, w[4] = {1, 1, 1, 1}, out[4];
    float zero_expected[4] = {0};
    rmsnorm(out, zeros, w, 4);
    failures += check_array("RMSNorm of a zero vector", out, zero_expected, 4);

    /* matmul with a 1x1 matrix and with zero weights */
    float x1[1] = {3.0f}, w1[1] = {-2.0f}, y1[1];
    float y1_expected[1] = {-6.0f};
    matmul(y1, x1, w1, 1, 1);
    failures += check_array("Matmul 1x1", y1, y1_expected, 1);

    /* larger matmul against a straightforward double-precision loop */
    enum { N = 257, D = 129 };
    static float wl[N * D], xl[N], yl[D], ref[D];
    for (int i = 0; i < N * D; ++i) {
        wl[i] = (float)((i * 37) % 101 - 50) / 50.0f;
    }
    for (int i = 0; i < N; ++i) {
        xl[i] = (float)((i * 13) % 17 - 8) / 8.0f;
    }
    for (int r = 0; r < D; ++r) {
        double acc = 0.0;
        for (int c = 0; c < N; ++c) {
            acc += (double)wl[r * N + c] * xl[c];
        }
        ref[r] = (float)acc;
    }
    matmul(yl, xl, wl, N, D);
    int ok = 1;
    for (int r = 0; r < D; ++r) {
        ok &= fabsf(yl[r] - ref[r]) < 1e-3f;
    }
    printf("%s: %s\n", "Matmul 129x257 vs double reference",
           ok ? "PASS" : "FAIL");
    failures += !ok;

    return failures;
}

int main(void)
{
    int failures = 0;

    printf("=== Mathematical Operations Test Suite ===\n");

    failures += test_rmsnorm();
    failures += test_softmax();
    failures += test_matmul();
    failures += test_edge_cases();

    printf("\n===========================================\n");

    if (failures == 0) {
        printf("ALL TESTS PASSED\n");
        return 0;
    }

    printf("%d TEST(S) FAILED\n", failures);
    return 1;
}