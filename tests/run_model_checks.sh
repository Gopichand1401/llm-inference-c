#!/usr/bin/env bash
#
# run_model_checks.sh - verify ONE model file end to end.
#
# Usage (from the project root, with .venv-gguf-ref active):
#   bash tests/run_model_checks.sh <model.gguf | model.bin> [expected word]
#
# llama2.c .bin models (tokenizer.bin next to the model or in the project):
#   1. forward() smoke test and input-validation tests
#   2. logits for a prompt match an independent NumPy forward pass that
#      reads the .bin weights directly (examples/06_forward_pass/reference_stories.py)
#   3. greedy generation runs to completion (or contains the expected word)
#
# GGUF models - checks (each C step is compared with gguf-py / NumPy):
#   1. random blocks of every supported type decode bit-exactly (no model)
#   2. tensor report: every tensor present, shaped, aligned, decodable
#   3. row decoding of all weights matches gguf-py bit for bit
#   4. matrix-vector products: reference kernel bit-exact, fast kernel
#      within float32 rounding of float64
#   5. full forward pass on "The capital of France is" matches NumPy
#   6. greedy generation contains the expected word (default: Paris)
# Logs: build/model_checks/<model>/<step>.log

set -u

MODEL="${1:-}"
EXPECT="${2:-}"

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "Usage: bash tests/run_model_checks.sh <model.gguf | model.bin> [expected word]"
    exit 2
fi
if [ ! -x build/llm_generate ] || [ ! -x build/gguf_quant_check ]; then
    echo "Build first: make && make gguf-tools"
    exit 2
fi

LOGDIR="build/model_checks/$(basename "$MODEL")"
mkdir -p "$LOGDIR"
PASS=0
FAIL=0

run() {
    local name="$1" expect="$2"
    shift 2
    printf "  %-24s " "$name"
    if "$@" > "$LOGDIR/$name.log" 2>&1 && grep -q "$expect" "$LOGDIR/$name.log"; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL  (see $LOGDIR/$name.log)"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== $(basename "$MODEL") ==="

# ---- llama2.c .bin models ----
if [ "$(head -c 4 "$MODEL" | tr -d '\000')" != "GGUF" ]; then
    # The prompt is echoed, so check that generation ran to completion.
    EXPECT="${EXPECT:-stopped at}"
    for t in test_forward test_validation; do
        if [ ! -x "build/$t" ]; then
            gcc -Iinclude -std=c11 -Wall -Wextra -Wpedantic -Werror -O2 -fopenmp \
                "tests/$t.c" build/libllm.a -lm -o "build/$t" || exit 2
        fi
    done
    run forward_smoke      "PASS: all logits are finite" ./build/test_forward "$MODEL"
    run input_validation   "ALL VALIDATION TESTS PASSED" ./build/test_validation "$MODEL"
    run logits_dump        "COMPLETED"               ./build/llm_logits_dump "$MODEL" "Once upon a time there was a little cat"
    run forward_numpy      "PASSED"                  python examples/06_forward_pass/reference_stories.py "$MODEL"
    run generate_greedy    "$EXPECT"                 ./build/llm_generate "$MODEL" --sampler greedy -n 20 "Once upon a time"
    echo "  Summary: $PASS passed, $FAIL failed"
    [ "$FAIL" -eq 0 ]
    exit $?
fi

# ---- GGUF models ----
EXPECT="${EXPECT:-Paris}"
./build/gguf_tensor_report "$MODEL" 2>/dev/null | sed -n '/Types used/,/^$/p' | grep -v "^$" | sed 's/^/  /'

run random_blocks      "PASSED"                  python examples/03_quantization/compare_quant_random.py 5000
run tensor_report      "Structure check: PASSED" ./build/gguf_tensor_report "$MODEL"
run tensor_report_py   "PASSED"                  python examples/04_model_loading/compare_tensor_report.py "$MODEL"
run row_decoding       "PASSED"                  ./build/gguf_model_demo "$MODEL"
run row_decoding_py    "PASSED"                  python examples/04_model_loading/compare_gguf_rows.py "$MODEL"
run matvec             "PASSED"                  ./build/gguf_matvec_demo "$MODEL"
run matvec_numpy       "PASSED"                  python examples/05_matrix_math/compare_matvec.py "$MODEL"
run forward            "COMPLETED"               ./build/gguf_forward_demo "$MODEL" "The capital of France is" 0
run forward_numpy      "PASSED"                  python examples/06_forward_pass/reference_forward.py "$MODEL"
run generate_greedy    "$EXPECT"                 ./build/llm_generate "$MODEL" --sampler greedy -n 12 --ctx 256 "The capital of France is"

echo "  Summary: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
