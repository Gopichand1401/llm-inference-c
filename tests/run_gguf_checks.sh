#!/usr/bin/env bash
#
# run_gguf_checks.sh - rerun every GGUF verification from Phases 1-8.
#
# Usage (from the project root, with .venv-gguf-ref active):
#   bash tests/run_gguf_checks.sh models/gguf/tinyllama.gguf [tokenizer.bin]
#
# Builds everything, then runs each C test program and its Python reference
# comparison. Prints PASS/FAIL per check and a summary; the exit status is
# non-zero if anything failed. Full output of each step goes to
# build/checks/<name>.log.
#
# The Hugging Face tokenizer check needs internet access the first time and
# the 'tokenizers' package; it is skipped if that package is missing.

set -u

MODEL="${1:-}"
TOKBIN="${2:-tokenizer.bin}"

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "Usage: bash tests/run_gguf_checks.sh <model.gguf> [tokenizer.bin]"
    exit 2
fi

LOGDIR=build/checks
PASS=0
FAIL=0
SKIP=0

run() {
    # run <name> <text that must appear in the output> <command...>
    local name="$1" expect="$2"
    shift 2
    printf "%-34s " "$name"
    if "$@" > "$LOGDIR/$name.log" 2>&1 && grep -q "$expect" "$LOGDIR/$name.log"; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL  (see $LOGDIR/$name.log)"
        FAIL=$((FAIL + 1))
    fi
}

skip() {
    printf "%-34s SKIP  (%s)\n" "$1" "$2"
    SKIP=$((SKIP + 1))
}

echo "=== Building ==="
if ! { make clean && make && make gguf-tools; } > /tmp/gguf_checks_build.log 2>&1; then
    echo "Build FAILED - see /tmp/gguf_checks_build.log"
    exit 1
fi
mkdir -p "$LOGDIR"
echo "Build OK"
echo

echo "=== Checks ==="
run unit_tests              "PASS: tokenizer tests"   make test

# Phase 1: quantization formats
run q4k_synthetic           "SUCCESS"                 ./build/gguf_q4k_demo
run q6k_synthetic           "PASSED"                  ./build/gguf_q6k_demo
run q4k_real_tensor         "PASSED"                  ./build/gguf_q4k_real_demo "$MODEL"
run q4k_vs_gguf_py          "PASSED"                  python examples/03_quantization/compare_q4k_reference.py "$MODEL"
run q6k_real_tensor         "PASSED"                  ./build/gguf_q6k_real_demo "$MODEL"
run q6k_vs_gguf_py          "PASSED"                  python examples/03_quantization/compare_q6k_reference.py "$MODEL"

# Phase 2: tensor report
run tensor_report           "Structure check: PASSED" ./build/gguf_tensor_report "$MODEL"
run tensor_report_vs_py     "PASSED"                  python examples/04_model_loading/compare_tensor_report.py "$MODEL"

# Phase 3: loader and row decoding
run model_load_rows         "PASSED"                  ./build/gguf_model_demo "$MODEL"
run rows_vs_gguf_py         "PASSED"                  python examples/04_model_loading/compare_gguf_rows.py "$MODEL"

# Phase 4 + 8: matrix-vector products (reference bit-exact, fast accurate)
run matvec                  "PASSED"                  ./build/gguf_matvec_demo "$MODEL"
run matvec_vs_numpy         "PASSED"                  python examples/05_matrix_math/compare_matvec.py "$MODEL"

# Phase 5: forward pass
run forward_pass            "COMPLETED"               ./build/gguf_forward_demo "$MODEL" "The capital of France is" 0
run forward_vs_numpy        "PASSED"                  python examples/06_forward_pass/reference_forward.py "$MODEL"

# Phase 6: tokenizer
if [ -f "$TOKBIN" ]; then
    run tokenizer           "PASSED"                  ./build/gguf_tokenizer_demo "$MODEL" "$TOKBIN"
else
    run tokenizer           "PASSED"                  ./build/gguf_tokenizer_demo "$MODEL" -
fi
if python -c "import sentencepiece, google.protobuf" 2>/dev/null; then
    run tokenizer_vs_spm    "PASSED"                  python examples/07_tokenizer/compare_tokenizer_spm.py "$MODEL"
else
    skip tokenizer_vs_spm "pip install sentencepiece protobuf"
fi
if python -c "import tokenizers" 2>/dev/null; then
    run tokenizer_vs_hf     "MATCH"                   python examples/07_tokenizer/compare_tokenizer_hf.py "$MODEL"
else
    skip tokenizer_vs_hf "pip install tokenizers"
fi

# Phase 7: generation (greedy is deterministic)
run generate_greedy         "Paris"                   ./build/gguf_generate "$MODEL" --sampler greedy -n 8 "The capital of France is"

echo
echo "=== Summary: $PASS passed, $FAIL failed, $SKIP skipped ==="
[ "$FAIL" -eq 0 ]
