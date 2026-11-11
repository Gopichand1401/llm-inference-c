#!/usr/bin/env bash
#
# run_model_matrix.sh - try every model file in a folder with llm_generate.
#
# Usage (from the project root, after make):
#   bash tests/run_model_matrix.sh ~/models [tokens]
#
# For each *.bin / *.gguf file it runs a short greedy generation and prints
# one row: OK with the speed and the start of the output, UNSUPPORTED with
# the missing quantization type, or FAIL with the error. Full output of each
# run goes to build/matrix/<file>.log.
#
# Prompts: llama2.c Stories models get "Once upon a time", GGUF models get
# "The capital of France is" (a working TinyLlama answers "Paris").

set -u

DIR="${1:-}"
TOKENS="${2:-24}"

if [ -z "$DIR" ] || [ ! -d "$DIR" ]; then
    echo "Usage: bash tests/run_model_matrix.sh <model folder> [tokens]"
    exit 2
fi
if [ ! -x build/llm_generate ]; then
    echo "build/llm_generate not found - run make first."
    exit 2
fi

LOGDIR=build/matrix
mkdir -p "$LOGDIR"

printf "%-40s %-12s %-8s %s\n" "MODEL" "RESULT" "TOK/S" "DETAILS"
printf "%-40s %-12s %-8s %s\n" "-----" "------" "-----" "-------"

ok=0; unsupported=0; failed=0

for path in "$DIR"/*.bin "$DIR"/*.gguf; do
    [ -f "$path" ] || continue
    name=$(basename "$path")
    log="$LOGDIR/$name.log"

    case "$name" in
        tokenizer*.bin|tok*.bin) continue ;;   # tokenizer files, not models
    esac

    case "$name" in
        *.gguf) prompt="The capital of France is" ;;
        *)      prompt="Once upon a time" ;;
    esac

    ./build/llm_generate "$path" --sampler greedy -n "$TOKENS" --ctx 256 \
        "$prompt" > "$log" 2>&1
    status=$?

    if [ $status -eq 0 ]; then
        speed=$(grep -o 'reply [0-9]* tokens [0-9.]* s ([0-9.]* tok/s)' "$log" \
                | grep -o '([0-9.]*' | tr -d '(')
        text=$(grep -v '^\[' "$log" | tr '\n' ' ' | sed 's/  */ /g' | cut -c1-60)
        printf "%-40s %-12s %-8s %s\n" "$name" "OK" "${speed:-?}" "$text"
        ok=$((ok + 1))
    elif grep -q "unsupported type" "$log"; then
        type=$(grep -o 'unsupported type [A-Z0-9_]* ([0-9]*)' "$log" | head -1)
        printf "%-40s %-12s %-8s %s\n" "$name" "UNSUPPORTED" "-" "$type"
        unsupported=$((unsupported + 1))
    else
        err=$(grep -v '^\[' "$log" | head -1 | cut -c1-70)
        printf "%-40s %-12s %-8s %s\n" "$name" "FAIL" "-" "$err"
        failed=$((failed + 1))
    fi
done

echo
echo "OK: $ok   UNSUPPORTED: $unsupported   FAIL: $failed   (logs in $LOGDIR/)"
