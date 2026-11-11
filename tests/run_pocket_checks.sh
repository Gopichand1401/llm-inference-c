#!/usr/bin/env bash
# run_pocket_checks.sh - verify the library interface and the Python package.
#
#   1. builds the shared library (make lib) and the command-line programs
#   2. tests/test_pocketllm.py: the Python package gives the same text as
#      llm_generate and llm_rag for the same model, prompt, sampler and seed,
#      and the same replies as llm_rag --retrieve-only for every conversation
#      of tests/rag_dialogues.txt
#   3. the same again with an AddressSanitizer + UBSan build of the library
#
# Uses tiny random test models, so no real model file is needed.
# Needs: python3 with openpyxl, gguf and numpy
#        (pip install --break-system-packages openpyxl gguf numpy).
# Usage: tests/run_pocket_checks.sh
set -u
cd "$(dirname "$0")/.." || exit 1

if ! python3 -c "import openpyxl, gguf, numpy" 2>/dev/null; then
    echo "missing Python modules: pip install --break-system-packages openpyxl gguf numpy" >&2
    exit 1
fi

echo "== build"
{ make && make rag-tools && make lib; } >/dev/null 2>build_pocket.err \
    || { cat build_pocket.err; rm -f build_pocket.err; echo "  FAIL  build"; exit 1; }
rm -f build_pocket.err
echo "  PASS  libpocketllm.so, llm_generate, llm_rag build with -Wall -Wextra -Wpedantic -Werror"

python3 tests/make_rag_test_xlsx.py build/rag_test_data >/dev/null || exit 1
python3 tests/make_tiny_gguf.py build/rag_tiny.gguf >/dev/null || exit 1
STORIES=""
if [ -f models/tokenizer.bin ]; then
    # the engine looks for tokenizer.bin next to the model file
    cp models/tokenizer.bin build/tokenizer.bin
    python3 tests/make_tiny_stories.py build/tiny_stories.bin >/dev/null && STORIES="build/tiny_stories.bin"
fi

echo "== Python package against the command-line programs"
POCKETLLM_LIB=build/libpocketllm.so PYTHONPATH=python \
    python3 tests/test_pocketllm.py build/rag_tiny.gguf $STORIES
STATUS=$?

echo "== the same with sanitizers"
gcc -std=c11 -Wall -Wextra -Wpedantic -Werror -g -O1 -fopenmp -fPIC -shared \
    -fsanitize=address,undefined -fno-sanitize-recover=all -Iinclude src/*.c \
    -lm -lz -o build/libpocketllm_asan.so || { echo "  FAIL  sanitizer build"; exit 1; }
ASAN_LIB=$(gcc -print-file-name=libasan.so)
LD_PRELOAD="$ASAN_LIB" ASAN_OPTIONS=detect_leaks=0 \
    POCKETLLM_LIB=build/libpocketllm_asan.so PYTHONPATH=python \
    python3 tests/test_pocketllm.py build/rag_tiny.gguf $STORIES > build/pocket_asan.log 2>&1
if [ $? = 0 ] && ! grep -q "ERROR: \|runtime error" build/pocket_asan.log; then
    echo "  PASS  $(tail -1 build/pocket_asan.log) (AddressSanitizer + UBSan clean)"
else
    echo "  FAIL  sanitizer run (see build/pocket_asan.log)"; tail -5 build/pocket_asan.log
    STATUS=1
fi
exit $STATUS
