#!/usr/bin/env bash
# run_xlsx_checks.sh - verify the XLSX reader (RAG Step 1).
#
#   1. builds xlsx_dump (strict flags) and a sanitizer build
#   2. builds 5 test workbooks, compares every cell with openpyxl
#   3. corrupt-file and fuzz tests (AddressSanitizer + UBSan)
#   4. mutation test: 14 planted bugs must all be detected
#
# Needs: gcc, zlib (zlib1g-dev), python3 with openpyxl (pip install openpyxl).
# Usage: tests/run_xlsx_checks.sh [fuzz iterations, default 200]
set -u
cd "$(dirname "$0")/.." || exit 1

ITER="${1:-200}"
DATA="build/xlsx_test_data"
PASS=0
FAIL=0

ok()  { echo "  PASS  $1"; PASS=$((PASS + 1)); }
bad() { echo "  FAIL  $1"; FAIL=$((FAIL + 1)); }

if ! python3 -c "import openpyxl" 2>/dev/null; then
    echo "openpyxl is missing: pip install openpyxl" >&2
    exit 1
fi

echo "== build"
make build/xlsx_dump >/dev/null 2>&1 || make rag-tools || exit 1
[ -x build/xlsx_dump ] && ok "xlsx_dump builds with -Wall -Wextra -Wpedantic -Werror" || bad "build"

gcc -std=c11 -Wall -Wextra -Wpedantic -Werror -g -O1 \
    -fsanitize=address,undefined -fno-sanitize-recover=all -Iinclude \
    src/xlsx_reader.c examples/09_rag/xlsx_dump.c -lz -o build/xlsx_dump_asan \
    && ok "sanitizer build" || bad "sanitizer build"

echo "== reader vs openpyxl (every cell, tab names, hidden flags)"
python3 tests/make_test_xlsx.py "$DATA" >/dev/null || exit 1
for f in ds_like cs_like types raw_features big; do
    ./build/xlsx_dump --tsv "$DATA/$f.xlsx" > "build/xlsx_c.tsv" 2>/dev/null
    python3 tests/compare_xlsx.py "$DATA/$f.xlsx" > "build/xlsx_py.tsv" 2>/dev/null
    if cmp -s build/xlsx_c.tsv build/xlsx_py.tsv; then
        ok "$f.xlsx: $(grep -vc '^#tab' build/xlsx_c.tsv) cells identical"
    else
        bad "$f.xlsx differs (diff build/xlsx_c.tsv build/xlsx_py.tsv)"
    fi
done

echo "== sanitizers on every test workbook"
for f in "$DATA"/*.xlsx; do
    ./build/xlsx_dump_asan --tsv "$f" >/dev/null 2>build/xlsx_asan.err \
        && ok "$(basename "$f") clean" || { bad "$(basename "$f")"; head -5 build/xlsx_asan.err; }
done

echo "== corrupt files and fuzzing ($ITER + $ITER files)"
python3 tests/fuzz_xlsx.py build/xlsx_dump_asan "$DATA/ds_like.xlsx" "$ITER" > build/xlsx_fuzz.log 2>&1 \
    && ok "no crash on bad or fuzzed files" || { bad "fuzz"; tail -15 build/xlsx_fuzz.log; }

echo "== mutation test"
python3 tests/mutation_xlsx.py "$PWD" "$PWD/$DATA" > build/xlsx_mut.log 2>&1 \
    && ok "$(tail -1 build/xlsx_mut.log)" || { bad "mutation"; cat build/xlsx_mut.log; }

echo
echo "xlsx checks: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
