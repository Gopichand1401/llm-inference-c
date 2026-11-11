#!/usr/bin/env bash
# run_rag_checks.sh - verify the RAG pipeline (reading, search, answers).
#
#   1. builds rag_inspect + llm_rag (strict flags) and sanitizer builds
#   2. chunking (.xlsx and .csv), file scan, search terms, typo and short-form
#      resolution and BM25 scores compared with the independent Python
#      reference (tests/rag_reference.py), byte for byte, on synthetic
#      edge-case files and on the real files in data/
#   3. retrieval test set: tests/rag_questions.tsv on data/ (right row given
#      to the model; off-topic questions find nothing)
#   4. conversations: tests/rag_dialogues.txt played against
#      `llm_rag --retrieve-only` (direct answers, sources, offers, lists,
#      counts, typos, short forms, sorry + related rows, CSV)
#   5. folder changes picked up while llm_rag is running (add, edit, remove,
#      broken file, lock file)
#   6. the three modes end to end with a tiny random test model under
#      AddressSanitizer + UBSan (needs: pip install gguf numpy; else skipped)
#   7. mutation test: planted bugs in rag_chunk.c / rag_index.c must be caught
#   8. optional, with the real model as argument: answers to
#      tests/rag_answers.tsv must contain the expected text
#
# Usage: tests/run_rag_checks.sh [models/gguf/tinyllama.gguf]
set -u
cd "$(dirname "$0")/.." || exit 1

MODEL="${1:-}"
TEST="build/rag_test_data"
PASS=0
FAIL=0

ok()  { echo "  PASS  $1"; PASS=$((PASS + 1)); }
bad() { echo "  FAIL  $1"; FAIL=$((FAIL + 1)); }

if ! python3 -c "import openpyxl" 2>/dev/null; then
    echo "openpyxl is missing: pip install openpyxl" >&2
    exit 1
fi
HAVE_DATA=0
ls data/*.xlsx >/dev/null 2>&1 && HAVE_DATA=1

echo "== build"
make rag-tools >/dev/null 2>build/rag_build.err \
    && ok "rag_inspect, llm_rag build with -Wall -Wextra -Wpedantic -Werror" \
    || { bad "build"; cat build/rag_build.err; exit 1; }
SAN="-std=c11 -Wall -Wextra -Wpedantic -Werror -g -O1 -fopenmp -fsanitize=address,undefined -fno-sanitize-recover=all -Iinclude"
gcc $SAN src/xlsx_reader.c src/rag_chunk.c src/rag_index.c examples/09_rag/rag_inspect.c \
    -lz -lm -o build/rag_inspect_asan \
 && gcc $SAN src/*.c examples/09_rag/llm_rag.c -lz -lm -o build/llm_rag_asan \
    && ok "sanitizer builds" || bad "sanitizer builds"

echo "== C vs Python reference (files, chunks, terms, BM25 scores)"
python3 tests/make_rag_test_xlsx.py "$TEST" >/dev/null || exit 1
FOLDERS="$TEST"
[ $HAVE_DATA = 1 ] && FOLDERS="$TEST data"
for d in $FOLDERS; do
    for opt in "" "--include-hidden"; do
        for what in files chunks; do
            ./build/rag_inspect_asan $what "$d" $opt > build/rag_c.tsv 2>build/rag_asan.err
            python3 tests/rag_reference.py $what "$d" $opt > build/rag_py.tsv
            if cmp -s build/rag_c.tsv build/rag_py.tsv && ! grep -q "ERROR\|runtime error" build/rag_asan.err; then
                ok "$d $what ${opt:-(visible tabs)}: $(grep -c "^$( [ $what = files ] && echo file || echo chunk )" build/rag_c.tsv) identical"
            else
                bad "$d $what $opt differs (diff build/rag_c.tsv build/rag_py.tsv)"
            fi
        done
    done
done

n_ok=0; n_all=0
while IFS= read -r t; do
    [ -z "$t" ] && continue
    n_all=$((n_all + 1))
    [ "$(./build/rag_inspect tokens "$t")" = "$(python3 tests/rag_reference.py tokens "$t")" ] && n_ok=$((n_ok + 1))
done < tests/rag_check_terms.txt
[ $n_ok = $n_all ] && ok "search terms: $n_ok of $n_all strings identical" \
                   || bad "search terms: only $n_ok of $n_all strings identical"

for d in $FOLDERS; do
    n_ok=0; n_all=0
    while IFS= read -r q; do
        [ -z "$q" ] && continue
        n_all=$((n_all + 1))
        a=$(./build/rag_inspect_asan search "$d" -q "$q" -k 8 2>/dev/null)
        b=$(python3 tests/rag_reference.py search "$d" -q "$q" -k 8)
        [ "$a" = "$b" ] && n_ok=$((n_ok + 1)) || echo "        differs: $q"
    done < tests/rag_check_queries.txt
    [ $n_ok = $n_all ] && ok "$d BM25 ranking and scores: $n_ok of $n_all questions identical" \
                       || bad "$d BM25: only $n_ok of $n_all questions identical"
    n_ok=0; n_all=0
    while IFS= read -r q; do
        [ -z "$q" ] && continue
        n_all=$((n_all + 1))
        a=$(./build/rag_inspect_asan resolve "$d" "$q" 2>/dev/null)
        b=$(python3 tests/rag_reference.py resolve "$d" "$q")
        [ "$a" = "$b" ] && n_ok=$((n_ok + 1)) || echo "        differs: $q"
    done < tests/rag_check_queries.txt
    [ $n_ok = $n_all ] && ok "$d typo / short-form resolution: $n_ok of $n_all questions identical" \
                       || bad "$d resolution: only $n_ok of $n_all questions identical"
done

echo "== retrieval test set (tests/rag_questions.tsv)"
if [ $HAVE_DATA = 1 ]; then
    python3 tests/check_rag_retrieval.py build/llm_rag data tests/rag_questions.tsv > build/rag_retrieval.log \
        && ok "$(tail -1 build/rag_retrieval.log)" || { bad "$(tail -1 build/rag_retrieval.log)"; head -n -1 build/rag_retrieval.log; }
else
    echo "  SKIP  no data/*.xlsx"
fi

echo "== rows given to the model (-k 1 keeps the full row, not a stub)"
if [ $HAVE_DATA = 1 ]; then
    # DS3010 is listed in the CSE sheet only as "Check DS sheet for course
    # details"; the full row (DS sheet) must win even when only 1 row is used.
    OUT=$(printf 'Who teaches DS3010?\n/exit\n' | ./build/llm_rag --retrieve-only --docs data --show-rows -k 1 2>&1)
    echo "$OUT" | grep -q '\[1\] 04_UG_DS_Aug-Dec-2026.xlsx, tab "Core", row 10' \
        && ok "-k 1: the full DS3010 row is given, not the CSE stub" \
        || { bad "-k 1 gave the wrong row"; echo "$OUT" | head -5; }
else
    echo "  SKIP  no data/*.xlsx"
fi

echo "== --compact prompt (key + asked columns only)"
if [ $HAVE_DATA = 1 ] && python3 -c "import gguf, numpy" 2>/dev/null; then
    python3 tests/make_tiny_gguf.py build/rag_tiny_compact.gguf >/dev/null
    OUT=$(printf 'When and where is the Introduction to Machine Learning class?\n/exit\n' |
          ./build/llm_rag build/rag_tiny_compact.gguf --docs data --rag -k 1 -n 2 --show-context --compact 2>&1)
    if echo "$OUT" | grep -q '^Slot: D \[Mon,Wed\]' && echo "$OUT" | grep -q '^Room: A01-212' &&
       ! echo "$OUT" | grep -q '^Credits:'; then
        ok "--compact: slot and room kept, other columns left out"
    else
        bad "--compact prompt"; echo "$OUT" | sed -n '/prompt given/,/^---/p' | head -12
    fi
else
    echo "  SKIP  needs data/*.xlsx and python3 gguf + numpy"
fi

echo "== conversations (tests/rag_dialogues.txt)"
SKIP=""
[ $HAVE_DATA = 1 ] || SKIP="--skip-data"
python3 tests/check_rag_dialogues.py build/llm_rag_asan tests/rag_dialogues.txt $SKIP > build/rag_dialogues.log 2>&1 \
    && ok "$(tail -1 build/rag_dialogues.log)" || { bad "$(tail -1 build/rag_dialogues.log)"; head -n -1 build/rag_dialogues.log; }
grep -q "ERROR: \|runtime error" build/rag_dialogues.log && bad "sanitizer report (build/rag_dialogues.log)"

echo "== folder changes while llm_rag is running"
LIVE="build/rag_live"
rm -rf "$LIVE"; mkdir -p "$LIVE"
cp "$TEST/marks.xlsx" "$LIVE/"
{
    echo "Who teaches the Design Studio?"; sleep 1
    cp "$TEST/edge.xlsx" "$LIVE/"
    echo "Who teaches the Design Studio?"; sleep 1
    printf 'lock' > "$LIVE/~\$edge.xlsx"; printf 'not a workbook' > "$LIVE/broken.xlsx"
    echo "Who teaches the Design Studio?"; sleep 1
    touch -d "2001-01-01" "$LIVE/marks.xlsx"
    echo "grade of Asha Menon"; sleep 1
    rm "$LIVE/edge.xlsx"
    echo "Who teaches the Design Studio?"
    echo "/exit"
} | ./build/llm_rag_asan --retrieve-only --docs "$LIVE" > build/rag_live.log 2>&1
cp "$TEST/books.csv" "$LIVE/"
{
    echo "author of BK-310"; sleep 1
    printf 'Book ID,Title,Author\nBK-999,New Arrivals,Tester\n' > "$LIVE/new.csv"
    echo "author of BK-999"
} | ./build/llm_rag_asan --retrieve-only --docs "$LIVE" > build/rag_live_csv.log 2>&1
grep -q "Silberschatz" build/rag_live_csv.log && grep -q "added new.csv" build/rag_live_csv.log \
    && grep -q "^Tester" build/rag_live_csv.log \
    && ok "a CSV file dropped in is answered from at the next question" || bad "live CSV"

check_live() { grep -q -- "$1" build/rag_live.log && ok "$2" || bad "$2"; }
check_live "documents changed: added edge.xlsx" "new file picked up at the next question"
check_live "documents changed: added broken.xlsx" "broken file noticed..."
check_live "broken.xlsx could not be read and is skipped" "...and skipped, the others still answer"
grep -q 'added ~\$edge' build/rag_live.log && bad "lock file not ignored" || ok "Excel lock file ignored"
check_live "documents changed: changed marks.xlsx" "edited file re-read"
check_live "documents changed: removed edge.xlsx" "deleted file's rows dropped"
[ "$(grep -c 'edge.xlsx, tab Batches' build/rag_live.log)" -ge 2 ] && \
    tail -3 build/rag_live.log | grep -q "couldn't find" \
    && ok "answers follow the folder (found while present, gone after)" \
    || bad "answers do not follow the folder"
grep -q "ERROR\|runtime error" build/rag_live.log && bad "sanitizer report (build/rag_live.log)" \
    || ok "sanitizers clean during the live session"

echo "== the three modes with tiny test models (sanitizers)"
if python3 -c "import gguf, numpy" 2>/dev/null; then
    python3 tests/make_tiny_gguf.py build/rag_tiny.gguf >/dev/null
    printf '%s\n' "Who is the instructor for LB205?" "Physics Lab slot" \
        "What is the capital of France?" "grade of 21DS003" "/files" "/context" \
        "Pens price" "/exit" \
        | ./build/llm_rag_asan build/rag_tiny.gguf --rag -v --docs "$TEST" -n 16 > build/rag_tiny.log 2>&1
    status=$?
    n_ans=$(grep -c "row(s) in prompt" build/rag_tiny.log)
    if [ $status = 0 ] && [ "$n_ans" = 4 ] && grep -q "could not find" build/rag_tiny.log \
       && ! grep -q "ERROR: \|runtime error\|context full" build/rag_tiny.log; then
        ok "--rag: 4 answers + 1 not-found in one session, KV cache reset each time"
    else
        bad "tiny-model session (see build/rag_tiny.log)"
    fi
    ./build/llm_rag build/rag_tiny.gguf --rag --docs "$TEST" --ctx 330 -n 50 "Who is the instructor for LB205?" \
        > build/rag_budget.log 2>&1 </dev/null
    grep -q "more than the .* available" build/rag_budget.log \
        && ok "context budget: clear error when even one row cannot fit" || bad "context budget error"
    ./build/llm_rag build/rag_tiny.gguf --rag -v --docs "$TEST" --ctx 900 -n 40 -k 3 "Who teaches Physics Lab and Circuits Lab?" \
        > build/rag_budget.log 2>&1 </dev/null
    grep -q "1 row(s) in prompt\|2 row(s) in prompt" build/rag_budget.log \
        && ok "context budget: lowest-ranked rows dropped to fit" || bad "context budget drop"
    # the model alone: chat keeps its history, /reset clears it
    printf '%s\n' "hello" "again" "/RESET" "hi" "/exit" \
        | ./build/llm_rag_asan build/rag_tiny.gguf -v -n 5 --sampler greedy > build/rag_chat.log 2>&1
    if [ $? = 0 ] && grep -q "context 39/" build/rag_chat.log && grep -q "context 78/" build/rag_chat.log \
       && grep -q "conversation reset" build/rag_chat.log && grep -q "context 36/" build/rag_chat.log \
       && ! grep -q "ERROR: \|runtime error" build/rag_chat.log; then
        ok "model only (chat): history kept between turns, /reset clears it, no files read"
    else
        bad "model-only chat (see build/rag_chat.log)"
    fi
    # the same tokens as llm_generate for the same chat prompt
    a=$(./build/llm_generate build/rag_tiny.gguf --chat -n 12 --sampler greedy "hello there" 2>/dev/null)
    b=$(./build/llm_rag build/rag_tiny.gguf -n 12 --sampler greedy "hello there" 2>/dev/null </dev/null)
    [ -n "$a" ] && [ "$a" = "$b" ] && ok "model only: same reply as llm_generate --chat" \
                                   || bad "model-only reply differs from llm_generate"
    ./build/llm_rag --rag > build/rag_usage.log 2>&1 </dev/null
    grep -q "needs a chat model" build/rag_usage.log && ok "--rag without a model: clear message" || bad "--rag usage"
else
    echo "  SKIP  pip install gguf numpy to run this part"
fi

echo "== mutation test"
MUT_DATA=""
[ $HAVE_DATA = 1 ] && MUT_DATA="$PWD/data"
python3 tests/mutation_rag.py "$PWD" "$PWD/$TEST" $MUT_DATA > build/rag_mut.log 2>&1 \
    && ok "$(tail -1 build/rag_mut.log)" || { bad "mutation"; cat build/rag_mut.log; }

if [ -n "$MODEL" ]; then
    echo "== answers from $MODEL (tests/rag_answers.tsv)"
    while IFS=$'\t' read -r q want; do
        case "$q" in ''|'#'*) continue ;; esac
        ans=$(./build/llm_rag "$MODEL" --rag --docs data "$q" 2>/dev/null </dev/null | sed '/^Sources:/,$d')
        if echo "$ans" | grep -qi -- "$want"; then
            ok "\"$q\" -> contains \"$want\""
        else
            bad "\"$q\" -> expected \"$want\", got: $(echo "$ans" | tr '\n' ' ' | cut -c1-120)"
        fi
    done < tests/rag_answers.tsv
fi

echo
echo "rag checks: $PASS passed, $FAIL failed"
[ $FAIL = 0 ]
