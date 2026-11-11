#!/usr/bin/env bash
#
# fix_example_paths.sh - after moving example files into subfolders by hand,
# update every path that refers to them.
#
# Usage (from the project root, after moving the files):
#   bash tests/fix_example_paths.sh
#
# For every file now inside a subfolder of examples/, it rewrites
#   examples/<file>   ->   examples/<subfolder>/<file>
# in the Makefile, tests/*.sh, include/*.h and every file under examples/.
# Paths that already include a subfolder are not touched, so running it
# twice is safe.

set -euo pipefail

if [ ! -f Makefile ] || [ ! -d examples ]; then
    echo "Run this from the project root (where the Makefile is)."
    exit 1
fi

TEXT_FILES=$(ls Makefile tests/*.sh include/*.h 2>/dev/null; \
             find examples -type f \( -name "*.c" -o -name "*.py" -o -name "*.h" \))

count=0
for path in $(find examples -mindepth 2 -type f \( -name "*.c" -o -name "*.py" \) \
              -not -path "*/__pycache__/*" | sort); do
    name=$(basename "$path")
    relative=${path#examples/}                       # e.g. 03_quantization/gguf_q6k_demo.c
    escaped=$(printf '%s' "$name" | sed 's/\./\\./g')
    for t in $TEXT_FILES; do
        # Only "examples/<name>" exactly (not already "examples/<folder>/<name>").
        sed -i "s#examples/${escaped}#examples/${relative}#g" "$t"
    done
    count=$((count + 1))
done

left=$(find examples -maxdepth 1 -type f \( -name "*.c" -o -name "*.py" \))

echo "Updated paths for $count files in subfolders."
if [ -n "$left" ]; then
    echo "Still directly in examples/ (paths unchanged):"
    echo "$left"
fi

# Report any Makefile rule that now points at a file that does not exist.
missing=0
for src in $(grep -o 'examples/[^ ]*\.c' Makefile | sort -u); do
    if [ ! -f "$src" ]; then
        echo "Makefile refers to a missing file: $src"
        missing=1
    fi
done
for src in $(grep -oh 'examples/[^ "]*\.py' tests/*.sh | sort -u); do
    if [ ! -f "$src" ]; then
        echo "A test script refers to a missing file: $src"
        missing=1
    fi
done

if [ "$missing" -eq 0 ]; then
    echo "All paths in the Makefile and test scripts point to existing files."
    echo "Now run:  make clean && make && make gguf-tools && make test"
else
    exit 1
fi
