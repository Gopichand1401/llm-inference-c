#!/bin/sh
# Builds the engine if needed, then runs llm_generate. Extra arguments are
# passed on, and a later option overrides an earlier default.
# Default model: models/stories42M.bin (change with MODEL=...),
# 150 new tokens, greedy sampling. Run it from anywhere.
#   sh run.sh                                   interactive mode
#   sh run.sh "Once upon a time"                one prompt
#   sh run.sh "Once upon a time" -n 200         choose the length
#   sh run.sh "Once upon a time" --sampler temperature --temperature 0.8
#   sh run.sh "Once upon a time" --sampler top-k --top-k 40 --seed 7
#   MODEL=~/models/stories110M.bin sh run.sh "Once upon a time"
#   MODEL=~/models/tinyllama.gguf sh run.sh            chat with TinyLlama
cd "$(dirname "$0")" || exit 1
make -s >/dev/null || { echo "Build failed."; exit 1; }
MODEL="${MODEL:-models/stories42M.bin}"
if [ ! -f "$MODEL" ]; then
    echo "Model not found: $MODEL  (set MODEL=path/to/model)"
    exit 1
fi
# No arguments: interactive mode.
if [ $# -eq 0 ]; then
    set -- -i
fi
exec ./build/llm_generate "$MODEL" -n 150 --sampler greedy "$@"
