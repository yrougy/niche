#!/bin/sh
# name, then extra llama-cli args; STATIC=0 runs the LRU cache instead of the static one
name=$1; shift
extra="--moe-cache-static"
if [ "$STATIC" = "0" ]; then extra=""; fi
R=$(cd "$(dirname "$0")" && pwd); LLAMA=${LLAMA:-$R/../llama.cpp}; MODEL=${MODEL:-$R/../models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf}
cd "$LLAMA"
./build/bin/llama-cli -v -m "$MODEL" -cmoe -ngl 99 \
  --moe-cache-mib 1024 --moe-profile /tmp/prof2.csv $extra -c 2048 -n 128 \
  --temp 0 -s 42 -p "Q: What is the capital of France? A:" --no-display-prompt "$@" \
  > /tmp/$name.txt 2>/tmp/$name.log &
pid=$!
echo "PID=$pid"
python3 $R/majflt2.py "$pid" 900 sda /tmp/$name.txt
wait "$pid"
echo "EXIT=$?"
