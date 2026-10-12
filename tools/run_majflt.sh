#!/bin/sh
cd /data/o35/llama.cpp
./build/bin/llama-cli -v -m /data/models/unsloth/Qwen3.6-35B-A3B-UD-IQ2_M.gguf -cmoe -ngl 99 \
  --moe-cache-mib 1024 --moe-profile /tmp/prof2.csv --moe-cache-static -c 2048 -n 128 \
  --temp 0 -s 42 -p "Q: What is the capital of France? A:" --no-display-prompt \
  > /tmp/lru128b.txt 2>/tmp/lru128b.log &
pid=$!
echo "PID=$pid"
python3 /data/o35/o35/tools/majflt.py "$pid" 200
wait "$pid"
echo "EXIT=$?"
