#!/bin/sh
# Trace every read() of the process so the disk reads of the generation window can be attributed to a file.
name=$1; n=$2; pin=$3
R=$(cd "$(dirname "$0")" && pwd); LLAMA=${LLAMA:-$R/../llama.cpp}; MODEL=${MODEL:-$R/../models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf}
cd "$LLAMA"
strace -f -e trace=read,pread64,readv,preadv -yy -tt -s 32 -o /tmp/$name.strace \
  ./build/bin/llama-cli -v -m "$MODEL" -cmoe -ngl 99 \
  --moe-cache-mib 1024 --moe-profile /tmp/prof2.csv --moe-cache-static -c 2048 -n $n --temp 0 -s 42 \
  -p "Q: What is the capital of France? A:" --no-display-prompt > /tmp/$name.txt 2> /tmp/$name.log
