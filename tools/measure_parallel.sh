#!/bin/bash
# Count the real compiler processes during a rebuild, and the CPU time they burn.
HERE=${HERE:-$(cd "$(dirname "$0")/.." && pwd)}; LLAMA=${LLAMA:-$HERE/llama.cpp}
touch "$LLAMA/src/llama-moe-cache.cpp"
( for i in $(seq 1 240); do
    a=$(pgrep -cx cc1plus 2>/dev/null); b=$(pgrep -cx ptxas 2>/dev/null); c=$(pgrep -cx cicc 2>/dev/null)
    echo "$((a+b+c))"
    sleep 0.5
  done ) > /tmp/par2.log &
cd "$LLAMA" && /snap/bin/cmake --build build --config Release -j8 > /tmp/par_build2.log 2>&1
echo "exit=$?"
wait
awk '{s+=$1; if($1>m) m=$1} END {print "samples="NR" max="m" avg="s/NR}' /tmp/par2.log
grep -c "^\[" /tmp/par_build2.log
grep "^\[" /tmp/par_build2.log | tail -3
