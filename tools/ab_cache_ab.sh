#!/bin/bash
# ab_session2.sh TAG SPEC — TAG ∈ {q-up,q-ours,lq-up,lq-ours} ; q=Qwen35B IQ2_M (CTX4096 bank5632), lq=LFM Q3_K_XL (CTX131072 bank5632)
set -u
pgrep -x llama-server >/dev/null && { pkill -x llama-server; sleep 3; }
TAG=$1; SPEC=${2:-none}
BIN=/data/o35/o35/llama.cpp/build/bin/llama-server
LOG=/tmp/ab-$TAG.log
case $TAG in
  q-*)  MODEL=/data/models/unsloth/Qwen3.6-35B-A3B-UD-IQ2_M.gguf; CTX=4096;  BANK=5632; PIN=1990; PROFILE=/data/o35/o35/tools/prof3.csv;;
  lq-*) MODEL=/data/o35/o35/models/LFM2.5-24B-A2B-UD-Q3_K_XL.gguf; CTX=131072; BANK=5632; PIN=1975; PROFILE=/data/tests/niche/tools/prof-lfm.csv;;
  *) echo "TAG invalide"; exit 1;;
esac
[[ $TAG == *-up ]] && BIN=/data/o35/o35/llama.cpp/build/bin-upstream/llama-server
COMMON="-m $MODEL -ngl 99 -cmoe -c $CTX -fa on --cache-type-k q4_0 --cache-type-v q4_0 -ub 512 -t 8 --threads-batch 8 --parallel 1 --host 127.0.0.1 --port 8003 --metrics -v"
if [[ $TAG == *-ours ]]; then
  CACHE="--moe-cache-mib $BANK --moe-profile $PROFILE --moe-cache-static --moe-cache-pin-mib $PIN --moe-cache-adapt 512 --moe-stats /tmp/moe-stats-$TAG.csv"
else
  CACHE="--moe-cache-mib $BANK"
fi
SPECARGS=""; [ "$SPEC" != "none" ] && SPECARGS="--spec-type $SPEC"
rm -f "$LOG"
setsid "$BIN" $COMMON $CACHE $SPECARGS > "$LOG" 2>&1 < /dev/null &
SRV=$!
for i in $(seq 1 240); do grep -q "listening on" "$LOG" && break; sleep 1; done
if ! grep -q "listening on" "$LOG"; then echo "ECHEC demarrage:"; grep -iE "error|oom|abort" "$LOG" | tail -5; kill -INT $SRV 2>/dev/null; exit 1; fi
grep -E "MoE cache size|slots \(|static MoE cache" "$LOG" | head -3
PORT=8003 timeout 480 python3 -u /tmp/ab_traffic.py "$TAG"
kill -INT $SRV; wait $SRV
echo "--- stats cache ($TAG):"
grep -E "hit rate" "$LOG" | tail -3
