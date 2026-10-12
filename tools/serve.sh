#!/bin/bash
# serve.sh — llama-server avec le cache MoE par expert, config optimale mesurée.
# Avant-plan, host 0.0.0.0, port 8002. Ctrl-C pour arrêter.
# Config de référence : static bank 5632 + MTP à CTX 4096 → tg ~39,8 / pp ~446.
set -u

CTX=${CTX:-4096}   # ← taille du contexte : à éditer ici, ou CTX=8192 tools/serve.sh

LLAMA=/data/o35/llama.cpp
BIN=$LLAMA/build/bin/llama-server
MODEL=/data/models/unsloth/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
PROFILE=/data/o35/o35/tools/prof2.csv
HOST=0.0.0.0
PORT=8002

# bank GPU max avec MTP (mesuré sur cette machine, VRAM 8 GiB) :
# 5632 tient à CTX 4096, 5120 à CTX 8192 ; au-delà on réduit d'environ 512 MiB par 8k tokens.
if [ "$CTX" -le 4096 ]; then
    BANK=5632
elif [ "$CTX" -le 8192 ]; then
    BANK=5120
else
    BANK=$((5120 - (CTX - 8192) / 8192 * 512))
    [ "$BANK" -ge 4096 ] || BANK=4096
fi
BANK=${BANK_OVERRIDE:-$BANK}   # BANK_OVERRIDE=5120 tools/serve.sh pour forcer le bank

echo "llama-server cache MoE : CTX=$CTX bank=${BANK} MiB http://$HOST:$PORT (Ctrl-C pour arrêter)"

exec "$BIN" -m "$MODEL" \
    -ngl 99 -cmoe \
    --moe-cache-mib "$BANK" --moe-profile "$PROFILE" --moe-cache-static --moe-cache-pin-mib 1990 \
    --spec-type draft-mtp \
    --host "$HOST" --port "$PORT" \
    -c "$CTX" -fa on --cache-type-k q4_0 --cache-type-v q4_0 -ub 512 \
    -t 8 --threads-batch 8 --parallel 1
