#!/bin/bash
# try.sh — essaie le port : cache MoE statique + pin RAM + fallback CPU par expert.
# Usage : tools/try.sh [warm|cold]   (cold = vide le page cache avant le run, si root)
# Résultat attendu : majflt = 0 pendant le décode et sortie bit-identique à masters/clean128.txt.
set -u

LLAMA=/data/o35/llama.cpp
BIN=$LLAMA/build/bin/llama-cli
MODEL=/data/models/unsloth/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
HERE=/data/o35/o35
PROFILE=$HERE/tools/prof2.csv
MASTER=$HERE/masters
PROMPT="Q: What is the capital of France? A:"
OUT=/tmp/try
MODE=${1:-warm}

fail() { echo "ERREUR : $*" >&2; exit 1; }

# --- prérequis ---
[ -x "$BIN" ] || fail "binaire $BIN absent — construire llama.cpp sur moe-expert-cache"
[ -f "$MODEL" ] || fail "modèle $MODEL absent"
[ -f "$PROFILE" ] || fail "profil $PROFILE absent"
BR=$(cd "$LLAMA" && git branch --show-current)
[ "$BR" = "moe-expert-cache" ] || fail "branche llama.cpp = $BR, attendu moe-expert-cache"

if [ "$MODE" = cold ]; then
    if [ "$(id -u)" = 0 ]; then sync; echo 3 > /proc/sys/vm/drop_caches; echo "page cache vidé"
    elif sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then echo "page cache vidé (sudo)"
    else echo "AVERTISSEMENT : pas root, page cache non vidé — le run cold sera approximatif"
    fi
fi

mkdir -p "$OUT" && rm -f "$OUT"/*

echo "=== 1. Run : cache statique + fallback CPU + pin 1990 MiB (majflt mesuré) ==="
cd "$LLAMA"
timeout 1200 "$BIN" -v -m "$MODEL" -cmoe -ngl 99 \
    --moe-cache-mib 1024 --moe-profile "$PROFILE" \
    --moe-cache-static --moe-cache-cpu-fallback --moe-cache-pin-mib 1990 \
    -c 2048 -n 128 --temp 0 -s 42 -p "$PROMPT" --no-display-prompt \
    > "$OUT/cpufb.txt" 2> "$OUT/cpufb.log" &
pid=$!
echo "PID=$pid (mesure majflt via tools/majflt2.py)"
python3 "$HERE/tools/majflt2.py" "$pid" 900 sda "$OUT/cpufb.txt" > "$OUT/majflt.samples"
wait "$pid"; rc=$?
echo "exit=$rc"
grep -Ei "ABORT|CUDA error|hit rate|uploaded" "$OUT/cpufb.log" | tail -5
echo "--- derniers samples majflt (t majflt minflt disk_read outfile_size) ---"
tail -5 "$OUT/majflt.samples"

echo
echo "=== 2. Comparaison bit-à-bit avec la baseline clean128 ==="
python3 "$HERE/tools/cmp.py" "$MASTER/clean128.txt" "$OUT/cpufb.txt"

echo
echo "Fait. Tout est dans $OUT/ (cpufb.txt, cpufb.log, majflt.samples)."
echo "Régression complète (tous les modes + prompt B) : sh $HERE/tools/verify.sh"
