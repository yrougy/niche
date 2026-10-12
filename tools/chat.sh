#!/bin/bash
# chat.sh — session interactive avec le cache MoE porté (llama-cli en mode conversation).
# Usage : tools/chat.sh [fb|static|lru|off]   (défaut : fb = port complet)
#   fb     : --moe-cache-static + --moe-cache-cpu-fallback + pin 1990 (config mesurée)
#   static : cache statique + pin, sans fallback CPU
#   lru    : cache LRU (sans --moe-cache-static)
#   off    : -cmoe sans cache (baseline amont)
# Env : CACHE_MIB (taille du bank GPU, défaut 4096).
# Sortie de la session : Ctrl+C. Un résumé (hit rate, majflt, lectures disque) s'affiche après.
set -u

LLAMA=/data/o35/llama.cpp
BIN=$LLAMA/build/bin/llama-cli
MODEL=/data/models/unsloth/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
HERE=/data/o35/o35
PROFILE=$HERE/tools/prof2.csv
OUT=/tmp/chat
MODE=${1:-fb}
CACHE_MIB=${CACHE_MIB:-4096}

fail() { echo "ERREUR : $*" >&2; exit 1; }
[ -x "$BIN" ] || fail "binaire $BIN absent — construire llama.cpp sur experimental/ornith-expert-cache"
[ -f "$MODEL" ] || fail "modèle $MODEL absent"
[ -f "$PROFILE" ] || fail "profil $PROFILE absent"

case "$MODE" in
    fb)     CACHE="--moe-cache-mib $CACHE_MIB --moe-profile $PROFILE --moe-cache-static --moe-cache-cpu-fallback --moe-cache-pin-mib 1990" ;;
    static) CACHE="--moe-cache-mib $CACHE_MIB --moe-profile $PROFILE --moe-cache-static --moe-cache-pin-mib 1990" ;;
    lru)    CACHE="--moe-cache-mib $CACHE_MIB --moe-profile $PROFILE" ;;
    off)    CACHE="" ;;
    *) fail "mode inconnue : $MODE (fb|static|lru|off)" ;;
esac

mkdir -p "$OUT" && rm -f "$OUT"/*
MON=$OUT/monitor.txt
LOG=$OUT/session.log

echo "Mode : $MODE"
echo "Chargement du modèle (~12 Go depuis le HDD, plusieurs minutes), puis session interactive."
echo "Tapez votre message puis Entrée. Pour quitter : Ctrl+C."
echo

# moniteur d'I/O en arrière-plan : majflt du processus + lectures disque (sda)
(
    pid=""
    while :; do pid=$(pgrep -f "build/bin/llama-cli" | head -1); [ -n "$pid" ] && break; sleep 1; done
    while kill -0 "$pid" 2>/dev/null; do
        s=$(cat "/proc/$pid/stat" 2>/dev/null) || break
        r=${s##*) }; set -- $r
        d=$(awk '$3=="sda"{printf "%d", $6*512}' /proc/diskstats)
        echo "$(date +%s) minflt=$8 majflt=${10} disk_bytes=$d" >> "$MON"
        sleep 2
    done
) &
WATCHPID=$!

cd "$LLAMA"
"$BIN" -m "$MODEL" -cmoe -ngl 99 $CACHE --moe-stats "$OUT/stats.csv" \
    -c 2048 --temp 0 -s 42 2> >(tee -a "$LOG" >&2)

kill "$WATCHPID" 2>/dev/null

echo
echo "=== Résumé de la session ==="
echo "--- hit rate / erreurs (log) ---"
grep -Ei "hit rate|ABORT|CUDA error" "$LOG" | tail -10
if [ -s "$MON" ]; then
    echo "--- I/O sur toute la session (moniteur 2 s) ---"
    awk '{split($3,a,"="); split($4,c,"=")}
         NR==1{m0=a[2]; d0=c[2]}
         {m=a[2]; d=c[2]}
         END{printf "majflt delta = %d, lectures disque = %.2f MiB, samples = %d\n", m-m0, (d-d0)/1048576, NR}' "$MON"
fi
echo "Fichiers : $LOG, $MON, $OUT/stats.csv (--moe-stats CSV)."
