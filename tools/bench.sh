#!/bin/bash
# bench.sh — bench de llama-server avec le cache MoE porté : pp et tg mesurés via les
# timings de /completion (stream=false). Comparaison des modes de cache, avec ou sans MTP.
# Flags serveur alignés sur le config du billet rougy.net (26/05/2026) : -fa on, KV q4_0,
# -ub 512, -t 8 --threads-batch 8, --parallel 1 — SAUF --no-mmap/--cache-ram 0 incompatibles
# avec le port (populate/pin travaillent sur le mmap).
# Usage : tools/bench.sh [fb|static|lru|off] [mtp]
#   fb     : --moe-cache-static + --moe-cache-cpu-fallback + pin 1990 (config mesurée)
#   static : cache statique + pin, sans fallback CPU
#   lru    : cache LRU (sans --moe-cache-static)
#   off    : config exacte du billet (auto-offload amont, sans -cmoe ni -ngl)
#   mtp    : + --spec-type draft-mtp (tête MTP native du GGUF, nextn_predict_layers=1)
# Env : PORT (défaut 8002), CACHE_MIB (taille du bank GPU, défaut 4096), COLD=1 (vide le
# page cache avant, si root/sudo), CTX (défaut 8192), PP_N (tokens du test pp, défaut 2048),
# TG_N (tokens du test tg, défaut 128), DEPTH (tokens de remplissage KV avant mesure, défaut 0),
# TAIL_SKIP (avec fb : skip les picks non résidents au rang ≥ N, défaut 0 = désactivé).
set -u

LLAMA=/data/o35/llama.cpp
BIN=$LLAMA/build/bin/llama-server
MODEL=${MODEL:-/data/models/unsloth/Qwen3.6-35B-A3B-UD-IQ2_M.gguf}
HERE=/data/o35/o35
PROFILE=${PROFILE:-$HERE/tools/prof3.csv}
OUT=/tmp/bench
MODE=${1:-fb}
CACHE_MIB=${CACHE_MIB:-4096}
CTX=${CTX:-8192}
PP_N=${PP_N:-2048}
TG_N=${TG_N:-128}
THREADS=${THREADS:-8}
DEPTH=${DEPTH:-0}
TAIL_SKIP=${TAIL_SKIP:-0}
SPEC=""
[ "${2:-}" = mtp ] && SPEC="--spec-type draft-mtp"
PORT=${PORT:-8002}

fail() { echo "ERREUR : $*" >&2; exit 1; }
[ -x "$BIN" ] || fail "binaire $BIN absent — construire llama.cpp sur experimental/ornith-expert-cache"
[ -f "$MODEL" ] || fail "modèle $MODEL absent"
[ -f "$PROFILE" ] || fail "profil $PROFILE absent"

case "$MODE" in
    fb)     BASE="-ngl 99 -cmoe"; CACHE="--moe-cache-mib $CACHE_MIB --moe-profile $PROFILE --moe-cache-static --moe-cache-cpu-fallback --moe-cache-pin-mib 1990"; [ "$TAIL_SKIP" != 0 ] && CACHE="$CACHE --moe-cache-tail-skip $TAIL_SKIP" ;;
    static) BASE="-ngl 99 -cmoe"; CACHE="--moe-cache-mib $CACHE_MIB --moe-profile $PROFILE --moe-cache-static --moe-cache-pin-mib 1990"; [ "$TAIL_SKIP" != 0 ] && CACHE="$CACHE --moe-cache-tail-skip $TAIL_SKIP" ;;
    lru)    BASE="-ngl 99 -cmoe"; CACHE="--moe-cache-mib $CACHE_MIB --moe-profile $PROFILE"; [ "$TAIL_SKIP" != 0 ] && CACHE="$CACHE --moe-cache-tail-skip $TAIL_SKIP" ;;
    off)    BASE=""; CACHE="" ;;
    *) fail "mode inconnue : $MODE (fb|static|lru|off)" ;;
esac

mkdir -p "$OUT" && rm -f "$OUT"/*
MON=$OUT/monitor.txt
LOG=$OUT/server.log

if [ "${COLD:-0}" = 1 ]; then
    if [ "$(id -u)" = 0 ]; then sync; echo 3 > /proc/sys/vm/drop_caches; echo "page cache vidé"
    elif sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; then echo "page cache vidé (sudo)"
    else echo "AVERTISSEMENT : pas root, page cache non vidé"
    fi
fi

cleanup() {
    if [ -n "${SRV:-}" ]; then
        kill "$SRV" 2>/dev/null
        # attendre la fin du serveur : les stats hit rate sont émises par ~llama_moe_cache à la libération du contexte
        local i=0
        while kill -0 "$SRV" 2>/dev/null && [ $i -lt 150 ]; do sleep 0.2; i=$((i+1)); done
        kill -9 "$SRV" 2>/dev/null
    fi
    [ -n "${WATCHPID:-}" ] && kill "$WATCHPID" 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "=== démarrage llama-server (mode : $MODE${TAIL_SKIP:+ tail-skip $TAIL_SKIP}${SPEC:++ MTP}) ==="
cd "$LLAMA"
"$BIN" -m "$MODEL" $BASE $CACHE $SPEC --host 127.0.0.1 --port "$PORT" \
    -c "$CTX" -fa on --cache-type-k q4_0 --cache-type-v q4_0 -ub 512 \
    -t "$THREADS" --threads-batch "$THREADS" --parallel 1 --temp 0 -s 42 \
    --moe-stats "$OUT/stats.csv" > "$LOG" 2>&1 &
SRV=$!

for _ in $(seq 1 300); do
    curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && break
    kill -0 "$SRV" 2>/dev/null || fail "serveur mort au démarrage — voir $LOG"
    sleep 2
done
curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" || fail "serveur pas prêt (chargement > 10 min ?) — voir $LOG"
echo "serveur prêt (pid $SRV, port $PORT)"

# moniteur d'I/O : majflt du serveur + lectures disque (sda)
(
    while kill -0 "$SRV" 2>/dev/null; do
        s=$(cat "/proc/$SRV/stat" 2>/dev/null) || break
        r=${s##*) }; set -- $r
        d=$(awk '$3=="sda"{printf "%d", $6*512}' /proc/diskstats)
        echo "$(date +%s) minflt=$8 majflt=${10} disk_bytes=$d" >> "$MON"
        sleep 2
    done
) &
WATCHPID=$!

req() {
    curl -s -X POST "http://127.0.0.1:$PORT/completion" \
        -H "Content-Type: application/json" -d "$1"
}

show() {
    python3 - "$1" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
t = d.get("timings", {})
if not t:
    print("PAS DE TIMINGS dans la réponse :", str(d)[:300]); sys.exit(1)
print("prompt_n=%s  pp=%.2f t/s  predicted_n=%s  tg=%.2f t/s  (%.1f s)" % (
    t.get("prompt_n"), t.get("prompt_per_second", 0),
    t.get("predicted_n"), t.get("predicted_per_second", 0),
    (t.get("prompt_ms", 0) + t.get("predicted_ms", 0)) / 1000.0))
EOF
}

SENT="The quick brown fox jumps over the lazy dog. "   # ~10 tokens par répétition
LONGP=$(python3 -c "print('$SENT' * $((PP_N / 10)))")
DEEPP=$(python3 -c "print('$SENT' * $((DEPTH / 10)))")

echo "=== warmup (16 tokens) ==="
req '{"prompt":"Hello.","max_tokens":16,"temperature":0,"stream":false}' > "$OUT/warmup.json"

if [ "$DEPTH" -gt 0 ]; then
    echo "=== remplissage KV ($DEPTH tokens) ==="
    req "{\"prompt\":\"$DEEPP\",\"max_tokens\":1,\"temperature\":0,\"stream\":false}" > "$OUT/depth.json"
    show "$OUT/depth.json"
fi

echo "=== prompt processing ($PP_N tokens, max_tokens=1) ==="
req "{\"prompt\":\"$LONGP\",\"max_tokens\":1,\"temperature\":0,\"stream\":false}" > "$OUT/pp.json"
show "$OUT/pp.json"

echo "=== génération ($TG_N tokens, ignore_eos) ==="
req "{\"prompt\":\"Q: What is the capital of France? A:\",\"max_tokens\":$TG_N,\"temperature\":0,\"ignore_eos\":true,\"stream\":false}" > "$OUT/tg.json"
show "$OUT/tg.json"

cleanup; SRV=""; WATCHPID=""

echo
echo "=== Résumé de la session ==="
if [ -s "$OUT/stats.csv" ]; then
    echo "--- stats cache (--moe-stats CSV) ---"
    cat "$OUT/stats.csv"
fi
grep -Ei "hit rate" "$LOG" | tail -5
if [ -n "$SPEC" ]; then
    echo "--- acceptance du draft MTP (log) ---"
    grep -Ei "accept|draft" "$LOG" | tail -5
fi
if [ -s "$MON" ]; then
    echo "--- I/O sur toute la session (moniteur 2 s) ---"
    awk '{split($3,a,"="); split($4,c,"=")}
         NR==1{m0=a[2]; d0=c[2]}
         {m=a[2]; d=c[2]}
         END{printf "majflt delta = %d, lectures disque = %.2f MiB, samples = %d\n", m-m0, (d-d0)/1048576, NR}' "$MON"
fi
echo "Fichiers : $LOG, $MON, $OUT/{warmup,pp,tg}.json"
