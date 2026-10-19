#!/bin/bash
# profile.sh — génère un profil de routage MoE COMPLET (tous les picks de tous les tokens).
# Le hook --moe-profile-output est ignoré quand le cache MoE est actif (il changerait les
# splits du scheduler) : on profile donc dans la config off (auto-offload, sans -cmoe ni -ngl),
# le routage ne dépend pas de l'endroit où vivent les poids d'experts.
# Le trafic rejoue celui de bench.sh (warmup, pp, tg) pour que le seed corresponde au bench.
# Usage : tools/profile.sh [fichier de sortie]   (défaut : tools/prof3.csv)
# Env : PORT (8002), PP_N (4096), TG_N (256), TG_ROUNDS (2), CTX (8192), THREADS (8)
set -u

HERE=${HERE:-$(cd "$(dirname "$0")/.." && pwd)}
LLAMA=${LLAMA:-$HERE/llama.cpp}
BIN=$LLAMA/build/bin/llama-server
MODEL=${MODEL:-$HERE/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf}
OUT=${1:-$HERE/tools/prof3.csv}
case "$OUT" in /*) ;; *) OUT=$PWD/$OUT ;; esac
LOG=/tmp/profile/server.log
PORT=${PORT:-8002}
PP_N=${PP_N:-4096}
TG_N=${TG_N:-256}
TG_ROUNDS=${TG_ROUNDS:-2}
CTX=${CTX:-8192}
THREADS=${THREADS:-8}

fail() { echo "ERREUR : $*" >&2; exit 1; }
[ -x "$BIN" ] || fail "binaire $BIN absent"
[ -f "$MODEL" ] || fail "modèle $MODEL absent"

mkdir -p /tmp/profile && rm -f "$LOG" "$OUT" "$OUT.tmp"

cleanup() {
    if [ -n "${SRV:-}" ]; then
        kill "$SRV" 2>/dev/null
        # le CSV est écrit à la libération du contexte : attendre la fin du serveur
        local i=0
        while kill -0 "$SRV" 2>/dev/null && [ $i -lt 150 ]; do sleep 0.2; i=$((i+1)); done
        kill -9 "$SRV" 2>/dev/null
    fi
}
trap cleanup EXIT INT TERM

echo "=== démarrage llama-server (config off, profil -> $OUT) ==="
cd "$LLAMA"
"$BIN" -m "$MODEL" --moe-profile-output "$OUT" --host 127.0.0.1 --port "$PORT" \
    -c "$CTX" -fa on --cache-type-k q4_0 --cache-type-v q4_0 -ub 512 \
    -t "$THREADS" --threads-batch "$THREADS" --parallel 1 --temp 0 -s 42 \
    > "$LOG" 2>&1 &
SRV=$!

for _ in $(seq 1 300); do
    curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" && break
    kill -0 "$SRV" 2>/dev/null || fail "serveur mort au démarrage — voir $LOG"
    sleep 2
done
curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" || fail "serveur pas prêt — voir $LOG"
echo "serveur prêt (pid $SRV)"

req() {
    curl -s -X POST "http://127.0.0.1:$PORT/completion" \
        -H "Content-Type: application/json" -d "$1"
}

SENT="The quick brown fox jumps over the lazy dog. "   # ~10 tokens par répétition
LONGP=$(python3 -c "print('$SENT' * $((PP_N / 10)))")

echo "=== warmup (16 tokens) ==="
req '{"prompt":"Hello.","max_tokens":16,"temperature":0,"stream":false}' > /tmp/profile/warmup.json

echo "=== prompt processing ($PP_N tokens, max_tokens=1) ==="
req "{\"prompt\":\"$LONGP\",\"max_tokens\":1,\"temperature\":0,\"stream\":false}" > /tmp/profile/pp.json

echo "=== génération ($TG_N tokens x $TG_ROUNDS) ==="
for i in $(seq 1 "$TG_ROUNDS"); do
    req "{\"prompt\":\"Q: What is the capital of France? A:\",\"max_tokens\":$TG_N,\"temperature\":0,\"ignore_eos\":true,\"stream\":false}" > "/tmp/profile/tg$i.json"
done

cleanup; SRV=""

echo
grep -Ei "profile" "$LOG" | tail -3
if [ -f "$OUT" ]; then
    python3 - "$OUT" <<'EOF'
import sys, collections
per_layer = collections.defaultdict(int)
n = 0
for line in open(sys.argv[1]):
    if line.startswith('layer') or not line.strip(): continue
    l, e, c, w = line.split(',')
    per_layer[int(l)] += int(c); n += 1
tot = sum(per_layer.values())
vals = sorted(per_layer.values())
print("paires=%d couches=%d sum(count)=%d  min/max par couche=%d/%d" % (n, len(per_layer), tot, vals[0], vals[-1]))
print("tokens implicites (count/8) min/max par couche : %d/%d" % (vals[0]//8, vals[-1]//8))
EOF
else
    echo "AUCUN CSV produit — voir $LOG"
fi
