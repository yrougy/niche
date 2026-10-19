#!/bin/bash
# serve-lfm.sh — llama-server LFM2.5-24B-A2B avec le cache MoE par expert, grand contexte,
# API OpenAI-compatible sur 0.0.0.0:8002 (mêmes flags d'API que serve.sh). Ctrl-C pour arrêter.
# Config mesurée (GTX 1070 8 GiB / 16 GB RAM, seed prof-lfm.csv) :
#   CTX 65536 + bank 6656 → tg 47,4 / pp 972 / hit 99,2 %, 0 majflt en décode.
#   CTX 131072 (max natif du modèle) + bank 6144 → tg 46,5 / pp 949 / hit 98,2 % (16/10 ; 6400 = OOM).
#   Bank 6912 = OOM CUDA à 65536 ; 7168 = OOM même à 8192.
#   Plate de 8k à 64k, puis pente ≈ 48 MiB/8k (modèle hybride : ~11/40 couches KV, q4_0).
# Pas de tête MTP dans ce GGUF → pas de --spec-type.
# Note : le template chat LFM2.5 active le mode raisonnement par défaut → la sortie textuelle
# arrive dans reasoning_content du JSON, content peut rester vide (test API 14/10/2026).
# Env : CTX (65536), BANK_OVERRIDE, MODEL, PROFILE, PIN, PORT, HOST, ENVFILE=tools/machine-lfm.env,
#       MOE_STATS=/tmp/moe-stats-lfm.csv (csv hit rate par couche/expert : écrit au chargement,
#       compteurs finaux flushés à l'arrêt propre SIGINT/Ctrl-C)
# Échantillonnage conforme à la model card LFM2.5 (relevé 16/10) : TEMP=0.1 TOPK=50 RPEN=1.05.
# (bench/profile/verify gardent --temp 0 -s 42 : déterminisme des mesures.)
set -u

HERE=${HERE:-$(cd "$(dirname "$0")/.." && pwd)}
ENVFILE=${ENVFILE:-$HERE/tools/machine-lfm.env}   # tools/calibrate.sh lfm … écrit ce fichier
[ -f "$ENVFILE" ] && source "$ENVFILE"

LLAMA=${LLAMA:-$HERE/llama.cpp}
BIN=$LLAMA/build/bin/llama-server
MODEL=${MODEL:-$HERE/models/LFM2.5-24B-A2B-UD-IQ3_XXS.gguf}
PROFILE=${PROFILE:-$HERE/tools/prof-lfm.csv}   # toujours absolu : le serveur ne résout pas le relatif
CTX=${CTX:-65536}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8002}
PIN=${PIN:-1990}
MOE_STATS=${MOE_STATS:-/tmp/moe-stats-lfm.csv}
TEMP=${TEMP:-0.1}     # model card LFM2.5
TOPK=${TOPK:-50}      # model card LFM2.5
RPEN=${RPEN:-1.05}    # model card LFM2.5

fail() { echo "ERREUR : $*" >&2; exit 1; }
[ -x "$BIN" ] || fail "binaire $BIN absent — construire llama.cpp (branche moe-expert-cache)"
[ -f "$MODEL" ] || fail "modèle $MODEL absent"
[ -f "$PROFILE" ] || fail "profil $PROFILE absent — tools/profile.sh MODEL=$MODEL tools/prof-lfm.csv"

# plafond de bank mesuré sur cette machine : 6656 tient de CTX 8192 à 65536 ; à 131072 le plafond
# tombe à ~6272 (6144 OK, 6400 OOM) → pente réelle ≈ 48 MiB/8k, on retire 64 MiB/8k (marge incluse).
if [ "$CTX" -le 65536 ]; then
    BANK=6656
else
    BANK=$(( 6656 - (CTX - 65536) / 8192 * 64 ))
    [ "$BANK" -ge 4096 ] || BANK=4096
fi
BANK=${BANK_OVERRIDE:-$BANK}
[ -n "${BANK_MAX:-}" ] && [ "$BANK" -gt "$BANK_MAX" ] && BANK=$BANK_MAX   # plafond mesuré par calibrate.sh

echo "llama-server LFM2.5-24B-A2B cache MoE : CTX=$CTX bank=${BANK} MiB pin=${PIN} MiB http://$HOST:$PORT (Ctrl-C pour arrêter)"

CORSARGS=""; [ -n "${CORS:-}" ] && CORSARGS="--cors-origins $CORS"   # ex. CORS=http://hote:8002 pour autoriser une UI web distante
exec "$BIN" -m "$MODEL" \
    -ngl 99 -cmoe \
    --moe-cache-mib "$BANK" --moe-profile "$PROFILE" --moe-cache-static --moe-cache-pin-mib "$PIN" \
    --moe-stats "$MOE_STATS" \
    --host "$HOST" --port "$PORT" \
    -c "$CTX" -fa on --cache-type-k q4_0 --cache-type-v q4_0 -ub 512 \
    -t "${THREADS:-8}" --threads-batch "${THREADS:-8}" --parallel 1 \
    --temp "$TEMP" --top-k "$TOPK" --repeat-penalty "$RPEN" --metrics --slots \
    --ui-mcp-proxy --webui-mcp-proxy $CORSARGS --tools exec_shell_command
