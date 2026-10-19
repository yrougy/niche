#!/bin/bash
# serve.sh — llama-server avec le cache MoE par expert, config optimale mesurée.
# Avant-plan, host 0.0.0.0, port 8002. Ctrl-C pour arrêter.
# Config de référence : static bank 5632 + MTP à CTX 4096 → tg ~39,8 / pp ~446.
# Spec : SPEC= vaut pour --spec-type (liste virgules autorisée, ex. SPEC=draft-mtp,ngram-mod) ;
# défaut : draft-mtp si MODEL_HAS_NEXTN, sinon rien. SPEC=none désactive toute spec.
# Multi-tour : la réutilisation du KV du contexte est DÉJÀ active par défaut (fiche idle slot
# + cache de prompt -cram 8192) ; mesure 19/10 : tours 2+ ≈ 100 tokens pp seulement, branche
# (édition/reroll) ≈ 55. Inutile de passer --cache-reuse dans nos usages.
set -u

CTX=${CTX:-4096}   # ← taille du contexte : à éditer ici, ou CTX=8192 tools/serve.sh

HERE=${HERE:-$(cd "$(dirname "$0")/.." && pwd)}
LLAMA=$(realpath -m "${LLAMA:-$HERE/llama.cpp}")   # absolus : le serveur ne résout pas le relatif
BIN=$LLAMA/build/bin/llama-server
ENVFILE=${ENVFILE:-$HERE/tools/machine-qwen.env}   # tools/calibrate.sh qwen … écrit ce fichier
[ -f "$ENVFILE" ] && source "$ENVFILE"
MODEL=$(realpath -m "${MODEL:-$HERE/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf}")
PROFILE=$(realpath -m "${PROFILE:-$HERE/tools/prof3.csv}")
# tête MTP : honorée seulement si le GGUF a des couches nextn (calibrate.sh sonde le
# header et écrit MODEL_HAS_NEXTN). MTP=0/1 à l'export force malgré tout.
MTP=${MTP:-${MODEL_HAS_NEXTN:-1}}
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
[ -n "${BANK_MAX:-}" ] && [ "$BANK" -gt "$BANK_MAX" ] && BANK=$BANK_MAX   # plafond mesuré par calibrate.sh
PIN=${PIN:-1990}

echo "llama-server MoE cache: CTX=$CTX bank=${BANK} MiB http://$HOST:$PORT (Ctrl-C to stop)"

SPEC=${SPEC:-}
[ -z "$SPEC" ] && [ "$MTP" = 1 ] && SPEC=draft-mtp
SPECARGS=""; [ -n "$SPEC" ] && SPECARGS="--spec-type $SPEC"
CORSARGS=""; [ -n "${CORS:-}" ] && CORSARGS="--cors-origins $CORS"   # ex. CORS=http://hote:8002 pour autoriser une UI web distante
exec "$BIN" -m "$MODEL" \
    -ngl 99 -cmoe \
    --moe-cache-mib "$BANK" --moe-profile "$PROFILE" --moe-cache-static --moe-cache-pin-mib "$PIN" \
    $SPECARGS \
    --host "$HOST" --port "$PORT" \
    -c "$CTX" -fa on --cache-type-k q4_0 --cache-type-v q4_0 -ub 512 \
    -t 8 --threads-batch 8 --parallel 1 --metrics --slots --ui-mcp-proxy --webui-mcp-proxy $CORSARGS --tools exec_shell_command
