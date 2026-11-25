#!/bin/bash
# serve-generic.sh — dépanneur générique : sert n'importe quel modèle installé par le menu
# niche.sh qui n'a PAS de script dédié mesuré (serve.sh/serve-lfm.sh gardent leurs bank et
# profils mesurés sur la machine de dev). Conçu pour les modèles du catalogue récents
# (Gemma-4-26B-A4B, GLM-4.7-Flash) et pour ceux piochés dans la liste unsloth sur HF.
# Usage : TAG=<tag> tools/serve-generic.sh   — le menu niche.sh passe TAG et MODEL tout seul.
# Sans calibrage, bank conservateur 3072 MiB + tools/prof-<TAG>.csv : c'est tools/calibrate.sh
# <TAG> <GGUF> qui mesure le plafond VRAM réel (BANK_MAX/BANK_BEST) et la pente KV
# (KV_MIB_PER_8K), écrits dans machine-<TAG>.env (sourcé ci-dessous).
# Hiérarchie (comme les autres serve) : env exporté > tools/settings-<TAG>.env > machine-<TAG>.env.
# KV : KVQ= q8_0 | q4_0 (défaut) | f16. Spec : SPEC= (défaut : draft-mtp si le GGUF a des
# couches nextn — MODEL_HAS_NEXTN écrit par calibrate.sh — sinon ngram-mod ; none désactive).
# Adapt : ADAPT= (--moe-cache-adapt), défaut 512, 0 désactive.
# Dense : les modèles SANS experts (ex. Gemma-4 12B) sont sondés dans le GGUF (tools/gguf_meta.py
# --flags) et servis sans cache MoE ni profil : pas de --moe-*, pas de bank, pas de PIN. Forcer :
# IS_MOE=1|0 IS_NEXTN=1|0.
# EXTRA= options llama-server personnelles, ajoutées en dernier (donc prioritaires) : ex.
# brouillon MTP séparé de Gemma-4 : EXTRA=--model-draft models/mtp-gemma-4-26B-A4B-it.gguf
set -u

fail() { echo "error: $*" >&2; exit 1; }

HERE=${HERE:-$(cd "$(dirname "$0")/.." && pwd)}
TAG=${TAG:?usage: TAG=<tag> tools/serve-generic.sh — le menu niche.sh le passe automatiquement}
# choix enregistrés par le menu niche.sh (env exporté gagne, puis settings, puis calibrage) :
SETFILE=${SETFILE:-$HERE/tools/settings-$TAG.env}
[ -f "$SETFILE" ] && source "$SETFILE"
ENVFILE=${ENVFILE:-$HERE/tools/machine-$TAG.env}   # tools/calibrate.sh $TAG … écrit ce fichier
[ -f "$ENVFILE" ] && source "$ENVFILE"

LLAMA=$(realpath -m "${LLAMA:-$HERE/llama.cpp}")   # absolus : le serveur ne résout pas le relatif
BIN=$LLAMA/build/bin/llama-server
. "$HERE/tools/cache_gen.sh" "$BIN" "${HOST:-0.0.0.0}"
MODEL=$(realpath -m "${MODEL:?MODEL= requis — choisissez le poids dans le menu niche.sh}")
PROFILE=$(realpath -m "${PROFILE:-$HERE/tools/prof-$TAG.csv}")
CTX=${CTX:-8192}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8002}
PIN=${PIN:-1990}
MOE_STATS=${MOE_STATS:-/tmp/moe-stats-$TAG.csv}
KVQ=${KVQ:-q4_0}   # quant du KV : q8_0 = moitié de f16 sans perte mesurable ; q4_0 = défaut ; f16 = aucun
KVARGS="--cache-type-k $KVQ --cache-type-v $KVQ"
EXTRA=${EXTRA:-}   # options personnelles en texte libre (menu niche.sh, clé EXTRA), en dernier

[ -x "$BIN" ] || fail "binary $BIN missing — build llama.cpp (branch moe-expert-cache)"
[ -f "$MODEL" ] || fail "model $MODEL missing"

# sonde d'architecture sur le GGUF RÉEL (la calibration du tag a pu être faite pour un autre
# modèle, ex. tag gemma4 calibré sur le 26B-A4B MoE puis MODEL changé pour le 12B dense) :
if [ -z "${IS_MOE:-}" ] || [ -z "${IS_NEXTN:-}" ]; then
    FL=$(python3 "$HERE/tools/gguf_meta.py" --flags "$MODEL" 2>/dev/null) \
        || fail "gguf probe failed on $MODEL — set IS_MOE=0|1 IS_NEXTN=0|1 manually"
    [ -n "$FL" ] || fail "empty gguf probe on $MODEL — set IS_MOE=0|1 IS_NEXTN=0|1 manually"
    eval "$FL"
fi
IS_MOE=${IS_MOE:-${MOE:-1}}; IS_NEXTN=${IS_NEXTN:-${NEXTN:-${MODEL_HAS_NEXTN:-0}}}
if [ "$IS_MOE" -lt 1 ]; then
    ADAPT=0
else
    [ "$MOE_GEN" = up ] || { [ -f "$PROFILE" ] || fail "profile $PROFILE missing — run tools/calibrate.sh $TAG $MODEL first (also sizes the bank)"; }
fi
if [ -f "$ENVFILE" ]; then
    CALMODEL=$(grep -m1 '^MODEL=' "$ENVFILE" | sed -E 's/^MODEL=\$\{MODEL:-(.*)\}$/\1/')
    if [ -n "$CALMODEL" ] && [ "$(realpath -m "$CALMODEL")" != "$MODEL" ]; then
        echo "WARNING: $(basename "$ENVFILE") calibrated for $CALMODEL — bank/pin not measured for this model (tools/calibrate.sh $TAG $MODEL)" >&2
    fi
fi

# bank : l'optimum MESURÉ quand calibré (BANK_BEST), sinon 3072 MiB conservateur ; au-delà
# du contexte de référence la pente KV mesurée le fait fondre ; BANK_MAX (plafond calibré) borne.
BANK=${BANK_OVERRIDE:-${BANK_BEST:-3072}}
CTXR=${CTX_REF:-8192}
if [ "${KV_MIB_PER_8K:-0}" -gt 0 ] 2>/dev/null && [ "$CTX" -gt "$CTXR" ]; then
    BANK=$(( BANK - (CTX - CTXR) / 8192 * KV_MIB_PER_8K ))
fi
[ "$BANK" -ge 1024 ] || BANK=1024
[ -n "${BANK_MAX:-}" ] && [ "$BANK" -gt "$BANK_MAX" ] && BANK=$BANK_MAX

SPEC=${SPEC:-}
if [ -z "$SPEC" ]; then
    if [ "$IS_NEXTN" -ge 1 ] 2>/dev/null; then SPEC=draft-mtp; else SPEC=ngram-mod; fi
fi
SPECARGS=""; [ -n "$SPEC" ] && SPECARGS="--spec-type $SPEC"
ADAPT=${ADAPT-512}   # --moe-cache-adapt N : ré-rankage live de la tier semée par le profil ; ADAPT=0 désactive
ADAPTARGS=""; [ "${ADAPT:-0}" -ge 1 ] 2>/dev/null && ADAPTARGS="--moe-cache-adapt $ADAPT"
CORSARGS=""; [ -n "${CORS:-}" ] && CORSARGS="--cors-origins $CORS"

if [ "$IS_MOE" -ge 1 ]; then
    # cache amont (upstream-sync) : SLRU, --moe-cache-mib seul — même args au 1er essai et au retry OOM
    if [ "$MOE_GEN" = moex ]; then
        moe_args() { echo "-cmoe --moe-cache-mib $BANK --moe-profile $PROFILE --moe-cache-static --moe-cache-pin-mib $PIN $ADAPTARGS --moe-stats $MOE_STATS"; }
    else
        moe_args() { echo "-cmoe --moe-cache-mib $BANK"; }
    fi
    MOEARGS=$(moe_args)
    echo "llama-server MoE cache ($TAG): CTX=$CTX bank=${BANK} MiB pin=${PIN} MiB spec=${SPEC:-none} http://$HOST:$PORT (Ctrl-C to stop)"
else
    MOEARGS=""
    echo "llama-server dense, no MoE cache ($TAG): CTX=$CTX spec=${SPEC:-none} http://$HOST:$PORT (Ctrl-C to stop)"
fi
# retry OOM VRAM (bank −512) comme dans serve-lfm.sh — voir commentaires là-bas. Dense : pas de
# bank à baisser, pas de retry (l'OOM d'un dense vient du KV/CTX, pas d'un paramètre réglable ici).
OOMLOG=/tmp/.serve-generic-oom-probe.log
SRV=""
trap '[ -n "$SRV" ] && kill -INT "$SRV" 2>/dev/null' INT TERM
while :; do
    "$BIN" -m "$MODEL" \
        -ngl 99 \
        $MOEARGS \
        $SPECARGS \
        --host "$HOST" --port "$PORT" \
        -c "$CTX" -fa on $KVARGS -ub 512 \
        -t "${THREADS:-8}" --threads-batch "${THREADS:-8}" --parallel 1 \
        --metrics --slots --ui-mcp-proxy --webui-mcp-proxy $CORSARGS --tools exec_shell_command $TOOLSGATEARGS \
        $EXTRA 2> >(tee "$OOMLOG" >&2) &
    SRV=$!
    wait "$SRV"; rc=$?
    if [ "$rc" -ne 0 ] && [ "$IS_MOE" -ge 1 ] && grep -qiE 'cudaMalloc failed|out of memory|failed to allocate' "$OOMLOG" && [ "$BANK" -gt 2048 ]; then
        OLD=$BANK; BANK=$(( BANK - 512 )); [ "$BANK" -lt 2048 ] && BANK=2048
        MOEARGS=$(moe_args)
        echo "WARNING: VRAM OOM au bank ${OLD} (CTX=$CTX) — nouvelle tentative au bank $BANK. tools/calibrate.sh $TAG $MODEL mesure le plafond réel." >&2
        continue
    fi
    exit "$rc"
done
