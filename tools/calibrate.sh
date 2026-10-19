#!/bin/bash
# calibrate.sh — mesure les paramètres du port cache-MoE pour un couple (machine, modèle).
# Usage : tools/calibrate.sh TAG CHEMIN_ABSOLU_GGUF
#   ex : tools/calibrate.sh lfm /data/models/unsloth/LFM2.5-24B-A2B-UD-IQ3_XXS.gguf
# Produit :
#   tools/prof-<TAG>.csv     profil de routage complet (config off, sans cache)
#   tools/machine-<TAG>.env  paramètres machine (sourceés par serve.sh / serve-lfm.sh) :
#     MODEL PROFILE PIN THREADS BANK_MAX BANK_BEST CTX_REF KV_MIB_PER_8K OFF_TG BEST_TG BEST_PP BEST_HIT
#   Chaque ligne est VAR=${VAR:-valeur} : une variable déjà exportée a la priorité.
# Env : CTX (8192) PP_N (2048) TG_N (256) PP_N_PROBE (1024) TG_N_PROBE (128) THREADS (nproc)
#       PIN (auto : min(1990, RLIMIT_MEMLOCK − 16 MiB))   SKIP_PROFILE (0)   VERIFY_CEILING (1)
# Déroulé : faits machine → bench off (baseline) → profil → recherche du plafond de bank VRAM
# (doublage puis recherche binaire, OOM détecté dans le log serveur) → mesure tg/pp/hit par paliers
# → meilleur bank → machine-<TAG>.env. ~10-20 min.
set -u

HERE=${HERE:-$(cd "$(dirname "$0")/.." && pwd)}
LLAMA=${LLAMA:-$HERE/llama.cpp}
BIN=$LLAMA/build/bin/llama-server

fail() { echo "ERREUR : $*" >&2; exit 1; }
TAG=${1:?TAG requis (ex. qwen, lfm)}
MODEL=${2:?chemin absolu du GGUF requis}
case "$MODEL" in
    /*) ;;
    *) fail "MODEL doit être un CHEMIN ABSOLU : le serveur est lancé depuis $LLAMA et ne résout pas les chemins relatifs (le static retombe sinon en LRU sans warn visible)" ;;
esac
[ -f "$MODEL" ] || fail "modèle $MODEL absent"
[ -x "$BIN" ] || fail "binaire $BIN absent — outils/setup.sh construit le port"

CTX=${CTX:-8192}
PP_N=${PP_N:-2048}; TG_N=${TG_N:-256}
PPNP=${PP_N_PROBE:-1024}; TGNP=${TG_N_PROBE:-128}
THREADS=${THREADS:-$(nproc)}
PROFILE=${PROFILE_ABS:-$HERE/tools/prof-$TAG.csv}
OUT_ENV=$HERE/tools/machine-$TAG.env
OUT=/tmp/bench

# ---------- 1. faits machine ----------
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
[ -n "$VRAM" ] || fail "aucun GPU (nvidia-smi) — le port place le bank sur le GPU ; sans CUDA, utiliser la config off amont"
RAM=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
RLIM=$(ulimit -l)
if [ "$RLIM" = "unlimited" ]; then PIN=1990; else PIN=$(( RLIM / 1024 - 16 )); [ "$PIN" -gt 1990 ] && PIN=1990; [ "$PIN" -lt 0 ] && PIN=0; fi
MODEL_MIB=$(( $(stat -c %s "$MODEL") / 1048576 ))
HAS_NEXTN=$(head -c 262144 "$MODEL" | grep -ac nextn_predict_layers || true)
[ "$MODEL_MIB" -gt $(( RAM - 2048 )) ] && echo "ATTENTION : modèle ${MODEL_MIB} MiB ≈ RAM dispo ${RAM} MiB — risque de cliff disque (cf. Qwen Q3_K_M tg 6,7). Prendre un quant plus bas."

echo "== machine : VRAM=${VRAM} MiB  RAM dispo=${RAM} MiB  pin=${PIN} MiB  threads=${THREADS}  nextn=${HAS_NEXTN} =="

# ---------- helpers ----------
run_bench() {  # $1=bank (0 = mode off) $2=PPN $3=TGN → $OUT rempli ; rc = 0 ok, sinon échec serveur
    rm -f "$OUT/tg.json" "$OUT/pp.json"   # éviter un vieux tg.json qui ferait passer un OOM pour OK
    if [ "$1" = 0 ]; then
        CTX=$CTX PP_N=$2 TG_N=$3 THREADS=$THREADS MODEL=$MODEL VERB=${VERB:-1} \
            "$HERE/tools/bench.sh" off >/dev/null 2>&1 && return 0 || return 1
    else
        CTX=$CTX PP_N=$2 TG_N=$3 THREADS=$THREADS MODEL=$MODEL PROFILE=$PROFILE CACHE_MIB=$1 VERB=${VERB:-1} \
            "$HERE/tools/bench.sh" static >/dev/null 2>&1 && return 0 || return 1
    fi
}
measure() {  # parse $OUT → "PP TG HIT" sur stdout (HIT = -1 si absent)
    python3 - "$OUT" <<'PY'
import json, sys, os
o = sys.argv[1]
def num(f):
    t = json.load(open(os.path.join(o, f + '.json')))['timings']
    if not t.get('predicted_ms') or not t.get('prompt_ms'): return 0.0, 0.0
    return (t['prompt_n']/t['prompt_ms']*1000, t['predicted_n']/t['predicted_ms']*1000)
pp, _ = num('pp'); _, tg = num('tg')
u = r = 0
try:
    for line in open(os.path.join(o, 'stats.csv')):
        if line.startswith('layer'): continue
        p = line.split(','); u += int(p[2]); r += int(p[2])*int(p[3])
except FileNotFoundError: pass
hit = 100.0*r/u if u else -1.0
print("%.1f %.2f %.1f" % (pp, tg, hit))
PY
}
is_oom() { grep -qiE "out of memory|resource allocation failed" "$OUT/server.log" 2>/dev/null; }

# ---------- 2. baseline off ----------
echo "== baseline off =="
run_bench 0 "$PP_N" "$TG_N" || fail "bench off (baseline) impossible — voir $OUT/server.log"
read -r OFF_PP OFF_TG _ <<<"$(measure)"
echo "  off : pp=$OFF_PP tg=$OFF_TG"

# ---------- 3. profil de routage ----------
if [ "${SKIP_PROFILE:-0}" != 1 ]; then
    echo "== profil de routage (config off) -> $PROFILE =="
    CTX=$CTX PP_N=$PP_N TG_N=$TG_N THREADS=$THREADS MODEL=$MODEL "$HERE/tools/profile.sh" "$PROFILE" >/dev/null || fail "profile.sh a échoué"
fi
[ -f "$PROFILE" ] || fail "profil $PROFILE absent"

# ---------- 4. plafond de bank (VRAM) ----------
CAP=$(( VRAM - 640 ))   # réserve non-experts + KV + compute
CEIL=1024; FAIL=$(( CAP < 1024 ? 1024 : CAP + 1 ))
if [ "${VERIFY_CEILING:-1}" = 1 ]; then
    echo "== recherche du plafond de bank (sonde : pp=$PPNP tg=$TGNP) =="
    for b in 2048 4096 6144 8192 12288 16384 32768; do
        [ "$b" -le "$CAP" ] || { FAIL=$b; break; }
        run_bench "$b" "$PPNP" "$TGNP"
        if [ $? -ne 0 ] || is_oom; then FAIL=$b; echo "  sonde $b : OOM"; break; fi
        CEIL=$b; echo "  sonde $b : OK"
    done
    if [ $(( FAIL - CEIL )) -gt 256 ]; then
        while [ $(( FAIL - CEIL )) -gt 256 ]; do
            mid=$(( (CEIL + FAIL) / 2 ))
            run_bench "$mid" "$PPNP" "$TGNP"
            if [ $? -ne 0 ] || is_oom; then FAIL=$mid; else CEIL=$mid; fi
            echo "  binaire $mid : $([ -e $OUT/tg.json ] && echo OK || echo OOM)"
        done
    fi
fi
echo "  plafond mesuré : bank ${CEIL} MiB passe, ${FAIL} MiB OOM (CTX=$CTX)"

# ---------- 5. mesure par paliers -> meilleur bank ----------
echo "== mesure tg/pp/hit par paliers (pp=$PP_N tg=$TG_N) =="
POINTS=$(python3 - <<PY
seen = set()
for b in (1024, 2048, 4096, $CEIL - 1024, $CEIL - 512, $CEIL):
    if b >= 1024 and b <= $CEIL and b not in seen:
        seen.add(b); print(b)
PY
)
BEST_TG=-1; BEST_BANK=0; BEST_PP=0; BEST_HIT=0; ROWS=""; LAST_BANK_OK=0
for b in $POINTS; do
    run_bench "$b" "$PP_N" "$TG_N" || { echo "  bank $b : échec (ignoré)"; continue; }
    read -r pp tg hit <<<"$(measure)"
    ROWS="$ROWS  bank $b : tg=$tg pp=$pp hit=$hit\n"
    LAST_BANK_OK=$b
    awk -v a="$tg" -v b="$BEST_TG" 'BEGIN{exit !(a>b)}' && { BEST_TG=$tg; BEST_PP=$pp; BEST_HIT=$hit; BEST_BANK=$b; }
done
printf '%b' "$ROWS"
# KV réel : colonne context de la breakdown VRAM du dernier run (log -v ; ligne
# "CUDA0 ... ( self = model + context + compute )"). En mode cache le bank est alloué dans le
# buft « context » : on retranche le bank du dernier run réussi pour isoler le KV.
CTXKV=$(grep -oE '\( *[0-9]+ = *[0-9]+ +\+ *[0-9]+ +\+ *[0-9]+\)' "$OUT/server.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+' | awk 'NR==3')
KV8=0
if [ -n "$CTXKV" ]; then
    KVBYTES=$(( CTXKV - LAST_BANK_OK ))
    [ "$KVBYTES" -ge 0 ] || KVBYTES=$CTXKV   # breakdown sans bank (mode off) : context = KV
    KV8=$(( KVBYTES * 8192 / CTX ))
fi

# ---------- 6. machine-TAG.env ----------
cat > "$OUT_ENV" <<EOF
# calibré le $(date -Is) sur $(hostname) — tools/calibrate.sh $TAG (CTX_REF=$CTX, sonde static sans MTP)
# Une variable déjà exportée dans l'environnement a la priorité sur la valeur calibrée.
MODEL=\${MODEL:-$MODEL}
PROFILE=\${PROFILE:-$PROFILE}
PIN=\${PIN:-$PIN}
THREADS=\${THREADS:-$THREADS}
BANK_MAX=\${BANK_MAX:-$CEIL}
BANK_BEST=\${BANK_BEST:-$BEST_BANK}
CTX_REF=\${CTX_REF:-$CTX}
KV_MIB_PER_8K=\${KV_MIB_PER_8K:-$KV8}
MODEL_HAS_NEXTN=\${MODEL_HAS_NEXTN:-$HAS_NEXTN}
OFF_PP=\${OFF_PP:-$OFF_PP}
OFF_TG=\${OFF_TG:-$OFF_TG}
BEST_TG=\${BEST_TG:-$BEST_TG}
BEST_PP=\${BEST_PP:-$BEST_PP}
BEST_HIT=\${BEST_HIT:-$BEST_HIT}
EOF

echo "== résumé =="
echo "baseline off      : pp=$OFF_PP tg=$OFF_TG"
echo "plafond bank      : $CEIL MiB (CTX=$CTX, KV≈${KV8} MiB/8k tokens)"
echo "meilleur bank     : $BEST_BANK MiB → tg=$BEST_TG pp=$BEST_PP hit=$BEST_HIT"
echo "écrit             : $OUT_ENV"
echo "exemples          : MODEL=$MODEL PROFILE=$PROFILE tools/bench.sh static $([ "$HAS_NEXTN" -gt 0 ] && echo mtp)"
echo "                  : tools/serve.sh   (qwen)   ou   tools/serve-lfm.sh   (lfm) — ils sourcent machine-$TAG.env"
