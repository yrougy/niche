#!/bin/bash
# setup.sh — préparer une machine pour le port cache-MoE niche : build du port, téléchargement
# des modèles, mesure des paramètres (bank max, pin, profil de routage) via calibrate.sh.
# Usage : tools/setup.sh [qwen|lfm|both]   (défaut : qwen — un seul modèle, both pour les deux)
# Env :
#   MODEL_DIR   répertoire d'installation des GGUF (défaut $HERE/models)
#   LLAMA       checkout llama.cpp de la branche moe-expert-cache (défaut $HERE/llama.cpp,
#               cloné automatiquement depuis le fork si absent — dossier gitignoré)
#   FORK        URL du fork llama.cpp portant le cache MoE (défaut ci-dessous)
#   SKIP_BUILD / SKIP_DOWNLOAD / SKIP_CALIB = 1 pour sauter une étape
#   CMAKE_EXTRA flags cmake additionnels (ex. -DCMAKE_CUDA_ARCHITECTURES=61 ; CUDA forcé par défaut)
#   CTX PP_N TG_N paramètres passés à calibrate.sh
set -u

HERE=${HERE:-$(cd "$(dirname "$0")/.." && pwd)}
LLAMA=${LLAMA:-$HERE/llama.cpp}
MODEL_DIR=${MODEL_DIR:-$HERE/models}
FORK=${FORK:-https://github.com/yrougy/llama-moe-cache.git}
WHICH=${1:-qwen}

fail() { echo "error: $*" >&2; exit 1; }

# catalogue : tag -> repo huggingface, fichier, taille minimale de décence
model_info() {
    case "$1" in
        qwen) REPO=unsloth/Qwen3.6-35B-A3B-GGUF;   FILE=Qwen3.6-35B-A3B-UD-IQ2_M.gguf;   MINSIZE=$((10*1024*1024*1024)) ;;
        lfm)  REPO=unsloth/LFM2.5-24B-A2B-GGUF;    FILE=LFM2.5-24B-A2B-UD-IQ3_XXS.gguf;  MINSIZE=$((7*1024*1024*1024)) ;;
        *) fail "unknown model: $1 (choices: qwen, lfm, both)" ;;
    esac
}

# ---------- 1. dépôt llama.cpp (port) + build ----------
if [ -d "$LLAMA/.git" ]; then
    BRANCH=$(git -C "$LLAMA" rev-parse --abbrev-ref HEAD)
    if [ "$BRANCH" != "moe-expert-cache" ]; then
        echo "WARNING: $LLAMA is on '$BRANCH', not 'moe-expert-cache' — the ported MoE cache is not in it." >&2
    fi
elif [ ! -e "$LLAMA" ]; then
    echo "cloning the port: $FORK (branch moe-expert-cache) -> $LLAMA"
    git clone -b moe-expert-cache --single-branch "$FORK" "$LLAMA" || fail "git clone failed (network? URL?)"
else
    echo "WARNING: $LLAMA exists but is not a git repo — empty that path and setup.sh will clone the port ($FORK)." >&2
fi
if [ "${SKIP_BUILD:-0}" != 1 ]; then
    nv="$(command -v nvcc 2>/dev/null || ls -1 /usr/local/cuda-*/bin/nvcc /opt/cuda-*/bin/nvcc 2>/dev/null | tail -1)"
    if [ ! -d "$LLAMA/build" ] || ! grep -q "GGML_CUDA:BOOL=ON" "$LLAMA/build/CMakeCache.txt" 2>/dev/null; then
        [ -n "$nv" ] || echo "WARNING: no nvcc found — cmake may not enable CUDA (the MoE cache needs a GPU backend)" >&2
        cmake -S "$LLAMA" -B "$LLAMA/build" -DGGML_CUDA=ON ${nv:+-DCMAKE_CUDA_COMPILER=$nv} ${CMAKE_EXTRA:-} || fail "cmake configure failed"
    fi
    cmake --build "$LLAMA/build" -j"$(nproc)" || fail "llama.cpp build failed"
fi
[ -x "$LLAMA/build/bin/llama-server" ] || fail "llama-server missing in $LLAMA/build/bin"
[ -e "$LLAMA/build/bin/libggml-cuda.so" ] || grep -q "GGML_CUDA:BOOL=ON" "$LLAMA/build/CMakeCache.txt" 2>/dev/null || fail "build has no CUDA backend — the MoE cache requires it; install the CUDA toolkit and rebuild"
echo "binary OK: $LLAMA/build/bin/llama-server"

# ---------- 2. modèles ----------
install_model() {
    model_info "$1"
    local target="$MODEL_DIR/$FILE"
    mkdir -p "$MODEL_DIR"
    if [ -f "$target" ]; then
        echo "$1: $target already exists ($(stat -c %s "$target") bytes) — download skipped"
    elif [ "${SKIP_DOWNLOAD:-0}" = 1 ]; then
        echo "$1: SKIP_DOWNLOAD=1 but $target missing — calibrate will be impossible" ; return 1
    else
        local url="https://huggingface.co/$REPO/resolve/main/$FILE"
        echo "$1: downloading $url -> $target (auto-resume if .part exists)"
        curl -L -C - --retry 5 --retry-delay 3 -o "$target.part" "$url" || { echo "error: curl failed (re-run setup.sh to resume)" >&2; return 1; }
        local sz=$(stat -c %s "$target.part")
        [ "$sz" -ge "$MINSIZE" ] || { echo "error: suspicious size ($sz bytes)" >&2; return 1; }
        mv "$target.part" "$target"
    fi
    local mib=$(( $(stat -c %s "$target") / 1048576 )) ram=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$mib" -gt $(( ram - 2048 )) ]; then
        echo "WARNING: ${mib} MiB model vs ${ram} MiB free RAM — the hot set will crack on disk (measured cliff e.g. tg 6.7). Lower quant recommended." >&2
    fi
    echo "$1: ready ($mib MiB)"
}

run_calibrate() {
    model_info "$1"
    [ "${SKIP_CALIB:-0}" = 1 ] && { echo "$1: calibrate skipped (SKIP_CALIB=1)"; return 0; }
    CTX=${CTX:-8192} PP_N=${PP_N:-2048} TG_N=${TG_N:-256} \
        "$HERE/tools/calibrate.sh" "$1" "$MODEL_DIR/$FILE" || fail "calibrate($1) failed"
}

case "$WHICH" in
    both) install_model qwen && run_calibrate qwen; install_model lfm && run_calibrate lfm ;;
    *)    install_model "$WHICH" && run_calibrate "$WHICH" ;;
esac

echo
echo "setup done. Relaunch the calibrated config:"
echo "  qwen : tools/serve.sh            (sources tools/machine-qwen.env if present)"
echo "  lfm  : tools/serve-lfm.sh        (sources tools/machine-lfm.env if present)"
echo "  rebench: tools/bench.sh static   with MODEL/PROFILE exported"
