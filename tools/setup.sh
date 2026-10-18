#!/bin/bash
# setup.sh — préparer une machine pour le port cache-MoE o35 : build du port, téléchargement
# des modèles, mesure des paramètres (bank max, pin, profil de routage) via calibrate.sh.
# Usage : tools/setup.sh [qwen|lfm|both]   (défaut : both)
# Env :
#   MODEL_DIR   répertoire d'installation des GGUF (défaut /data/models/unsloth)
#   LLAMA       checkout llama.cpp de la branche moe-expert-cache (défaut /data/o35/llama.cpp,
#               cloné automatiquement depuis le fork si le chemin n'existe pas)
#   FORK        URL du fork llama.cpp portant le cache MoE (défaut ci-dessous)
#   SKIP_BUILD / SKIP_DOWNLOAD / SKIP_CALIB = 1 pour sauter une étape
#   CMAKE_EXTRA flags cmake additionnels (ex. -DGGML_CUDA=ON — détection auto par défaut)
#   CTX PP_N TG_N paramètres passés à calibrate.sh
set -u

HERE=${HERE:-$(cd "$(dirname "$0")/.." && pwd)}
LLAMA=${LLAMA:-/data/o35/llama.cpp}
MODEL_DIR=${MODEL_DIR:-/data/models/unsloth}
FORK=${FORK:-https://github.com/yrougy/llama-moe-cache.git}
WHICH=${1:-both}

fail() { echo "ERREUR : $*" >&2; exit 1; }

# catalogue : tag -> repo huggingface, fichier, taille minimale de décence
model_info() {
    case "$1" in
        qwen) REPO=unsloth/Qwen3.6-35B-A3B-GGUF;   FILE=Qwen3.6-35B-A3B-UD-IQ2_M.gguf;   MINSIZE=$((10*1024*1024*1024)) ;;
        lfm)  REPO=unsloth/LFM2.5-24B-A2B-GGUF;    FILE=LFM2.5-24B-A2B-UD-IQ3_XXS.gguf;  MINSIZE=$((7*1024*1024*1024)) ;;
        *) fail "modèle inconnu : $1 (choix : qwen, lfm, both)" ;;
    esac
}

# ---------- 1. dépôt llama.cpp (port) + build ----------
if [ -d "$LLAMA/.git" ]; then
    BRANCH=$(git -C "$LLAMA" rev-parse --abbrev-ref HEAD)
    if [ "$BRANCH" != "moe-expert-cache" ]; then
        echo "ATTENTION : $LLAMA est sur '$BRANCH', pas 'moe-expert-cache' — le cache MoE porté n'y est pas." >&2
    fi
elif [ ! -e "$LLAMA" ]; then
    echo "clonage du port : $FORK (branche moe-expert-cache) -> $LLAMA"
    git clone -b moe-expert-cache --single-branch "$FORK" "$LLAMA" || fail "git clone a échoué (réseau ? URL ?)"
else
    echo "ATTENTION : $LLAMA existe mais n'est pas un dépôt git — vider ce chemin laisserait setup.sh cloner le port ($FORK)." >&2
fi
if [ "${SKIP_BUILD:-0}" != 1 ]; then
    [ -d "$LLAMA/build" ] || cmake -S "$LLAMA" -B "$LLAMA/build" ${CMAKE_EXTRA:-} || fail "cmake configure a échoué"
    cmake --build "$LLAMA/build" -j"$(nproc)" || fail "build llama.cpp a échoué"
fi
[ -x "$LLAMA/build/bin/llama-server" ] || fail "llama-server absent dans $LLAMA/build/bin"
echo "binaire OK : $LLAMA/build/bin/llama-server"

# ---------- 2. modèles ----------
install_model() {
    model_info "$1"
    local target="$MODEL_DIR/$FILE"
    mkdir -p "$MODEL_DIR"
    if [ -f "$target" ]; then
        echo "$1 : $target existe déjà ($(stat -c %s "$target") octets) — téléchargement sauté"
    elif [ "${SKIP_DOWNLOAD:-0}" = 1 ]; then
        echo "$1 : SKIP_DOWNLOAD=1 mais $target absent — calibrate sera impossible" ; return 1
    else
        local url="https://huggingface.co/$REPO/resolve/main/$FILE"
        echo "$1 : téléchargement $url -> $target (reprise automatique si .part existant)"
        curl -L -C - --retry 5 --retry-delay 3 -o "$target.part" "$url" || { echo "ERREUR : curl a échoué (relancer setup.sh pour reprendre)" >&2; return 1; }
        local sz=$(stat -c %s "$target.part")
        [ "$sz" -ge "$MINSIZE" ] || { echo "ERREUR : taille suspecte ($sz octets)" >&2; return 1; }
        mv "$target.part" "$target"
    fi
    local mib=$(( $(stat -c %s "$target") / 1048576 )) ram=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$mib" -gt $(( ram - 2048 )) ]; then
        echo "ATTENTION : ${mib} MiB de modèle pour ${ram} MiB RAM dispo — le hot set va feler sur disque (cliff mesuré type tg 6,7). Quant plus bas recommandé." >&2
    fi
    echo "$1 : prêt ($mib MiB)"
}

run_calibrate() {
    model_info "$1"
    [ "${SKIP_CALIB:-0}" = 1 ] && { echo "$1 : calibrate sauté (SKIP_CALIB=1)"; return 0; }
    CTX=${CTX:-8192} PP_N=${PP_N:-2048} TG_N=${TG_N:-256} \
        "$HERE/tools/calibrate.sh" "$1" "$MODEL_DIR/$FILE" || fail "calibrate($1) a échoué"
}

case "$WHICH" in
    both) install_model qwen && run_calibrate qwen; install_model lfm && run_calibrate lfm ;;
    *)    install_model "$WHICH" && run_calibrate "$WHICH" ;;
esac

echo
echo "setup terminé. Relancer la config calibrée :"
echo "  qwen : tools/serve.sh            (lit tools/machine-qwen.env s'il existe)"
echo "  lfm  : tools/serve-lfm.sh        (lit tools/machine-lfm.env s'il existe)"
echo "  rebench : tools/bench.sh static   avec MODEL/PROFILE exportés"
