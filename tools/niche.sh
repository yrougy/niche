#!/bin/bash
# niche.sh — l'installateur écran du port cache-MoE niche (à la maya.sh de project-maya).
#   tools/niche.sh              l'écran complet : machine, modèles, build, calibrage, serveur
#   tools/niche.sh --plain      la même suite en texte brut (pipes, ssh sans écran)
# Textual vient des roues locales (tools/wheels) dans tools/.venv : rien n'est téléchargé.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python3}
if ! "$PY" -c "import textual" 2>/dev/null; then
    if [ ! -x "$HERE/.venv/bin/python" ]; then
        "$PY" -m venv "$HERE/.venv" || { echo "venv impossible : installer python3-venv" >&2; exit 1; }
    fi
    "$HERE/.venv/bin/pip" install -q --no-index --find-links "$HERE/wheels" textual || {
        echo "installation de textual depuis les roues a échoué — les re-générer :" >&2
        echo "  python3 -m pip download textual==8.2.8 -d tools/wheels" >&2; exit 1; }
    PY="$HERE/.venv/bin/python"
fi
exec "$PY" "$HERE/niche_tui.py" "$@"
