# cache_gen.sh — à sourcer par les scripts de lancement (serve*, bench, chat, calibrate) après
# avoir défini BIN : détecte la génération de cache MoE du binaire et pose MOE_GEN=moex|up.
#   moex = fork branche moe-expert-cache : tier static semée (--moe-profile), --moe-cache-adapt,
#          pin, cpu-fallback, tail-skip, stats CSV
#   up   = cache amont (master/upstream-sync) : SLRU en ligne, --moe-cache-mib seul
# Usage : . "$HERE/tools/cache_gen.sh" "$BIN" "$HOST"
MOE_GEN=up
if [ -x "${1:-}" ] && "$1" --help 2>/dev/null | grep -q -- "--moe-cache-static"; then
    MOE_GEN=moex
fi

# binaire llama.cpp >= v1.0.0 : refuse server tools/MCP proxy sur hôte non-loopback sans API key
# (fatal au démarrage). Nos scripts lient 0.0.0.0 et activent toujours --tools/--*-mcp-proxy sur
# réseau LAN de confiance → on ajoute l'option explicite quand le binaire la connaît.
TOOLSGATEARGS=""
if [ -x "${1:-}" ] && "$1" --help 2>/dev/null | grep -q -- "--allow-unauthenticated-tools"; then
    case "${2:-127.0.0.1}" in
        127.0.0.1|localhost|::1) ;;
        *) TOOLSGATEARGS="--allow-unauthenticated-tools" ;;
    esac
fi
