# cache_gen.sh — à sourcer par les scripts de lancement (serve*, bench, chat, calibrate) après
# avoir défini BIN : détecte la génération de cache MoE du binaire et pose MOE_GEN=moex|up.
#   moex = fork branche moe-expert-cache : tier static semée (--moe-profile), --moe-cache-adapt,
#          pin, cpu-fallback, tail-skip, stats CSV
#   up   = cache amont (master/upstream-sync) : SLRU en ligne, --moe-cache-mib seul
# Usage : . "$HERE/tools/cache_gen.sh" "$BIN"
MOE_GEN=up
if [ -x "${1:-}" ] && "$1" --help 2>/dev/null | grep -q -- "--moe-cache-static"; then
    MOE_GEN=moex
fi
