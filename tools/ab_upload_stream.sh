#!/bin/bash
# ab_upload_stream.sh — démarre serve-lfm.sh (prod LFM, CTX 131072) avec MOE_UPLOAD_STREAM on/off,
# joue le trafic de /tmp/probe_traffic.py, rapporte tg + texte (déterminisme), arrête le serveur.
# Usage : tools/ab_upload_stream.sh off|on [PROBE_CSV]
set -u
cd /data/o35/o35
MODE=${1:?off|on}
TAG=$MODE
PROBE_CSV=${2:-/tmp/moe-stall-$TAG.csv}
rm -f "$PROBE_CSV" /tmp/ab-out-$TAG.txt
export CTX=131072
export MOE_STALL_PROBE="$PROBE_CSV"
case "$MODE" in on*) export MOE_UPLOAD_STREAM=1;; esac
[ "$MODE" = off ] && unset MOE_UPLOAD_STREAM

setsid nohup tools/serve-lfm.sh > /tmp/serve-ab-$TAG.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do
    curl -sf -o /dev/null http://127.0.0.1:8002/health && break
    kill -0 "$SRV" 2>/dev/null || { echo "server died"; tail -5 /tmp/serve-ab-$TAG.log; exit 1; }
    sleep 2
done
curl -sf -o /dev/null http://127.0.0.1:8002/health || { echo "server timeout"; exit 1; }
echo "== $TAG: server up (pid $SRV), upload_stream log:"
grep -i "upload stream" /tmp/serve-ab-$TAG.log || echo "(no upload stream line = serial path)"

python3 /tmp/probe_traffic.py $TAG > /tmp/ab-out-$TAG.txt 2>&1
cat /tmp/ab-out-$TAG.txt

kill -INT "$SRV"
i=0; while kill -0 "$SRV" 2>/dev/null && [ $i -lt 30 ]; do sleep 1; i=$((i+1)); done
echo "== $TAG: done, probe=$PROBE_CSV out=/tmp/ab-out-$TAG.txt"
