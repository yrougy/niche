#!/bin/bash
# sweep_session.sh TAG [EXTRA flags...] — démarre serve-lfm.sh (CTX 131072) avec EXTRA optionnel,
# joue probe_traffic.py, rapporte le résumé, arrête le serveur proprement.
set -u
cd /data/o35/o35
TAG=${1:?tag}; shift
export CTX=${CTX:-131072}
export EXTRA="$*"
rm -f /tmp/ab-out-$TAG.txt /tmp/ab-text-$TAG.txt
setsid nohup tools/serve-lfm.sh > /tmp/serve-sw-$TAG.log 2>&1 &
SRV=$!
for i in $(seq 1 75); do
    curl -sf -o /dev/null http://127.0.0.1:8002/health && break
    kill -0 "$SRV" 2>/dev/null || { echo "server died"; tail -5 /tmp/serve-sw-$TAG.log; exit 1; }
    sleep 2
done
curl -sf -o /dev/null http://127.0.0.1:8002/health || { echo "server timeout"; exit 1; }
echo "== $TAG: up (pid $SRV) EXTRA='$EXTRA'"
if [ "${TRAFFIC:-full}" = text ]; then
    python3 "$(dirname "$0")/text_only.py" $TAG > /tmp/ab-out-$TAG.txt 2>&1
    cat /tmp/ab-out-$TAG.txt
else
    python3 "$(dirname "$0")/probe_traffic.py" $TAG > /tmp/ab-out-$TAG.txt 2>&1
    cat /tmp/ab-out-$TAG.txt
fi
kill -INT "$SRV"
i=0; while kill -0 "$SRV" 2>/dev/null && [ $i -lt 30 ]; do sleep 1; i=$((i+1)); done
echo "== $TAG: done"
