#!/bin/bash
cd /data/o35/o35
for mode in cpufb clean; do
  echo "=== EVICT 13 GiB, mode=$mode ==="
  python3 tools/evict.py 13
  python3 tools/runtrial.py trial$mode 1024 $mode 1990 > /tmp/trial$mode.samples 2>&1
  echo "--- t/s, stats ---"
  grep -E "eval time|hit rate|ABORT|CUDA error" /tmp/trial$mode.log | tail -3
  echo "--- fenetre de decode ---"
  python3 tools/phase6.py /tmp/trial$mode.samples /tmp/trial$mode.log
done
echo TRIAL_DONE
