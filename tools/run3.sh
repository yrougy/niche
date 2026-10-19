#!/bin/sh
R=$(cd "$(dirname "$0")" && pwd)
python3 $R/evict.py 13
STATIC=1 sh $R/run_pin.sh pin7 --moe-cache-pin-mib 1990 > /tmp/pin7.samples 2>&1
python3 $R/evict.py 13
STATIC=1 sh $R/run_pin.sh pin8 --moe-cache-pin-mib 1990 -n 512 > /tmp/pin8.samples 2>&1
echo ALL_DONE
