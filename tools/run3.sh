#!/bin/sh
python3 /data/o35/o35/tools/evict.py 13
STATIC=1 sh /data/o35/o35/tools/run_pin.sh pin7 --moe-cache-pin-mib 1990 > /tmp/pin7.samples 2>&1
python3 /data/o35/o35/tools/evict.py 13
STATIC=1 sh /data/o35/o35/tools/run_pin.sh pin8 --moe-cache-pin-mib 1990 -n 512 > /tmp/pin8.samples 2>&1
echo ALL_DONE
