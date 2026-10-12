#!/usr/bin/env python3
"""Sample major page faults and real block-device reads of a running process.

Usage: python3 tools/majflt.py <pid> <seconds>
Prints one line per change: t majflt minflt read_bytes write_bytes
"""
import sys, time

pid = int(sys.argv[1])
dur = float(sys.argv[2])
t0 = time.time()
last = None
while time.time() - t0 < dur:
    try:
        with open("/proc/%d/stat" % pid) as f:
            parts = f.read().split()
        i = next(k for k, x in enumerate(parts) if x.endswith(")"))
        f = parts[i + 1:]
        minflt, majflt = int(f[6]), int(f[7])
        io = open("/proc/%d/io" % pid).read().splitlines()
        kv = dict((x.split(":", 1) + [""])[:2] for x in io if ":" in x)
        rb, wb = int(kv.get("read_bytes", 0) or 0), int(kv.get("write_bytes", 0) or 0)
        vals = (majflt, minflt, rb, wb)
        if last is None or vals != last[1:]:
            print("%.2f %d %d %d %d" % (time.time() - t0, *vals), flush=True)
            last = (0, *vals)
    except FileNotFoundError:
        print("process exited")
        break
    time.sleep(0.1)
