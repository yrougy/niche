#!/usr/bin/env python3
"""Sample major page faults of a process and the block reads of the HDD.

Usage: python3 tools/majflt2.py <pid> <seconds> [device] [outfile]
Prints one line per change: t majflt minflt disk_read_bytes outfile_size
/proc/PID/io needs ptrace rights, so the disk reads are read from /proc/diskstats instead.
The size of the output file grows during decode, so it marks the end of the model load.
"""
import os, sys, time

pid = int(sys.argv[1])
dur = float(sys.argv[2])
dev = sys.argv[3] if len(sys.argv) > 3 else "sda"
out = sys.argv[4] if len(sys.argv) > 4 else None

def disk_reads():
    try:
        for line in open("/proc/diskstats"):
            f = line.split()
            if len(f) > 5 and f[2] == dev:
                return int(f[5]) * 512
    except Exception:
        return -1
    return -1

def proc(pid):
    with open("/proc/%d/stat" % pid) as f:
        parts = f.read().split()
    i = next(k for k, x in enumerate(parts) if x.endswith(")"))
    f = parts[i + 1:]
    return int(f[7]), int(f[8])

t0 = time.time()
last = None
while time.time() - t0 < dur:
    try:
        size = os.path.getsize(out) if out else 0
        vals = proc(pid) + (disk_reads(), size)
    except FileNotFoundError:
        print("process exited")
        break
    if last is None or vals != last[1:]:
        print("%.2f %d %d %d %d" % (time.time() - t0, *vals), flush=True)
        last = (0, *vals)
    time.sleep(0.2)
