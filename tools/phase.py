#!/usr/bin/env python3
"""Analyse a majflt2 sample file: report the deltas in the decode window.

Usage: python3 tools/phase.py <samples>
Columns: t minflt majflt disk outfile_size
Decode starts when the output file first grows, ends when it stops growing.
"""
import sys

rows = []
for line in open(sys.argv[1]):
    f = line.split()
    if len(f) == 5 and f[0] != "PID=":
        rows.append([float(f[0]), int(f[1]), int(f[2]), int(f[4-1]), int(f[4])])

start = None
end = None
for r in rows:
    if r[4] > 0 and start is None:
        start = r
    if r[4] > 0:
        end = r

def delta(a, b):
    return b[2] - a[2], b[3] - a[3], b[1] - a[1]

print("first sample: t=%.2f minflt=%d majflt=%d disk=%d" % (rows[0][0], rows[0][1], rows[0][2], rows[0][3]))
print("decode start: t=%.2f minflt=%d majflt=%d disk=%d" % (start[0], start[1], start[2], start[3]))
print("decode end:   t=%.2f minflt=%d majflt=%d disk=%d" % (end[0], end[1], end[2], end[3]))
mj, d, mn = delta(start, end)
print("decode deltas: majflt=%d disk_bytes=%d minflt=%d" % (mj, d, mn))
print("whole run:     majflt=%d disk_bytes=%d" % (rows[-1][2]-rows[0][2], rows[-1][3]-rows[0][3]))
