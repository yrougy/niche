#!/usr/bin/env python3
"""Fenêtrage d'un run mesuré par runio7.py : lectures disque + résidence par bloc de GiB.

Index corrects de la ligne de sample : f[0]=t, f[1]=minflt, f[2]=majflt, f[3]=rss,
f[4]=read_bytes, f[5]=secteurs lus sur sda (x512), f[6:]=résidence par bloc de 1 GiB.
(phase4.py filtre len(f)==7 et phase5.py lit f[6] comme disque : les deux sont faux ici.)

Usage : python3 tools/phase6.py <samples> <log>
"""
import sys

samples, log = sys.argv[1], sys.argv[2]


def logsec(t):
    p = t.split(".")
    return int(p[0]) * 60 + int(p[1]) + int(p[2]) / 1000.0


eval_ms, t_end = None, None
for line in open(log):
    if "eval time = " in line:
        t_end = logsec(line.split()[0])
        eval_ms = float(line.split("eval time = ")[1].split()[0].replace(",", "."))
t_start = t_end - eval_ms / 1000.0

rows = []
for line in open(samples):
    f = line.split()
    if len(f) < 7:
        continue
    rows.append([float(f[0])] + [int(x) for x in f[1:6]] + [int(x) for x in f[6:]])

MiB = 1048576.0
w = [r for r in rows if r[0] >= t_start]
a, b = w[0], w[-1]


def delta(x, y):
    return "minflt+%d majflt+%d read_bytes+%.2f MiB disk+%.2f MiB rss %.2f->%.2f GiB" % (
        y[1] - x[1], y[2] - x[2], (y[4] - x[4]) / MiB, (y[5] - x[5]) / MiB,
        x[3] / 1073741824.0, y[3] / 1073741824.0)


print("fenêtre de décode %.2f .. %.2f s, eval = %.0f ms, %d samples" % (t_start, t_end, eval_ms, len(w)))
print("décode : " + delta(a, b))
print("run entier : " + delta(rows[0], rows[-1]))

print("résidence par bloc de GiB (début -> fin de fenêtre, delta en MiB) :")
for i in range(6, len(a)):
    print("  bloc %2d : %4d -> %4d  %+d MiB" % (i - 6, a[i], b[i], b[i] - a[i]))

prev = a
for r in w:
    dr = (r[4] - prev[4]) / MiB
    dk = (r[5] - prev[5]) / MiB
    if dr > 1.0 or dk > 1.0:
        print("  t=%7.3f  read+%.2f MiB  disk+%.2f MiB" % (r[0], dr, dk))
    prev = r
