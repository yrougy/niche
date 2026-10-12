import sys, re

samples = sys.argv[1]
log = sys.argv[2]

rows = []
for line in open(samples):
    f = line.split()
    if len(f) == 5:
        rows.append((float(f[0]), int(f[1]), int(f[2]), int(f[3]), int(f[4])))

def logsec(line):
    p = line.split()[0].split('.')
    return int(p[0]) * 60 + int(p[1]) + int(p[2]) / 1000.0

end = None
eval_ms = None
for line in open(log):
    m = re.search(r'eval time\s*=\s*([\d,]+)\s*ms', line)
    if m:
        eval_ms = float(m.group(1).replace(',', '.'))
        end = logsec(line)

if end is None:
    print("no eval time in log")
    sys.exit(1)

t0 = end - rows[-1][0]
start = end - eval_ms / 1000.0
lo, hi = start - t0, end - t0

def at(t):
    best = min(rows, key=lambda r: abs(r[0] - t))
    return best

a, b = at(lo), at(hi)
mib = 1024.0 * 1024.0
print("sampler t=0 = log %.3f s" % t0)
print("decode window sampler t = %.2f .. %.2f (log %.2f .. %.2f)" % (lo, hi, start, end))
print("start sample: t=%.2f minflt=%d majflt=%d disk=%d" % a[:4])
print("end   sample: t=%.2f minflt=%d majflt=%d disk=%d" % b[:4])
print("decode deltas: majflt=%d minflt=%d disk=%.2f MiB" %
      (b[2] - a[2], b[1] - a[1], (b[3] - a[3]) / mib))
print("whole run:     majflt=%d minflt=%d disk=%.2f MiB" %
      (rows[-1][2] - rows[0][2], rows[-1][1] - rows[0][1], (rows[-1][3] - rows[0][3]) / mib))
