import sys, re

samples = sys.argv[1]
log = sys.argv[2]

rows = []
for line in open(samples):
    f = line.split()
    if len(f) == 8:
        rows.append(tuple(float(x) for x in f))

def logsec(line):
    p = line.split()[0].split('.')
    return int(p[0]) * 60 + int(p[1]) + int(p[2]) / 1000.0

end = None
eval_ms = None
for line in open(log):
    m = re.search(r'\|\s*eval time\s*=\s*([\d,]+)\s*ms', line)
    if m:
        eval_ms = float(m.group(1).replace(',', '.'))
        end = logsec(line)

if end is None:
    print("no eval time in log")
    sys.exit(1)

t0 = end - rows[-1][0]
lo, hi = (end - eval_ms / 1000.0) - t0, end - t0

def at(t):
    return min(rows, key=lambda r: abs(r[0] - t))

a, b = at(lo), at(hi)
mib = 1024.0 * 1024.0

def fmt(r):
    return "t=%.2f minflt=%d majflt=%d rss=%.2f GiB read=%.2f MiB disk=%.2f MiB" % (
        r[0], int(r[1]), int(r[2]), r[3] / 1073741824.0, r[4] / mib, r[7] / mib)

def delta(x, y):
    return "majflt=%d read=%.2f MiB disk=%.2f MiB rss=%.2f GiB" % (
        int(y[2]) - int(x[2]), (y[4] - x[4]) / mib, (y[7] - x[7]) / mib, (y[3] - x[3]) / 1073741824.0)

print("sampler t=0 = log %.3f s" % t0)
print("decode window sampler t = %.2f .. %.2f (log %.2f .. %.2f)" % (lo, hi, end - eval_ms / 1000.0, end))
print("start " + fmt(a))
print("end   " + fmt(b))
print("decode: " + delta(a, b))
print("whole:  " + delta(rows[0], rows[-1]))
