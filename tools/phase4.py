import sys, re

samples, log = sys.argv[1], sys.argv[2]
rows = []
for line in open(samples):
    f = line.split()
    if len(f) == 7:
        rows.append(tuple(float(x) for x in f))

def logsec(line):
    p = line.split()[0].split('.')
    return int(p[0]) * 60 + int(p[1]) + int(p[2]) / 1000.0

end = eval_ms = None
for line in open(log):
    m = re.search(r'\|\s*eval time\s*=\s*([\d,]+)\s*ms', line)
    if m:
        eval_ms = float(m.group(1).replace(',', '.'))
        end = logsec(line)

t0 = end - rows[-1][0]
lo, hi = (end - eval_ms / 1000.0) - t0, end - t0
def at(t):
    return min(rows, key=lambda r: abs(r[0] - t))

a, b = at(lo), at(hi)
def fmt(r):
    return "t=%.2f majflt=%d rss=%.2f GiB read=%.2f MiB resident=%.2f GiB" % (
        r[0], int(r[2]), r[3] / 1073741824.0, r[4] / 1048576.0, r[6] / 1024.0)
def delta(x, y):
    return "majflt=%d read=%.2f MiB disk=%.2f MiB resident=%+.2f GiB" % (
        int(y[2]) - int(x[2]), (y[4] - x[4]) / 1048576.0, (y[5] - x[5]) / 1048576.0,
        (y[6] - x[6]) / 1024.0)

print("decode window sampler t = %.2f .. %.2f" % (lo, hi))
print("start " + fmt(a))
print("end   " + fmt(b))
print("decode: " + delta(a, b))
print("whole:  " + delta(rows[0], rows[-1]))
