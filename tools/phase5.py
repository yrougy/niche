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
    rows.append([float(f[0])] + [int(x) for x in f[1:7]])

MiB = 1048576.0
print("decode window %.3f .. %.3f s, eval = %.0f ms, %d samples" % (t_start, t_end, eval_ms, len(rows)))
prev = None
for r in rows:
    if r[0] < t_start - 1.0:
        prev = r
        continue
    if prev is None:
        prev = r
    d_mn, d_mj = r[1] - prev[1], r[2] - prev[2]
    d_rb, d_dk = r[4] - prev[4], r[6] - prev[6]
    if d_rb or d_dk or d_mj:
        print("t=%7.3f  minflt+=%d majflt+=%d  read_bytes+=%.2f MiB  disk+=%.2f MiB  rss=%.2f GiB"
              % (r[0], d_mn, d_mj, d_rb / MiB, d_dk / MiB, r[3] / 1073741824.0))
    prev = r
last = rows[-1]
first = [r for r in rows if r[0] >= t_start][0]
print("window totals: read_bytes %.2f MiB, disk %.2f MiB, majflt %d, minflt %d"
      % ((last[4] - first[4]) / MiB, (last[6] - first[6]) / MiB, last[2] - first[2], last[1] - first[1]))
