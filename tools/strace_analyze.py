import sys, re, collections

samples, log = sys.argv[1], sys.argv[2]

def logsec(t):
    p = t.split(".")
    return int(p[0]) * 60 + int(p[1]) + int(p[2]) / 1000.0

def wall(ts):
    p = ts.split(":")
    return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])

eval_ms, t_end = None, None
for line in open(log):
    if "eval time = " in line:
        t_end = logsec(line.split()[0])
        eval_ms = float(line.split("eval time = ")[1].split()[0].replace(",", "."))
if eval_ms is None:
    print("no eval time found"); sys.exit(1)
t_start = t_end - eval_ms / 1000.0

pat = re.compile(r"(\d+)\s+(\d\d:\d\d:\d\d\.\d+)\s+(read|pread64|readv|preadv)\((\d+)<([^>]*)>,")
first = None
tot = collections.Counter(); cnt = collections.Counter()
win = collections.Counter(); wcnt = collections.Counter()
for line in open(samples):
    m = pat.search(line)
    if not m:
        continue
    pid, ts, call, fd, path = m.groups()
    r = line.rsplit(" = ", 1)
    if len(r) < 2:
        continue
    try:
        n = int(r[1].split()[0])
    except ValueError:
        continue
    if n <= 0:
        continue
    w = wall(ts)
    if first is None:
        first = w
    tot[path] += n; cnt[path] += 1
    rel = w - first
    if t_start <= rel <= t_end:
        win[path] += n; wcnt[path] += 1

print("process start (wall) = %.3f, decode window = %.3f .. %.3f s after start (%.0f ms)"
      % (first, t_start, t_end, eval_ms))
print("--- all reads by file ---")
for k, v in tot.most_common(12):
    print("%10.2f MiB  %6d calls  %s" % (v / 1048576.0, cnt[k], k))
print("--- reads inside the decode window ---")
for k, v in win.most_common(12):
    print("%10.2f MiB  %6d calls  %s" % (v / 1048576.0, wcnt[k], k))
