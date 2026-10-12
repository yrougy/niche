import sys, subprocess, time, os

# python is the parent of llama-cli, so /proc/PID/io is readable under ptrace_scope=1
name = sys.argv[1]
n = sys.argv[2]
pin = sys.argv[3]

args = ["./build/bin/llama-cli", "-v",
        "-m", "/data/models/unsloth/Qwen3.6-35B-A3B-UD-IQ2_M.gguf",
        "-cmoe", "-ngl", "99", "--moe-cache-mib", "1024",
        "--moe-profile", "/tmp/prof2.csv", "--moe-cache-static",
        "-c", "2048", "-n", n, "--temp", "0", "-s", "42",
        "-p", "Q: What is the capital of France? A:", "--no-display-prompt"]
if pin != "0":
    args += ["--moe-cache-pin-mib", pin]

out = open("/tmp/%s.txt" % name, "wb")
log = open("/tmp/%s.log" % name, "wb")
p = subprocess.Popen(args, cwd="/data/o35/llama.cpp", stdout=out, stderr=log)
pid = p.pid

def stat():
    f = open("/proc/%d/stat" % pid).read().split(")", 1)[1].split()
    return int(f[7]), int(f[8]), int(f[21]) * 4096

def io():
    d = {}
    for line in open("/proc/%d/io" % pid):
        k, v = line.split(":")
        d[k] = int(v)
    return d["read_bytes"], d["rchar"], d["write_bytes"]

def disk():
    for l in open("/proc/diskstats"):
        f = l.split()
        if f[2] == "sda":
            return int(f[5]) * 512

t0 = time.time()
last = None
while p.poll() is None:
    try:
        mn, mj, rss = stat()
        rb, rc, wb = io()
        d = disk()
    except Exception:
        break
    row = (round(time.time() - t0, 2), mn, mj, rss, rb, rc, wb, d)
    if row[4:] != last:
        print(" ".join(str(x) for x in row), flush=True)
        last = row[4:]
    time.sleep(0.2)
out.close()
log.close()
