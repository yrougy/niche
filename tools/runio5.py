import sys, subprocess, time, os, glob

name, n, pin = sys.argv[1], sys.argv[2], sys.argv[3]
path = "/data/models/unsloth/Qwen3.6-35B-A3B-UD-IQ2_M.gguf"
args = ["./build/bin/llama-cli", "-v", "-m", path, "-cmoe", "-ngl", "99",
        "--moe-cache-mib", "1024", "--moe-profile", "/tmp/prof2.csv", "--moe-cache-static",
        "-c", "2048", "-n", n, "--temp", "0", "-s", "42",
        "-p", "Q: What is the capital of France? A:", "--no-display-prompt"]
if pin != "0":
    args += ["--moe-cache-pin-mib", pin]

out = open("/tmp/%s.txt" % name, "wb")
log = open("/tmp/%s.log" % name, "wb")
p = subprocess.Popen(["strace", "-f", "-e", "trace=read,pread64,readv,preadv", "-yy", "-tt",
                      "-s", "32", "-o", "/tmp/%s.strace" % name] + args,
                     cwd="/data/o35/llama.cpp", stdout=out, stderr=log)
sp = p.pid

pid = None
t0 = time.time()
while pid is None and p.poll() is None:
    for d in glob.glob("/proc/[0-9]*"):
        try:
            st = open(d + "/stat").read()
            comm = st.split("(", 1)[1].split(")", 1)[0]
            ppid = int(st.split(")", 1)[1].split()[1])
            if comm == "llama-cli" and ppid == sp:
                pid = int(d.split("/")[-1]); break
        except Exception:
            pass
    if pid is None:
        time.sleep(0.05)

def stat():
    f = open("/proc/%d/stat" % pid).read().split(")", 1)[1].split()
    return int(f[7]), int(f[8]), int(f[21]) * 4096

def ioread():
    d = {}
    for line in open("/proc/%d/io" % pid):
        k, v = line.split(":")
        d[k] = int(v)
    return d["read_bytes"], d["rchar"]

def disk():
    for l in open("/proc/diskstats"):
        f = l.split()
        if f[2] == "sda":
            return int(f[5]) * 512

print("pid=%d" % pid, flush=True)
while p.poll() is None:
    try:
        mn, mj, rss = stat()
        rb, rc = ioread()
        d = disk()
    except Exception:
        break
    print("%.3f %d %d %d %d %d %d" % (time.time() - t0, mn, mj, rss, rb, rc, d), flush=True)
    time.sleep(0.02)
out.close(); log.close()
