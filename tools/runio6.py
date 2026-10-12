import sys, subprocess, time, os, glob, ctypes

libc = ctypes.CDLL("libc.so.6", use_errno=True)
mm = libc.mmap; mm.restype = ctypes.c_void_p
mm.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
path = "/data/models/unsloth/Qwen3.6-35B-A3B-UD-IQ2_M.gguf"
fd = os.open(path, os.O_RDONLY); size = os.fstat(fd).st_size
addr = mm(None, size, 1, 1, fd, 0)
W = 64*1024*1024
wins = [ (i*1024*1024*1024) % (size - W) for i in range(8) ]
vecs = []
for o in wins:
    v = ctypes.create_string_buffer(W // 4096)
    vecs.append(ctypes.cast(v, ctypes.POINTER(ctypes.c_ubyte)))

def res(i):
    if libc.mincore(ctypes.c_void_p(addr + wins[i]), ctypes.c_size_t(W), vecs[i]) != 0:
        return -1
    n = 0
    for k in range(W // 4096):
        n += vecs[i][k] & 1
    return n * 4096 // 1048576

name, n, pin = sys.argv[1], sys.argv[2], sys.argv[3]
args = ["./build/bin/llama-cli", "-v", "-m", path, "-cmoe", "-ngl", "99",
        "--moe-cache-mib", "1024", "--moe-profile", "/tmp/prof2.csv", "--moe-cache-static",
        "-c", "2048", "-n", n, "--temp", "0", "-s", "42",
        "-p", "Q: What is the capital of France? A:", "--no-display-prompt"]
if pin != "0":
    args += ["--moe-cache-pin-mib", pin]

out = open("/tmp/%s.txt" % name, "wb"); log = open("/tmp/%s.log" % name, "wb")
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
    return d["read_bytes"]

def disk():
    for l in open("/proc/diskstats"):
        f = l.split()
        if f[2] == "sda":
            return int(f[5]) * 512

print("pid=%d windows=%s" % (pid, wins), flush=True)
i = 0
while p.poll() is None:
    try:
        mn, mj, rss = stat()
        rb = ioread(); d = disk()
        r = res(i)
    except Exception:
        break
    print("%.3f %d %d %d %d %d %d %d" % (time.time() - t0, mn, mj, rss, rb, d, i, r), flush=True)
    i = (i + 1) % 8
    time.sleep(0.02)
out.close(); log.close()
