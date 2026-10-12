import sys, subprocess, time, os, ctypes

libc = ctypes.CDLL("libc.so.6", use_errno=True)
mm = libc.mmap
mm.restype = ctypes.c_void_p
mm.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
PROT_READ, MAP_SHARED, page = 1, 1, 4096

path = "/data/models/unsloth/Qwen3.6-35B-A3B-UD-IQ2_M.gguf"
fd = os.open(path, os.O_RDONLY)
size = os.fstat(fd).st_size
addr = mm(None, size, PROT_READ, MAP_SHARED, fd, 0)
vec = ctypes.create_string_buffer(size // page)
vp = ctypes.cast(vec, ctypes.POINTER(ctypes.c_ubyte))

def resident():
    if libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec) != 0:
        return -1
    return sum(vp[i] & 1 for i in range(size // page)) * page // 1048576

name, n, pin = sys.argv[1], sys.argv[2], sys.argv[3]
args = ["./build/bin/llama-cli", "-v", "-m", path, "-cmoe", "-ngl", "99",
        "--moe-cache-mib", "1024", "--moe-profile", "/tmp/prof2.csv", "--moe-cache-static",
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

t0 = time.time()
last = None
while p.poll() is None:
    try:
        mn, mj, rss = stat()
        rb = ioread()
        d = disk()
        res = resident()
    except Exception:
        break
    row = (round(time.time() - t0, 2), mn, mj, rss, rb, d, res)
    if row[4:] != last:
        print(" ".join(str(x) for x in row), flush=True)
        last = row[4:]
    time.sleep(0.2)
out.close()
log.close()
