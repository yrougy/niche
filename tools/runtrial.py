#!/usr/bin/env python3
"""Premier essai realiste : run long (prompt B, -n N) avec echantillonnage des lectures disque.

Python est le parent de llama-cli, donc /proc/PID/io est lisible. Pas de strace : il
fausse le t/s. Sortie format compatible tools/phase6.py.

Usage : python3 tools/runtrial.py <name> <n> <mode> [pin]
  mode = clean (bank GPU) | cpufb (fallback CPU) | allcpu (q = 0, tout sur le CPU)
"""
import sys, subprocess, time, os, glob, ctypes

libc = ctypes.CDLL("libc.so.6", use_errno=True)
mm = libc.mmap; mm.restype = ctypes.c_void_p
mm.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
path = "/data/models/unsloth/Qwen3.6-35B-A3B-UD-IQ2_M.gguf"
fd = os.open(path, os.O_RDONLY); size = os.fstat(fd).st_size
addr = mm(None, size, 1, 1, fd, 0)
PG = 4096
B = 1024 * 1024 * 1024
nb = (size + B - 1) // B
vec = ctypes.create_string_buffer((size + PG - 1) // PG)
ptr = ctypes.cast(vec, ctypes.POINTER(ctypes.c_ubyte))


def sweep():
    if libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), ptr) != 0:
        return [-1] * nb
    raw = vec.raw
    out = []
    for i in range(nb):
        lo = i * (B // PG)
        out.append(raw[lo:lo + B // PG].count(1) * PG // 1048576)
    return out


name, n, mode = sys.argv[1], sys.argv[2], sys.argv[3]
pin = sys.argv[4] if len(sys.argv) > 4 else "0"
args = ["./build/bin/llama-cli", "-v", "-m", path, "-cmoe", "-ngl", "99",
        "--moe-cache-mib", "1024", "--moe-profile", "/tmp/prof2.csv",
        "--moe-cache-static",
        "-c", "2048", "-n", n, "--temp", "0", "-s", "42",
        "-p", "Q: Explain how a diesel engine works. A:", "--no-display-prompt"]
if mode in ("cpufb", "allcpu"):
    args += ["--moe-cache-cpu-fallback"]
if mode == "allcpu":
    args += ["--moe-cache-mib", "32"]
if pin != "0":
    args += ["--moe-cache-pin-mib", pin]

out = open("/tmp/%s.txt" % name, "wb"); log = open("/tmp/%s.log" % name, "wb")
p = subprocess.Popen(args, cwd="/data/o35/llama.cpp", stdout=out, stderr=log)
pid = p.pid
t0 = time.time()


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


print("pid=%d blocks=%d" % (pid, nb), flush=True)
while p.poll() is None:
    try:
        mn, mj, rss = stat()
        rb = ioread(); d = disk()
        r = sweep()
    except Exception:
        break
    print("%.3f %d %d %d %d %d %s" % (time.time() - t0, mn, mj, rss, rb, d,
          " ".join(str(x) for x in r)), flush=True)
    time.sleep(0.13)
out.close(); log.close()
