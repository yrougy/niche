import ctypes, sys, os

libc = ctypes.CDLL("libc.so.6", use_errno=True)
mm = libc.mmap
mm.restype = ctypes.c_void_p
mm.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
PROT_READ = 1
MAP_SHARED = 1
page = 4096

path = sys.argv[1]
fd = os.open(path, os.O_RDONLY)
size = os.fstat(fd).st_size
addr = mm(None, size, PROT_READ, MAP_SHARED, fd, 0)
if addr is None or addr == ctypes.c_void_p(-1).value:
    print("mmap failed errno=%d (%s)" % (ctypes.get_errno(), os.strerror(ctypes.get_errno())))
    sys.exit(1)
vec = ctypes.create_string_buffer(size // page)
vp = ctypes.cast(vec, ctypes.POINTER(ctypes.c_ubyte))
r = libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(size), vec)
if r != 0:
    print("mincore failed errno=%d" % ctypes.get_errno())
    sys.exit(1)
tot = 0
blocks = []
run = 0
for i in range(size // page):
    res = vp[i] & 1
    tot += res
    run += res
    if i % 262144 == 262143:
        blocks.append((i // 262144, run))
        run = 0
mib = 1048576.0
print("file %.2f GiB, resident %.2f GiB (%.2f MiB)" % (size / 1073741824.0, tot * page / 1073741824.0, tot * page / mib))
for b, c in blocks:
    print("GiB block %d: %.2f MiB resident" % (b, c * page / mib))
