#!/usr/bin/env python3
"""Evict the page cache by touching more anonymous memory than RAM can hold.

Usage: python3 tools/evict.py <GiB>
"""
import sys, mmap

size = int(float(sys.argv[1]) * 1024**3)
m = mmap.mmap(-1, size)
chunk = b"\x01" * (1024**2)
for off in range(0, size, len(chunk)):
    m.write(chunk)
print("touched %d GiB" % (size // 1024**3))
