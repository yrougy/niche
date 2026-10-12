#!/usr/bin/env python3
"""Minimal GGUF metadata + tensor-info dumper (no numpy needed)."""
import sys, struct

TYPES = ["I8", "U8", "I16", "U16", "I32", "U32", "F32", "BOOL", "STRING", "ARR", "U64", "I64", "F64"]

GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4", 40: "NVFP4",
    41: "Q1_0", 42: "Q2_0",
}

def ggml_type_name(t):
    return GGML_TYPES.get(t, f"TYPE{t}")

def read_string(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", "replace")

def read_value(f, t):
    if t == 0: return struct.unpack("b", f.read(1))[0]
    if t == 1: return struct.unpack("B", f.read(1))[0]
    if t == 2: return struct.unpack("<h", f.read(2))[0]
    if t == 3: return struct.unpack("<H", f.read(2))[0]
    if t == 4: return struct.unpack("<i", f.read(4))[0]
    if t == 5: return struct.unpack("<I", f.read(4))[0]
    if t == 6: return struct.unpack("<f", f.read(4))[0]
    if t == 7: return struct.unpack("<?", f.read(1))[0]
    if t == 8: return read_string(f)
    if t == 9:
        et = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        return [read_value(f, et) for _ in range(n)]
    if t == 10: return struct.unpack("<Q", f.read(8))[0]
    if t == 11: return struct.unpack("<q", f.read(8))[0]
    if t == 12: return struct.unpack("<d", f.read(8))[0]
    raise ValueError(t)

QK_K = 256
GGML_QUANT_SIZES = {
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2),
    "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22), "Q5_1": (32, 24),
    "Q8_0": (32, 34), "Q8_1": (32, 36),
    "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144), "Q5_K": (256, 156),
    "Q6_K": (256, 168), "Q8_K": (256, 292),
    "IQ2_XXS": (256, 66), "IQ2_XS": (256, 74), "IQ3_XXS": (256, 98),
    "IQ1_S": (256, 68), "IQ4_NL": (32, 18), "IQ3_S": (256, 104),
    "IQ2_S": (256, 72), "IQ4_XS": (256, 132), "IQ1_M": (256, 56),
    "I8": (1, 1), "I16": (1, 2), "I32": (1, 4), "I64": (1, 8), "F64": (1, 8),
    "TQ1_0": (256, 54), "TQ2_0": (256, 66), "MXFP4": (32, 17), "NVFP4": (64, 36),
    "Q1_0": (128, 18), "Q2_0": (64, 18),
}

def tensor_bytes(dims, tname):
    n = 1
    for d in dims:
        n *= d
    bs, ts = GGML_QUANT_SIZES[tname]
    return (n // bs) * ts

def main(path, tensor_filter):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF", "not GGUF"
    ver = struct.unpack("<I", f.read(4))[0]
    if ver >= 3:
        n_tensors_hdr = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
    else:
        n_kv = struct.unpack("<Q", f.read(8))[0]
        n_tensors_hdr = None
    meta = {}
    for _ in range(n_kv):
        k = read_string(f)
        t = struct.unpack("<I", f.read(4))[0]
        meta[k] = (TYPES[t], read_value(f, t))
    n_tensors = n_tensors_hdr if n_tensors_hdr is not None else struct.unpack("<Q", f.read(8))[0]
    tensors = []
    for _ in range(n_tensors):
        name = read_string(f)
        n_d = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_d)]
        dt = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        tensors.append((name, dims, ggml_type_name(dt), off))
    f.close()

    print(f"# version={ver} n_tensors_hdr={n_tensors_hdr} n_kv={n_kv} n_tensors={n_tensors}")
    for k in sorted(meta):
        if tensor_filter(k):
            print(f"{k} [{meta[k][0]}] = {meta[k][1]}")
    print("# ---- tensors ----")
    for name, dims, dt, off in tensors:
        if tensor_filter(name):
            b = tensor_bytes(dims, dt)
            per_expert = b // dims[2] if len(dims) == 3 else 0
            print(f"{name} {dt} dims={dims} bytes={b} per_expert={per_expert}")

if __name__ == "__main__":
    filt = lambda s: True
    if len(sys.argv) > 2:
        pat = sys.argv[2]
        filt = lambda s: pat in s
    main(sys.argv[1], filt)
