#!/usr/bin/python3
"""Comparaison de sorties de llama-cli : diff de contenu + prefixe commun.

Les lignes de bruit (spinner de chargement, ligne `build      :`, ligne de
t/s, `Exiting`, `>`, lignes vides) sont retirees avant toute comparaison,
sinon le diff compte 4 lignes de differences qui ne sont pas du texte genere.

Usage:
  tools/cmp.py A.txt B.txt [C.txt ...]   # toutes les paires
  tools/cmp.py --prefix A.txt B.txt      # texte filtre de A (pour lecture)
"""
import subprocess
import sys


def filt(path):
    out = []
    for line in open(path, encoding="utf-8", errors="replace"):
        if any(s in line for s in ("t/s", "Loading model", "Exiting")):
            continue
        if line.strip() in (">", ""):
            continue
        if line.startswith("build      :"):
            continue
        out.append(line)
    return "".join(out)


def diff_lines(a, b):
    fa, fb = "/tmp/cmp_a.txt", "/tmp/cmp_b.txt"
    open(fa, "w").write(filt(a))
    open(fb, "w").write(filt(b))
    r = subprocess.run(["diff", "-u", fa, fb], capture_output=True, text=True)
    return sum(1 for l in r.stdout.splitlines()
               if l.startswith(("+", "-")) and not l.startswith(("+++", "---")))


def prefix(a, b):
    A, B = filt(a), filt(b)
    i = 0
    n = min(len(A), len(B))
    while i < n and A[i] == B[i]:
        i += 1
    return i, len(A), len(B)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--prefix":
        A, B = args[1], args[2]
        i, la, lb = prefix(A, B)
        print("prefixe commun %s/%s = %d chars (A=%d B=%d)" % (A, B, i, la, lb))
        sys.exit(0)
    for i in range(len(args)):
        for j in range(i + 1, len(args)):
            a, b = args[i], args[j]
            d = diff_lines(a, b)
            p, la, lb = prefix(a, b)
            print("%s vs %s : diff=%d  prefixe=%d  (lenA=%d lenB=%d)"
                    % (a, b, d, p, la, lb))
