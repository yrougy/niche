#!/usr/bin/env python3
"""Analyse a MoE routing profile written by llama.cpp's --moe-profile-output.

The CSV has the header "layer,expert,count,weight_sum" where count is the number of
times the router picked that expert and weight_sum is the sum of the routing weights
of those picks. This script answers the question the MoE cache needs: how many experts
must stay resident to cover how much of the routing, and is a per-layer quota better
than a global ranking.

No numpy. Pure stdlib.

usage: python3 tools/analyze_moe_profile.py prof.csv [--sizes 10,20,30,40,50]
"""

import argparse
import sys


def read_profile(path):
    rows = []
    with open(path) as f:
        header = f.readline()
        if header.strip() != "layer,expert,count,weight_sum":
            raise SystemExit("unexpected header: %r" % (header.strip(),))
        for line in f:
            line = line.strip()
            if not line:
                continue
            il, e, c, w = line.split(",")
            rows.append((int(il), int(e), int(c), float(w)))
    return rows


def coverage(picked, total_selections, total_weight):
    hits = sum(r[2] for r in picked)
    mass = sum(r[3] for r in picked)
    return hits, mass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--sizes", default="10,20,30,40,50",
                    help="cache sizes as a percentage of all experts, comma separated")
    ap.add_argument("--top", type=int, default=10, help="how many hottest experts to print")
    ap.add_argument("--expert-used", type=int, default=0,
                    help="experts the router picks per token (default: infer from the counts)")
    args = ap.parse_args()

    rows = read_profile(args.csv)
    if not rows:
        raise SystemExit("no rows in %s" % args.csv)

    n_layer = max(r[0] for r in rows) + 1
    n_expert = max(r[1] for r in rows) + 1
    total_experts = n_layer * n_expert

    total_selections = sum(r[2] for r in rows)
    total_weight = sum(r[3] for r in rows)
    per_layer_sel = {}
    for r in rows:
        per_layer_sel[r[0]] = per_layer_sel.get(r[0], 0) + r[2]
    # the router picks n_expert_used experts per token in every layer, so the counts of a
    # layer add up to n_expert_used * tokens_of_that_layer. all layers see the same tokens.
    layers = sorted(per_layer_sel)
    n_expert_used = args.expert_used
    if n_expert_used <= 0:
        n_tokens = min(per_layer_sel[il] for il in layers)
        for il in layers:
            k = per_layer_sel[il] // n_tokens
            if n_expert_used == 0 or k > n_expert_used:
                n_expert_used = k
    tokens = [per_layer_sel[il] / n_expert_used for il in layers]
    n_tokens = min(tokens)

    print("file            %s" % args.csv)
    print("layers          %d (profiled: %d)" % (n_layer, len(layers)))
    print("experts/layer   %d" % n_expert)
    print("expert_used     %d" % n_expert_used)
    print("tokens          %d" % n_tokens)
    print("selections      %d" % total_selections)
    print("weight mass     %.1f" % total_weight)
    print("(layer,expert) seen %d of %d = %.1f%%" %
          (len(rows), total_experts, 100.0 * len(rows) / total_experts))
    print()

    by_count = sorted(rows, key=lambda r: (-r[2], -r[3], r[0], r[1]))

    print("hottest experts (layer, expert, count, weight, share of all selections)")
    for r in by_count[:args.top]:
        print("  %2d %3d  %8d  %8.2f  %.3f%%" %
              (r[0], r[1], r[2], r[3], 100.0 * r[2] / total_selections))
    print()

    sizes = [float(x) for x in args.sizes.split(",")]

    print("cache size as %% of all %d experts:" % total_experts)
    print("  %6s %10s %8s %8s %8s %8s" %
          ("size%", "experts", "slots/ly", "hit%", "hit%(q)", "mass%"))
    for pct in sizes:
        k = int(round(total_experts * pct / 100.0))
        if k > total_experts:
            k = total_experts
        hits, mass = coverage(by_count[:k], total_selections, total_weight)

        # per-layer quota: keep the hottest k/layer experts of each layer, spread the
        # budget over the layers that have routing data
        quota = k // len(layers) if layers else 0
        picked = []
        for il in layers:
            layer_rows = [r for r in rows if r[0] == il]
            layer_rows.sort(key=lambda r: (-r[2], -r[3], r[1]))
            picked.extend(layer_rows[:quota])
        q_hits, q_mass = coverage(picked, total_selections, total_weight)

        print("  %6.1f %10d %8d %8.2f %8.2f %8.2f" %
              (pct, k, quota, 100.0 * hits / total_selections,
               100.0 * q_hits / total_selections, 100.0 * mass / total_weight))
    print()

    # compulsory misses: the first time an expert is seen, it cannot be resident
    seen = set()
    compulsory = 0
    ordered = sorted(rows, key=lambda r: (r[0], r[1]))
    for r in ordered:
        seen.add((r[0], r[1]))
        compulsory += 1
    print("distinct experts used %d -> compulsory misses %.2f%% of selections" %
          (len(seen), 100.0 * compulsory / total_selections))

    # distribution of heat per layer
    counts = sorted((r[2] for r in rows), reverse=True)
    for q in (1, 4, 8, 16, 32, 64):
        if q <= len(counts):
            top = sum(counts[:q])
            print("  top %3d experts overall cover %.2f%% of selections" %
                  (q, 100.0 * top / total_selections))

    # per-layer spread: how uneven is the routing across layers
    print()
    print("per-layer selections (normalised to the first layer):")
    for il in layers:
        print("  layer %2d  %8d  %.2f tokens" %
              (il, per_layer_sel[il], per_layer_sel[il] / n_expert_used))


if __name__ == "__main__":
    main()
