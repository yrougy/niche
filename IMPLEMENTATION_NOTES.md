# IMPLEMENTATION_NOTES — what was ported from Strata, where it lives, what was refused

Branch `experimental/ornith-expert-cache` in `/data/o35/llama.cpp` (base `master` @ `23b0202a1`).
Commits: `5bfb641e4` (routing instrumentation), `1a1811266` (profile-seeded static cache + metrics),
`de0531d8f` (RAM complement pinning). Test model
`/data/models/unsloth/Qwen3.6-35B-A3B-UD-IQ2_M.gguf` (`qwen35moe`, 256 experts, 8 used, 41 blocks,
11.07 GiB, 0.913 MiB per expert, experts = 95.6 % of the file). Box: GTX 1070 sm_61, i7-4790,
15 GiB RAM, **spinning disk**, `RLIMIT_MEMLOCK` = 1.945 GiB, kernel 6.8.

The port extends `llama_moe_cache` — llama.cpp already had the core Strata idea (per-expert slots,
host slot map, device banks, copy-callback upload). No new model format, no tokenizer change, no
rewrite of llama.cpp.

## 1. Mechanism map

| Strata source (file:line) | Mechanism | Verdict | llama.cpp implementation |
|---|---|---|---|
| `include/strata/core/expert_cache.hpp:38` `kNotResident = -1` | sentinel for a non-resident expert | **reuse** | llama.cpp already used `-1` in `slot_map`; unchanged |
| `src/core/expert_cache.cpp:633-636` `slot_of(layer, expert)` | bounds-checked lookup | **reuse** | `llama_moe_cache::get_slot_map()` `src/llama-moe-cache.cpp:797` |
| `src/core/expert_cache.cpp:638-656` `admit()` — never evicts, `if (next_free_ >= slots_) return kNotResident` | static admission | **adapt** | `moe_cache_lru::seed()` `src/llama-moe-cache.cpp:68`, `plan()` `:124` with the `static_mode` branch at `:174` (scratch slots for misses, no eviction of seeded slots) |
| `src/core/expert_cache.cpp:141-168` `rank_learned_profile` — sort resident, then descending heat, then prior rank, then index | ranking of the (layer, expert) list | **reuse** | ranking built in `llama_moe_cache_load_profile()` `src/llama-moe-cache.cpp:214` |
| `src/core/expert_cache.cpp:170-210` `write_expert_profile` — `STRP` binary, atomic `.tmp` + rename | profile persistence | **adapt** | CSV instead of `STRP`: `src/llama-moe-profile.cpp` writes `layer,expert,count,weight_sum` (numpy-free, no float formatting so the French locale cannot corrupt it); atomic write on destruction |
| `src/core/expert_cache.cpp:624-631` `layer_slot_range` + `expert_cache.hpp:170` `set_per_layer_admission` | per-layer quota so one layer cannot starve another | **adapt — the key structural fix** | per-layer disjoint slot ranges `[lo[il], hi[il])`: seeded slots `[lo[il], lo[il]+qi)` are never evicted, `[lo[il]+qi, hi[il])` is that layer's own LRU. `plan()` takes `head[chain[il]]` only inside the layer's range, `touch()` `:103` only touches `slots[e] >= lo[il]`. Cross-layer eviction is structurally impossible |
| `src/core/expert_cache.cpp:567-568` round each slot to 256-byte alignment | alignment for vector loads | **reuse** | llama.cpp already adds one extra slot (`src/llama-moe-cache.cpp:270-278`); unchanged |
| `src/core/expert_cache.cpp:527-533` `cudaMemset(base_, 0, want)` so stale slots are deterministic | deterministic stale slots | **reuse** | existing llama.cpp behaviour, verified |
| `src/core/expert_source.cpp:3562-3564` `d.usage[layers*n_expert + ids[i]] += 1.0f` + `adapt_decay` 0.7 (`generate.cpp:578, 8277`) | live (layer, expert) counters | **adapt** | counters in `src/llama-moe-profile.cpp` (count + weight sum per `(layer, expert)`), exported as CSV by `--moe-profile-output`; hook points `src/llama-graph.cpp:2245-2261` (host `ids`/`weights` copy nodes) and `src/llama-context.cpp:2686-2688` (accumulate after the split) |
| `src/core/expert_source.cpp:2526-2586` `pin_cache_complement` | fault the complement into RAM, lock the hottest, drop the pages of VRAM-resident experts | **adapt — this is requirement 5** | `cache_complement()` `src/llama-moe-cache.cpp:354-476`, called at `:761` under `static_mode && !model.hparams.no_alloc`. Order: `release()` resident ranges → `populate()` complement (contiguous ranges merged per mapping) → **`populate()` the gaps between expert ranges** (see below) → `pin()` hottest complement experts in `ranked[il]` order, one `mlock` per expert, stop at `pin_bytes` (`:457`) or when `pin()` fails. Log line `MoE complement: ...` |
| *(no Strata analogue)* — filling the file between the expert ranges | the tail of the GGUF (MTP block, attention weights, embeddings) is read during decode on a cold cache | **llama.cpp addition, measured necessary** | after the complement populate, walk all expert ranges sorted by `(mapping, first)` and `populate()` every gap from 0 to `mapping->size()` (`src/llama-moe-cache.cpp`, `rest_bytes`). Accessors `llama_model::n_mappings()` / `mapping(i)` (`src/llama-model.h:793-795`). Cold-run reads drop 207,52 → 112,69 MiB, all six verify diffs zero |
| `include/strata/core/pinned.hpp` `PinnedArena` (`LargePages`, `PinnedByCuda`, `locked_bytes`, `STRATA_ARENA_PIN_GIB`) | pinned arena | **adapt** | `llama_mmap::pin()` `src/llama-mmap.h:65`, Linux `mlock` `src/llama-mmap.cpp:604-613` (warns and returns false), Windows `VirtualLock` `:702`. Huge pages unavailable on this box, so ranges are 4 KB pages |
| `src/llama-mmap` (llama.cpp's own) `unmap_fragment` `src/llama-mmap.h:59` | drop a mapped range | **rejected as dangerous** | `copy_experts()` reads host expert tensors straight from the mmap during prefill (`src/llama-context.cpp:2696-2771`) → unmapping = SIGSEGV. `MADV_DONTNEED` (`release()`, `src/llama-mmap.cpp:618-626`) keeps the mapping valid; a later read re-faults from disk (prefill pays, decode does not) |
| `include/strata/kernels/dp4a.hpp` | `__dp4a` | **available on sm_61** | not needed: llama.cpp's MMQ path already runs on Pascal |
| `include/strata/kernels/pdl.hpp:40-41` — `__restrict__` is what lets the compiler use the read-only data path on Pascal, "without it decode on sm_61 halves" | codegen rule | **reuse as a rule** | applies to any new sm_61 kernel; no new kernel was written |
| `src/core/expert_source.cpp:3522-3760` `expert_pool_dispatch_multi` with per-entry `kind[]` (`-1` CPU, `0` VRAM, `1` PCIe, `2` peer, `-2` tail-skipped), GPU share published first (`:3566`) | per-expert CPU/GPU dispatch | **adapt conceptually, kernel rejected (sm_61)** | open blocker: `ggml_mul_mat_id` has a fixed `n_expert_used`, so experts cannot be skipped per token. MVP (A) = layer granularity via the `-cmoe`/`-ncmoe` budget chosen from the profile; MVP (B) = a new ggml op taking a per-slot `kind[]` |
| `expert_source.cpp:3596-3615` `tail_rank` (`STRATA_ROUTE_TAIL_SKIP`, default 7) | skip experts routed only at rank >= R | **defer** | needs the per-token skip capability, i.e. MVP (B) |
| `src/core/vmm.cpp`, `cpp:305-306` VMM | elastic VRAM | **reject** | Turing+, not Pascal |
| `src/prefill/moe_fused.cu` (`mma.sync m16n8k32`, `cp.async`), `src/kernels/cuda/qsa_prompt_attn.cu:31-34`, `s2_expert_grouped.cu`, `native_moe.cu`, `verify_kernels.cu:839` (clusters) | prefill/grouped kernels | **reject** | sm_75/sm_80/sm_90 only; Strata's CMake refuses `CMAKE_CUDA_ARCHITECTURES < 75` (`CMakeLists.txt:146-193`) |
| `src/core/peer_experts.cpp`, `remote_experts.cpp`, `remote_expert_opt.cu` | multi-GPU | **reject** | one GPU |
| `generate.cpp:8153-8279` `adapt` lambda, `peer_experts.cpp:276-319`, `foresight_swap.hpp:37-54` | adaptive swap tier | **defer** | post-MVP |
| `expert_cache.hpp:11-17, 45-50` — `h_expert = 0.6447` leave-one-out vs `h_layer = 0.0456`, compulsory miss `h = 0.4864` | evidence that per-expert granularity is mandatory | **reuse as design guidance** | confirmed on this model: at 1024 MiB the cache serves 48.77 % of decode expert picks; the coverage curve from `/tmp/prof2.csv` (1026 tokens, 82 % of pairs seen) is in `PROGRESS.md` |
| `expert_source.hpp:18-21` `H=2560, FF=640, BLOB=1 382 400` with compile-time blob offsets, other geometry refused | layout constants | **reject** | Strata's geometry is Qwen3.5-Next 48x512, not qwen35moe 2048x512 |

## 2. New code, by file

**`src/llama-moe-profile.{h,cpp}`** (new, 38 + 251 lines) — per-`(layer, expert)` counters, CSV writer.
Hooked through `src/llama-graph.{h,cpp}` (host `ids`/`weights` copy nodes in `build_moe_ffn`,
`:2245-2261`), `src/llama-context.{h,cpp}` (accumulate at `:2686-2688`, creation at `:501-502`),
`src/llama-cparams.h`, `include/llama.h`, `common/{common.h,common.cpp,arg.cpp,speculative.cpp}`.
Option `--moe-profile-output` (`common/arg.cpp:2790`).

**`src/llama-moe-cache.{h,cpp}`** — `llama_moe_cache_load_profile()` `:214`, `seed()` `:68`,
`init_chains()` `:82` (must be given `g.layers`, not the model's layer list), `touch()` `:103`,
`plan()` `:124` + `static_mode` branch `:174`, `static_mode` `:321`, `pin_bytes` `:324`,
`cache_complement()` `:354-476`, constructor `:479`, static block `:689`, call site `:761`,
`get_slot_map()` `:797`, merged miss uploads `:909`, `log_stats()` `:927`, `export_stats()` `:985`,
wrapper `:1025-1027`. Options `--moe-profile` (`arg.cpp:2799`), `--moe-cache-static` (`:2808`),
`--moe-cache-pin-mib` (`:2816`), `--moe-stats` (`:2825`).

**`src/llama-mmap.{h,cpp}`** — `populate()` `llama-mmap.h:62`, `pin()` `:65`, `release()` `:68`,
`POPULATE_SUPPORTED` `:73`, `PIN_SUPPORTED` `:74`. Linux: `align_full()` `llama-mmap.cpp:580`
(full page alignment, clamped to the file — the existing `align_range()` `:531` skips the first
partial page and is unusable here), `MADV_POPULATE_READ` with `POSIX_MADV_WILLNEED` fallback
`:587-600`, `mlock` `:604-613`, `MADV_DONTNEED` `:618-626`. Windows: `VirtualLock` `:702`,
populate/release are no-ops. Fallback: `throw std::runtime_error("mmap not supported")`.

**`src/llama-model.{h,cpp}`** — `struct llama_mmap;` forward declaration (`llama-model.h` does not
include `llama-mmap.h`), `mmap_range()` declared `llama-model.h:791`, defined
`llama-model.cpp:2382`: finds `t->data` inside `pimpl->mappings` (declared `llama-model.cpp:1242`,
filled `:2013-2014`, `init_mappings(true, ...)` `:1870`) and returns the mapping plus the
`[first, last)` byte range within it. `n_mappings()` / `mapping(i)` (`llama-model.h:793-795`,
defined after `mmap_range()` in `llama-model.cpp`) expose the mmap'ed files so the cache can fault
in the bytes between the expert ranges.

**Plumbing for `--moe-cache-pin-mib`** — `include/llama.h` (`int64_t moe_cache_pin_mib`),
`src/llama-cparams.h`, `src/llama-context.cpp` (assign `:305`, default `0` `:3937`, construction
`:497-498`), `common/common.h`, `common/common.cpp`, `common/arg.cpp` (the `common_arg` lambdas
have no `int64_t`/`size_t` overloads, so the option takes `int`).

**Built-in metric** — `llama_read_majflt()` `src/llama-context.cpp:37`, `src/llama-context.h:308`,
printed at `src/llama-context.cpp:553`: `major page faults in the first decode ubatch`.

**CPU fallback (`--moe-cache-cpu-fallback`)** — plumbing: `include/llama.h`
(`bool moe_cache_cpu_fallback`), `src/llama-cparams.h`, `src/llama-context.cpp` (assign `:303`,
construction `:496`, default `:3937`), `common/common.{h,cpp}`, `common/arg.cpp` (option).

`src/llama-moe-cache.{h,cpp}` — constructor gains `bool cpu_fallback`; `n_extra = cpu_fallback ? 2 : 1`
extra slots in the bank (`:671`): slot `n_slots` is the **all-zero slot** the GPU reads for experts
computed on the CPU, `n_slots+1` is the padding CUDA reads past the end; the GPU buffer is cleared to 0
at `:724` (`ggml_backend_buffer_clear`) so the zero slot is truly zero. Per-layer host map `l.cpu_map`
(I32 `[1, n_expert]`) created `:691-695`, cleared to `0xff` at `:733`; `cpu[e] = -1` when `e` is
resident, `= e` otherwise; the resident slot of non-resident experts is set to `g.n_slots` `:779-792`.
`copy()` short-circuited `:992-1005` (counts hits/misses, uploads nothing, copies the slot map);
`get_slot_map()` `:895-905` skips the `n_tokens*n_expert_used > hi-lo` limit in this mode; the static
block sets `q = g.n_slots / n_layers` and `g.max_batch_cached = 4` (constant `:325`, block `:760-795`) —
**no LRU, no scratch**, the seeded residents stay for the whole run.

`src/llama-graph.{h,cpp}` — `build_lora_mm_id(..., ggml_tensor * ids_cpu = nullptr)`,
`build_moe_cache_slots(..., int il, ggml_tensor ** ids_cpu)`. `ids_cpu_rows = ggml_get_rows(ctx0,
cpu_map, ids)` forced to `backend_cpu` (`:2522`), reshaped `[8, n_tokens]`;
`res_cpu = ggml_mul_mat_id(ctx0, w, cur, ids_cpu)` forced to `backend_cpu` (`:1630`), then
`res = ggml_add(res, res_cpu)`. Wired at `:2288`, `:2307`, `:2320`, `:2421`; early return when
`slot_map == nullptr` (`:2494`) so **prefill is untouched**.

`ggml/src/ggml-cpu/ggml-cpu.c` — `if (i02 < 0) continue;` in the row grouping (`:1655`) so `-1` ids are
never claimed by a matrix, and a loop that **zeroes** the `dst` rows no matrix claimed (`:1667-1681`),
addressing `dst` as `id*nb1 + iid1*nb2` — the **same order** as the compute path at `:1534`. No CUDA
kernel is touched: `-1` never reaches the GPU.

Strata analogue: **none** — the zero slot has no counterpart in Strata. The closest concept is the
per-entry `kind[]` dispatch in `expert_source.cpp:3522-3760` (GPU share published first at `:3566`, tail
skip at `:3596-3615`), a *prefill* path whose kernels need `sm_75+` and are unusable on `sm_61`.

## 3. Invariants found the hard way

- **The copy callback runs only for host weights** (`ggml_backend.cpp:1835-1838`) and
  `ggml_backend_sched_compute_splits` copies **all** inputs of a split before computing it. Two
  layers of the same group in one split share the bank, so a later upload clobbered bytes an
  earlier layer was about to read → corrupted text. Per-layer disjoint ranges are the fix.
- **Eviction must not write `-1` into `slot_map`**: `get_rows` yields row `-1`, `mul_mat_id` reads
  row `-1` of the bank, and the CUDA illegal access surfaces at
  `ggml_backend_cuda_synchronize` (`ggml-cuda.cu:2633`).
- `init_chains()` must be given `g.layers`. Given the model's 41 layers, the non-group layers have
  `lo=0, hi=n_slots, chain=0` and the function rewrote head/queue of chain 0 across `[0, n_slots)`,
  destroying the per-layer chains.
- `max_batch_cached = 1` per group → **the cache is decode-only; prefill is not cached**.
- Only layers whose **every** expert tensor is in host memory are cached, so `-cmoe` is mandatory
  (without it 10 GB of experts go to the GPU and loading fails, exit 1).
- `GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il)` fires when
  static scratch is exhausted, or when top-k and the slot-map lookup land in the same split.
- Two `llama_context` objects exist (main + MTP); the first writes a profile/stats file with 0 pairs.
- `MADV_POPULATE_READ` needs no privilege on kernel 6.8; `mlock` is capped by
  `RLIMIT_MEMLOCK` = 1.945 GiB, so `--moe-cache-pin-mib 1990` locks 1986.96 MiB in 6635 experts and
  then stops cleanly on a 405504-byte `ENOMEM`.
- **`majflt` is not a disk-read indicator on this kernel**: a cold 100 MiB touch of the GGUF mmap
  gives `minflt += 1604`, `majflt += 0`, `read_bytes += 100.25 MiB`. Only `read_bytes`
  (`/proc/PID/io`) and `/proc/diskstats` are trustworthy.
- `strace` shows **zero `read()` calls on the GGUF inside the generation window** (20.98 MiB of
  `read()` on the GGUF, all during load). The cold-run reads in the first seconds of decode are
  **evicted pages being re-faulted** (RSS drops 11.41 → 10.29 GiB at t = 99 s, then climbs back),
  and the reads stop at t ≈ 102 s: the last 3.6 s of generation read nothing.

## 4. Numerical correctness

The cache changes tensor layout, so results are not bit-exact against the no-cache path — upstream
`master` shows the same divergence. "Preserve numerical correctness" here means: no dropped expert
contributions, no garbage. `bash tools/verify.sh` (delete `/tmp/*.done` first) on the current build:

| run | config | result |
|---|---|---|
| clean8 | static, `-n 8` | diff 0 vs `base8`, hit 39.58 % |
| clean128 | static, `-n 128` | diff 0 vs `base128`, hit 48.77 % |
| clean512 | static, `-n 512` | diff 0 vs `base512`, hit 48.94 % |
| lru8 | LRU, `-n 8` | diff 0 vs `base8`, hit 41.19 % + 5.43 % |
| lru128 | LRU, `-n 128` | bit-identical to `/tmp/lru_master.txt`, hit 52.95 % + 5.43 % |
| **pin128** | static + `--moe-cache-pin-mib 1990`, `-n 128` | **diff 0 vs `clean128`**, hit 48.77 % |
| **cpufb** | static + `--moe-cache-cpu-fallback`, `-n 128` | **diff 0 vs `clean128`**, hit 30.01 %, **uploaded 0.00 MiB** |

Seven runs, seven diffs = 0. Zero bytes dequantize to zero weights for every quant type in the model
(`iq2_xxs`, `iq3_xxs`, `iq4_xs`, `iq2_s`, `q2_K`) — verified by `/tmp/zeroq.c`, which dequantizes a block
of zero bytes per type. That is what makes the zero slot numerically valid.

Raw `diff` reports 4 lines per run; the only difference is the `build      : b11553-<hash>` line,
which changes with every commit. After filtering `^build      :`, every diff is 0. Use `tools/cmp.py`
for any output comparison — it strips the noise lines before diffing and reports the common prefix.

**Strongest exactness result**: `allcpuA` (`--moe-cache-mib 32 --moe-cache-static
--moe-cache-cpu-fallback` → `q = 0`, **every expert of every token computed on the CPU**) on prompt A at
`-n 128` has **content diff 0** vs `base128` (1121 identical chars). The CPU expert path is
mathematically exact, not an approximation. On prompt B the cache runs diverge from `baseB` because
`baseB` runs dense layers on the CPU (no `-cmoe`, no `-ngl`); the clean cache-vs-fallback comparison
(`cleanB` vs `cpufbB`, same CUDA dense path) shares **1514 of 9027 chars** (16.8 %) with 121 diff lines.

## 5. Measured decode behaviour

Decode window aligned to the log's `eval time` line (log timestamps are `M.SS.mmm`). Cold page cache
via `tools/evict.py 13` (12.3 GB → 1.68 GB). Full table in `PROGRESS.md`.

- **Warm page cache, pinned, `-n 1024` (pin17): `read_bytes` delta 0.00 MiB, disk delta 0.04 MiB.**
  Requirement 5 is met when the page cache is warm.
- Cold page cache: ~140 MiB read during the first ~5.4 s of generation, then silence. The number is
  insensitive to pin 1990, populate-only, `release()` disabled, LRU vs static, and KV size
  (80–89 MiB across configurations).
- **`strace` shows zero `read()` on the GGUF inside the decode window** (20.98 MiB / 5372 calls, all
  during load). The cold reads are mmap page-cache misses, not syscalls.
- **`majflt` is not a disk-read indicator on kernel 6.8**: a controlled cold 100 MiB touch of the
  GGUF mmap gives `minflt += 1604`, `majflt += 0`, `read_bytes += 100.25 MiB`. Only `/proc/PID/io`
  (`read_bytes`) and `/proc/diskstats` are trustworthy.
- **Residency of the expert ranges never dips during decode** (`runio6.py`, 8 rotating 64 MiB
  `mincore` windows at 20 ms, all 64 MiB resident in every sample) — the complement works.
- **Filling the gaps between the expert ranges** (`runio7.py`, full-file `mincore` per 1 GiB block at
  130 ms) drops cold-run reads from **207,52 MiB (pin28) to 112,69 MiB (pin29)**, disk 113,43 MiB.
  The reads concentrate in the first ~1.2 s of generation (27+27+27+10 MiB) and then fall to 1–4 MiB
  per sample. RSS drops 11.44 → 10.32 GiB mid-generation: the kernel evicts mapped pages under
  pressure (11.2 GiB page cache + ~1 GiB anon on 15 GiB RAM).
- **First long-run trial without MTP (prompt B, `-n 1024`, cold, `pin 1990`)** — new tool
  **`tools/runtrial.py`** (python is the parent of `llama-cli` so `/proc/PID/io` is readable, **no strace**,
  output compatible with `tools/phase6.py`); driver `tools/trial.sh` evicts then runs per mode:

  | mode | prefill t/s | **decode t/s** | hit rate | uploaded | **decode disk reads** | RSS |
  |---|---|---|---|---|---|---|
  | `cpufb` | 24.20 | **16.26** | 30.75 % (101443/228437) | **0.00 MiB** | **0.05 MiB** (disk 0.13) | 10.57 → 5.10 GiB |
  | `clean` | 17.22 | **20.13** | 56.10 % (183663/143721) | 131 327.19 MiB | **114.81 MiB** (disk 115.34) | 11.40 → 0.56 GiB |

  **Requirement 5 holds on a long run**: the CPU fallback reads **nothing** from disk during the 62.9 s of
  decode, against **115 MiB** for the all-GPU path (expert uploads re-read evicted mmap pages). Cost is
  **−19 % t/s**; prefill is **faster** with the fallback (24.20 vs 17.22 — no uploads during prefill). The
  load reads the whole file (10 576.65 MiB) and the 11.88 GB file **stays resident** afterwards
  (blocks 0..10 at 1024/1024 MiB, block 11 at 68 MiB), which is what makes the decode read-free. Output is
  coherent in both modes.
- **`--moe-stats` takes a FILE argument**: passing it bare swallows the next flag (`-c`) and the parser
  fails with `error: invalid argument: 2048` (207-byte run, exit 1). Hit-rate lines come from
  `LLAMA_LOG_INFO` and are in the log **without** `--moe-stats`.
- Ambient disk reads with no process running: 0.05 MiB over 10 s.
- RSS at decode start 10.67 GiB; GGUF residency 11.07 GiB (the whole file resident).
- Complement log: `8618.06 MiB faulted into RAM, 1986.48 MiB of the rest faulted into RAM,
  1986.96 MiB pinned in 6635 experts, 727.94 MiB of resident pages dropped`.
- Pinning's measurable benefit is **survival under memory pressure** (the kernel cannot evict the
  hottest 1.945 GiB of the complement); it did not reduce the cold-run read count.
- **Per-layer heat is uniform**: the 40 cached layers each have 8384 selections (2.50 % of 335 184),
  the MTP layer 8208. Caching the 8 "hottest" layers covers 20.01 % of selections, 32 layers 80.04 %.
  Layer-granularity CPU fallback (MVP A) is therefore **measured worthless**; the lever is per-expert.
- Dominant decode cost is PCIe upload, not disk: `uploaded = 26137.55 MiB` for 28608 misses over a
  9.4 s generation = **2.78 GB/s of PCIe**, close to saturating PCIe gen3 x16.
- **CPU fallback, warm page cache**: `-n 128` → 17.78 t/s, diff 0 vs `clean128`; `-n 1024` → 176 tokens
  (EOS), **17.22 t/s**, diff 0 vs `clean512` (1715 identical bytes) against **19.39 t/s** static →
  **11 % cost**. `uploaded = 0.00 MiB` for the whole decode, hit 30.24 % (no LRU, no scratch: only the
  27 seeded residents per layer count), **99 341 expert computations on the CPU** against 58 520 bank
  rows (63 % of the total), `1115/10496` experts resident (1016.88 MiB), 40/41 layers cached.
- **CPU fallback, cold page cache** (`runio7.py cpufb 1024 1990 fb`): decode window 138.76 → 151.61 s
  reads **170.02 MiB (`read_bytes`) / 172.42 MiB (disk), `majflt` 0** — the static pinned baseline
  `pin29` reads **173.30 / 174.11 MiB**, so the fallback **adds no disk reads**. Residency moves only
  in the **GiB block 10** (`906 → 1024`, +118 MiB): the tail of the file, not the expert ranges. Reads
  are concentrated in the first 2 s of decode (27 MiB per sample up to t = 140.66) and are **0.00 MiB**
  from t = 143.24 to the end. Cold t/s = 13.62 (page-fault cost at decode start).
- `strace` inside the cpufb decode window: **0.15 MiB of `read()`** (prof2.csv) — the 170 MiB are mmap
  page faults, not syscalls.
- **Pinning matters in fallback mode**: the same cold run **without** `--moe-cache-pin-mib`
  (`runio7.py cpufb0 1024 0 fb`) reads **290.99 MiB (`read_bytes`) / 291.66 MiB (disk)** inside the decode
  window (97.36 → 110.42 s) against **170.02 / 172.42 MiB** with `pin 1990` → **−119 MiB (−41 %)**.
  Cold t/s 13.40 vs 13.62: pinning changes throughput not at all, it cuts reads by protecting 1.945 GiB
  of the complement from eviction.

**Second prompt (prompt B), `Q: Explain how a diesel engine works. A:`, `-c 2048 -n 256 --temp 0 -s 42`**
(`/tmp/runB.sh`, `/tmp/runAllCPU.sh`, comparisons via `tools/cmp.py`, which strips the `build      :`,
`t/s`, `Loading model`, spinner, `>` and blank lines **before** diffing — without that filter every run
"differs" in 4 lines and the common prefix stops at ~75 chars):

| pair | content diff lines | common prefix (chars) | filtered lengths |
|---|---|---|---|
| `cleanB` vs `baseB` | **84** | 699 | 9027 vs 1758 |
| `cpufbB` vs `baseB` | **75** | 699 | 8136 vs 1758 |
| `allcpuB` vs `baseB` | **81** | 699 | 7923 vs 1758 |
| `cleanB` vs `cpufbB` | **121** | **1514** (16.8 % of 9027) | 9027 vs 8136 |
| `cleanB` vs `allcpuB` | **137** | 795 (8.8 % of 9027) | 9027 vs 7923 |
| `cpufbB` vs `allcpuB` | **130** | 795 | 8136 vs 7923 |

**Length asymmetry**: `baseB` runs with `-n 256` (1758 filtered chars), the three cache runs run **without
`-n`** → they generate to EOS (9027, 8136, 7923 chars). Diffs against `baseB` are dominated by length, so
only the common prefix 699 is comparable. The figures 21, 19, 33, 8, 28, 28 recorded in the previous
session were **raw-text diffs** polluted by the `build      :` line, the spinner and the `t/s` line (4 noise
lines per run) — **they were wrong**; the values above are measured by `tools/cmp.py` on filtered text.

t/s: `baseB` **23.76**, `cleanB` **21.32**, `cpufbB` **18.64**, `allcpuB` **15.1**.
`cpufbB`: hits 31936, misses 52184, hit **37.96 %**, **uploaded 0.00 MiB**, 10515 tokens served,
**136 304 expert computations left to the CPU**, `1115/10496` resident (1016.88 MiB). `cleanB`: hit
**57.59 %**, **uploaded 31 654.70 MiB**, scratch 34617, `1118/10496` resident.

**These stats come from the `-n 256` runs.** The `verify.sh` masters are the runs **without `-n`** (generate
to EOS, ~1900 tokens): `cleanB` hits **321253**, misses 286771, hit **52.84 %**, uploaded **261800.45 MiB**;
`cpufbB` hits **153768**, misses 383792, hit **28.60 %**, uploaded **0.00 MiB**; `allcpuB` hits **0**,
misses **494024**, hit **0.00 %**, uploaded **0.00 MiB**. The masters are stored in
**`/data/o35/o35/masters/`** because `runB` overwrites `/tmp/$name.txt` on every run — comparing a run
against its own `/tmp` file was 0 by construction, i.e. a check with no content.

- **`baseB` is not the numeric baseline of the cache runs**: it runs **without `-cmoe` and without
  `-ngl`**, so its dense layers (attention, layernorm, dense projections) compute on the **CPU**, while
  the cache runs compute them on **CUDA**. The clean comparison is `cleanB` vs `cpufbB` — same dense
  path on CUDA, same expert ids, only the expert compute destination differs → **121 diff lines, common
  prefix 1514 of 9027** (16.8 % of the generation identical).
- **All-CPU proof of exactness**: `--moe-cache-mib 32 --moe-cache-static --moe-cache-cpu-fallback` gives
  `q = n_slots / n_layers = 0`, so **no expert is resident** and all 8 experts of every token compute on
  the CPU. On prompt A at `-n 128`, `allcpuA` content diff vs `base128` = **0** (1121 identical chars).
  **The CPU path reproduces the no-cache output bit-for-bit.**
- **Divergence point on prompt B**: at char 699, `baseB` writes `**Understand User Input:**` and **all
  three cache modes write** `**Analyze User Input:**` — the cache modes agree with each other there. A
  near-tie in greedy sampling picks the other token and the divergence amplifies from that point.
- **Probe built and run (this session)** — patch saved as `tools/moe_probe.patch` (224 lines, 3 files),
  enabled with `LLAMA_MOE_PROBE=1`, script `/tmp/runProbe4.sh`, output `/tmp/probeB4.log`. Per layer it
  adds a `ggml_cpy` of the GPU bank-slot result to a CPU tensor (`moe_probe_gpu_<il>`) and a CPU
  `ggml_mul_mat_id` on the original host weights with the same `ids` (`moe_probe_cpu_<il>`), then
  compares them row by row for **resident** experts (`ids_cpu == -1`) over the real `ubatch.n_tokens`.
  **Measured on prompt B, `-n 8`, cpufb mode — 300 layer-steps, 1 968 128 elements compared**:
  **global max abs 0,020653** (layer 0), **weighted mean abs 0,003236**, mean per-layer max abs 0,003144,
  **global max rel 7,455** (layer 0, on a near-zero value), **mean per-layer max rel 1,41**,
  max gpu 3,630216 vs max cpu 3,628572. **Not FP accumulation order** (that would be ~1e-6): the gap is
  **0,3 % of the typical magnitude** — a **kernel difference** (CPU `MMQ` dequantizing to fp32 vs CUDA
  `MMVQ` on the same `iq2_xxs`/`iq3_xxs`/`iq4_xs` bytes). **No bug in the CPU path**: it is exact against
  the CPU baseline (`allcpuA` = `base128`, diff 0), and CPU-vs-GPU output divergence is llama.cpp's
  standard `-ngl 0` vs `-ngl 99` behaviour. **Acceptable**: bounded, documented, and the mixed `cpufb`
  path stays closest to the all-GPU path (common prefix 1514 of 9027 chars).

## 6. Open work

1. **CPU/GPU dispatch (requirement 6) is DONE** (`47a1dbbfc`): per-layer heat is uniform, so layer
   granularity buys nothing; the shipped design is the **zero slot** — non-resident experts read an
   all-zero bank slot on the GPU and are computed on the CPU from host weights, merged with `ggml_add`.
   The `-1` guard is CPU-only (`ggml-cpu.c:1655`), so `mm_ids_helper` (`ggml/src/ggml-cuda/mmid.cu:33-108`)
   is untouched. Conceptual reference to `expert_source.cpp:3522-3760` (`kind[]` dispatch), rejected for
   `sm_61`. **Batch-size tuning is a measured no-op**: greedy decode processes **1 token per ubatch** (the
   `n_tokens = 1, 2, 4, 8, …` in the log are `graph_reserve` reservations, not real batches), so the
   `max_batch_cached = 4` limit never applies. **The lever is activating MTP** (`model has unused tensor
   blk.40.nextn.eh_proj.weight ... ignoring`), which is what would create multi-token decode batches.
2. **Prompt-B divergence is measured, not a bug** (see section 5): probe numbers recorded. **DONE this
   session**: probe removed (`git checkout -- src/llama-graph.h src/llama-graph.cpp src/llama-context.cpp`),
   clean rebuild at `47a1dbbfc`, `verify.sh` → **7 diffs = 0**; the patch is kept as `tools/moe_probe.patch`
   (`git apply` to re-run it). Probe gotchas: probe tensors must be **consumed**
   (`res = ggml_add(res, ggml_scale(ggml_add(a, b), 0.0f))`) to enter the graph; `ggml_set_input` is
   **illegal on op nodes** (`GGML_ASSERT(op == GGML_OP_NONE)`, `src/llama-context.cpp:654`) → use
   `ggml_set_output` to keep the buffers alive; clamp the loop to the real `ubatch.n_tokens` because
   `ne[2]` is the **reserved** token count.
3. **Prompt B is now in `tools/verify.sh` as a regression** (this session): `runB` / `checkB` compare
   through `tools/cmp.py`. Masters live in **`/data/o35/o35/masters/`** (`cleanB.txt`, `cpufbB.txt`,
   `allcpuB.txt`, `baseB.txt`) — **not** in `/tmp`, because `runB` overwrites `/tmp/$name.txt` on every
   run, so comparing a run against its own `/tmp` file was **0 by construction** and the check was
   meaningless. Expected content diffs: each mode **0** vs its own master, `cpufbB` vs `cleanB` = **121**
   (prefix 1514), `cpufbB` vs `allcpuB` = **130** (prefix 795), `allcpuB` vs `cleanB` = **137** (prefix
   795). The expectations **8 and 28** were **wrong** (raw-text measurement) and produced two
   `REGRESSION FAIL` lines; the full run is green with the corrected constants. Future changes are
   checked against the prompt-B reference, not `baseB`.
4. **Convergence cost on a cold cache**: filling the gaps between the expert ranges cut the reads to
   112.69 MiB. The residual is eviction under memory pressure, and pinning is capped at 1.945 GiB by
   `RLIMIT_MEMLOCK`, so ~6.6 GiB of the complement stays evictable. Reducing the resident working set
   (smaller cache budget, fewer cached layers) is the remaining lever.
5. **Prefill is not cached** (`max_batch_cached = 1`), so prefill pays the disk. Requirement 5 only
   covers decode.
6. Optional GPU/CPU overlap via the two-phase hit hook (`hit_hook.hpp:28,39`).
