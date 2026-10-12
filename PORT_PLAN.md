# PORT_PLAN — Strata per-expert MoE cache → llama.cpp `qwen35moe`

Branch: `experimental/ornith-expert-cache` (in `llama.cpp`).
Target: `Ornith-1.5-35B-A3B` GGUF (Q2/Q3) on a GTX 1070 (sm_61, 8192 MiB) + i7-4790 (Haswell, AVX2/FMA, no AVX-512) + 15 GiB RAM + HDD.
Context 4k–8k. **Zero disk access in the decode hot path after load.**

Only these Strata ideas are ported: per-expert cache granularity, (layer, expert) usage
statistics, hot set in VRAM, complementary set pinned in RAM, no mmap/HDD dependency in
decode, GPU experts on CUDA + absent experts on CPU. Everything else in Strata (prefill
mma kernels, PDL, VMM, peer GPUs, remote experts, adaptive tier) is out of scope.

---

## 1. Measured model geometry (verified from a real GGUF, not assumed)

`Qwen3.6-35B-A3B-UD-IQ2_M.gguf` is the same architecture as Ornith-1.5-35B-A3B
(`general.architecture = qwen35moe`), so the numbers below are ground truth, not guesses.

| key | value |
|---|---|
| `general.architecture` | `qwen35moe` |
| `qwen35moe.expert_count` | 256 |
| `qwen35moe.expert_used_count` | 8 |
| `qwen35moe.expert_feed_forward_length` | 512 |
| `qwen35moe.expert_shared_feed_forward_length` | 512 |
| `qwen35moe.full_attention_interval` | 4 (10 full-attention layers, 30 gated-delta-net layers) |
| `qwen35moe.nextn_predict_layers` | 1 (`blk.40` = MTP) |
| blocks | 41 (`blk.0..blk.39` trunk + `blk.40` MTP) |
| `n_embd` | 2048 |
| file size | 11.88 GB |
| expert bytes | 10.06 GB (95.6 % of the model) |
| shared-expert bytes | 0.08 GB |
| non-expert block bytes | 0.92 GB |
| per block, all experts | 245.3 MB |
| **per expert (gate+up+down)** | **0.958 MB (0.913 MiB)** |

Per-expert byte sizes for `blk.0` (IQ2_XXS gate/up, IQ3_XXS down):
gate 270 336 B, up 270 336 B, down 401 408 B → 942 080 B = 0.898 MiB.
For `blk.40` (Q2_K gate/up, Q3_K down): 344 064 + 344 064 + 450 560 = 1 138 688 B = 1.086 MiB.

Budget arithmetic on this box:

- VRAM 8192 MiB − ~1.0 GB non-expert weights − KV/compute buffers ⇒ **1–2 GiB usable for
  the expert cache**.
- 1 GiB ≈ 1121 experts = **10.7 %** of the 10 496 experts.
- 2 GiB ≈ 2242 experts = **21.4 %** = ~54 slots per layer.
- Strata measured (`Strata/include/strata/core/expert_cache.hpp:162-164`) that 8 slots per
  layer gives 21.4 % hit and 64 slots per layer gives 70.4 % hit. So a 2 GiB cache on this
  box should land in the 60–70 % expert-hit range.
- RAM: 11.88 GB mmap + 2 GiB VRAM copy = 13.9 GiB, which fits in ~14 GiB available. With
  page release (see §6) the duplication disappears.

## 2. Mandatory item A — exact expert tensor shapes and `expert_id` → memory slice

GGUF/llama.cpp names (`src/llama-arch.cpp:471-474`), created in
`src/models/qwen35moe.cpp:95-97` (trunk) and `:122-124` (MTP). `ggml` `ne[]` is
fastest-varying-first.

| tensor | `ne[]` | memory order | per-expert stride |
|---|---|---|---|
| `blk.N.ffn_gate_exps` | `{2048, 512, 256}` | `[256][512][2048]` | `nb[2]` |
| `blk.N.ffn_up_exps`   | `{2048, 512, 256}` | `[256][512][2048]` | `nb[2]` |
| `blk.N.ffn_down_exps` | `{512, 2048, 256}` | `[256][2048][512]` | `nb[2]` |
| `blk.N.ffn_gate_up_exps` (merged, optional, `TENSOR_NOT_REQUIRED`, `src/llama-model.cpp:3463-3478`) | `{2048, 1024, 256}` | gate at offset 0, up at `512*nb[0]` | `nb[2]` |
| `blk.N.ffn_gate_inp` | `{2048, 256}` | router, F32/BF16 | — |
| `blk.N.ffn_gate_inp_shexp` | `{2048}` | shared-expert gate | — |
| `blk.N.ffn_{gate,up}_shexp` | `{2048, 512}` | shared expert | — |
| `blk.N.ffn_down_shexp` | `{512, 2048}` | shared expert | — |

**The rule:** expert `e` of tensor `w` is the contiguous block

```
(uint8_t *) w->data + e * w->nb[2]        // size = w->nb[2] bytes
```

This is exactly what llama.cpp already does:
- upload: `src/llama-moe-cache.cpp:568` → `(const uint8_t *) w->data + fills[i].expert*expert_size`, `expert_size = w->nb[2]` (:562)
- large-batch copy: `src/llama-moe-cache.cpp:503` → `slots[e]*bank->nb[2]`
- bank creation: `src/llama-moe-cache.cpp:364-365` → `ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], n_slots+1)` with `GGML_ASSERT(bank->nb[2] == t->nb[2])`

It matches Strata's `ExpertLayout::blob_offset(layer, expert) = layer_offset(layer) +
expert * blob_bytes(layer)` (`Strata/src/kernels/cpu/expert_layout.hpp:37-40`).

Consequences:
- experts of a layer are **contiguous** in the GGUF file, so a resident set can be released
  page-range by page-range (§6) and a miss can be uploaded with one `memcpy`-style copy;
- the three tensors of a layer have **different** `nb[2]` when the quantization differs
  (gate IQ2_XXS vs down IQ3_XXS), so banks must be per-tensor (llama.cpp already does this)
  and a per-expert byte count is `sum over the layer's expert tensors of nb[2]`;
- a slot is a triple of offsets (one per tensor), not one address.

## 3. Mandatory item B — where the final top-8 (ids + weights + layer) is available

`llm_graph_context::build_moe_ffn` — `src/llama-graph.cpp:2077-2446`. `il` is a function
argument, `n_tokens = cur->ne[1]`. qwen35moe uses `LLM_FFN_SILU`, `norm_w = true`,
`LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX`, no `exp_probs_b`
(`src/models/qwen35moe.cpp:494-512`).

```
router      logits = build_lora_mm(gate_inp, cur)                        :2109  [256, n_tokens]
            probs  = ggml_soft_max(logits)                               :2127  [256, n_tokens]
            selection_probs = probs                                      :2148
top-k       selected_experts = ggml_argsort_top_k(selection_probs, 8)   :2193  [8, n_tokens]  <-- IDS
            cb(selected_experts, "ffn_moe_topk", il)                    :2196
weights     weights = ggml_get_rows(probs, selected_experts)            :2207  [1, 8, n_tokens]
            normalize by row sum (norm_w)                               :2218-2232
            weights = ggml_scale(weights, w_scale)                      :2233-2236
            ggml_build_forward_expand(gf, weights)                      :2239  <-- TOP-8 PUBLISHED HERE
gather      slots = build_moe_cache_slots(selected_experts, up, gate, down, gate_up, il)  :2242
            up    = build_lora_mm_id(up_exps,   cur, selected_experts, ..., slots)        :2277
            gate  = build_lora_mm_id(gate_exps, cur, selected_experts, ..., slots)        :2290
            swiglu/silu                                                                  :2308-2389
            down  = build_lora_mm_id(down_exps, cur, selected_experts, ..., slots)        :2391
reduce      experts = ggml_mul(experts, weights)                        :2408
            cur_experts[i] = view_2d(experts, n_embd, n_tokens, nb[2], i*nb[1])           :2424
            moe_out = Σ cur_experts[0..7]                              :2430-2436
```

The top-8 is fully local: `il` + `selected_experts` (ids, `int32`, `[8, n_tokens]`) +
`weights` (`[1, 8, n_tokens]`). `selected_experts` is contiguous
(`GGML_ASSERT(ggml_is_contiguous(sel))`, `src/llama-moe-cache.cpp:546`). **No architectural
rewrite is needed to observe routing.**

The existing cache hook is `build_moe_cache_slots` (`src/llama-graph.cpp:2448-2485`): it
makes `ids` 1-D contiguous, does `slots = ggml_get_rows(slot_map, ids)`, forces the slots
tensor onto the cache device, and marks `cb(slots, "ffn_moe_slots", il)`. Because
`slot_map` is a host weight, the scheduler **starts a new split** at that `get_rows` and
invokes the copy callback (`src/llama-context.cpp:2645`), which reads the ids from the
device (`src/llama-moe-cache.cpp:549-550`), plans, uploads misses, and updates the slot map.

**This split + copy-callback pattern is the template for the instrumentation hook.**

## 4. What llama.cpp already has (the base to extend, not to rewrite)

`llama_moe_cache` (`src/llama-moe-cache.h`, `src/llama-moe-cache.cpp`), CLI
`--moe-cache-mib N` (`common/arg.cpp:2780-2788`), `moe_cache_size`
(`common/common.h:596`, `src/llama-cparams.h:58`), construction
(`src/llama-context.cpp:468-469`).

Already implemented, and correct for this port:
- per-expert granularity: `slot_map[il][e] → slot`, `key_of[slot] = il*n_expert + e`
  (`llama-moe-cache.cpp:20-21`);
- one device bank per expert tensor, `n_slots+1` slots, `nb[2]` preserved (:364-365);
- host slot maps on the CPU buffer, cleared to `0xff` = all `-1` (:418);
- copy callback that reads ids from the device, uploads misses with merged consecutive
  runs, updates the slot map (:515-583);
- `get_slot_map` returns `nullptr` for `n_tokens > 32` (:469) → cache is decode-only,
  prefill uses `copy_experts` (:480-513, `src/llama-context.cpp:2698-2709`);
- hit/miss/bytes stats per ubatch size (:190-202, `log_stats` :585-599);
- `memory_breakdown()` (:628-639);
- layers are cached only when **all** their expert tensors are host weights
  (`llama_moe_cache_is_host_weight`, :145-149, :242-246) → `-cmoe` / `-ncmoe` is required.

Missing for the requirements:
1. **No (layer, expert) statistics.** Only aggregate hits/misses. Requirement 2.
2. **LRU, dynamic.** `plan()` evicts `head` on every miss (:109-117). There is no static,
   profile-seeded, never-evicting mode. Requirement 3 needs determinism.
3. **No guarantee of zero disk access.** Non-resident experts are read from the mmap on
   every miss → HDD page faults. Requirement 5.
4. **No per-expert CPU/GPU compute split inside a layer.** A cached layer computes all 8
   experts on GPU; a non-cached layer computes all 8 on CPU. Requirement 6 is met only at
   layer granularity today.
5. **Duplication**: host weights of cached layers stay in RAM *and* in VRAM.

## 5. Strata mechanisms — reuse / adapt / reject

| Strata source | Mechanism | Verdict for this port |
|---|---|---|
| `include/strata/core/expert_cache.hpp:38` `kNotResident = -1` | sentinel | **reuse** (llama.cpp already uses `-1`) |
| `src/core/expert_cache.cpp:633-636` `slot_of(layer, expert)` | bounds-checked lookup | **reuse** |
| `src/core/expert_cache.cpp:638-656` `admit()` — **never evicts**, `if (next_free_ >= slots_) return kNotResident` | static admission | **adapt** → `plan_static()` in `llama-moe-cache.cpp` |
| `expert_cache.hpp:42-43` + `cpp:72-139` `read_expert_profile` / `cpp:170-210` `write_expert_profile` — `STRP` magic, `version, n_layers, n_expert, slots, n_ranked`, ranked `(uint16 layer, uint16 expert)` pairs, then `n_layers × n_expert` int32 rank table | profile file format | **adapt** (adopt the format, add geometry gate checks) |
| `cpp:141-168` `rank_learned_profile` — sort resident, then descending heat, then prior rank, then index | ranking | **reuse** |
| `cpp:567-568` round each slot to 256-byte alignment "so every blob starts where the kernels' vector loads expect it" | alignment | **reuse** (CUDA MMQ reads past the last expert, llama.cpp adds one extra slot at :270-278) |
| `cpp:489-503` free-VRAM check before allocation; `cpp:527-533` `cudaMemset(base_, 0, want)` so stale slots are deterministic | safety | **reuse** |
| `cpp:40-66` UMA detection (`cudaDevAttrIntegrated`) + `/proc/meminfo` MemAvailable with `STRATA_UMA_HEADROOM_GIB` default 6 GiB | RAM headroom | **reuse** — critical on a 15 GiB box |
| `cpp:624-631` `layer_slot_range` + `hpp:170` `set_per_layer_admission` — per-layer quota `q = slots_/n_layers_` | quota | **adapt** — llama.cpp splits the budget by group size (:310-329), add a per-layer quota so one layer cannot starve another |
| `src/core/expert_source.cpp:3562-3564` `d.usage[layers*n_expert + ids[i]] += 1.0f` + `adapt_decay` (default 0.7, `generate.cpp:578, 8277`) | live (layer, expert) counters | **adapt** → counters in `llama_moe_cache` |
| `src/core/expert_source.cpp:3522-3760` `expert_pool_dispatch_multi` — per-entry `kind[]`: `-1` CPU, `0` VRAM, `1` PCIe, `2` peer, `-2` tail-skipped; GPU share published **first** so the GPU starts while the CPU works (:3566) | per-expert dispatch | **adapt conceptually**; the kernel itself is **rejected** (sm_61) |
| `expert_source.cpp:3596-3615` `tail_rank` (`STRATA_ROUTE_TAIL_SKIP`, default 7): an expert routed only at rank ≥ R by every token of the window is neither copied nor computed | tail skip | **adapt** — cheap, high value, pure host logic |
| `expert_source.cpp:2526-2586` `pin_cache_complement` + `expert_source.hpp:104-108` `make_cache_complement_plan` + `expert_source.hpp:183` `release()` | pin the complement in RAM, drop pages of VRAM-resident experts | **adapt** — this is the mechanism that satisfies requirement 5, see §6 |
| `include/strata/core/pinned.hpp` `PinnedArena` (`LargePages`, `PinnedByCuda`, `locked_bytes`, `registered_bytes`, `STRATA_ARENA_PIN_GIB`) — 4 KB pages = 8.3 M TLB entries → TLB misses on every matvec | pinned arena | **adapt** (mlock ranges; huge pages unavailable on this box) |
| `include/strata/kernels/dp4a.hpp` | `__dp4a` | **available on sm_61** (compute cap ≥ 6.1) |
| `include/strata/kernels/pdl.hpp:40-41` — "On Pascal (`__CUDA_ARCH__ < 700`, no PDL) `__restrict__` is what lets the compiler use the read-only data path (LDG.CI): without it decode on sm_61 halves" | `__restrict__` | **reuse as a rule** for any new sm_61 kernel |
| `src/core/vmm.cpp` + `cpp:305-306` `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` | elastic VRAM | **reject** — VMM is Turing+, not on Pascal |
| `src/prefill/moe_fused.cu` (`mma.sync m16n8k32`, `cp.async`, `STAGES=4`), `src/kernels/cuda/qsa_prompt_attn.cu:31-34` | prefill MoE | **reject** — sm_75/sm_80 only |
| `src/kernels/cuda/s2_expert_grouped.cu`, `native_moe.cu`, `verify_kernels.cu:839` (clusters sm_90+) | grouped/cluster kernels | **reject** |
| `src/core/peer_experts.cpp`, `remote_experts.cpp`, `remote_expert_opt.cu` | multi-GPU | **reject** (one GPU) |
| `generate.cpp:8153-8279` `adapt` lambda, `PeerExperts::adapt` (`peer_experts.cpp:276-319`), `ForesightSwap` (`foresight_swap.hpp:37-54`) | adaptive swap tier | **defer** (post-MVP) |
| `expert_cache.hpp:11-17, 45-50` — `h_expert = 0.6447` leave-one-out, `h_layer = 0.0456`, compulsory miss `h = 0.4864`; "only 4.6 % of (layer, token) pairs have all ten experts resident — so this cannot be a per-layer grouped kernel" | hit-rate evidence | **reuse as design guidance**: per-expert granularity is mandatory, per-layer residency is hopeless |
| `expert_source.hpp:18-21` `H=2560, FF=640, BLOB=1 382 400` with compile-time blob offsets, other geometry **refused** | layout | **reject** — Strata's geometry is Qwen3.5-Next 48×512, not qwen35moe 2048×512 |

## 6. Requirement 5 — zero disk access in decode

Two complementary mechanisms, both native to llama.cpp:

**(a) `--mlock` (exists today).** `llama_mlock` (`src/llama-mmap.h:68-80`,
`src/llama-mmap.cpp:694-834`, `SUPPORTED == true` on Linux) locks the whole mmap. With
11.88 GB model + 2 GiB cache this fits in ~14 GiB. Simple, guaranteed, no code needed.

**(b) Complement pinning + release (the Strata port).** Because the cache is static and
never evicts, the host pages of VRAM-resident experts are never read again after the
initial upload. So:

1. Seed the cache from the profile → the resident set is known at load time.
2. Pin the pages of the **complement** (non-resident experts), hottest first, within a RAM
   budget — Strata's `pin_cache_complement` (`expert_source.cpp:2526-2586`).
   Implementation: add `llama_mlock::pin(void * ptr, size_t size)` (a thin `mlock` wrapper)
   in `src/llama-mmap.{h,cpp}`, and drive it from the cache using
   `llama_mmap::addr() + tensor_offset + e*nb[2]`.
3. `llama_mmap::unmap_fragment(first, last)` (`src/llama-mmap.h:59`) the page ranges of the
   resident experts → RAM for those experts is freed, so the VRAM copy is not duplicated.
   This is Strata's `release()` (`expert_source.hpp:183`).

Ordering matters: pin the complement **before** releasing, and only release ranges that are
seeded resident. A resident expert that is never uploaded (never touched) can be released
immediately; one that is uploaded on the first miss must be released after that upload.
The MVP releases only ranges that were uploaded during seeding.

## 7. sm_61 (GTX 1070) hardware audit

| feature | status on sm_61 | consequence |
|---|---|---|
| `__dp4a` | **available** (≥ 6.1) | int8/int4 dot products usable |
| PDL (`cudaTriggerProgrammaticLaunchCompletion`) | sm_90+ only | unavailable; use `__restrict__` — `pdl.hpp:40-41` says decode on sm_61 **halves** without it |
| `mma.sync m16n8k32` | sm_75+ | no tensor-core MoE kernel |
| `cp.async` | sm_80+ | no async copy pipeline |
| CUDA VMM (`CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`) | Turing+ | no elastic VRAM |
| bf16 | not supported | keep F32/F16 paths |
| `__nanosleep` | sm_70+ | spin loops must busy-spin on sm_6x |
| llama.cpp MMVQ fusion | **disabled on Pascal** (`ggml/src/ggml-cuda/ggml-cuda.cu:1839`) | **MMQ path is what runs** — `mmvq.cu:288` has explicit Pascal branches |
| Strata CMake | refuses `CMAKE_CUDA_ARCHITECTURES < 75` unless `-DSTRATA_EXPERIMENTAL_SM60=ON` (`CMakeLists.txt:146-193`) | upstream Strata does not support Pascal; only its host-side logic ports |

CPU (i7-4790, Haswell): AVX2 + FMA yes; **no AVX-512, no AVX-512BW, no AVX-VNNI, no fast
gather**. Strata's `q2_avx2.cpp` / `iq_avx2.cpp` are the AVX2 floor and will run; llama.cpp's
CPU kernels for Q2_K/IQ2_XXS/IQ3_XXS are AVX2.

## 8. Implementation steps

### Step 1 — audit + this document
Done. Files: `PORT_PLAN.md` (this file), `tools/gguf_meta.py` (numpy-free GGUF dumper used
to measure the geometry above).

### Step 2 — routing instrumentation
Goal: a CSV of `(layer, expert_id, count, weight_sum)` over a real run, with **no change to
numerics**.

Hook: reuse the split + copy-callback pattern of `src/llama-moe-cache.cpp:515-583`.

- `src/llama-graph.cpp` — in `build_moe_ffn`, right after
  `ggml_build_forward_expand(gf, weights)` (:2239), add a `ggml_cpy(weights, prof_host)`
  node where `prof_host` is a per-layer host tensor of shape
  `[1, n_expert_used, n_tokens]` on the CPU buffer. The scheduler starts a split at the
  `cpy` because `prof_host` is a host weight.
- `src/llama-context.cpp:2641` `sched_copy_experts` — add
  `lctx->moe_profile && lctx->moe_profile->copy(...)` **before** the moe_cache branch, so
  profiling works even with no cache. In the callback, read `ids` from
  `lookup->src[1]` (`selected_experts`) with `ggml_backend_tensor_get_async` +
  `ggml_backend_synchronize` (the exact pattern at `llama-moe-cache.cpp:549-550`), read the
  weights from `src->data`, and accumulate
  `count[il*n_expert + e] += 1`, `wsum[il*n_expert + e] += w`.
- New files: `src/llama-moe-profile.{h,cpp}` — counters, CSV writer on destruction
  (atomic `.tmp` + rename, as Strata does at `expert_cache.cpp:170-210`).
- New args: `--moe-profile-output <file>` in `common/arg.cpp` next to `--moe-cache-mib`
  (:2780), `moe_profile_output` in `common/common.h:596` region, `moe_profile_output` in
  `src/llama-cparams.h:58` region, wired in `src/llama-context.cpp:278` + `:468`.

Verification: run with and without the flag, compare logits/token output → identical.

### Step 3 — per-expert abstraction
Make the per-expert identity explicit in `llama-moe-cache.cpp` so the rest of the work has
a clean handle:

- `expert_bytes(il)` = `Σ over layer il's expert tensors of nb[2]` (needed for budgeting
  and for the metrics);
- `resident(il, e)` = `slot_map[il][e] >= 0`;
- `n_resident_total`, `resident_per_layer`;
- expose `slot_of(il, e)` (Strata `expert_cache.cpp:633-636`).

No behaviour change; this is the abstraction the later steps build on.

### Step 4 — static profile-driven cache, no eviction
- `plan_static(il, ids, n_ids, fills, n_hit)`: same as `plan()` (:74-119) but **never
  evicts**. A miss on a non-resident expert is recorded as a "cold miss" and the expert is
  computed from host memory (see step 5), not admitted. Mirrors Strata's `admit()`
  (`expert_cache.cpp:638-656`).
- `seed(profile)`: walk the ranked `(layer, expert)` list, `admit` while
  `next_free < n_slots`, set `slot_map[il][e] = slot`, `key_of[slot] = il*n_expert + e`,
  build the LRU tail in rank order. Deterministic.
- New args: `--moe-profile <file>` (load), `--moe-cache-static` (no eviction).
- Per-layer quota (Strata `layer_slot_range`, `expert_cache.cpp:624-631`): cap slots per
  layer at `ceil(n_slots / n_cached_layers)` so hot layers cannot starve cold ones.
- Keep LRU as the default path — `--moe-cache-static` selects the static mode. No dead
  code: both paths are exercised.

### Step 5 — CPU fallback for non-resident experts
The hard part, and the honest blocker:

`ggml_mul_mat_id` has a **fixed** `n_expert_used` in the graph. All 8 slots of a token are
computed by the same op. Masking a slot's weight to zero still performs the full work.
There is **no way to skip experts per token** with the current op set.

MVP (A) — layer granularity, which is what llama.cpp does today and satisfies requirement 6
in the practical sense:
- cached layers: all 8 experts on GPU (misses uploaded first);
- non-cached layers: all 8 experts on CPU via `-cmoe` / `-ncmoe`
  (`common/arg.cpp:2763-2792`, `LLM_FFN_EXPS_REGEX` at `common/common.h:1237`);
- the static profile decides **which layers** get cache budget, so the hottest layers run
  on GPU and the coldest run on CPU.

MVP (B) — true per-expert split, requires a new ggml op:
- a `MUL_MAT_ID` variant taking a per-slot `kind[]` (Strata's `expert_pool_dispatch_multi`,
  `expert_source.cpp:3522-3760`) that computes GPU slots on the device and CPU slots on the
  host, then combines. Design: publish the GPU share **first** (`expert_source.cpp:3566`)
  so the GPU starts while the CPU works; skip tail-rank experts
  (`expert_source.cpp:3596-3615`).
- This is a `ggml` op + CUDA + CPU backend addition. It is scoped as a separate step, not
  part of the first working prototype.

### Step 6 — metrics
Extend `stats` (`llama-moe-cache.cpp:190-194`) and `log_stats` (:585-599) to report, per
run:

- total experts selected, experts found in cache, experts uploaded (existing);
- **GPU expert-work vs CPU expert-work** (count of expert matmuls per side);
- resident experts total, per layer, resident bytes, cache bytes;
- **disk accesses during decode = 0** — measured by counting page-fault deltas
  (`/proc/self/stat` `minflt`/`majflt`) across the decode loop, and by reporting the pinned
  complement bytes;
- tokens/s (already reported by `llama-cli` / `llama-bench`).

Export via `llama_context` so `llama-cli`/`llama-server` print it, plus a
`--moe-stats <file>` CSV.

### Step 7 — optional: GPU/CPU overlap
Two-phase hit hook (Strata `HitPhase { Launch, Combine }`,
`include/strata/core/hit_hook.hpp:28,39`): launch GPU work for resident slots, then
combine after CPU misses finish. Only after steps 1-6 are green.

## 9. Build (no ninja installed)

```
cd /data/o35/llama.cpp
/snap/bin/cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=61 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
/snap/bin/cmake --build build --config Release -j8
```

`ninja` is absent → cmake uses the default Makefile generator. Driver reports CUDA 13.0,
nvcc is 12.8; `GGML_CUDA_CC_PASCAL` is explicitly handled in `ggml-cuda.cu`.

## 10. Test plan

- `tests/test-llama-archs.cpp` already covers `-a qwen35moe` (:100) and the CPU-override
  regex (:880). No MoE-cache tests exist; add a small one that builds a cache, seeds a
  profile, and asserts `slot_of` results.
- Real run: `llama-cli -m <ornith-q2.gguf> -cmoe --moe-cache-mib 2048
  --moe-profile-output prof.csv -ngl 99 -c 8192` then
  `python3 tools/analyze_moe_profile.py prof.csv`.
- `tools/analyze_moe_profile.py`: for cache sizes 10/20/30/40/50 % of experts, report
  `% experts kept`, `% of selections covered`, `% of routing weight mass covered`, and the
  per-layer quota curve, so the 2 GiB budget can be justified with data.
- Numerics: greedy output must be byte-identical with and without `--moe-profile-output`.

## 11. Risks

- `max_batch = 32` (`llama-moe-cache.cpp:196`) → cache is decode-only; prefill goes through
  `copy_experts`. Prefill MoE kernels from Strata are unusable on sm_61, so prefill stays on
  llama.cpp's MMQ. Do not touch prefill until decode is green.
- `GGML_ABORT` at `llama-moe-cache.cpp:543` if the top-k and the slot lookup land in the
  same split. The instrumentation hook must be placed so its split is strictly after the
  top-k split.
- Only layers with **all** expert tensors in host memory are cached → `-cmoe` is mandatory.
- RAM duplication (11.88 GB + cache) — solved by §6(b), which is the reason step 4 must
  land before step 6's RAM claims are made.
- No Ornith-1.5-35B-A3B GGUF is present on this box. `Qwen3.6-35B-A3B-UD-IQ2_M.gguf`
  (same `qwen35moe` arch, 256 experts, top-8, n_embd 2048, n_ff_exp 512) is present and is
  the test model.
- Quantization: start with Q2 (`IQ2_XXS`/`IQ2_S`/`Q2_K` gate/up + `IQ3_XXS`/`Q3_K` down).
  Do not design for all variants in the first prototype.
