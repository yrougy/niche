# Niche — MoE expert cache for llama.cpp on small machines

Niche is the tooling and measurement record for running **large sparse MoE models on
consumer hardware** (dev machine: 15 GiB RAM, GTX 1070 8 GiB) using a fork of
[llama.cpp](https://github.com/ggml-org/llama.cpp) with a **MoE expert cache**:
hot experts are pinned to GPU + locked RAM, cold experts stay in pageable RAM and are
paged on demand, instead of requiring the whole model to be resident.

Fork: **[yrougy/llama-moe-cache](https://github.com/yrougy/llama-moe-cache)**,
branch **`moe-expert-cache`** (9 commits on top of upstream master).
This repo holds the installer TUI, calibration and benchmark scripts, the server
configurations, and the full experiment journal — everything except the C++ port itself.

## Quick start

```sh
git clone https://github.com/yrougy/niche.git
niche/tools/niche.sh            # interactive TUI (offline: Textual wheels are vendored)
niche/tools/niche.sh --plain    # same steps as plain text, for ssh/pipes
```

The installer walks through, and reuses anything already present:

1. **Machine checks** — git checkout of the fork, build, RAM, GPU, installed GGUFs.
2. **Build** — clones the fork automatically (branch `moe-expert-cache`) if
   `$LLAMA` (default `/data/o35/llama.cpp`) is absent, then cmake + build.
3. **Models** — picks the GGUF quant that fits the machine: it queries the Hugging Face
   API for real file sizes, computes a budget of `MemTotal − 3 GiB` (the constraint is
   the hot set in RAM, not VRAM), and prefers a quant whose quality step was already
   measured. Downloads resume from `.part`.
4. **Calibration** — `tools/calibrate.sh` measures the cache pin size and bank size for
   the machine and writes `tools/machine-<model>.env`.
5. **Serve** — starts `llama-server` with the measured parameters; you pick the context
   size from annotated choices.

## Models and reference results (15 GiB RAM / GTX 1070)

| Model | Quant | Generation |
|---|---|---|
| [unsloth/LFM2.5-24B-A2B-GGUF](https://huggingface.co/unsloth/LFM2.5-24B-A2B-GGUF) | UD-IQ3_XXS (8.3 GiB) | **47.2 tok/s** (reference config) |
| [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | UD-IQ2_M | 41.5 tok/s |
| same | UD-Q2_K_XL | 40.3 tok/s |
| same | UD-Q3_K_M | 6.7 tok/s — cliff: hot set no longer fits the RAM budget |

The cliff at Q3_K_M is why quant selection is machine-adapted: one GiB too much and
throughput collapses by 6×.

## llama.cpp flags used

From the fork (see its branch for implementation details):

- `--moe-cache-mib N` — expert cache bank size in MiB
- `--moe-cache-pin-mib N` — amount locked (mlock) in RAM
- `--moe-cache-static` — fixed bank instead of adaptive tier
- `--moe-profile FILE` — per-layer expert routing profile measured at runtime

## Layout

```
tools/niche.sh          entry point (TUI bootstrap, offline wheels in tools/wheels)
tools/niche_setup.py    installer engine (checks, models, quant choice, calibration)
tools/niche_tui.py      Textual UI   ·   tools/niche_bridge.py  UI bridge (+ --plain)
tools/setup.sh          headless variant of the same install (no TUI)
tools/calibrate.sh      pin/bank measurement → machine-<model>.env
tools/serve.sh          Qwen server (env overrides: MODEL CTX BANK_OVERRIDE PIN CORS…)
tools/serve-lfm.sh      LFM2.5 server
tools/try.sh bench.sh   one-shot latency / benchmark harness
tools/machine-lfm.env   measured parameters of the dev machine
WORKLOG.md              full experiment journal (French) — measurements, dead ends, fixes
```

## Requirements

Linux, `cmake` ≥ 3.14, `curl`, Python ≥ 3.10 (venv bootstrapped automatically),
CUDA driver optional, ~10–15 GiB free disk for one model.

## License

MIT — see [LICENSE](LICENSE). The `tools/wheels/` directory vendors
Textual/Rich and dependencies (MIT) for offline installation.
