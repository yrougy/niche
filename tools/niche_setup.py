#!/usr/bin/env python3
"""niche_setup — le moteur sans écran du port cache-MoE niche : état de la machine,
choix du MODÈLE (un seul à la fois : Qwen3.6-35B-A3B ou LFM2.5-24B-A2B) puis d'une
QUANTIZATION ADAPTÉE À LA MACHINE (l'API HF
donne la liste et les tailles exactes ; le budget est le hot set RAM, pas la VRAM —
cliff mesuré 12/10 : au-delà, le tail chaud faulte sur disque, tg ~7), réutilisation
des poids déjà présents, reprise .part, build, calibrage, serveur. Le moteur ne parle
qu'à un objet ui (voir niche_bridge.py : Bridge pour l'écran, PlainUI en texte brut).

Variables d'environnement : MODEL_DIR, GGUF_DIRS, LLAMA.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent          # tools/
ROOT = HERE.parent                              # le dépôt niche


def find_nvcc():
    """nvcc sur le PATH, sinon toolkit le plus récent sous /usr/local ou /opt.
    Sans ça, cmake ne trouve pas nvcc hors PATH et configure en CPU-only sans rien
    dire — et le cache MoE exige un backend GPU (llama-moe-cache.cpp jette)."""
    nv = shutil.which("nvcc")
    if nv:
        return nv
    cands = sorted(str(p) for base in ("/usr/local", "/opt") for p in Path(base).glob("cuda-*/bin/nvcc"))
    return cands[-1] if cands else None

# catalogue : familles de quants connues. « mesure »/« evite » = retours mesurés sur la
# machine de dev (15 GiB RAM / 8 GiB VRAM, WORKLOG 12-14/10) ; ils guident la reco mais
# ne remplacent pas le budget calculé sur la machine de l'utilisateur.
MODELS = [
    dict(tag="lfm", nom="LFM2.5-24B-A2B", prefix="LFM2.5-24B-A2B",
         repo="unsloth/LFM2.5-24B-A2B-GGUF",
         note="24B MoE (2B active), flat 8k-64k, ctx max 131k — the fastest: tg 47.2 reference",
         fallback=("LFM2.5-24B-A2B-UD-IQ3_XXS.gguf", 7 * 2**30),
         serve="serve-lfm.sh", envfile="machine-lfm.env", mtp=False, ctx_defaut=65536,
         ctxs=[(8192, "measured plateau: bank 6656 from 8k to 64k"),
               (16384, "bank 6656 (plateau)"),
               (32768, "bank 6656 (plateau)"),
               (65536, "script default, measured plateau, bank 6656"),
               (131072, "GGUF native max: bank 6144, tg −3 % (measured 16/10)")],
         mesure={"UD-IQ3_XXS": "tg 47.2 measured 14/10 — the machine's reference config"},
         evite={"UD-Q4_K_XL": "> 13.8 GiB: hot set would crack on 15 GiB RAM",
                "UD-Q5_K_XL": "> 16.6 GiB: does not fit in RAM",
                "UD-Q6_K_XL": "> 20 GiB: does not fit in RAM",
                "UD-Q8_0": "> 23.7 GiB: does not fit in RAM",
                "UD-Q2_K_XL": "possible but below the IQ3_XXS quality step (measured)"}),
    dict(tag="qwen", nom="Qwen3.6-35B-A3B", prefix="Qwen3.6-35B-A3B",
         repo="unsloth/Qwen3.6-35B-A3B-GGUF",
         note="35B MoE (3B active), MTP head, usual ctx 4-32k",
         fallback=("Qwen3.6-35B-A3B-UD-IQ2_M.gguf", 10 * 2**30),
         serve="serve.sh", envfile="machine-qwen.env", mtp=True, ctx_defaut=4096,
         ctxs=[(4096, "script default, bank ~5632 measured on this machine"),
               (8192, "bank ~5120 (−512 MiB/8k)"),
               (16384, "bank ~4608"),
               (32768, "bank floored to 4096 by serve.sh — watch the hit rate")],
         mesure={"UD-IQ2_M": "tg 41.5 measured 12/10",
                 "UD-Q2_K_XL": "measured viable step: tg 40.3, better quality and MTP acceptance"},
         evite={"UD-Q3_K_M": "measured wall tg 6.7 on 15 GiB RAM",
                "UD-Q3_K_S": "measured wall tg 6.7 on 15 GiB RAM",
                "UD-Q3_K_XL": "> 15 GiB: same wall as Q3_K_M",
                "UD-IQ3_S": "> 12.7 GiB: near the measured wall",
                "UD-IQ3_XXS": "> 12.3 GiB: near the measured wall"}),
]
BRANCHE = "moe-expert-cache"
FORK = "https://github.com/yrougy/llama-moe-cache.git"


def model_dir() -> Path:
    return Path(os.environ.get("MODEL_DIR", str(ROOT / "models"))).resolve()


def llama_dir() -> Path:
    return Path(os.environ.get("LLAMA", str(ROOT / "llama.cpp"))).resolve()


def gguf_dirs() -> list:
    """où chercher un poids déjà téléchargé ailleurs que MODEL_DIR (GGUF_DIRS, ':' séparé)."""
    home = Path.home()
    d = os.environ.get("GGUF_DIRS", "")
    dirs = [model_dir()] + [Path(p) for p in d.split(":") if p]
    dirs += [Path("/data/models"), home / "Downloads", home / "models"]
    return [x for x in dict.fromkeys(dirs) if x.is_dir() and x != model_dir()]


def quant_of(fname: str, prefix: str) -> str:
    """le suffixe de quant d'un fichier : UD-Q2_K_XL ggé de « prefix-UD-Q2_K_XL.gguf »."""
    return fname[len(prefix) + 1:-len(".gguf")]


def deja_present(m) -> Path | None:
    """quant de la famille déjà en place (> 1 GiB, sans mmproj). S'il y en a plusieurs : MODEL_DIR
    d'abord, puis ceux mesurés bons sur la machine de dev, puis le plus gros."""
    matches = []
    for d in [model_dir()] + gguf_dirs():
        matches += [c for c in d.glob(m["prefix"] + "-*.gguf")
                    if "mmproj" not in c.name and c.stat().st_size > 2**30]
    if not matches:
        return None
    return max(matches, key=lambda c: (c.parent == model_dir(),
                                       quant_of(c.name, m["prefix"]) in m.get("mesure", {}),
                                       c.stat().st_size))


def hf_listing(ui, m) -> list | None:
    """[(fichier, octets)] des .gguf du dépôt (hors mmproj), du plus gros au plus petit ; None si l'API est injoignable."""
    url = f"https://huggingface.co/api/models/{m['repo']}/tree/main"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            ents = json.load(r)
    except Exception as e:
        ui.warn(f"{m['tag']}: HF API unreachable ({e.__class__.__name__}) — falling back to the default quant.")
        return None
    out = [(e["path"], int(e.get("size", 0))) for e in ents
           if e.get("path", "").endswith(".gguf") and "/" not in e["path"] and "mmproj" not in e["path"]]
    return sorted(out, key=lambda x: -x[1]) or None


def budget_octets(info) -> tuple:
    """taille maxi de GGUF pour que le hot set tienne en RAM — la contrainte mesurée (12/10),
    pas la VRAM (le bank d'experts se règle au calibrage). Réserve : OS + KV + marge."""
    ram_gib = info.get("ram") or 16
    b = int((ram_gib - 3) * 2**30)
    return b, f"RAM {ram_gib} GiB − 3 GiB (OS+KV+slack) ≈ {b / 2**30:.1f} GiB of hot set"


def _label(m, f, s, b, vram=0):
    q = quant_of(f, m["prefix"])
    lbl = f"{q} — {s / 2**30:.2f} GiB"
    if q in m.get("mesure", {}):
        lbl += f"  ({m['mesure'][q]})"
    if q in m.get("evite", {}):
        lbl += f"  ⚠ {m['evite'][q]}"
    elif s > b:
        lbl += "  ⚠ over the hot-set budget: decoding will collapse if the tail faults on disk"
    elif s >= 0.95 * b:
        lbl += "  (tight fit, cut it close)"
    if vram and s > vram:
        lbl += "  ⚠ bigger than VRAM (fine on this MoE port: weights stay in RAM, the bank holds the hot experts)"
    return lbl


def choix_quant(ui, m, info):
    """propose les quants qui tiennent dans le budget, PUIS les plus gros qui ne tiennent pas
    (signalés ⚠, confirmation obligatoire) ; rend (fichier, octets) ou None (passer)."""
    lst = hf_listing(ui, m) or [m["fallback"]]
    b, expl = budget_octets(info)
    vram = int(info["vram"]) * 2**20 if info.get("vram") else 0
    note_vram = ""
    if vram and info["vram"] < 4 * 1024:
        note_vram = f" ⚠ {info['vram']} MiB GPU: prefer a small quant (less PCIe traffic)."
    ui.say(f"{m['tag']} ({m['nom']}): budget — {expl}.{note_vram} "
           "A heavier KV quant (q8_0/q4_0, set in the server settings) also buys RAM back.")
    fits = [(f, s) for f, s in lst if 0 < s <= b]
    reco = None
    if fits:
        reco = fits[0]                                   # le plus gros qui tient
        for f, s in fits:                                # un palier MESURÉ faisant presque
            if quant_of(f, m["prefix"]) in m.get("mesure", {}) and s >= 0.75 * reco[1]:
                reco = (f, s)
                break
    haut = fits[:2]
    gros = [(f, s) for f, s in lst if s > b][:2]         # les plus gros au-delà du budget, proposés quand même
    choix = [_label(m, f, s, b, vram) for f, s in haut + gros]
    choix += ["another quant (full list)", "skip this model"]
    defaut = _label(m, *reco, b, vram) if reco else (choix[0] if haut + gros else "skip this model")
    c = ui.ask(f"{m['tag']}: which quant to download? (⚠ = over the measured budget — will warn, then confirm)",
               choix, defaut)
    if c in ("skip this model", "another quant (full list)"):
        if c == "skip this model":
            return None
        complet = [_label(m, f, s, b, vram) for f, s in lst] + ["back (skip)"]
        c = ui.ask(f"{m['tag']}: full list ({len(lst)} quants, ⚠ = avoid on this machine)",
                   complet, complet[0] if lst else "back (skip)")
        if c == "back (skip)":
            return None
    for f, s in lst:
        if _label(m, f, s, b, vram) == c:
            if s > b:
                w = ui.ask(f"{f} exceeds the budget ({s / 2**30:.1f} GiB > {b / 2**30:.1f} GiB) — "
                           f"decoding will collapse if the hot set does not fit in RAM. Confirm?",
                           ["confirm anyway", "give up"], "give up")
                if w == "give up":
                    return None
            elif vram and s > vram:
                ui.warn(f"{f} is bigger than the GPU VRAM ({info['vram']} MiB): fine on this MoE port "
                        "(weights live in RAM), but the expert bank shrinks — calibration will size it.")
            return (f, s)
    return None


def download(ui, m, fname: str, target: Path, minsize: int) -> bool:
    """curl -C - avec reprise du .part, contrôle de taille, renommage (identique à setup.sh)."""
    part = Path(str(target) + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{m['repo']}/resolve/main/{fname}"
    ui.say(f"{m['tag']}: {url}  ({target.name})")
    ui.progress(f"waiting for {part.name}", 0.0)
    rc = ui.run(["curl", "-L", "-C", "-", "--retry", "5", "--retry-delay", "3",
                 "-o", str(part), url])
    if ui.stop_flag.is_set():
        return False
    if rc != 0:
        ui.fail("curl failed", "re-run the tool: .part resume is automatic")
        return False
    sz = part.stat().st_size if part.is_file() else 0
    if sz < minsize:
        ui.fail(f"suspicious size: {sz} bytes (< {minsize})", "the HF repo may have changed — check the URL")
        return False
    part.rename(target)
    ui.ok(f"{m['tag']}: {sz // 2**20} MiB in place")
    return True


def ram_warn(ui, m, path: Path):
    try:
        ram = int(subprocess.run("awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo",
                                 shell=True, capture_output=True, text=True).stdout)
    except Exception:
        return
    mib = path.stat().st_size // 2**20
    if mib > ram - 2048:
        ui.warn(f"{mib} MiB model vs {ram} MiB free RAM: the hot set will crack on disk "
                "(measured cliff tg ~7). Take a smaller quant, or free RAM.")


def choix_modele(ui) -> list:
    """on installe UN modèle à la fois : on demande lequel d'abord, on ne descend
    jamais les deux familles l'une après l'autre."""
    etat = {m["tag"]: deja_present(m) for m in MODELS}
    labels = [f"{m['tag']} — {m['nom']} ({m['note']})"
              + (f" — weights already there: {etat[m['tag']].name}" if etat[m["tag"]] else "")
              for m in MODELS]
    defaut = next((i for i, m in enumerate(MODELS) if etat[m["tag"]]), 0)
    c = ui.ask("which model to install?", labels, labels[defaut])
    return [MODELS[labels.index(c)]]


def models_step(ui, info, mods: list) -> dict:
    """le coeur : quant adapté à la machine, réutilisation des poids présents, téléchargement
    repris — pour le(s) modèle(s) choisi(s) uniquement."""
    prets = {}
    for m in mods:
        deja = deja_present(m)
        if deja is not None and model_dir() in deja.parents:
            ui.ok(f"{m['tag']} ({m['nom']}): quant already present, reused — {deja.name}")
            q = quant_of(deja.name, m["prefix"])
            if q in m.get("evite", {}):
                ui.warn(f"{m['tag']}: this quant is marked avoid on the dev machine "
                        f"({q}: {m['evite'][q]}) — calibration will tell whether it fits here, otherwise pick another one.")
            prets[m["tag"]] = deja
            continue
        if deja is not None:                       # poids d'un quant trouvé dans une autre arborescence
            ui.say(f"{m['tag']} ({m['nom']}): weights found elsewhere → {deja}")
            cible = model_dir() / deja.name
            c = ui.ask(f"{m['tag']}: what to do with these already-downloaded weights ({deja.name})?",
                       ["link into MODEL_DIR (symlink, no copy)",
                        "copy (doubles the bytes)",
                        "ignore and pick a fitted quant", "do nothing for this model"],
                       "link into MODEL_DIR (symlink, no copy)")
            if c.startswith("link"):
                cible.parent.mkdir(parents=True, exist_ok=True)
                cible.symlink_to(deja)
                ui.ok(f"link: {cible} -> {deja}")
                prets[m["tag"]] = cible
                continue
            if c.startswith("copy"):
                cible.parent.mkdir(parents=True, exist_ok=True)
                ui.progress(f"copying {deja.name}", None)
                if ui.run(["cp", "-f", str(deja), str(cible)]) == 0:
                    ui.ok(f"copied into {cible}")
                    prets[m["tag"]] = cible
                    continue
                ui.fail("copy interrupted")
                continue
            if c == "do nothing for this model":
                continue
        # ni poids complet ni envie de celui trouvé : choisir un quant adapté et le descendre
        part = None
        for p in model_dir().glob(m["prefix"] + "-*.gguf.part"):
            part = p
            break
        if part is not None:
            ui.warn(f"{m['tag']}: interrupted download already there — {part.name}")
            if ui.ask(f"{m['tag']}: resume this download?", ["resume", "skip"], "resume") == "skip":
                continue
            fname = part.name[:-len(".part")]
            if download(ui, m, fname, model_dir() / fname, max(2**30, int(part.stat().st_size * 0.5))):
                prets[m["tag"]] = model_dir() / fname
            if m["tag"] in prets:
                ram_warn(ui, m, prets[m["tag"]])
            continue
        ch = choix_quant(ui, m, info)
        if ch is None:
            continue
        fname, octets = ch
        cible = model_dir() / fname
        if download(ui, m, fname, cible, max(2**30, int(octets * 0.99))):
            prets[m["tag"]] = cible
            if quant_of(fname, m["prefix"]) != quant_of(m["fallback"][0], m["prefix"]):
                ui.warn(f"{m['tag']}: quant differs from the seed in {m['envfile']} — the routing profile "
                        "depends on the quant: re-run calibration after download.")
        if m["tag"] in prets:
            ram_warn(ui, m, prets[m["tag"]])
    if not prets:
        ui.fail("no model ready", "relaunch and pick a quant (or 'another quant (full list)')")
    return prets


def pc_checks(ui) -> dict:
    """état de la machine — des avertissements, jamais d'arrêt (comme setup.sh)."""
    ui.say(f"niche repository: {ROOT}")
    llama = llama_dir()
    info = {}
    if (llama / ".git").is_dir():
        br = subprocess.run(["git", "-C", str(llama), "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
        if br != BRANCHE:
            ui.warn(f"{llama} is on '{br}', not '{BRANCHE}': the ported MoE cache is not in it.")
        else:
            ui.ok(f"llama.cpp on the right branch: {llama}")
    else:
        ui.warn(f"{llama} is not a git repo — the Build step will offer to clone the port "
                f"({FORK}, branch {BRANCHE}).")
    info["bin"] = llama / "build" / "bin" / "llama-server"
    if info["bin"].is_file():
        ui.ok(f"binary: {info['bin']}")
    else:
        ui.warn("llama-server missing — Build step.")
    for tool in ("cmake", "curl"):
        if not shutil.which(tool):
            ui.warn(f"{tool} missing — install it (Build/Models steps impossible).")
    try:
        ram = int(subprocess.run("awk '/MemTotal/{print int($2/1024/1024)}' /proc/meminfo",
                                 shell=True, capture_output=True, text=True).stdout)
        ui.ok(f"total RAM: {ram} GiB")
        info["ram"] = ram
    except Exception:
        pass
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi:
        out = subprocess.run([nvsmi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        if out:
            ui.ok(f"GPU : {out.replace(', ', ' ')}")
            try:
                info["vram"] = int(out.split(",")[1].split()[0])     # MiB
            except Exception:
                pass
        else:
            ui.warn("nvidia-smi answers nothing — driver?")
    else:
        ui.warn("nvidia-smi missing: GGML will see no CUDA (CPU build → tiny tg on a MoE).")
    return info


def build_step(ui, info) -> bool:
    b = info["bin"]
    if b.is_file():
        if (b.parent / "libggml-cuda.so").is_file():
            ui.ok(f"build already there: {b}")
            return True
        ui.warn("existing build lacks libggml-cuda.so — the MoE cache requires a CUDA backend; reconfiguring")
    c = ui.ask("compile llama.cpp (the build may take 10-30 min)?", ["compile", "skip"], "compile")
    if c != "compile":
        return b.is_file()
    if not (llama_dir() / "CMakeLists.txt").is_file():
        if llama_dir().exists() and any(llama_dir().iterdir()):
            ui.fail(f"{llama_dir()} exists but has no CMakeLists.txt — bring the repo (branch {BRANCHE})")
            return False
        if ui.ask(f"clone the port llama-moe-cache ({FORK}, ~450 MB)?", ["clone", "skip"], "clone") != "clone":
            return False
        if ui.run(["git", "clone", "-b", BRANCHE, "--single-branch", FORK, str(llama_dir())]) != 0:
            ui.fail("git clone failed (network?)")
            return False
        info["bin"] = llama_dir() / "build" / "bin" / "llama-server"
    build = llama_dir() / "build"
    cache = build / "CMakeCache.txt"
    if not cache.is_file() or "GGML_CUDA:BOOL=ON" not in cache.read_text():
        cfg = ["cmake", "-S", str(llama_dir()), "-B", str(build), "-DGGML_CUDA=ON"]
        nv = find_nvcc()
        if nv:
            cfg.append(f"-DCMAKE_CUDA_COMPILER={nv}")
        else:
            ui.warn("nvcc not found on PATH or under /usr/local — cmake may fail to enable CUDA (install the CUDA toolkit)")
        if ui.run(cfg) != 0:
            ui.fail("cmake configure failed")
            return False
    if ui.run(["cmake", "--build", str(build), "-j", str(os.cpu_count() or 8)]) != 0:
        ui.fail("build failed")
        return False
    if b.is_file():
        if not (b.parent / "libggml-cuda.so").is_file():
            ui.fail("build has no CUDA backend (libggml-cuda.so) — the MoE cache requires it; install the CUDA toolkit and rebuild")
            return False
        ui.ok(f"binary: {b}")
        return True
    ui.fail("build finished with no llama-server — check the CUDA warnings above")
    return False


def calibrate_step(ui, info, prets):
    """calibrate.sh par modèle : plafond de bank, pin, profil de routage → machine-<tag>.env."""
    if not prets or not info["bin"].is_file():
        ui.warn("calibration skipped (no ready model or no binary)")
        return
    for m in MODELS:
        tag = m["tag"]
        if tag not in prets:
            continue
        envf = HERE / m["envfile"]
        if envf.is_file():
            ui.ok(f"{tag}: settings already calibrated ({envf.name}) — re-run calibration only if GPU/RAM changed or the quant changed")
        if ui.ask(f"calibrate {tag} (bank/pin/profile measurements, ~10 min, GPU busy)?",
                  ["calibrate", "skip"], "skip") == "calibrate":
            rc = ui.run([str(HERE / "calibrate.sh"), tag, str(prets[tag])], cwd=str(ROOT))
            ui.ok(f"calibration {tag}: {HERE / ('machine-' + tag + '.env')}") if rc == 0 else \
                ui.fail(f"calibrate.sh {tag} failed (rc={rc})")


def ctx_choix(ui, m, saved=None):
    """taille de contexte demandée avant de servir ; None = laisser le défaut du script."""
    ctxs = m.get("ctxs")
    if not ctxs:
        return None
    labels = [f"{n} tokens" + (f"  ({note})" if note else "") for n, note in ctxs]
    pairs = dict(zip(labels, (n for n, _ in ctxs)))
    defaut = None
    if saved:
        defaut = next((l for l in labels if l.startswith(f"{saved} tokens")), None)
    defaut = defaut or next((l for l in labels if l.startswith(f"{m['ctx_defaut']} tokens")), labels[0])
    c = ui.ask(f"{m['tag']}: which context size? (context ↑ = expert bank ↓)",
               labels, defaut)
    return pairs.get(c, m.get("ctx_defaut"))


# ---------------------------------------------------- choix enregistrés (settings-<tag>.env)

def settings_path(tag) -> Path:
    return HERE / f"settings-{tag}.env"


def settings_get(tag) -> dict:
    """les choix enregistrés par le menu : lignes KEY=${KEY:-v} — un export garde la priorité."""
    p = settings_path(tag)
    if not p.is_file():
        return {}
    return dict(re.findall(r"(?m)^(\w+)=\$\{\w+:-([^}]*)\}$", p.read_text(encoding="utf-8")))


def settings_save(tag, updates: dict):
    cur = settings_get(tag)
    cur.update({k: str(v) for k, v in updates.items() if v is not None})
    head = (f"# choices saved by the niche.sh menu for {tag} — exported env wins over these,\n"
            f"# and these win over machine-{tag}.env from calibration. Edit or delete to reset.\n")
    lines = "".join(f"{k}=${{{k}:-{v}}}\n" for k, v in sorted(cur.items()))
    settings_path(tag).write_text(head + lines, encoding="utf-8")


def serve_port(tag) -> int:
    try:
        return int(settings_get(tag).get("PORT") or 8002)
    except ValueError:
        return 8002


def pid_path(tag) -> Path:
    return HERE / f".niche-server-{tag}.pid"


def running_pid(tag):
    """pid du serveur en arrière-plan lancé par le menu (survit au TUI), None sinon."""
    pf = pid_path(tag)
    if not pf.is_file():
        return None
    try:
        pid = int(pf.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        pf.unlink(missing_ok=True)
        return None
    except OSError:
        return None


def health_ok(port: int) -> bool:
    """200 seulement : un 503 « loading model » répond avant la fin du chargement."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def setup_existe(m, info) -> bool:
    """un setup prêt à servir tel quel : poids + binaire + profil de routage (calibré ou défaut)."""
    if not info["bin"].is_file() or deja_present(m) is None:
        return False
    envf = HERE / m["envfile"]
    if envf.is_file() and "PROFILE=" in envf.read_text():
        return True
    return (HERE / f"prof-{m['tag']}.csv").is_file() or (m["tag"] == "qwen" and (HERE / "prof3.csv").is_file())


# ---------------------------------------------------- questions de réglage du serveur

def kvq_choix(ui, m, saved=None):
    opts = [("q8_0 — half of f16, no measurable quality loss (recommended when RAM/VRAM are tight)", "q8_0"),
            ("q4_0 — the project's measured reference: biggest savings, small quality hit", "q4_0"),
            ("f16 — no quantization: 2x the KV memory of q8_0, only for small contexts", "f16")]
    labels = [l for l, _ in opts]
    defaut = next((l for l, v in opts if v == saved), labels[1])
    c = ui.ask(f"{m['tag']}: KV cache quantization? (smaller KV = more room for the expert bank and the hot set)",
               labels, defaut)
    return dict(opts)[c]


def spec_choix(ui, m, saved=None):
    if m["mtp"]:
        opts = [("keep the script default (draft-mtp when the GGUF has the MTP head)", ""),
                ("draft-mtp — native MTP head, ~93 % acceptance on repeats", "draft-mtp"),
                ("draft-mtp,ngram-mod — x3 on re-sent prompts, neutral on fresh text (measured 20/10)", "draft-mtp,ngram-mod"),
                ("ngram-mod — lookup only", "ngram-mod"),
                ("none — no speculation", "none")]
    else:
        opts = [("ngram-mod — lookup speculation, script default (~2.4x on code-like sessions)", "ngram-mod"),
                ("none — no speculation", "none")]
    labels = [l for l, _ in opts]
    defaut = next((l for l, v in opts if v == saved), labels[0])
    c = ui.ask(f"{m['tag']}: speculation policy?", labels, defaut)
    return dict(opts)[c]


def adapt_choix(ui, m, saved=None):
    opts = [("512 — measured default: re-rank the tier every 512 cached tokens", "512"),
            ("0 — off: pure static tier, only the routing profile", "0"),
            ("128 — very reactive (more GPU↔RAM swaps)", "128"),
            ("2048 — gentle (fewer swaps)", "2048")]
    labels = [l for l, _ in opts]
    defaut = next((l for l, v in opts if v == saved), labels[0])
    c = ui.ask(f"{m['tag']}: live tier adaptation (--moe-cache-adapt)?", labels, defaut)
    return dict(opts)[c]


def port_choix(ui, m, saved=None):
    ports = ["8002", "8003", "8004", "8005", "8080"]
    defaut = str(saved) if str(saved or "") in ports else "8002"
    c = ui.ask(f"{m['tag']}: which server port?", ports, defaut)
    return c


def model_choix(ui, m, saved=None):
    """parmi les poids de la famille déjà téléchargés (MODEL_DIR + GGUF_DIRS)."""
    cands, seen = [], set()
    for d in [model_dir()] + gguf_dirs():
        for c in d.glob(m["prefix"] + "-*.gguf"):
            if "mmproj" not in c.name and c.name not in seen and c.stat().st_size > 2**30:
                seen.add(c.name)
                cands.append(c)
    if not cands:
        ui.warn(f"{m['tag']}: no weights of this family found — use 'download another quant of this family'")
        return None
    labels = [f"{c}  ({c.stat().st_size / 2**30:.2f} GiB)" + ("  ← in use" if saved and str(c) == saved else "")
              for c in cands]
    defaut = next((l for l in labels if l.endswith("← in use")), labels[0])
    c = ui.ask(f"{m['tag']}: which weight file?", labels, defaut)
    return str(cands[labels.index(c)])


def serve_trio(ui, m) -> dict:
    """contexte + KV + spéculation demandés à chaque démarrage, puis enregistrés dans settings-<tag>.env."""
    st = settings_get(m["tag"])
    n = ctx_choix(ui, m, st.get("CTX"))
    k = kvq_choix(ui, m, st.get("KVQ"))
    sp = spec_choix(ui, m, st.get("SPEC"))
    settings_save(m["tag"], {kk: vv for kk, vv in (("CTX", n), ("KVQ", k), ("SPEC", sp)) if vv is not None})
    env = dict(os.environ)
    if n:
        env["CTX"] = str(n)
    env["KVQ"], env["SPEC"] = k, sp
    mm = st.get("MODEL") or (str(deja_present(m)) if deja_present(m) else None)
    if mm:
        env["MODEL"] = mm
    for kk in ("PORT", "ADAPT"):
        if st.get(kk) is not None:
            env[kk] = st[kk]
    return env


# ---------------------------------------------------- démarrer / arrêter un serveur

def serve_fg(ui, info, m):
    env = serve_trio(ui, m)
    rc = ui.run([str(HERE / m["serve"])], cwd=str(ROOT), env=env)
    if rc not in (0, -1) and not ui.stop_flag.is_set():
        ui.fail(f"{m['tag']} server exited with code {rc}",
                "check the output above: port busy? missing model? profile?")


def serve_bg(ui, info, m):
    tag = m["tag"]
    if health_ok(serve_port(tag)):
        ui.warn(f"something already answers /health on http://127.0.0.1:{serve_port(tag)} — "
                f"stop it first, or change the port in 'change the {tag} settings'")
        return
    env = serve_trio(ui, m)
    port = serve_port(tag)
    log = HERE / f"niche-server-{tag}.log"
    with open(log, "w", encoding="utf-8") as lf:
        p = subprocess.Popen([str(HERE / m["serve"])], cwd=str(ROOT), env=env,
                             stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                             start_new_session=True)     # survit à la sortie du TUI
    pid_path(tag).write_text(f"{p.pid}\n")   # exec() dans serve.sh garde le pid du script
    ui.say(f"{tag}: starting the server in the background (pid {p.pid}) — log tools/{log.name}")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 900 and not ui.stop_flag.is_set():
        if health_ok(port):
            ui.ok(f"{tag}: server up on http://127.0.0.1:{port} after {int(time.monotonic() - t0)} s — "
                  "it keeps running after niche.sh quits (the menu offers 'stop' for it)")
            return
        ui.progress(f"{tag}: loading the model on the GPU … {int(time.monotonic() - t0)} s", None)
        time.sleep(2)
    ui.fail(f"{tag}: server did not answer /health — see tools/{log.name}")
    try:
        for line in log.read_text(errors="replace").splitlines()[-4:]:
            ui.warn(line)
    except OSError:
        pass


def stop_server(ui, m):
    tag = m["tag"]
    pid = running_pid(tag)
    if pid is None:
        ui.warn(f"{tag}: no background server known (pid file missing or stale)")
        return
    try:
        os.kill(pid, signal.SIGINT)      # arrêt propre : les compteurs moe-stats se flushent
    except ProcessLookupError:
        pid_path(tag).unlink(missing_ok=True)
        ui.warn(f"{tag}: process {pid} was already gone")
        return
    t0 = time.monotonic()
    while time.monotonic() - t0 < 60:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.5)
    pid_path(tag).unlink(missing_ok=True)
    ui.ok(f"{tag}: background server stopped (SIGINT sent to pid {pid})")


def settings_step(ui, info, m):
    """modifier les paramètres d'un setup déjà configuré : tout est écrit dans settings-<tag>.env."""
    tag = m["tag"]
    while not ui.stop_flag.is_set():
        st = settings_get(tag)
        ui.say(f"{tag} current settings: CTX={st.get('CTX') or m['ctx_defaut']}  "
               f"KVQ={st.get('KVQ') or 'q4_0 (script default)'}  "
               f"SPEC={st.get('SPEC') or '(script default)'}  ADAPT={st.get('ADAPT') or '512'}  "
               f"PORT={st.get('PORT') or 8002}")
        choix = ["context size", "KV cache quantization", "speculation policy",
                 "live tier adaptation (ADAPT)", "server port", "weight file (already downloaded)",
                 "download another quant of this family", "back to the menu"]
        c = ui.ask(f"{tag} settings: what to change?", choix, choix[0])
        if c == "back to the menu":
            return
        if c == "context size":
            n = ctx_choix(ui, m, st.get("CTX"))
            if n:
                settings_save(tag, {"CTX": n})
                ui.ok(f"{tag}: CTX={n} saved")
        elif c == "KV cache quantization":
            k = kvq_choix(ui, m, st.get("KVQ"))
            settings_save(tag, {"KVQ": k})
            ui.ok(f"{tag}: KVQ={k} saved")
        elif c == "speculation policy":
            s = spec_choix(ui, m, st.get("SPEC"))
            settings_save(tag, {"SPEC": s})
            ui.ok(f"{tag}: SPEC={s or '(script default)'} saved")
        elif c == "live tier adaptation (ADAPT)":
            a = adapt_choix(ui, m, st.get("ADAPT"))
            settings_save(tag, {"ADAPT": a})
            ui.ok(f"{tag}: ADAPT={a} saved")
        elif c == "server port":
            p = port_choix(ui, m, st.get("PORT"))
            settings_save(tag, {"PORT": p})
            ui.ok(f"{tag}: PORT={p} saved — start the server again for it to take effect")
        elif c == "weight file (already downloaded)":
            mm = model_choix(ui, m, st.get("MODEL"))
            if mm:
                settings_save(tag, {"MODEL": mm})
                ui.ok(f"{tag}: MODEL={mm} saved")
                q = quant_of(Path(mm).name, m["prefix"])
                ui.warn(f"{tag}: the routing profile depends on the quant ({q}) — "
                        "if tg collapses or the hit rate drops, run calibration again")
        elif c == "download another quant of this family":
            ch = choix_quant(ui, m, info)
            if ch is None:
                continue
            fname, octets = ch
            cible = model_dir() / fname
            if download(ui, m, fname, cible, max(2**30, int(octets * 0.99))):
                settings_save(tag, {"MODEL": str(cible)})
                ui.ok(f"{tag}: MODEL={cible} saved")
                if ui.ask("the routing profile depends on the quant — recalibrate now (~10 min, GPU busy)?",
                          ["calibrate now", "later"], "calibrate now") == "calibrate now":
                    calibrate_step(ui, info, {tag: cible})


def full_setup(ui, info):
    """la suite de préparation d'un modèle : poids → build → calibrage (étapes 2-4)."""
    ui.step(2, "Model")
    prets = models_step(ui, info, choix_modele(ui))
    ui.step(3, "Build")
    built = build_step(ui, info)
    ui.step(4, "Calibration")
    calibrate_step(ui, info, prets if built else {})


def menu_menu(ui, info):
    """le menu central : démarrer un setup déjà configuré (avant-plan ou arrière-plan),
    modifier ses paramètres, ou préparer un nouveau modèle."""
    while not ui.stop_flag.is_set():
        choix = []
        for m in MODELS:
            tag = m["tag"]
            if not setup_existe(m, info):
                continue
            pid = running_pid(tag)
            if pid:
                choix.append(f"stop the {tag} background server (pid {pid})")
            else:
                choix.append(f"start the {tag} server in the foreground ({m['serve']})")
                choix.append(f"start the {tag} server in the background ({m['serve']})")
            choix.append(f"change the {tag} settings")
        choix += ["prepare a new model (weights, build, calibration)", "quit"]
        c = ui.ask("Ready. What to do?", choix, choix[0])
        if c == "quit":
            return
        if c == "prepare a new model (weights, build, calibration)":
            full_setup(ui, info)
            ui.step(5, "Ready")
            continue
        for m in MODELS:
            tag = m["tag"]
            if c.startswith(f"start the {tag} server in the foreground"):
                serve_fg(ui, info, m)
            elif c.startswith(f"start the {tag} server in the background"):
                serve_bg(ui, info, m)
            elif c.startswith(f"stop the {tag} background server"):
                stop_server(ui, m)
            elif c.startswith(f"change the {tag} settings"):
                settings_step(ui, info, m)


def body(ui):
    """machine d'abord ; si un setup est déjà configuré on ouvre le menu directement,
    sinon on enchaîne modèle → build → calibrage puis le menu."""
    ui.step(1, "Machine")
    info = pc_checks(ui)
    if not ui.stop_flag.is_set():
        if not any(setup_existe(m, info) for m in MODELS):
            full_setup(ui, info)
        if not ui.stop_flag.is_set():
            if any(setup_existe(m, info) for m in MODELS):
                ui.step(5, "Ready")
                menu_menu(ui, info)
            else:
                ui.fail("nothing to serve — no model + binary + profile combination found")
    ui._note("end")


if __name__ == "__main__":
    import sys
    from niche_bridge import PlainUI
    ui = PlainUI()
    body(ui)
    sys.exit(0)
