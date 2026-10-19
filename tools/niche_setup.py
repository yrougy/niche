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
import shutil
import subprocess
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent          # tools/
ROOT = HERE.parent                              # le dépôt niche

# catalogue : familles de quants connues. « mesure »/« evite » = retours mesurés sur la
# machine de dev (15 GiB RAM / 8 GiB VRAM, WORKLOG 12-14/10) ; ils guident la reco mais
# ne remplacent pas le budget calculé sur la machine de l'utilisateur.
MODELS = [
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
    dict(tag="lfm", nom="LFM2.5-24B-A2B", prefix="LFM2.5-24B-A2B",
         repo="unsloth/LFM2.5-24B-A2B-GGUF",
         note="24B MoE (2B active), flat 8k-64k, ctx max 131k",
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


def _label(m, f, s, b):
    q = quant_of(f, m["prefix"])
    lbl = f"{q} — {s / 2**30:.2f} GiB"
    if q in m.get("mesure", {}):
        lbl += f"  ({m['mesure'][q]})"
    if q in m.get("evite", {}):
        lbl += f"  ⚠ {m['evite'][q]}"
    elif s > b:
        lbl += "  ⚠ over the hot-set budget (measured wall beyond)"
    elif s >= 0.95 * b:
        lbl += "  (tight fit, cut it close)"
    return lbl


def choix_quant(ui, m, info):
    """propose les quants adaptés à la machine ; rend (fichier, octets) ou None (passer)."""
    lst = hf_listing(ui, m) or [m["fallback"]]
    b, expl = budget_octets(info)
    note_vram = ""
    if info.get("vram") is not None and info["vram"] < 4 * 1024:
        note_vram = f" ⚠ {info['vram']} MiB GPU: prefer a small quant (less PCIe traffic)."
    ui.say(f"{m['tag']} ({m['nom']}): budget — {expl}.{note_vram}")
    fits = [(f, s) for f, s in lst if 0 < s <= b]
    reco = None
    if fits:
        reco = fits[0]                                   # le plus gros qui tient
        for f, s in fits:                                # un palier MESURÉ faisant presque
            if quant_of(f, m["prefix"]) in m.get("mesure", {}) and s >= 0.75 * reco[1]:
                reco = (f, s)
                break
    haut = fits[:2] if len(fits) >= 2 else fits
    choix = [_label(m, f, s, b) for f, s in haut]
    choix += ["another quant (full list)", "skip this model"]
    defaut = _label(m, *reco, b) if reco else "skip this model"
    c = ui.ask(f"{m['tag']}: which quant to download?", choix, defaut)
    if c in ("skip this model", "another quant (full list)"):
        if c == "skip this model":
            return None
        complet = [_label(m, f, s, b) for f, s in lst] + ["back (skip)"]
        c = ui.ask(f"{m['tag']}: full list ({len(lst)} quants, ⚠ = avoid on this machine)",
                   complet, complet[0] if lst else "back (skip)")
        if c == "back (skip)":
            return None
    for f, s in lst:
        if _label(m, f, s, b) == c:
            if s > b:
                w = ui.ask(f"{f} exceeds the budget ({s / 2**30:.1f} GiB > {b / 2**30:.1f} GiB) — "
                           f"decoding will collapse if the hot set does not fit. Confirm?",
                           ["confirm anyway", "give up"], "give up")
                if w == "give up":
                    return None
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
        ui.ok(f"build already there: {b}")
        return True
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
    if not (llama_dir() / "build").exists():
        if ui.run(["cmake", "-S", str(llama_dir()), "-B", str(llama_dir() / "build")]) != 0:
            ui.fail("cmake configure failed")
            return False
    if ui.run(["cmake", "--build", str(llama_dir() / "build"), "-j", str(os.cpu_count() or 8)]) != 0:
        ui.fail("build failed")
        return False
    if b.is_file():
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


def ctx_choix(ui, m):
    """taille de contexte demandée avant de servir ; None = laisser le défaut du script."""
    ctxs = m.get("ctxs")
    if not ctxs:
        return None
    labels = [f"{n} tokens" + (f"  ({note})" if note else "") for n, note in ctxs]
    pairs = dict(zip(labels, (n for n, _ in ctxs)))
    defaut = next((l for l in labels if l.startswith(f"{m['ctx_defaut']} tokens")), labels[0])
    c = ui.ask(f"{m['tag']}: which context size? (context ↑ = expert bank ↓)",
               labels, defaut)
    return pairs.get(c, m.get("ctx_defaut"))


def ready_step(ui, info, prets):
    """le menu final : démarrer le serveur sur le quant installé (MODEL forcé), ou revenir en arrière."""
    while not ui.stop_flag.is_set():
        choix = [f"start the {m['tag']} server ({m['serve']})" for m in MODELS if m["tag"] in prets]
        choix += ["run calibration again", "quit"]
        c = ui.ask("Ready. What to do?", choix, choix[0])
        if c == "quit":
            return
        for m in MODELS:
            if c.startswith(f"start the {m['tag']} server"):
                if info["bin"].is_file():
                    env = dict(os.environ, MODEL=str(prets[m["tag"]]))
                    n = ctx_choix(ui, m)
                    if n:
                        env["CTX"] = str(n)
                    ui.run([str(HERE / m["serve"])], cwd=str(ROOT), env=env)
                break
        if c == "run calibration again":
            calibrate_step(ui, info, prets)


def body(ui):
    """la suite complète des étapes ; le TUI la lance dans un thread, --plain la court devant."""
    ui.step(1, "Machine")
    info = pc_checks(ui)
    ui.step(2, "Model")
    prets = models_step(ui, info, choix_modele(ui))
    ui.step(3, "Build")
    built = build_step(ui, info)
    ui.step(4, "Calibration")
    calibrate_step(ui, info, prets if built else {})
    ui.step(5, "Ready")
    if built and prets:
        ready_step(ui, info, prets)
    ui._note("end")


if __name__ == "__main__":
    import sys
    from niche_bridge import PlainUI
    ui = PlainUI()
    body(ui)
    sys.exit(0)
