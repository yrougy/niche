#!/usr/bin/env python3
"""niche_setup — le moteur sans écran du port cache-MoE niche : état de la machine,
choix d'une QUANTIZATION ADAPTÉE À LA MACHINE pour LFM2.5 ou Qwen3.6-35B (l'API HF
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
         fallback=("Qwen3.6-35B-A3B-UD-IQ2_M.gguf", 10 * 2**30),
         serve="serve.sh", envfile="machine-qwen.env", mtp=True, ctx_defaut=4096,
         ctxs=[(4096, "défaut du script, bank ~5632 mesuré sur ce PC"),
               (8192, "bank ~5120 (−512 MiB/8k)"),
               (16384, "bank ~4608"),
               (32768, "bank coupé à 4096 minimum par serve.sh — surveiller le hit")],
         mesure={"UD-IQ2_M": "tg 41,5 mesuré 12/10",
                 "UD-Q2_K_XL": "palier viable mesuré : tg 40,3, qualité et acceptance MTP meilleures"},
         evite={"UD-Q3_K_M": "mur mesuré tg 6,7 sur 15 GiB de RAM",
                "UD-Q3_K_S": "mur mesuré tg 6,7 sur 15 GiB de RAM",
                "UD-Q3_K_XL": "> 15 GiB : même mur que Q3_K_M",
                "UD-IQ3_S": "> 12,7 GiB : vers le mur mesuré",
                "UD-IQ3_XXS": "> 12,3 GiB : vers le mur mesuré"}),
    dict(tag="lfm", nom="LFM2.5-24B-A2B", prefix="LFM2.5-24B-A2B",
         repo="unsloth/LFM2.5-24B-A2B-GGUF",
         fallback=("LFM2.5-24B-A2B-UD-IQ3_XXS.gguf", 7 * 2**30),
         serve="serve-lfm.sh", envfile="machine-lfm.env", mtp=False, ctx_defaut=65536,
         ctxs=[(8192, "plateau mesuré : bank 6656 de 8k à 64k"),
               (16384, "bank 6656 (plateau)"),
               (32768, "bank 6656 (plateau)"),
               (65536, "défaut du script, plateau mesuré, bank 6656"),
               (131072, "max natif du GGUF : bank 6144, tg −3 % (mesuré 16/10)")],
         mesure={"UD-IQ3_XXS": "tg 47,2 mesuré 14/10 — config de référence de la machine"},
         evite={"UD-Q4_K_XL": "> 13,8 GiB : hot set félerait sur 15 GiB de RAM",
                "UD-Q5_K_XL": "> 16,6 GiB : ne tient pas en RAM",
                "UD-Q6_K_XL": "> 20 GiB : ne tient pas en RAM",
                "UD-Q8_0": "> 23,7 GiB : ne tient pas en RAM",
                "UD-Q2_K_XL": "possible mais sous le palier qualité d'IQ3_XXS (mesuré)"}),
]
BRANCHE = "moe-expert-cache"
FORK = "https://github.com/yrougy/llama-moe-cache.git"


def model_dir() -> Path:
    return Path(os.environ.get("MODEL_DIR", str(ROOT / "models")))


def llama_dir() -> Path:
    return Path(os.environ.get("LLAMA", str(ROOT / "llama.cpp")))


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
        ui.warn(f"{m['tag']} : API HF injoignable ({e.__class__.__name__}) — repli sur le quant d'office.")
        return None
    out = [(e["path"], int(e.get("size", 0))) for e in ents
           if e.get("path", "").endswith(".gguf") and "/" not in e["path"] and "mmproj" not in e["path"]]
    return sorted(out, key=lambda x: -x[1]) or None


def budget_octets(info) -> tuple:
    """taille maxi de GGUF pour que le hot set tienne en RAM — la contrainte mesurée (12/10),
    pas la VRAM (le bank d'experts se règle au calibrage). Réserve : OS + KV + marge."""
    ram_gib = info.get("ram") or 16
    b = int((ram_gib - 3) * 2**30)
    return b, f"RAM {ram_gib} GiB − 3 GiB (OS+KV+marge) ≈ {b / 2**30:.1f} GiB de hot set"


def _label(m, f, s, b):
    q = quant_of(f, m["prefix"])
    lbl = f"{q} — {s / 2**30:.2f} GiB"
    if q in m.get("mesure", {}):
        lbl += f"  ({m['mesure'][q]})"
    if q in m.get("evite", {}):
        lbl += f"  ⚠ {m['evite'][q]}"
    elif s > b:
        lbl += "  ⚠ dépasse le budget hot set (mur mesuré au-delà)"
    elif s >= 0.95 * b:
        lbl += "  (limite, compte serré)"
    return lbl


def choix_quant(ui, m, info):
    """propose les quants adaptés à la machine ; rend (fichier, octets) ou None (passer)."""
    lst = hf_listing(ui, m) or [m["fallback"]]
    b, expl = budget_octets(info)
    note_vram = ""
    if info.get("vram") is not None and info["vram"] < 4 * 1024:
        note_vram = f" ⚠ GPU de {info['vram']} MiB : préfère un quant petit (moins de trafic PCIe)."
    ui.say(f"{m['tag']} ({m['nom']}) : budget — {expl}.{note_vram}")
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
    choix += ["choisir un autre quant (liste complète)", "passer ce modèle"]
    defaut = _label(m, *reco, b) if reco else "passer ce modèle"
    c = ui.ask(f"{m['tag']} : quel quant descendre ?", choix, defaut)
    if c in ("passer ce modèle", "choisir un autre quant (liste complète)"):
        if c == "passer ce modèle":
            return None
        complet = [_label(m, f, s, b) for f, s in lst] + ["retour (passer)"]
        c = ui.ask(f"{m['tag']} : liste complète ({len(lst)} quants, ⚠ = à éviter sur cette machine)",
                   complet, complet[0] if lst else "retour (passer)")
        if c == "retour (passer)":
            return None
    for f, s in lst:
        if _label(m, f, s, b) == c:
            if s > b:
                w = ui.ask(f"{f} dépasse le budget ({s / 2**30:.1f} GiB > {b / 2**30:.1f} GiB) — "
                           f"le décodage s'effondrera si le hot set ne tient pas. Confirmer ?",
                           ["confirmer quand même", "renoncer"], "renoncer")
                if w == "renoncer":
                    return None
            return (f, s)
    return None


def download(ui, m, fname: str, target: Path, minsize: int) -> bool:
    """curl -C - avec reprise du .part, contrôle de taille, renommage (identique à setup.sh)."""
    part = Path(str(target) + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{m['repo']}/resolve/main/{fname}"
    ui.say(f"{m['tag']} : {url}  ({target.name})")
    ui.progress(f"en attente de {part.name}", 0.0)
    rc = ui.run(["curl", "-L", "-C", "-", "--retry", "5", "--retry-delay", "3",
                 "-o", str(part), url])
    if ui.stop_flag.is_set():
        return False
    if rc != 0:
        ui.fail("curl a échoué", "relancer l'outil : la reprise .part est automatique")
        return False
    sz = part.stat().st_size if part.is_file() else 0
    if sz < minsize:
        ui.fail(f"taille suspecte : {sz} octets (< {minsize})", "le dépôt HF a peut-être changé — vérifier l'URL")
        return False
    part.rename(target)
    ui.ok(f"{m['tag']} : {sz // 2**20} MiB en place")
    return True


def ram_warn(ui, m, path: Path):
    try:
        ram = int(subprocess.run("awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo",
                                 shell=True, capture_output=True, text=True).stdout)
    except Exception:
        return
    mib = path.stat().st_size // 2**20
    if mib > ram - 2048:
        ui.warn(f"{mib} MiB de modèle pour {ram} MiB RAM dispo : le hot set va féler sur disque "
                "(cliff mesuré tg ~7). Prendre un quant plus petit, ou libérer de la RAM.")


def models_step(ui, info) -> dict:
    """le coeur : quant adapté à la machine, réutilisation des poids présents, téléchargement repris."""
    prets = {}
    for m in MODELS:
        deja = deja_present(m)
        if deja is not None and model_dir() in deja.parents:
            ui.ok(f"{m['tag']} ({m['nom']}) : quant déjà présent, réutilisé — {deja.name}")
            q = quant_of(deja.name, m["prefix"])
            if q in m.get("evite", {}):
                ui.warn(f"{m['tag']} : ce quant est marqué à éviter sur la machine de dev "
                        f"({q} : {m['evite'][q]}) — un calibrage dira s'il tient ici, sinon choisis-en un autre.")
            prets[m["tag"]] = deja
            continue
        if deja is not None:                       # poids d'un quant trouvé dans une autre arborescence
            ui.say(f"{m['tag']} ({m['nom']}) : poids trouvé ailleurs → {deja}")
            cible = model_dir() / deja.name
            c = ui.ask(f"{m['tag']} : que faire de ce poids déjà téléchargé ({deja.name}) ?",
                       ["lier dans le MODEL_DIR (lien symbolique, pas de copie)",
                        "copier (doubles les octets)",
                        "ignorer et choisir un quant adapté", "ne rien faire pour ce modèle"],
                       "lier dans le MODEL_DIR (lien symbolique, pas de copie)")
            if c.startswith("lier"):
                cible.parent.mkdir(parents=True, exist_ok=True)
                cible.symlink_to(deja)
                ui.ok(f"lien : {cible} -> {deja}")
                prets[m["tag"]] = cible
                continue
            if c.startswith("copier"):
                cible.parent.mkdir(parents=True, exist_ok=True)
                ui.progress(f"copie de {deja.name}", None)
                if ui.run(["cp", "-f", str(deja), str(cible)]) == 0:
                    ui.ok(f"copié dans {cible}")
                    prets[m["tag"]] = cible
                    continue
                ui.fail("copie interrompue")
                continue
            if c == "ne rien faire pour ce modèle":
                continue
        # ni poids complet ni envie de celui trouvé : choisir un quant adapté et le descendre
        part = None
        for p in model_dir().glob(m["prefix"] + "-*.gguf.part"):
            part = p
            break
        if part is not None:
            ui.warn(f"{m['tag']} : téléchargement interrompu déjà là — {part.name}")
            if ui.ask(f"{m['tag']} : reprendre ce téléchargement ?", ["reprendre", "passer"], "reprendre") == "passer":
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
                ui.warn(f"{m['tag']} : quant différent du seed de {m['envfile']} — le profil de routage "
                        "dépend du quant : refais un calibrage après téléchargement.")
        if m["tag"] in prets:
            ram_warn(ui, m, prets[m["tag"]])
    if not prets:
        ui.fail("aucun modèle prêt", "relance et choisis un quant (ou « choisir un autre quant »)")
    return prets


def pc_checks(ui) -> dict:
    """état de la machine — des avertissements, jamais d'arrêt (comme setup.sh)."""
    ui.say(f"dépôt niche : {ROOT}")
    llama = llama_dir()
    info = {}
    if (llama / ".git").is_dir():
        br = subprocess.run(["git", "-C", str(llama), "rev-parse", "--abbrev-ref", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
        if br != BRANCHE:
            ui.warn(f"{llama} est sur « {br} », pas « {BRANCHE} » : le cache MoE porté n'y est pas.")
        else:
            ui.ok(f"llama.cpp sur la bonne branche : {llama}")
    else:
        ui.warn(f"{llama} n'est pas un dépôt git — l'étape Build proposera de cloner le port "
                f"({FORK}, branche {BRANCHE}).")
    info["bin"] = llama / "build" / "bin" / "llama-server"
    if info["bin"].is_file():
        ui.ok(f"binaire : {info['bin']}")
    else:
        ui.warn("llama-server absent — étape Build.")
    for tool in ("cmake", "curl"):
        if not shutil.which(tool):
            ui.warn(f"{tool} absent — installer (étape Build/Modèles impossible).")
    try:
        ram = int(subprocess.run("awk '/MemTotal/{print int($2/1024/1024)}' /proc/meminfo",
                                 shell=True, capture_output=True, text=True).stdout)
        ui.ok(f"RAM totale : {ram} GiB")
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
            ui.warn("nvidia-smi répond rien — pilote ?")
    else:
        ui.warn("nvidia-smi absent : GGML ne verra pas de CUDA (build CPU → tg minuscule sur un MoE).")
    return info


def build_step(ui, info) -> bool:
    b = info["bin"]
    if b.is_file():
        ui.ok(f"build déjà là : {b}")
        return True
    c = ui.ask("compiler llama.cpp (la build peut durer 10-30 min) ?", ["compiler", "passer"], "compiler")
    if c != "compiler":
        return b.is_file()
    if not (llama_dir() / "CMakeLists.txt").is_file():
        if llama_dir().exists() and any(llama_dir().iterdir()):
            ui.fail(f"{llama_dir()} existe mais sans CMakeLists.txt — apporter le dépôt (branche {BRANCHE})")
            return False
        if ui.ask(f"cloner le port llama-moe-cache ({FORK}, ~450 Mo) ?", ["cloner", "passer"], "cloner") != "cloner":
            return False
        if ui.run(["git", "clone", "-b", BRANCHE, "--single-branch", FORK, str(llama_dir())]) != 0:
            ui.fail("git clone a échoué (réseau ?)")
            return False
        info["bin"] = llama_dir() / "build" / "bin" / "llama-server"
    if not (llama_dir() / "build").exists():
        if ui.run(["cmake", "-S", str(llama_dir()), "-B", str(llama_dir() / "build")]) != 0:
            ui.fail("cmake configure a échoué")
            return False
    if ui.run(["cmake", "--build", str(llama_dir() / "build"), "-j", str(os.cpu_count() or 8)]) != 0:
        ui.fail("build échouée")
        return False
    if b.is_file():
        ui.ok(f"binaire : {b}")
        return True
    ui.fail("build finie sans llama-server — regarder les avertissements CUDA plus haut")
    return False


def calibrate_step(ui, info, prets):
    """calibrate.sh par modèle : plafond de bank, pin, profil de routage → machine-<tag>.env."""
    if not prets or not info["bin"].is_file():
        ui.warn("calibrage sauté (pas de modèle prêt ou pas de binaire)")
        return
    for m in MODELS:
        tag = m["tag"]
        if tag not in prets:
            continue
        envf = HERE / m["envfile"]
        if envf.is_file():
            ui.ok(f"{tag} : réglages déjà calibrés ({envf.name}) — relance un calibrage seulement si GPU/RAM changés ou si le quant a changé")
        if ui.ask(f"calibrer {tag} (mesures de bank/pin/profil, ~10 min, GPU occupé) ?",
                  ["calibrer", "passer"], "passer") == "calibrer":
            rc = ui.run([str(HERE / "calibrate.sh"), tag, str(prets[tag])], cwd=str(ROOT))
            ui.ok(f"calibrage {tag} : {HERE / ('machine-' + tag + '.env')}") if rc == 0 else \
                ui.fail(f"calibrate.sh {tag} a échoué (rc={rc})")


def ctx_choix(ui, m):
    """taille de contexte demandée avant de servir ; None = laisser le défaut du script."""
    ctxs = m.get("ctxs")
    if not ctxs:
        return None
    labels = [f"{n} tokens" + (f"  ({note})" if note else "") for n, note in ctxs]
    pairs = dict(zip(labels, (n for n, _ in ctxs)))
    defaut = next((l for l in labels if l.startswith(f"{m['ctx_defaut']} tokens")), labels[0])
    c = ui.ask(f"{m['tag']} : quelle taille de contexte ? (contexte ↑ = bank d'experts ↓)",
               labels, defaut)
    return pairs.get(c, m.get("ctx_defaut"))


def ready_step(ui, info, prets):
    """le menu final : démarrer le serveur sur le quant installé (MODEL forcé), ou revenir en arrière."""
    while not ui.stop_flag.is_set():
        choix = [f"démarrer le serveur {m['tag']} ({m['serve']})" for m in MODELS if m["tag"] in prets]
        choix += ["repasser le calibrage", "quitter"]
        c = ui.ask("Prêt. Que faire ?", choix, choix[0])
        if c == "quitter":
            return
        for m in MODELS:
            if c.startswith(f"démarrer le serveur {m['tag']}"):
                if info["bin"].is_file():
                    env = dict(os.environ, MODEL=str(prets[m["tag"]]))
                    n = ctx_choix(ui, m)
                    if n:
                        env["CTX"] = str(n)
                    ui.run([str(HERE / m["serve"])], cwd=str(ROOT), env=env)
                break
        if c == "repasser le calibrage":
            calibrate_step(ui, info, prets)


def body(ui):
    """la suite complète des étapes ; le TUI la lance dans un thread, --plain la court devant."""
    ui.step(1, "Ordinateur")
    info = pc_checks(ui)
    ui.step(2, "Modèles")
    prets = models_step(ui, info)
    ui.step(3, "Build")
    built = build_step(ui, info)
    ui.step(4, "Calibrage")
    calibrate_step(ui, info, prets if built else {})
    ui.step(5, "Prêt")
    if built and prets:
        ready_step(ui, info, prets)
    ui._note("end")


if __name__ == "__main__":
    import sys
    from niche_bridge import PlainUI
    ui = PlainUI()
    body(ui)
    sys.exit(0)
