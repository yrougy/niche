#!/usr/bin/env python3
"""niche_bridge — le pont entre le setup (thread) et l'écran (Textual).
Même découpage que project-maya : le moteur (niche_setup.py) ne parle qu'à un objet « ui »
(say/step/ok/warn/fail/progress/ask/run) ; ici Bridge pousse tout dans une file d'événements
que le TUI pompe, et attend les réponses aux questions. PlainUI fait les mêmes méthodes en
texte brut (input/print) pour --plain ou sans terminal.
"""
from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path


def curl_meter(line: str):
    """(fraction faite, texte lisible) depuis une ligne du compteur curl, sinon None.
    Colonnes : % Total, Total, % Reçu, Reçu, % Xferd, Xferd, Dload, Upload, Ttotal, Tspend, Tleft, Vitesse."""
    f = line.split()
    if len(f) != 12 or not (f[0].isdigit() and f[2].isdigit()):
        return None
    return int(f[0]) / 100, f"{f[3]}B sur {f[1]}B, {f[11]}B/s, reste {f[10]}"


def describe(cmd: list) -> str:
    """Ce que fait une commande, pour la carte."""
    joined = " ".join(cmd)
    name = Path(cmd[0]).stem.lower()
    if name == "cmake":
        return "configuration de la build (cmake)" if "-S" in joined else "compilation de llama.cpp (cmake --build)"
    if name == "curl":
        return "téléchargement du modèle (curl, reprise auto)"
    if "calibrate.sh" in joined:
        return f"calibrage machine : {joined.split('calibrate.sh')[-1].strip()}"
    if "llama-server" in joined:
        return "serveur llama-server (Ctrl+C ou 'q' pour l'arrêter)"
    return joined[:100]


class Bridge:
    """File d'événements pour le TUI + blocage sur les questions jusqu'à réponse."""

    def __init__(self, log_path: Path | None = None):
        self.q: queue.Queue = queue.Queue()
        self.logf = open(log_path, "a", encoding="utf-8") if log_path else None
        self.child: subprocess.Popen | None = None
        self.stop_flag = threading.Event()
        self._ask = None  # (question, [choix], defaut, Event, reponse)

    def _note(self, kind: str, text=None):
        if self.logf and kind in ("say", "ok", "warn", "fail", "out", "step"):
            print(f"[{kind}] {text}", file=self.logf, flush=True)
        self.q.put((kind, text))

    def say(self, msg=""):
        self._note("say", str(msg))

    def step(self, n, title):
        self._note("step", f"{n}\t{title}")

    def ok(self, msg):
        self._note("ok", msg)

    def warn(self, msg):
        self._note("warn", msg)

    def fail(self, msg, hint=None):
        self._note("fail", msg + (f" — {hint}" if hint else ""))

    def progress(self, msg=None, frac=None):
        self._note("progress", (msg, frac))

    def ask(self, question, choices, default):
        """Bloque le thread du moteur jusqu'à Bridge.answer(label) côté écran."""
        ev = threading.Event()
        self._ask = [question, list(choices), default, ev, None]   # liste, pas tuple : answer() écrit en case 4
        self._note("ask", (question, list(choices), default))
        ev.wait()
        reponse = self._ask[4]
        self._ask = None
        self._note("ask_done")
        return reponse

    def answer(self, label):
        if self._ask:
            self._ask[4] = label
            self._ask[3].set()

    def run(self, cmd, cwd=None, env=None) -> int:
        """Exécute, diffuse la sortie ligne à ligne ; le compteur curl devient une barre."""
        if self.stop_flag.is_set():
            return -1
        self._note("card_cmd", describe(cmd))
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, errors="replace", bufsize=1)
        self.child = p
        is_curl = Path(cmd[0]).stem.lower() == "curl"
        for line in p.stdout:
            line = line.rstrip()
            if is_curl:
                m = curl_meter(line)
                if m:
                    self.progress(m[1], m[0])
                    continue
            if line:
                self._note("out", line)
        p.wait()
        self.child = None
        return p.returncode

    def stop(self):
        self.stop_flag.set()
        if self.child:
            try:
                self.child.terminate()
            except OSError:
                pass

    def finish(self):
        if self.logf:
            self.logf.close()
            self.logf = None


class PlainUI(Bridge):
    """Même API, en texte brut : rien ne s'affiche ailleurs que le terminal."""

    def _note(self, kind, text=None):
        if kind == "step":
            print(f"\n== {text.split(chr(9))[1]} ==", flush=True)
        elif kind in ("say", "out"):
            print(text, flush=True)
        elif kind == "ok":
            print(f"  ok     : {text}", flush=True)
        elif kind == "warn":
            print(f"  ATTENT : {text}", flush=True)
        elif kind == "fail":
            print(f"  ERREUR : {text}", flush=True)

    def progress(self, msg=None, frac=None):
        if msg:
            print(f"  {frac * 100:5.1f} %  {msg}" if frac is not None else f"  {msg}", flush=True)

    def ask(self, question, choices, default):
        print(f"\n{question}", flush=True)
        for i, c in enumerate(choices, start=1):
            mark = " (défaut)" if c == default else ""
            print(f"  {i}. {c}{mark}", flush=True)
        try:
            r = input("> ").strip()
        except EOFError:
            r = ""
        if not r:
            return default
        if r.isdigit() and 1 <= int(r) <= len(choices):
            return choices[int(r) - 1]
        for c in choices:
            if c.lower().startswith(r.lower()):
                return c
        return default

    def run(self, cmd, cwd=None, env=None) -> int:
        print(f"$ {' '.join(cmd)}", flush=True)
        return subprocess.run(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL).returncode


if __name__ == "__main__":
    # petit test autonome : ask/answer en file + run curl fictif
    br = Bridge()
    threading.Timer(0.2, lambda: br.answer("oui")).start()
    print(br.ask("descendre ?", ["oui", "non"], "non"))
