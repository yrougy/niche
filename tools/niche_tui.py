#!/usr/bin/env python3
"""niche_tui — l'écran du setup niche (« la bête doit tenir dans le clapier »), dans l'esprit
de la TUI de project-maya : les étapes
en onglets en haut, une carte pour ce qui tourne maintenant (avec sa barre de progression),
la sortie sous forme de journal, les questions comme des menus, les touches dans le bas.
Le moteur (niche_setup.body) tourne dans son thread et ne parle qu'au Bridge ; l'écran
 pompe la file d'événements. Ctrl+C arrête l'enfant puis quitte ; la sortie complète
 reste dans niche-setup.log. --plain : sans écran (niche_setup en texte brut).
"""
from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.timer import Timer
from textual.widgets import OptionList, RichLog, Static
from textual.widgets.option_list import Option

ROOT = Path(__file__).resolve().parent.parent      # le dépôt niche (coin haut droit)

STEPS = {1: "Ordinateur", 2: "Modèle", 3: "Build", 4: "Calibrage", 5: "Prêt"}
SPLASH = "Niche — installeur du port cache-MoE pour llama.cpp"

CSS = """
    Screen { background: #0d1117; }
    #top { height: 3; border: solid #3d5a7a; border-title-align: center; padding: 0 1; }
    #tabs { width: 1fr; text-wrap: nowrap; text-overflow: ellipsis; }
    #where { width: auto; max-width: 45%; padding-left: 2; color: #7f97a9;
             text-style: italic; text-wrap: nowrap; }
    #main { height: 1fr; border: solid #3d5a7a; border-subtitle-align: center; padding: 1 2 0 2; }
    #page { height: 1fr; }
    #splash { width: 100%; height: auto; margin-bottom: 1; text-align: center; color: #75b6ff; }
    #card { width: 76; max-width: 100%; height: auto; border: solid #33424f;
            padding: 0 1; margin-bottom: 1; border-title-align: center; }
    #output { width: 100%; max-width: 120; height: 1fr; min-height: 3; border: solid #33424f;
              padding: 0 1; border-title-align: center; scrollbar-size-vertical: 1; }
    #ask { width: 100%; max-width: 120; height: auto; max-height: 70%; margin-top: 1;
           border: solid #3d5a7a; border-title-align: center; padding: 0 1;
           display: none; overflow-y: auto; scrollbar-size-vertical: 1; }
    #ask.on { display: block; }
    #intro { margin-bottom: 1; }
    #options, #options:focus { height: auto; max-height: 12; border: none; padding: 0;
                               background: #0d1117; }
    #options > .option-list--option-highlighted,
    #options:focus > .option-list--option-highlighted { background: #21415e; color: #d9ecff; text-style: bold; }
    #keys { height: 1; color: #7f97a9; }
"""
BINDINGS = [Binding("ctrl+c", "stop", show=False, priority=True),
            Binding("ctrl+q", "stop", show=False, priority=True)]


class NicheApp(App):
    CSS = CSS
    BINDINGS = BINDINGS
    ENABLE_COMMAND_PALETTE = False
    TITLE = "Niche · cache-MoE llama.cpp"

    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge
        self.state = {n: "wait" for n in STEPS}
        self.cur = 1
        self.action = ""
        self.prog: tuple | None = None      # (fraction, texte)
        self.ask_labels: list = []
        self._timer: Timer | None = None

    def compose(self) -> ComposeResult:
        with Vertical():
            with Vertical(id="top"):
                yield Static("", id="tabs")
            with Vertical(id="main"):
                with Vertical(id="page"):
                    yield Static(SPLASH, id="splash")
                    yield Static("", id="card")
                    yield RichLog(id="output", markup=False, wrap=True, max_lines=4000)
                    with Vertical(id="ask"):
                        yield Static("", id="intro")
                        yield OptionList(id="options")
                yield Static("↑↓ choix · Entrée valider · Ctrl+C arrêter puis quitter · "
                             "la sortie complète : tools/niche-setup.log", id="keys")

    def on_mount(self) -> None:
        self.query_one("#card").border_title = "en cours"
        self.query_one("#output").border_title = "sortie"
        self.query_one("#ask").border_title = "question"
        self._draw_tabs()
        self._timer = self.set_interval(1 / 12, self._poll)

    # --------------------------------------------------------- dessins (redessinés à chaque event)
    def _draw_tabs(self) -> None:
        t = Text()
        for n, name in STEPS.items():
            mark = {"done": "✓ ", "run": "▸ ", "wait": "· "}[self.state[n]]
            t.append(f" {mark}{name}", style="bold" if self.cur == n else "dim")
        self.query_one("#tabs", Static).update(t)

    def _draw_card(self) -> None:
        t = Text()
        t.append(f"étape {self.cur} — {STEPS[self.cur]}\n", style="bold #75b6ff")
        t.append(self.action or "(rien de lancé)\n")
        if self.prog is not None:
            frac, msg = self.prog
            if frac is not None:
                n = int(frac * 40)
                t.append(f"[{'█' * n}{'░' * (40 - n)}] {frac * 100:5.1f} %  {msg or ''}\n")
            else:
                t.append(f"{msg or ''}\n")
        self.query_one("#card", Static).update(t)

    # --------------------------------------------------------- pompe à événements du moteur
    def _poll(self) -> None:
        try:
            while True:
                kind, text = self.bridge.q.get_nowait()
                try:
                    self._apply(kind, text)
                except Exception:
                    import traceback
                    self.query_one("#output", RichLog).write(traceback.format_exc())
                    self.bridge.stop()
                    self._timer and self._timer.stop()
                    self.exit()
                    return
        except queue.Empty:
            pass

    def _apply(self, kind, text) -> None:
        out = self.query_one("#output", RichLog)
        if kind == "step":
            n, title = text.split("\t")
            n = int(n)
            for i in STEPS:
                if i < n:
                    self.state[i] = "done"
            self.state[n], self.cur = "run", n
            out.write(Text(f"\n== étape {n} : {title} ==", style="bold"))
            self._draw_tabs()
        elif kind == "say":
            out.write(text)
        elif kind == "ok":
            out.write(Text(f"ok     : {text}", style="green"))
        elif kind == "warn":
            out.write(Text(f"ATTENT : {text}", style="orange1"))
        elif kind == "fail":
            out.write(Text(f"ERREUR : {text}", style="bold red"))
        elif kind == "out":
            out.write(text)
        elif kind == "card_cmd":
            self.action = text
        elif kind == "progress":
            msg, frac = text
            self.prog = (frac, msg) if frac is not None else (self.prog[0] if self.prog else None, msg)
        elif kind == "ask":
            q, choices, default = text
            self.ask_labels = choices
            ol = self.query_one("#options", OptionList)
            ol.clear_options()
            ol.add_options([Option(f"{c}{'   [défaut]' if c == default else ''}", id=i)
                            for i, c in enumerate(choices)])
            ol.highlighted = choices.index(default) if default in choices else 0
            self.query_one("#intro", Static).update(q)
            self.query_one("#ask", Vertical).add_class("on")
            ol.focus()
        elif kind == "ask_done":
            self.query_one("#ask", Vertical).remove_class("on")
        elif kind == "end":
            self._timer and self._timer.stop()
            self.exit()
        self._draw_tabs()
        self._draw_card()

    # --------------------------------------------------------- réponses aux menus
    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        label = str(event.option.prompt).split("   [défaut]")[0]
        self.query_one("#ask", Vertical).remove_class("on")
        self.bridge.answer(label)

    def action_stop(self) -> None:
        self.bridge.stop()
        self.exit()


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from niche_bridge import Bridge, PlainUI
    import niche_setup

    plain = "--plain" in sys.argv or not sys.stdin.isatty()
    if plain:
        ui = PlainUI()
        try:
            niche_setup.body(ui)
        except KeyboardInterrupt:
            print("\ninterrompu")
            return 1
        return 0

    bridge = Bridge(ROOT / "tools" / "niche-setup.log")
    app = NicheApp(bridge)
    threading.Thread(target=niche_setup.body, args=(bridge,), daemon=True).start()
    try:
        app.run()          # le thread meurt avec le process ; stop_flag évite les relances
    finally:
        bridge.stop()
        bridge.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
