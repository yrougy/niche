#!/usr/bin/env python3
"""test_niche_tui — sans écran (App.run_test) : rendu des événements du bridge ET le vrai
chemin de question Bridge.ask() (thread moteur bloqué → réponse clavier → déblocage).
Lancer : tools/.venv/bin/python tools/test_niche_tui.py
(Le 18/10, un _ask en tuple faisait planter answer() au premier menu ; et highlight
s'appelle « highlighted » en Textual 8.2.8 — ces deux pièges sont couverts ici.)
"""
import asyncio
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from niche_bridge import Bridge            # noqa: E402
from niche_tui import NicheApp            # noqa: E402
from textual.containers import Vertical     # noqa: E402


async def scenario_events(pilot, br, app):
    br.step(1, "Ordinateur"); br.ok("RAM 15 GiB"); br.warn("un souci"); br.say("ligne")
    await pilot.pause(0.4)
    assert app.state[1] == "run", app.state
    br._note("card_cmd", "curl -L -C - ...")
    br.progress("6.2MB sur 11.5GB", 0.55)
    await pilot.pause(0.3)
    assert app.prog[0] == 0.55
    print("  evts ok")


async def scenario_ask(pilot, br, app):
    """Le chemin réel : le moteur appelle Bridge.ask (bloque), l'utilisateur répond au clavier."""
    v = {}
    threading.Thread(target=lambda: v.update(
        a=br.ask("descendre ?", ["télécharger", "passer"], "passer")), daemon=True).start()
    await pilot.pause(0.4)
    assert app.query_one("#ask", Vertical).has_class("on")
    assert app.query_one("#options").highlighted == 1        # le défaut est pré-surligné
    await pilot.press("enter"); await pilot.pause(0.4)       # Entrée sèche = choix par défaut
    assert v.get("a") == "passer", v
    threading.Thread(target=lambda: v.update(
        b=br.ask("encore ?", ["a", "b"], "a")), daemon=True).start()
    await pilot.pause(0.4)
    await pilot.press("down"); await pilot.pause(0.2)
    await pilot.press("enter"); await pilot.pause(0.4)
    assert v.get("b") == "b", v
    assert not app.query_one("#ask", Vertical).has_class("on")
    print("  ask ok")


async def main():
    br = Bridge()
    app = NicheApp(br)
    async with app.run_test(size=(120, 40)) as pilot:
        await scenario_events(pilot, br, app)
        await scenario_ask(pilot, br, app)
        br._note("end")
        await pilot.pause(0.5)
    print("test_niche_tui : OK")


if __name__ == "__main__":
    asyncio.run(main())
