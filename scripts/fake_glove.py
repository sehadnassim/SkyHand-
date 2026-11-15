#!/usr/bin/env python3
"""Fake Rokoko glove for testing SkyHand without hardware.

  python scripts/fake_glove.py demo         # scripted demo: takeoff, fly, brake, point, draw a circle, land
  python scripts/fake_glove.py keys         # keyboard: arrows/WASD tilt, QE yaw, 1-9 hand shapes
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from skyhand.fake_glove import SHAPES, FakeGlove, circle_drawing  # noqa: E402

R = math.radians

# (duration s, shape, pitch, roll, yaw, label)
DEMO = [
    (2.0, "open", 0, 0, 0, "calibration: flat hand"),
    (1.6, "thumbs_up", 0, 0, 0, "thumbs-up -> take off"),
    (3.5, "open", 0, 0, 0, "hover"),
    (3.0, "open", R(-30), 0, 0, "tilt fingers down -> forward"),
    (2.5, "open", 0, R(30), 0, "roll right -> slide right"),
    (2.5, "open", 0, 0, R(40), "turn hand -> yaw right"),
    (2.0, "climb", 0, 0, 0, "thumb out -> climb"),
    (1.0, "fist", 0, 0, 0, "fist -> brake"),
    (1.4, "peace", 0, 0, 0, "peace -> waypoint 1"),
    (0.5, "open", 0, 0, 0, ""),
    (1.4, "point", 0, 0, R(-45), "point left (hold) -> fly there"),
    (6.0, "open", 0, 0, 0, "(flying to pointed spot)"),
    (1.4, "rock", 0, 0, 0, "rock -> start drawing"),
    ("draw", "point", 0, 0, 0, "draw a circle with the index finger"),
    (1.4, "rock", 0, 0, 0, "rock -> stop: drone flies the circle"),
    (16.0, "open", 0, 0, 0, "(flying the drawing)"),
    (1.6, "pinky", 0, 0, 0, "pinky -> land"),
    (4.0, "open", 0, 0, 0, "done"),
]


def demo(g):
    for dur, shape, p, r, y, label in DEMO:
        print(f"[fake glove] {label}", flush=True)
        t0 = time.time()
        total = 4.0 if dur == "draw" else dur
        while (t := time.time() - t0) < total:
            extra = circle_drawing(t) if dur == "draw" else {}
            g.send(pitch=p, roll=r, yaw=y, curls=SHAPES[shape], **extra)
            time.sleep(1 / 60)


def keys(g):
    import curses

    def ui(scr):
        scr.nodelay(True)
        names = list(SHAPES)
        st = {"p": 0.0, "r": 0.0, "y": 0.0, "shape": "open"}
        while True:
            k = scr.getch()
            if k in (ord("x"), 27):
                return
            step = R(5)
            if k in (curses.KEY_UP, ord("w")): st["p"] -= step
            if k in (curses.KEY_DOWN, ord("s")): st["p"] += step
            if k in (curses.KEY_RIGHT, ord("d")): st["r"] += step
            if k in (curses.KEY_LEFT, ord("a")): st["r"] -= step
            if k == ord("e"): st["y"] += step
            if k == ord("q"): st["y"] -= step
            if k == ord(" "): st.update(p=0.0, r=0.0, y=0.0)
            if ord("1") <= k <= ord("9") and k - ord("1") < len(names):
                st["shape"] = names[k - ord("1")]
            g.send(pitch=st["p"], roll=st["r"], yaw=st["y"], curls=SHAPES[st["shape"]])
            scr.erase()
            scr.addstr(0, 0, "Fake Rokoko glove | w/s pitch, a/d roll, q/e yaw, space level, x quit")
            scr.addstr(1, 0, "shapes: " + "  ".join(f"{i + 1}={n}" for i, n in enumerate(names)))
            scr.addstr(3, 0, f"pitch {math.degrees(st['p']):+5.0f}  roll {math.degrees(st['r']):+5.0f}  "
                             f"yaw {math.degrees(st['y']):+5.0f}  shape {st['shape']}")
            time.sleep(1 / 60)

    curses.wrapper(ui)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    port = int(os.environ.get("ROKOKO_PORT", 14043))
    (demo if mode == "demo" else keys)(FakeGlove(port=port))
