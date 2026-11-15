#!/usr/bin/env python3
"""Check the VLM obstacle advice before a demo: right answer? same answer every time? fast enough?

  export VERDA_INFERENCE_KEY=...
  python3 scripts/vlm_avoid_check.py                 # rendered first-person test scenes with known answers
  python3 scripts/vlm_avoid_check.py --images DIR    # your own photos, named <expected>_anything.jpg,
                                                     # expected = continue | avoid_left | avoid_right | stop
  python3 scripts/vlm_avoid_check.py --save DIR      # also write the rendered scenes to DIR to look at

Take photos from the drone's camera at the demo spot (clear, obstacle left/right, person) and run --images:
if a scene is not right 3 times out of 3, don't rely on it in the demo.
"""
import argparse
import glob
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from skyhand import vlm  # noqa: E402

W, H, HORIZON = 640, 480, 210


def _base():
    img = np.zeros((H, W, 3), np.uint8)
    for y in range(HORIZON):  # sky
        img[y] = (235 - y // 4, 200 - y // 6, 150)
    img[HORIZON:] = (95, 125, 110)  # grass/floor
    for x in range(-W, 2 * W, 80):  # perspective lines to show depth
        cv2.line(img, (W // 2, HORIZON), (x, H), (85, 110, 98), 1)
    for k in range(1, 8):
        y = HORIZON + int((H - HORIZON) * (k / 8) ** 2)
        cv2.line(img, (0, y), (W, y), (85, 110, 98), 1)
    return img


def _box(img, x0, x1, bottom, height):
    cv2.rectangle(img, (x0, bottom - height), (x1, bottom), (60, 120, 170), -1)  # cardboard brown (BGR)
    cv2.rectangle(img, (x0, bottom - height), (x1, bottom), (40, 80, 120), 3)
    mid = bottom - height // 2
    cv2.line(img, (x0, mid), (x1, mid), (150, 190, 215), 6)  # tape
    return img


def _person(img, cx, bottom, h):
    w = h // 4
    cv2.rectangle(img, (cx - w // 2, bottom - int(h * 0.82)), (cx + w // 2, bottom - int(h * 0.42)), (160, 60, 40), -1)
    cv2.rectangle(img, (cx - w // 2, bottom - int(h * 0.42)), (cx - 4, bottom), (50, 50, 60), -1)
    cv2.rectangle(img, (cx + 4, bottom - int(h * 0.42)), (cx + w // 2, bottom), (50, 50, 60), -1)
    cv2.circle(img, (cx, bottom - int(h * 0.9)), h // 10, (140, 170, 210), -1)
    return img


def _banner(img, x0, x1, bottom, height):
    """Roll-up banner on a stand with a printed person and text on it."""
    top = bottom - height
    cv2.rectangle(img, (x0 + 10, bottom - 8), (x1 - 10, bottom + 4), (60, 60, 60), -1)  # stand foot
    cv2.rectangle(img, (x0, top), (x1, bottom - 8), (245, 245, 245), -1)
    cv2.rectangle(img, (x0, top), (x1, top + height // 6), (150, 70, 20), -1)  # header band
    cv2.putText(img, "NEST", (x0 + 12, top + height // 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    _person(img, (x0 + x1) // 2, bottom - 30, int(height * 0.6))
    cv2.rectangle(img, (x0, top), (x1, bottom - 8), (90, 90, 90), 2)
    return img


SCENES = {
    "continue_clear": lambda: _base(),
    "avoid_left_box_right": lambda: _box(_base(), 290, 600, 470, 230),
    "avoid_right_box_left": lambda: _box(_base(), 40, 350, 470, 230),
    "continue_box_far": lambda: _box(_base(), 300, 340, 228, 26),
    "stop_wall": lambda: _box(_base(), 0, W, 475, 330),
    "stop_person": lambda: _person(_base(), 320, 440, 260),
    "avoid_left_banner_right": lambda: _banner(_base(), 300, 470, 460, 300),
    "avoid_right_banner_left": lambda: _banner(_base(), 170, 340, 460, 300),
}


def expected(name):
    for a in vlm.ACTIONS:
        if os.path.basename(name).startswith(a):
            return a
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--images", help="folder of your own photos named <expected>_*.jpg")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--save", help="write the rendered scenes here")
    a = p.parse_args()

    if a.images:
        scenes = {os.path.basename(f): (lambda f=f: cv2.imread(f))
                  for f in sorted(glob.glob(os.path.join(a.images, "*"))) if expected(f)}
    else:
        scenes = SCENES
    if a.save:
        os.makedirs(a.save, exist_ok=True)
        for n, make in scenes.items():
            cv2.imwrite(os.path.join(a.save, n + ".jpg"), make())

    lat, right, total = [], 0, 0
    for name, make in scenes.items():
        b64 = vlm.jpeg_b64(make())
        want, got = expected(name), []
        for _ in range(a.runs):
            t = time.time()
            try:
                r = vlm.avoid(b64)
            except Exception as e:  # noqa: BLE001
                r = {"action": f"ERROR {type(e).__name__}", "confidence": 0, "reason": str(e)[:60]}
            lat.append(time.time() - t)
            got.append(r)
        hits = sum(r["action"] == want for r in got)
        right += hits
        total += len(got)
        mark = "OK  " if hits == len(got) else "FLAKY" if hits else "WRONG"
        print(f"{mark} {name:26s} want {want:12s} got {[r['action'] for r in got]}  "
              f"conf {got[0]['confidence']}  '{got[0].get('reason', '')}'")
    lat.sort()
    print(f"\naccuracy {right}/{total}   latency median {statistics.median(lat):.2f}s  max {lat[-1]:.2f}s")


if __name__ == "__main__":
    main()
