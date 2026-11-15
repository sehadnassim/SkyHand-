#!/usr/bin/env python3
"""Guided glove check: walks you through every SkyHand gesture and shows what is recognized.

    ./fly.sh --check        (starts driver + bridge, then this)

No model training is involved; gestures are rules on finger curls (gestures.py). This tells you
whether those rules fit your hand, and suggests OPEN_T / CLOSED_T if they don't.
"""
import math
import os
import statistics as st
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from skyhand.gestures import CLOSED_T, FINGER_ORDER, OPEN_T, classify  # noqa: E402
from skyhand.hand import DEFAULT_CURL_RANGE, HandModel  # noqa: E402
from skyhand.sdk_convert import SDK_FINGER_RANGE  # noqa: E402
from skyhand.rokoko import RokokoReceiver  # noqa: E402

STEPS = [  # (instruction, expected gesture or None, check)
    ("FIST: curl all fingers tightly", "fist", "fist"),
    ("THUMBS-UP: fist with the thumb sticking up", "thumbs_up", None),
    ("POINT: index finger straight, others curled", "point", None),
    ("PEACE: index + middle straight, others curled", "peace", None),
    ("ROCK: index + little straight, others curled", "rock", None),
    ("PINKY: only the little finger straight", "pinky", None),
    ("OPEN hand, tilt FINGERS DOWN ~30 deg (= fly forward)", "open", ("pitch", -1)),
    ("OPEN hand, ROLL to the RIGHT ~30 deg (= slide right)", "open", ("roll", +1)),
    ("OPEN hand, TURN hand to the RIGHT ~40 deg (= yaw right)", "open", ("yaw", +1)),
    ("OPEN hand, THUMB stuck OUT away from the hand (= climb)", "open", ("thumb", -1)),
]


def sample(rx, seconds=1.0):
    frames, t = [], time.time()
    while time.time() - t < seconds:
        f = rx.latest()
        if f is not None and (not frames or f is not frames[-1]):
            frames.append(f)
        time.sleep(0.01)
    return frames


def wait_for_glove(rx):
    for _ in range(50):
        if rx.latest() is not None:
            return
        time.sleep(0.1)
    sys.exit("no glove data on UDP 14043 (is scripts/sdk_bridge.py running?)")


def main():
    rx = RokokoReceiver(14043)
    wait_for_glove(rx)
    model = HandModel({**DEFAULT_CURL_RANGE, **{f: SDK_FINGER_RANGE for f in FINGER_ORDER if f != "Thumb"}})
    input("\nCalibration: hand FLAT, palm down, fingers straight toward the screen. Hold it and press Enter...")
    model.calibrate(sample(rx, 0.5)[-1])
    print("  calibrated\n")

    results, curls_by_step = [], {}
    for i, (text, expected, check) in enumerate(STEPS, 1):
        input(f"[{i}/{len(STEPS)}] {text}. Hold it and press Enter...")
        frames = sample(rx)
        if check == "fist":
            model.calibrate_fist(frames[-1])
        states = [model.update(f) for f in frames]
        curls = {k: st.median(s.curls[k] for s in states) for k in FINGER_ORDER}
        gestures = [classify(s.curls).value for s in states]
        got = max(set(gestures), key=gestures.count)
        ok = got == expected
        line = f"  seen: {got:<10} curls " + " ".join(f"{k[:2]}={v:.2f}" for k, v in curls.items())
        if isinstance(check, tuple):
            name, sign = check
            val = st.median(getattr(s, name) for s in states)
            shown = f"{math.degrees(val):+.0f} deg" if name != "thumb" else f"{val:+.2f}"
            dir_ok = val * sign > (0.15 if name != "thumb" else 0.1)
            line += f" | {name} {shown} {'OK' if dir_ok else 'WRONG/TOO SMALL'}"
            ok = ok and dir_ok
        print(("  OK  " if ok else "  FAIL") + line[1:])
        results.append((text.split(":")[0], ok))
        curls_by_step[text.split(":")[0]] = curls

    print("\nSummary:", ", ".join(f"{n} {'ok' if ok else 'FAIL'}" for n, ok in results))
    straight = [v for c in curls_by_step.values() for v in c.values() if v < 0.5]
    bent = [v for c in curls_by_step.values() for v in c.values() if v >= 0.5]
    if straight and bent:
        print(f"Your straight fingers reached up to {max(straight):.2f}, curled ones down to {min(bent):.2f} "
              f"(current OPEN_T={OPEN_T}, CLOSED_T={CLOSED_T} in skyhand/gestures.py).")
    if all(ok for _, ok in results):
        print("Everything recognized: you are good to fly (./fly.sh --sim).")
    else:
        print("Retry failed ones more exaggerated; if a finger is always in between, adjust OPEN_T/CLOSED_T.")
    rx.close()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
