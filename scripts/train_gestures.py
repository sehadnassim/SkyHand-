#!/usr/bin/env python3
"""Teach SkyHand YOUR gestures: guided recording + training, with automatic gesture selection.

    ./fly.sh --train              full sweep: every candidate gesture in every hand pose (~5 min)
    ./fly.sh --train --quick      2 poses only (~2 min)
    ./fly.sh --train --add        load earlier recordings, ask only for the weak (pose, gesture) combos

A window shows each gesture as a 3D hand next to your live hand (--text = terminal prompts).
After recording (skyhand/gesture_select.py):
  - each pose is left out once and predicted by a model trained on the other poses (honest accuracy),
  - frames that clearly look like another gesture are dropped (transitions, wrong shape),
  - gestures that get mixed up are dropped, the ones the app needs are always kept,
  - a PCA plot of the clusters is saved next to the dataset.
The final model is saved to models/gestures.npz, the raw data to recordings/.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from skyhand import gesture_select as S  # noqa: E402
from skyhand.gesture_model import N_FEATURES, N_IMU, GestureNet, features  # noqa: E402
from skyhand.rokoko import RokokoReceiver  # noqa: E402

GESTURES = [(n, t) for n, _, t in S.CANDIDATES]
GTEXT = dict(GESTURES)
PTEXT = dict(S.CONDITIONS)


def record(rx, seconds):
    xs, last = [], None
    t_end = time.time() + seconds
    while time.time() < t_end:
        f = rx.latest()
        if f is not None and f.stamp != last:
            last = f.stamp
            x = features(f, imu=True) if f.raw else None
            x = x if x is not None else features(f)
            if x is not None:
                xs.append(x)
        time.sleep(1 / 60)
    return xs


def load_old(n_feat):
    """Earlier datasets with the same features. Old files had rounds 1/2 instead of poses."""
    out = []
    for f in sorted(glob.glob("recordings/gesture_data_*.npz")):
        d = np.load(f, allow_pickle=True)
        if d["X"].shape[1] != n_feat:
            continue
        groups = d["groups"] if "groups" in d.files else np.array([f"old_r{r}" for r in d["rounds"]])
        out += list(zip(groups, d["labels"], d["X"]))
    return out


def plan_steps(a):
    """[(pose, gesture)] to record, grouped by pose so you set up each pose once."""
    names = [n for n, _ in GESTURES if not a.gestures or n in a.gestures.split(",")]
    if a.add:
        old = load_old(N_FEATURES + N_IMU)
        if old:
            groups, labels, X = (np.array(v) for v in zip(*old))
            classes = sorted(set(labels) | set(names))
            P = S.oof_proba(X, labels, groups, classes)
            cells = [(g, c) for g, c, _ in S.weak_cells(labels, groups, P, classes) if c in names]
            print(f"{len(old)} old frames. weak or missing combos: {len(cells)}")
            for g, c, acc in S.weak_cells(labels, groups, P, classes):
                if c in names:
                    print(f"  {g:<9} {c:<10} " + ("not recorded" if acc is None else f"{acc:.0%} correct"))
            if cells:
                return cells
            print("nothing weak, doing a quick sweep for extra variety")
    poses = [p for p, _ in S.CONDITIONS if not a.quick or p in S.QUICK]
    return [(p, n) for p in poses for n in names]


def train_and_report(data, out, log=print, add=False):
    """data: list of (pose, label, features). Saves dataset + model + cluster plot; returns the report."""
    imu = all(len(d[2]) > N_FEATURES for d in data)  # IMU only if EVERY sample has it
    groups = np.array([d[0] for d in data])
    labels = np.array([d[1] for d in data])
    X = np.array([d[2] if imu else d[2][:N_FEATURES] for d in data])
    old = load_old(X.shape[1]) if add else []  # before saving, so the new file isn't loaded twice
    os.makedirs("recordings", exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    ds = f"recordings/gesture_data_{stamp}.npz"
    np.savez(ds, X=X, labels=labels, groups=groups)
    lines = [f"new data: {len(X)} frames -> {ds}"]
    if old:
        og, ol, oX = (np.array(v) for v in zip(*old))
        groups, labels, X = np.r_[og, groups], np.r_[ol, labels], np.r_[oX, X]
        lines[0] += f"  + {len(old)} old frames"
    lines[0] += "  (finger rotations + IMU)" if imu else "  (finger rotations only: restart the SDK bridge for IMU)"
    classes = sorted(set(labels))

    if len(set(groups)) > 1:
        P = S.oof_proba(X, labels, groups, classes)
        keep = S.clean(labels, P, classes)
        lines.append(f"dropped {int((~keep).sum())} frames that looked like another gesture")
        C = S.confusion(labels[keep], P[keep], classes)
        kept, dropped = S.pick(C, classes)
        acc = float(np.mean(P[keep].argmax(1) == [classes.index(c) for c in labels[keep]]))
        lines.append(f"pose left out, all gestures: {acc:.0%} correct")
        lines.append("kept:    " + ", ".join(c for c in kept))
        for c, (like, mix) in dropped.items():
            lines.append(f"dropped: {c:<10} mixed up with {like} {mix:.0%}")
        for c in S.REQUIRED:
            i = classes.index(c) if c in classes else None
            if i is not None and (C[i].sum() - C[i, i]) > 0.05:
                lines.append(f"warning: {c} is needed but only {C[i, i]:.0%} right. Record more with --add")
        weak = S.weak_cells(labels[keep], groups[keep], P[keep], classes)
        weak = [w for w in weak if w[1] in kept and w[2] is not None]
        if weak:
            lines.append("weak spots (./fly.sh --train --add asks for these): " +
                         ", ".join(f"{c} in {g} {a_:.0%}" for g, c, a_ in weak[:6]))
        _plot(X[keep], labels[keep], kept, f"recordings/gesture_clusters_{stamp}.png")
        lines.append(f"cluster plot -> recordings/gesture_clusters_{stamp}.png")
        m = keep & np.isin(labels, kept)
        X, labels = X[m], labels[m]
    else:
        kept = classes
    net = GestureNet(kept, n_in=X.shape[1]).fit(X, labels)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    net.save(out)
    lines.append(f"model saved -> {out}  ({len(kept)} classes, training accuracy {net.accuracy(X, labels):.0%})")
    lines.append("now fly with:  ./fly.sh --sim")
    text = "\n".join(lines)
    log(text)
    return text


def _plot(X, labels, kept, path):
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    Y, var = S.pca_2d(X)
    fig, ax = plt.subplots(figsize=(7, 6))
    for c in sorted(set(labels)):
        m = labels == c
        ax.scatter(Y[m, 0], Y[m, 1], s=6, alpha=0.5, label=c + ("" if c in kept else " (dropped)"),
                   marker="o" if c in kept else "x")
        ax.annotate(c, Y[m].mean(0), fontsize=9, weight="bold")
    ax.set_title(f"gesture clusters, PCA of features ({var:.0%} of variance shown)", fontsize=10)
    ax.legend(fontsize=7, markerscale=2)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def run_text(rx, a, steps):
    print("\nI'll ask for each gesture. Press Enter, then HOLD it for a few seconds.")
    data, pose = [], None
    for g, label in steps:
        if g != pose:
            pose = g
            input(f"\n=== pose: {PTEXT[g]} ===  Enter to start ")
        input(f"  {GTEXT[label]}\n    make it ({PTEXT[g]}), then press Enter... ")
        time.sleep(0.4)  # let the hand settle after pressing Enter
        xs = record(rx, a.seconds)
        print(f"    got {len(xs)} samples")
        data += [(g, label, x) for x in xs]
    train_and_report(data, a.out, add=a.add)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=14043)
    ap.add_argument("--seconds", type=float, default=2.5, help="per gesture per pose")
    ap.add_argument("--quick", action="store_true", help="only the flat and tilt poses")
    ap.add_argument("--add", action="store_true", help="keep old recordings, record only the weak combos")
    ap.add_argument("--gestures", default="", help="comma list to record, default: all candidates")
    ap.add_argument("--out", default="models/gestures.npz")
    ap.add_argument("--text", action="store_true", help="terminal prompts instead of the 3D window")
    a = ap.parse_args()

    rx = RokokoReceiver(a.port)
    for _ in range(50):
        if rx.latest() is not None:
            break
        time.sleep(0.1)
    else:
        sys.exit("no glove data (is scripts/sdk_bridge.py running?)")
    try:
        steps = plan_steps(a)
        print(f"{len(steps)} recordings of {a.seconds:g} s")
        if a.text:
            run_text(rx, a, steps)
        else:
            from skyhand.train_gui import run_gui
            run_gui(rx, a, steps, GTEXT, PTEXT, lambda d, out: train_and_report(d, out, add=a.add))
    finally:
        rx.close()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\ncancelled, nothing saved")
