"""Gesture-training window: the expected gesture as a 3D hand (left) next to your live glove hand (right).

Both are drawn from the same viewpoint: behind and above your right hand, fingers pointing away from you.
SPACE records the current gesture (~2.5 s), BACKSPACE redoes the previous one. After the last gesture the
neural network is trained and the result is shown.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .gesture_model import features
from .rokoko import FINGERS

UP = np.array([0.0, 1.0, 0.0])
ORDER4 = ("Index", "Middle", "Ring", "Little")

# ---------- target hand model (local frame: x = right, y = up/back of hand, z = forward) ----------
_BASE = {"Index": (-0.028, 0.0, 0.09), "Middle": (-0.009, 0.0, 0.094), "Ring": (0.01, 0.0, 0.09),
         "Little": (0.027, 0.0, 0.082)}
_LEN = {"Index": (0.045, 0.026, 0.02), "Middle": (0.05, 0.03, 0.022), "Ring": (0.046, 0.028, 0.021),
        "Little": (0.035, 0.021, 0.018), "Thumb": (0.042, 0.032, 0.027)}
_CURL = np.radians([80, 100, 60])  # MCP, PIP, DIP when fully curled
_THUMB_BASE = np.array([-0.025, -0.012, 0.025])
_THUMB_DIRS = {
    "out": [(-0.75, 0.05, 0.66), (-0.7, 0.1, 0.7), (-0.65, 0.12, 0.75)],      # sticking out sideways
    "relaxed": [(-0.45, -0.05, 0.89), (-0.3, -0.05, 0.95), (-0.2, -0.05, 0.98)],
    "tucked": [(0.2, -0.55, 0.81), (0.75, -0.55, 0.37), (0.95, -0.3, -0.1)],  # folded over the curled fingers
}
# per gesture: which fingers are straight (index..little), thumb style, how to turn the drawing for clarity
TARGETS = {
    "open": ((1, 1, 1, 1), "relaxed", 0),
    "fist": ((0, 0, 0, 0), "tucked", 0),
    "thumbs_up": ((0, 0, 0, 0), "out", -90),  # hand on its side so the thumb points up
    "pinky": ((0, 0, 0, 1), "tucked", 0),
    "point": ((1, 0, 0, 0), "tucked", 0),
    "peace": ((1, 1, 0, 0), "tucked", 0),
    "rock": ((1, 0, 0, 1), "tucked", 0),
    "call": ((0, 0, 0, 1), "out", 0),
    "gun": ((1, 0, 0, 0), "out", 0),
    "three": ((1, 1, 1, 0), "tucked", 0),
    "relax": ((0.6, 0.55, 0.5, 0.45), "relaxed", 0),  # half bent
}


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def target_hand(name):
    """Return {chain_name: Nx3 points} (plot coords x right, y forward, z up) and the palm polygon."""
    straight, thumb, roll = TARGETS[name]
    chains = {}
    for f, s in zip(ORDER4, straight):
        p = np.array(_BASE[f])
        pts, ang = [p], 0.0
        for seg, curl in zip(_LEN[f], _CURL):
            ang += (1 - s) * curl
            p = p + seg * np.array([0.0, -np.sin(ang), np.cos(ang)])
            pts.append(p)
        chains[f] = np.array(pts)
    p = _THUMB_BASE.copy()
    pts = [p]
    for seg, d in zip(_LEN["Thumb"], _THUMB_DIRS[thumb]):
        p = p + seg * _unit(d)
        pts.append(p)
    chains["Thumb"] = np.array(pts)
    palm = np.array([(-0.022, 0, 0), _BASE["Index"], _BASE["Middle"], _BASE["Ring"], _BASE["Little"], (0.022, 0, 0)])
    if roll:
        c, s_ = np.cos(np.radians(roll)), np.sin(np.radians(roll))
        R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])  # about the forward (z) axis
        chains = {k: v @ R.T for k, v in chains.items()}
        palm = palm @ R.T
    sw = lambda a: a[:, [0, 2, 1]]  # noqa: E731  (x, y_up, z_fwd) -> plot (x, fwd, up)
    return {k: sw(v) for k, v in chains.items()}, sw(palm)


# ---------- live hand from glove bones (Studio Y-up world), yaw removed so forward = away from you ----------
def live_hand(frame):
    b = frame.bones
    need = ["Hand"] + [f + j for f in FINGERS for j in ("Proximal", "Medial", "Distal")]
    if any(n not in b for n in need):
        return None
    o = b["Hand"].position
    fwd = b["MiddleProximal"].position - o
    fwd = fwd - UP * (fwd @ UP)
    if np.linalg.norm(fwd) < 1e-6:
        return None
    fwd = _unit(fwd)
    right = np.cross(fwd, UP)
    to_plot = lambda p: np.array([(p - o) @ right, (p - o) @ fwd, (p - o) @ UP])  # noqa: E731
    chains = {}
    for f in FINGERS:
        ps = [b[f + j].position for j in ("Proximal", "Medial", "Distal")]
        ps.append(ps[-1] + 0.8 * (ps[-1] - ps[-2]))  # fingertip (not streamed): extend the last bone
        chains[f] = np.array([to_plot(p) for p in ps])
    palm = np.array([to_plot(o)] + [chains[f][0] for f in ORDER4] + [to_plot(o)])
    return chains, palm


COLORS = {"Thumb": "tab:purple", "Index": "tab:red", "Middle": "tab:orange", "Ring": "tab:green",
          "Little": "tab:blue"}


def _setup_ax(ax, title, span):
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlim(-span, span); ax.set_ylim(-span * 0.4, span * 1.6); ax.set_zlim(-span, span)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel("right ->", fontsize=8); ax.set_ylabel("forward ->", fontsize=8); ax.set_zlabel("up", fontsize=8)
    ax.view_init(elev=28, azim=-115)
    lines = {f: ax.plot([], [], [], "-o", color=COLORS[f], lw=4, ms=4)[0] for f in FINGERS}
    (palm,) = ax.plot([], [], [], "-", color="#8d5524", lw=3)
    return lines, palm


def _draw(lines, palm_line, chains, palm):
    for f, ln in lines.items():
        p = chains[f]
        ln.set_data_3d(p[:, 0], p[:, 1], p[:, 2])
    palm_line.set_data_3d(palm[:, 0], palm[:, 1], palm[:, 2])


def run_gui(rx, a, plan, gtext, ptext, train_and_report):
    """plan: [(pose, gesture)] in recording order."""
    steps = [(p, g, gtext[g]) for p, g in plan]
    st = {"i": 0, "phase": "wait", "t": 0.0, "xs": [], "last": None, "result": None}
    data = []  # (pose, label, features)

    fig = plt.figure("SkyHand - teach your gestures", figsize=(13, 7.5))
    ax_t = fig.add_subplot(1, 2, 1, projection="3d")
    ax_l = fig.add_subplot(1, 2, 2, projection="3d")
    t_lines, t_palm = _setup_ax(ax_t, "COPY THIS", 0.1)
    l_lines, l_palm = _setup_ax(ax_l, "YOUR HAND (live)", 0.1)
    head = fig.text(0.5, 0.965, "", ha="center", va="top", fontsize=16, weight="bold")
    sub = fig.text(0.5, 0.915, "", ha="center", va="top", fontsize=11)
    foot = fig.text(0.5, 0.1, "", ha="center", va="top", fontsize=12, family="monospace")
    fig.text(0.5, 0.025, "colors: thumb=purple index=red middle=orange ring=green little=blue   |   "
             "same finger shape in every POSE   |   SPACE record   BACKSPACE redo   Q quit",
             ha="center", fontsize=9)

    def show_target():
        _, g, text = steps[st["i"]]
        chains, palm = target_hand(g)
        _draw(t_lines, t_palm, chains, palm)
        pose = steps[st["i"]][0]
        head.set_text(f"{st['i'] + 1}/{len(steps)}  {g.upper().replace('_', '-')}")
        sub.set_text(f"{text}\nPOSE: {ptext[pose]}")

    def on_key(ev):
        if st["phase"] in ("train", "done"):
            return
        if ev.key == " " and st["phase"] == "wait":
            st["phase"], st["t"], st["xs"] = "countdown", time.time(), []
        elif ev.key == "backspace" and st["phase"] == "wait" and st["i"] > 0:
            st["i"] -= 1
            r, g, _ = steps[st["i"]]
            data[:] = [d for d in data if not (d[0] == r and d[1] == g)]
            show_target()

    def update(_):
        f = rx.latest()
        live = live_hand(f) if f is not None else None
        if live:
            _draw(l_lines, l_palm, *live)
        now = time.time()
        if st["phase"] == "wait":
            foot.set_text("make the gesture like the left hand, then press SPACE" if live
                          else "NO GLOVE DATA - is the glove on and the bridge running?")
            foot.set_color("black" if live else "red")
        elif st["phase"] == "countdown":
            left = 1.0 - (now - st["t"])
            foot.set_text(f"get ready... {left:.1f}"); foot.set_color("tab:orange")
            if left <= 0:
                st["phase"], st["t"] = "rec", now
        elif st["phase"] == "rec":
            if f is not None and f.stamp != st["last"]:
                st["last"] = f.stamp
                x = features(f, imu=True) if f.raw else None
                x = x if x is not None else features(f)
                if x is not None:
                    st["xs"].append(x)
            done = (now - st["t"]) / a.seconds
            bar = "#" * int(30 * min(done, 1)) + "-" * (30 - int(30 * min(done, 1)))
            foot.set_text(f"RECORDING - hold it ({ptext[steps[st['i']][0]]})  [{bar}]"); foot.set_color("tab:red")
            if done >= 1:
                r, g, _ = steps[st["i"]]
                data.extend((r, g, x) for x in st["xs"])
                st["i"] += 1
                if st["i"] < len(steps):
                    st["phase"] = "wait"
                    show_target()
                else:
                    st["phase"] = "train"  # trained on the next frame, so this text is drawn first
                    head.set_text("training the neural network...")
                    foot.set_text("")
        elif st["phase"] == "train":
            st["phase"] = "done"
            report = train_and_report(data, a.out)
            head.set_text("DONE - your gesture model is ready (close this window)")
            sub.set_text("")
            foot.set_text(report); foot.set_color("black"); foot.set_fontsize(12); foot.set_multialignment("left")
            foot.set_position((0.5, 0.82))
            ax_t.set_visible(False); ax_l.set_visible(False)

    fig.canvas.mpl_connect("key_press_event", on_key)
    show_target()
    fig.subplots_adjust(left=0.02, right=0.98, top=0.83, bottom=0.12, wspace=0.05)
    ani = FuncAnimation(fig, update, interval=33, cache_frame_data=False)  # noqa: F841 (keep ref)
    plt.show()
