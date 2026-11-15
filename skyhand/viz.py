"""Live view: 3D quadcopter (tilts with motion) + trail, on-screen flying instructions, air-sketch preview."""
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from .gestures import CLOSED_T, FINGER_ORDER, OPEN_T
from .hand_icons import make_cards

GESTURES = [("open", "OPEN\nfly"), ("fist", "FIST\nstop"), ("thumbs_up", "THUMB UP\ntake off"),
            ("pinky", "PINKY\nland"), ("point", "POINT\ngo there"), ("peace", "PEACE\nwaypoint"),
            ("rock", "ROCK\ndraw")]

ARM = 0.55       # drawn arm length [m] (exaggerated so the drone is visible)
ROTOR = 0.2      # drawn rotor radius [m]
VIEW = 3.5       # half-width of the follow camera [m]
MAX_TILT = math.radians(25)

def _drone_lines(st):
    """World-frame (x fwd, y right, z up) polylines of the quad: arms, rotors, nose."""
    yaw = st["yaw"]
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    right = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    v = np.array([st.get("vx", 0.0), st.get("vy", 0.0), st.get("vz", 0.0)])
    # a quad leans into its motion: nose down when going forward, right side down when sliding right
    pitch = float(np.clip(v @ fwd / 1.5, -1, 1)) * MAX_TILT
    roll = float(np.clip(v @ right / 1.5, -1, 1)) * MAX_TILT
    pos = np.array([st["x"], st["y"], st["z"]])

    def w(f, r, u=0.0):
        u = u - f * math.sin(pitch) - r * math.sin(roll)
        return pos + f * fwd + r * right + u * np.array([0, 0, 1.0])

    d = ARM / math.sqrt(2)
    tips = [(d, d), (d, -d), (-d, -d), (-d, d)]  # FR, FL, BL, BR
    arms = [np.array([w(*tips[0]), w(*tips[2])]), np.array([w(*tips[1]), w(*tips[3])])]
    th = np.linspace(0, 2 * math.pi, 24)
    rotors = [np.array([w(f + ROTOR * math.cos(t), r + ROTOR * math.sin(t), 0.04) for t in th]) for f, r in tips]
    nose = np.array([w(0, 0), w(ARM * 1.3, 0)])
    return arms, rotors, nose


def build_viz(app):
    fig = plt.figure("SkyHand - glove drone sim", figsize=(15, 8.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1])
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    right = gs[0, 1].subgridspec(5, 3, height_ratios=[1, 1, 1, 0.42, 0.9], hspace=0.5, wspace=0.08)
    ax_help = fig.add_subplot(right[3, :])
    ax_curl = fig.add_subplot(right[4, :2])
    ax2 = fig.add_subplot(right[4, 2])

    # --- 3D scene (plotted as x fwd, -y, z so "right" is on the right of the view) ---
    P = lambda a: (a[..., 0], -a[..., 1], a[..., 2])  # noqa: E731
    ax.xaxis.set_pane_color((0.93, 0.93, 0.93, 1)); ax.yaxis.set_pane_color((0.93, 0.93, 0.93, 1))
    ax.zaxis.set_pane_color((0.80, 0.90, 0.78, 1))  # ground
    ax.set_xlabel("north / start-forward [m]"); ax.set_ylabel("west [m]"); ax.set_zlabel("altitude [m]")
    ax.plot([0], [0], [0], "kX", ms=10)
    ax.text(0, 0, 0.05, " home", fontsize=8)
    (trail,) = ax.plot([], [], [], "-", color="tab:blue", lw=1.2, alpha=0.7)
    arms = [ax.plot([], [], [], "-", color="k", lw=3)[0] for _ in range(2)]
    rotors = [ax.plot([], [], [], "-", color=c, lw=2)[0] for c in ("red", "red", "dimgray", "dimgray")]
    (nose,) = ax.plot([], [], [], "-", color="red", lw=3)
    (shadow,) = ax.plot([], [], [], "o", color="gray", alpha=0.5, ms=10)
    (drop,) = ax.plot([], [], [], ":", color="gray", lw=1)
    (vel,) = ax.plot([], [], [], "-", color="tab:orange", lw=2)
    (ray,) = ax.plot([], [], [], "g--", lw=1)
    (path_ln,) = ax.plot([], [], [], "-", color="magenta", lw=2.5, alpha=0.8)
    (path_pt,) = ax.plot([], [], [], "o", color="magenta", ms=9)
    wps = ax.scatter([], [], [], c="orange", marker="^", s=60)
    banner = fig.text(0.02, 0.97, "", fontsize=15, weight="bold", va="top")
    hud = fig.text(0.02, 0.92, "", family="monospace", fontsize=9, va="top")

    # --- instructions: hand pictograms (blue arrow = what the drone does), lit green when active ---
    simple = getattr(app.args, "simple", False)
    update_cards = make_cards(fig, right, simple)
    fig.text(0.6, 0.975, "HOW TO FLY  (right hand, palm down; flat open hand = hold position)",
             fontsize=11, weight="bold", va="top")
    # --- what the glove sees: detected gesture (green box) + live finger bend bars ---
    ax_help.axis("off")
    ax_help.set_title("GESTURE THE DRONE SEES (hold 1 s for take off / land / extras)", fontsize=9, weight="bold",
                      loc="left", pad=2)
    g_boxes = {}
    shown = [g for g in GESTURES if not simple or g[0] in ("open", "fist", "thumbs_up", "peace")]
    if simple:
        new = {"thumbs_up": "THUMB UP\ntake off / land", "open": "OPEN\nfwd / back", "peace": "PEACE\nturn"}
        shown = [(k, new.get(k, lb)) for k, lb in shown]
    for i, (key, label) in enumerate(shown):
        g_boxes[key] = ax_help.text((i + 0.5) / len(shown), 0.45, label, ha="center", va="center", fontsize=8,
                                    weight="bold", transform=ax_help.transAxes,
                                    bbox=dict(boxstyle="round,pad=0.35", fc="#f7f7f7", ec="#cccccc", lw=1))
    fig.text(0.01, 0.012, "keys in terminal:  c = recalibrate (hand flat, fingers forward)   "
             "f = calibrate fist (make a tight fist, press f)   q = quit", fontsize=8, family="monospace")
    ax_curl.set_title("finger bend (left of yellow = straight, right of red = curled)", fontsize=8, pad=2)
    bars = ax_curl.barh(range(5), [0] * 5, color="tab:gray")
    ax_curl.set_yticks(range(5), FINGER_ORDER, fontsize=8); ax_curl.invert_yaxis()
    ax_curl.set_xlim(0, 1); ax_curl.tick_params(axis="x", labelsize=7)
    ax_curl.axvline(OPEN_T, color="gold", lw=2); ax_curl.axvline(CLOSED_T, color="tab:red", lw=2)

    # --- air drawing panel ---
    (sk_raw,) = ax2.plot([], [], "c-", lw=1, label="live drawing")
    (sk_fit,) = ax2.plot([], [], "m.-", lw=2, label="drone path")
    ax2.set_title("air drawing (finger ray)", fontsize=9)
    ax2.set_aspect("equal"); ax2.set_xlim(-1.2, 1.2); ax2.set_ylim(-1.2, 1.2)
    ax2.legend(loc="lower right", fontsize=7)

    hist = []

    def update(_=None):
        if not getattr(app, "running", True):
            plt.close(fig)
            return
        st = app.drone.state()
        s = app.status
        c = s["cmd"]
        mode = s["mode"]
        if st and not st.get("airborne", True):
            text, color = "ON GROUND - thumbs-up 1 s to take off", "tab:gray"
        elif mode.startswith("BRAKE"):
            text, color = "BRAKE - hovering", "tab:red"
        elif mode.startswith("DIRECT"):
            text, color = "FLYING - open hand control", "tab:green"
        else:
            text, color = mode.upper(), "tab:blue"
        banner.set_text(text); banner.set_color(color)
        conf = f"  (model {s['conf']:.0%} sure)" if "conf" in s else "  (rules)"
        lines = [f"gesture : {s['gesture']}{conf}",
                 f"cmd     : fwd {c.vx:+.2f}  right {c.vy:+.2f}  up {c.vz:+.2f} m/s  turn {c.yaw_rate:+.2f} rad/s"]
        if st:
            lines.append(f"drone   : alt {st['z']:.2f} m   heading {math.degrees(st['yaw']):+.0f} deg   "
                         f"speed {math.hypot(st.get('vx', 0), st.get('vy', 0)):.2f} m/s")
        if s["msg"]:
            lines.append(f"copilot : {s['msg'][:80]}" + ("  (thinking...)" if app.llm_busy else ""))
        hud.set_text("\n".join(lines))
        update_cards(s)
        for key, t in g_boxes.items():
            on = key == s["gesture"]
            t.get_bbox_patch().set(fc="#b6f2b0" if on else "#f7f7f7", ec="green" if on else "#cccccc",
                                   lw=3 if on else 1)
        hs = s.get("hs")
        if hs is not None:
            for bar, f in zip(bars, FINGER_ORDER):
                v = hs.curls[f]
                bar.set_width(max(v, 0.01))
                bar.set_color("tab:green" if v < OPEN_T else "tab:red" if v > CLOSED_T else "tab:orange")

        pts, here = app.path_preview() if hasattr(app, "path_preview") else (None, None)
        path_ln.set_data_3d(*P(pts)) if pts is not None else path_ln.set_data_3d([], [], [])
        path_pt.set_data_3d(*P(here[None])) if here is not None else path_pt.set_data_3d([], [], [])
        path_ln.set_linestyle("-" if here is not None else "--")
        if st:
            p = np.array([st["x"], st["y"], st["z"]])
            hist.append(p); del hist[:-3000]
            trail.set_data_3d(*P(np.array(hist)))
            a, r, n = _drone_lines(st)
            for ln, seg in zip(arms, a):
                ln.set_data_3d(*P(seg))
            for ln, seg in zip(rotors, r):
                ln.set_data_3d(*P(seg))
            nose.set_data_3d(*P(n))
            g = np.array([[st["x"], st["y"], 0.0]])
            shadow.set_data_3d(*P(g))
            drop.set_data_3d(*P(np.array([p, g[0]])))
            v = np.array([st.get("vx", 0), st.get("vy", 0), st.get("vz", 0)])
            vel.set_data_3d(*P(np.array([p, p + 0.8 * v])))
            py, pp = app.ctx.point_yaw, app.ctx.point_pitch
            d = 3.0
            ray.set_data_3d(*P(np.array([p, p + d * np.array([math.cos(pp) * math.cos(py),
                                                               math.cos(pp) * math.sin(py),
                                                               math.sin(pp)])])) if s["gesture"] == "point"
                            else ([], [], []))
            if app.ctx.waypoints:
                wp = np.array(list(app.ctx.waypoints.values()))
                wps._offsets3d = P(wp)
            # follow camera
            cx, cy, half, top = p[0], p[1], VIEW, p[2] + 1.5
            if pts is not None:  # keep the whole shape in view
                lo, hi = np.minimum(pts.min(0), p), np.maximum(pts.max(0), p)
                cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
                half, top = max(VIEW, (hi[:2] - lo[:2]).max() / 2 + 0.7), max(top, hi[2] + 0.7)
            ax.set_xlim(cx - half, cx + half); ax.set_ylim(-cy - half, -cy + half)
            ax.set_zlim(0, max(3.0, top))
        raw = app.sketch.raw
        if raw:
            q = np.array(raw) - raw[0]
            e = max(np.ptp(q, axis=0).max(), 1e-3)
            sk_raw.set_data(q[:, 0] / e, q[:, 1] / e)
        if app.ctx.sketch:
            q = np.array(app.ctx.sketch)
            sk_fit.set_data(q[:, 0], q[:, 1])

    ax.view_init(elev=22, azim=-135)  # chase-cam-ish: looking from behind-left of the start heading
    fig.subplots_adjust(left=0.0, right=0.99, top=0.93, bottom=0.03, wspace=0.02)
    return fig, update


def run_viz(app):
    fig, update = build_viz(app)
    ani = FuncAnimation(fig, update, interval=50, cache_frame_data=False)  # noqa: F841 (keep ref)
    from matplotlib.widgets import TextBox
    fig.subplots_adjust(bottom=0.1)
    box = TextBox(fig.add_axes([0.13, 0.015, 0.55, 0.045]), "Mission (Enter): ",
                  initial="", color="#fffbe6", hovercolor="#fff3c4")

    def submit(text):
        if text.strip():
            app.inbox.put(text.strip())
            box.set_val("")
    box.on_submit(submit)
    fig._skyhand_box = box  # keep a reference
    fig.text(0.70, 0.03, "e.g. fly a 2 m square / turn around / make it bigger   -> thumbs up to fly",
             fontsize=8, color="0.35", va="center")
    plt.show()
