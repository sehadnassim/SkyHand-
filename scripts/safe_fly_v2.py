#!/usr/bin/env python3
"""Safe glove test, version 2: take off / land + very gentle forward-back and turn with the hand. No LLM.

  ./fly.sh --safe                          # starts the glove driver, then this
  python3 scripts/safe_fly_v2.py --dry-run # nothing is sent; prints what would be

Limits that cannot be changed from the GUI or the glove:
  - only forward/back (default max 0.4 m/s, hard cap 0.4) and turn (0.35 rad/s cap). Sideways is always 0; up/down only via the point gesture angled up or down.
  - velocity is 0 unless: airborne flag set by TAKE OFF, 5 s settle passed, "Hand control" on (auto after settle),
    fresh glove data (<0.15 s), and an open hand (forward/back) or peace sign (turn) held steady 0.25 s.
  - uncertain gesture or glove dropout = zero velocity (hover), and it resumes by itself.
  - acceleration is slew-limited (0.2 per s), so no sudden jumps.
  - LAND is always available (button, Esc, pinky 1 s or fist 1 s).
Gestures: point (index only) angled up = up a little, angled down = down a little (0.15 m/s, max 2 s each)   thumbs-up 1 s on the ground = take off   pinky or fist held 1 s = land.
AI assist (--ai): velocity goes to skyhand/pilot_vel; the drone's ai_mux (scripts/ai_mux.py) forwards it and,
  when "AI assist" is ticked and you push forward, the cloud VLM may slide the drone left/right around an
  obstacle or stop it. The panel shows what the AI saw and decided. Take-off/land are unchanged.
Keep the DJI remote in your hand: flipping to RC control is your real emergency stop.
"""
import argparse
import math
import os
import sys
import time
import tkinter as tk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from safe_takeoff import AMBER, BG, CARD, DIM, FG, GREEN, RED, Link  # noqa: E402
from skyhand.gestures import G, GestureTracker, classify  # noqa: E402
from skyhand.hand import DEFAULT_CURL_RANGE, HandModel  # noqa: E402
from skyhand.mapping import HandToVelocity, MapConfig, VelCmd  # noqa: E402
from skyhand.rokoko import FINGERS, RokokoReceiver  # noqa: E402

STEADY_S = 0.25       # gesture must be stable this long before it can move the drone
MIN_CONF = 0.85
MAX_AGE_S = 0.15      # glove data older than this = uncertain
SETTLE_S = 5.0        # after takeoff before hand control can be enabled
HARD_MAX_SPEED = 0.4
POINT_TILT_DEG = 25.0  # point + finger angled this far up/down = climb/descend
VERT_SPEED = 0.15     # m/s, index pointed up = up a bit, down = down a bit
VERT_MAX_S = 2.0      # one gesture moves vertically at most this long; release and repeat for more
YAW_CAP = 0.35        # rad/s
ACCEL = 0.2           # m/s^2 and rad/s^2 slew limit
BLUE = "#3b82f6"
F = "DejaVu Sans"


def clamp(v, lim):
    return max(-lim, min(lim, v))


AI_BANNER = {"continue": ("CLEAR", GREEN), "avoid_left": ("\u2190  AVOID LEFT", AMBER),
             "avoid_right": ("AVOID RIGHT  \u2192", AMBER), "stop": ("STOP", RED)}


def make_ai_panel(parent, link):
    """AI assist panel: big decision banner, the frame the AI judged, and whether it is steering right now."""
    import base64
    import io
    import json

    from PIL import Image, ImageTk

    st = {"status": None, "status_t": 0.0, "thumb": None, "img": None}
    enabled = tk.BooleanVar(value=False)

    def on_msg(_c, _u, m):
        if m.topic == "skyhand/ai_status":
            try:
                st["status"], st["status_t"] = json.loads(m.payload), time.time()
            except ValueError:
                pass
        elif m.topic == "skyhand/ai_view":
            st["thumb"] = m.payload

    def subscribe(c):
        c.subscribe([("skyhand/ai_status", 0), ("skyhand/ai_view", 0)])

    if link.client:
        prev = link.client.on_connect

        def on_connect(c, u, f, rc, p=None):
            prev(c, u, f, rc, p)
            if rc == 0:
                subscribe(c)
                c.publish("skyhand/ai_enable", "1" if enabled.get() else "0", retain=True)
        link.client.on_connect = on_connect
        link.client.on_message = on_msg
        if link.connected:
            subscribe(link.client)
    if link.client:  # always start with AI assist OFF on the drone side too
        link.client.publish("skyhand/ai_enable", "0", retain=True)

    box = tk.Frame(parent, bg=CARD)
    box.pack(fill="x", padx=24, pady=(10, 4))
    tk.Checkbutton(box, text="AI assist  (AI may steer around obstacles while you push forward)", variable=enabled,
                   command=lambda: link.client and link.client.publish("skyhand/ai_enable",
                                                                       "1" if enabled.get() else "0", retain=True),
                   bg=CARD, fg=FG, selectcolor=BG, activebackground=CARD, activeforeground=FG, font=(F, 11),
                   anchor="w", wraplength=420, justify="left").pack(fill="x", padx=10, pady=(8, 4))
    banner = tk.Label(box, text="AI: no data", bg=DIM, fg="white", font=(F, 18, "bold"), pady=8)
    banner.pack(fill="x", padx=10)
    row = tk.Frame(box, bg=CARD)
    row.pack(fill="x", padx=10, pady=8)
    # Without an image, Label width/height are in text characters (240 chars ~ 2000 px), which pushed the
    # gesture cards out of the window. A blank 240x135 image makes them pixels.
    blank = tk.PhotoImage(width=240, height=135)
    pic = tk.Label(row, bg=BG, image=blank, width=240, height=135)
    pic.image = blank
    pic.pack(side="left")
    detail = tk.Label(row, text="", bg=CARD, fg=FG, font=(F, 10), justify="left", anchor="nw", wraplength=180)
    detail.pack(side="left", fill="both", expand=True, padx=(10, 0))

    def update(now):
        if st["thumb"] is not None:
            try:
                im = Image.open(io.BytesIO(base64.b64decode(st["thumb"])))
                im.thumbnail((240, 135))
                st["img"] = ImageTk.PhotoImage(im)
                pic.config(image=st["img"], width=im.width, height=im.height)
            except Exception:  # noqa: BLE001 - a bad thumbnail must never break the flight GUI
                pass
            st["thumb"] = None
        sts = st["status"]
        if sts is None or now - st["status_t"] > 1.0:
            banner.config(text="AI: drone mux offline", bg=DIM)
            detail.config(text="no status from ai_mux")
            return
        age, action = sts.get("ai_age", 99), sts.get("action")
        # "live" = the VLM is answering at its normal rate; it may only ACT within act_window s of a decision
        fresh = age < float(sts.get("interval", 1.0)) + 1.5
        can_act = age <= float(sts.get("act_window", 1.0))
        text, color = AI_BANNER.get(action, ("AI: no data", DIM))
        if not fresh:
            text, color = "AI: no fresh view", DIM
        elif sts.get("stop_only") and action in ("avoid_left", "avoid_right"):
            text += "  (= BACK + HOVER)"  # hover-only mode: avoid advice backs off and hovers
        if sts.get("mode") == "backoff":
            text = "AI: BACKING OFF  (" + text + ")"
        elif sts.get("mode") in ("avoid_left", "avoid_right", "stop"):
            text = "AI STEERING:  " + text
        banner.config(text=text, bg=color if fresh else DIM)
        what = " ".join(x for x in (sts.get("obstacle", ""), f"({sts['distance']})" if sts.get("distance") else "") if x)
        dot = "\u25cf " + ("live" if fresh else "stale")
        detail.config(text=f"VLM: {action or '-'}  {what}\n{sts.get('reason', '')}\n\n"
                           f"conf {sts.get('conf', 0):.2f}   {sts.get('latency', 0):.1f}s\n"
                           f"{dot} ({age:.1f}s ago, every {float(sts.get('interval', 1.0)):.0f}s)\n"
                           f"{'can act now' if can_act else 'waiting for next view'}\n"
                           f"{'assist ON' if sts.get('enabled') else 'assist OFF'}"
                           f"{'  |  AI: back off + hover' if sts.get('stop_only') else '  |  AI: back off + avoid'}",
                      fg=GREEN if fresh else DIM)
    return update


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--host", "--mqtt-host", dest="host", default="192.168.4.2")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--vel-topic", default="drone/cmd_vel")
    p.add_argument("--cmd-topic", default="drone/command")
    p.add_argument("--max-speed", type=float, default=0.4, help=f"m/s, capped at {HARD_MAX_SPEED}")
    p.add_argument("--glove-port", type=int, default=14043)
    p.add_argument("--finger-range", type=float, default=1.5)
    p.add_argument("--gesture-model", default=os.path.join(os.path.dirname(HERE), "models", "gestures.npz"))
    p.add_argument("--ai", action="store_true", help="send to skyhand/pilot_vel via the drone's ai_mux + show AI panel")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.ai and a.vel_topic == "drone/cmd_vel":
        a.vel_topic = "skyhand/pilot_vel"
    cap = clamp(a.max_speed, HARD_MAX_SPEED)
    link = Link(a)

    rx = RokokoReceiver(port=a.glove_port)
    rng = {**DEFAULT_CURL_RANGE, **{f: a.finger_range for f in FINGERS if f != "Thumb"}}
    hand = HandModel(rng)
    gest = GestureTracker(hold_s=1.0)
    gmodel = None
    if os.path.exists(a.gesture_model):
        from skyhand.gesture_model import GestureModel
        gmodel = GestureModel(a.gesture_model, min_conf=MIN_CONF)
    mapper = HandToVelocity(MapConfig(max_xy=cap, max_yaw_rate=YAW_CAP, axis_lock=True))
    mapper.cfg.lock_axes = ("vx",)

    root = tk.Tk()
    root.title("Safe glove flight v2" + ("  (DRY RUN)" if a.dry_run else ""))
    root.configure(bg=BG)
    root.geometry("1100x900" if a.ai else "1100x720")
    left = tk.Frame(root, bg=BG)
    right = tk.Frame(root, bg=BG)
    right.pack(side="right", fill="both", padx=(0, 16), pady=16)  # packed first: gesture cards always get room
    left.pack(side="left", fill="both", expand=True)

    def lbl(text="", size=11, fg=FG, bold=False, **kw):
        w = tk.Label(left, text=text, bg=BG, fg=fg, font=(F, size, "bold" if bold else "normal"), **kw)
        w.pack(fill="x", padx=24)
        return w

    def btn(text, color, cmd, size=15):
        b = tk.Button(left, text=text, command=cmd, bg=color, fg="white", activebackground=color,
                      activeforeground="white", disabledforeground="#6b7280", relief="flat", bd=0,
                      font=(F, size, "bold"), cursor="hand2", pady=12)
        b.pack(fill="x", padx=24, pady=5)
        return b

    lbl("", 4)
    pill = tk.Label(left, text="", bg=DIM, fg="white", font=(F, 20, "bold"), pady=14)
    pill.pack(fill="x", padx=24)
    info = lbl("", 10, DIM, pady=6, anchor="w", justify="left", wraplength=440)

    s = {"flying": False, "takeoff_t": 0.0, "hand_on": False, "last": VelCmd(),
         "first_t": None, "g": G.NONE, "recal": False, "g_since": 0.0, "g_prev": G.NONE, "flat_t": None,
         "need_flat": True, "hand_t": 0.0, "fist_t": None, "vert_sign": 0.0, "vert_t": 0.0, "conf": 0.0, "why": "", "auto": False}
    hand_on = tk.BooleanVar(value=False)

    takeoff_b = btn("TAKE OFF", GREEN, lambda: takeoff())
    hand_chk = tk.Checkbutton(left, text="Hand control ON  (open hand tilt = fwd/back, peace + roll = turn)",
                              variable=hand_on, bg=BG, fg=FG, selectcolor=CARD, activebackground=BG,
                              activeforeground=FG, disabledforeground="#6b7280", font=(F, 11), anchor="w",
                              wraplength=420, justify="left")
    hand_chk.pack(fill="x", padx=24, pady=6)
    btn("LAND   (Esc)", RED, lambda: land())
    bars = tk.Canvas(left, bg=BG, height=52, highlightthickness=0)
    bars.pack(fill="x", padx=24, pady=(10, 0))
    log = lbl("", 10, DIM, pady=6, wraplength=440, anchor="w")
    ai_panel = make_ai_panel(left, link) if a.ai else None
    tk.Button(left, text="recalibrate hand (hold flat, fingers forward)", command=lambda: s.update(recal=True),
              bg=BG, fg=DIM, relief="flat", bd=0, font=(F, 9), cursor="hand2").pack(pady=(0, 10))

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    from skyhand.hand_icons import make_cards
    fig = Figure(figsize=(6.2, 7.0), facecolor=BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.04)
    update_cards = make_cards(fig, fig.add_gridspec(3, 3, hspace=0.45, wspace=0.12), safe=True)
    for ax in fig.axes:
        ax.title.set_color(FG)
        ax.xaxis.label.set_color("#b6bdc9")
    fig.text(0.5, 0.985, "HOW TO FLY  (right hand, palm down)", color=FG, ha="center", va="top", fontsize=11,
             weight="bold")
    fig.text(0.5, 0.012, "hold the gesture to act", color=DIM, ha="center", fontsize=9)
    cards = FigureCanvasTkAgg(fig, master=right)
    cards.get_tk_widget().pack(fill="both", expand=True)
    shown = [None]

    def note(msg):
        log.config(text=f"{time.strftime('%H:%M:%S')}  {msg}")

    def takeoff(by="button"):
        if s["flying"] or not link.connected:
            return
        link.pub(a.cmd_topic, "takeoff", qos=1)
        s.update(flying=True, takeoff_t=time.time())
        note(f"sent: takeoff ({by})")

    def land(by="button"):
        link.zero_vel()
        link.pub(a.cmd_topic, "land", qos=1)
        s["flying"] = False
        hand_on.set(False)
        mapper.reset()
        s["last"] = VelCmd()
        note(f"sent: land ({by})")

    def hand_cmd(frame, now):
        """Raw (unlimited) command from the glove; also fires gesture events."""
        if s["first_t"] is None:
            s["first_t"] = now
        if not hand.calibrated and now - s["first_t"] < 1.5:
            return VelCmd(), G.NONE, "calibrating: hold hand flat"
        if s["recal"] or not hand.calibrated:
            hand.calibrate(frame)
            mapper.reset()
            s["recal"] = False
            note("calibrated")
        hs = hand.update(frame)
        learned, conf = gmodel(frame) if gmodel else (None, 0.0)
        if gmodel and learned == G.NONE:
            # model unsure: finger-curl rules may still confirm an open hand / peace sign / point, nothing else
            rule = classify(hs.curls)
            learned = rule if rule in (G.OPEN, G.PEACE, G.POINT) else G.NONE
        s["conf"] = conf
        g, ev = gest.update(hs.curls, now, learned)
        if ev == G.THUMB:
            takeoff("thumbs-up")
        elif ev == G.PINKY:
            land("pinky")
        if g == G.FIST and s["flying"]:
            s["fist_t"] = s["fist_t"] or now
            if now - s["fist_t"] >= 1.0:
                s["fist_t"] = None
                land("fist")
        else:
            s["fist_t"] = None
        if g != s["g_prev"]:
            s["g_prev"], s["g_since"] = g, now
        steady = now - s["g_since"] >= STEADY_S
        level = g == G.OPEN and max(abs(hs.pitch), abs(hs.roll)) < math.radians(10)
        if level:
            s["flat_t"] = s["flat_t"] or now
            if now - s["flat_t"] >= STEADY_S:
                s["need_flat"] = False
        else:
            s["flat_t"] = None
        pd, rd = math.degrees(hs.pitch), math.degrees(hs.roll)
        if g not in (G.OPEN, G.PEACE, G.POINT):
            s["why"] = f"gesture is '{g.value}', need open hand"
        elif not steady:
            s["why"] = "hold the gesture steady"
        else:
            s["why"] = f"ready - tilt more (pitch {pd:+.0f}° roll {rd:+.0f}°, need ~12°+)"
        cmd = VelCmd()
        if steady and g == G.OPEN:
            cmd = mapper(hs)
        elif steady and g == G.PEACE:
            cmd = mapper.turn(hs)
        elif steady and g == G.POINT and abs(pd) >= POINT_TILT_DEG:
            # index finger angled up = climb, angled down = descend (level = nothing)
            sign = 1.0 if pd > 0 else -1.0
            if s["vert_sign"] != sign:
                s["vert_sign"], s["vert_t"] = sign, now
            cmd = VelCmd(vz=sign * VERT_SPEED if now - s["vert_t"] <= VERT_MAX_S else 0.0)
        else:
            mapper.reset()
        if not (g == G.POINT and abs(pd) >= POINT_TILT_DEG):
            s["vert_sign"] = 0.0
        return VelCmd(vx=cmd.vx, vz=cmd.vz, yaw_rate=cmd.yaw_rate), g, "glove ok"

    def step():
        now = time.time()
        dt = 0.033
        frame = rx.latest(MAX_AGE_S)
        raw, g, glove = VelCmd(), G.NONE, "no glove"
        if frame is None:
            mapper.reset()
        else:
            raw, g, glove = hand_cmd(frame, now)
        s["g"] = g

        settled = s["flying"] and now - s["takeoff_t"] >= SETTLE_S
        if not s["flying"]:
            s["auto"] = False
        elif settled and not s["auto"]:
            s["auto"] = True
            hand_on.set(True)
        if not settled and hand_on.get():
            hand_on.set(False)
        allowed = settled and hand_on.get() and frame is not None
        # turn direction is flipped for this drone
        tgt = VelCmd(vx=clamp(raw.vx, cap), vz=clamp(raw.vz, VERT_SPEED), yaw_rate=-clamp(raw.yaw_rate, YAW_CAP)) if allowed else VelCmd()

        last = s["last"]
        out = VelCmd(vx=last.vx + clamp(tgt.vx - last.vx, ACCEL * dt),
                     vz=last.vz + clamp(tgt.vz - last.vz, ACCEL * dt),
                     yaw_rate=last.yaw_rate + clamp(tgt.yaw_rate - last.yaw_rate, ACCEL * dt))
        s["last"] = out
        link.pub(a.vel_topic, '{"vx": %.3f, "vy": 0.0, "vz": %.3f, "yaw_rate": %.3f}' % (out.vx, out.vz, out.yaw_rate))

        ok = link.connected
        takeoff_b.config(state="normal" if ok and not s["flying"] else "disabled",
                         bg=GREEN if ok and not s["flying"] else "#3a414d")
        hand_chk.config(state="normal" if settled else "disabled")
        if not ok:
            text, color = "NOT CONNECTED", RED
        elif not s["flying"]:
            text, color = "ON GROUND", DIM
        elif not settled:
            text, color = f"TAKING OFF  {SETTLE_S - (now - s['takeoff_t']):.0f}s", AMBER
        elif allowed and (abs(out.vx) > 0.01 or abs(out.yaw_rate) > 0.01):
            text, color = "MOVING", BLUE
        elif allowed:
            text, color = "HAND CONTROL", GREEN
        else:
            text, color = "HOVER", GREEN
        pill.config(text=text, bg=color)
        why = "" if allowed and s["why"] else ("hand control OFF" if not hand_on.get() else "")
        if allowed and (abs(raw.vx) > 0.01 or abs(raw.yaw_rate) > 0.01):
            why = "moving"
        elif allowed:
            why = s["why"]
        info.config(text=f"{why}\n{a.host}:{a.port}   ·   {glove}   ·   max {cap:g} m/s   ·   gesture: {g.value}")

        bars.delete("all")
        w = max(bars.winfo_width(), 200)
        for i, (name, v, mx) in enumerate((("forward", out.vx, cap), ("turn", out.yaw_rate, YAW_CAP))):
            y = 6 + i * 24
            mid = 60 + (w - 64) / 2
            bars.create_text(0, y + 8, text=name, anchor="w", fill=DIM, font=(F, 9))
            bars.create_rectangle(60, y, w - 4, y + 16, fill="#2a303a", width=0)
            bars.create_rectangle(mid, y, mid + (w - 64) / 2 * clamp(v / mx, 1), y + 16, fill=BLUE, width=0)
            bars.create_line(mid, y - 2, mid, y + 18, fill=DIM)

        # cards follow the hand even on the ground (sending stays gated); shown as a fraction of our limits
        shown_cmd = VelCmd(vx=clamp(raw.vx, cap) / cap, vz=clamp(raw.vz, VERT_SPEED) / VERT_SPEED,
                           yaw_rate=clamp(raw.yaw_rate, YAW_CAP) / YAW_CAP)
        key = (g.value, round(shown_cmd.vx, 1), round(shown_cmd.vz, 1), round(shown_cmd.yaw_rate, 1))
        if key != shown[0]:
            shown[0] = key
            update_cards({"gesture": g.value, "cmd": shown_cmd, "intent": shown_cmd})
            cards.draw_idle()
        if ai_panel:
            ai_panel(now)
        root.after(33, step)

    def close():
        s["last"] = VelCmd()
        link.zero_vel()
        root.destroy()

    root.bind("<Escape>", lambda _e: land("Esc"))
    root.protocol("WM_DELETE_WINDOW", close)
    root.after(100, step)
    # macOS opens Tk windows from a terminal behind it: bring this one to the front once
    root.lift()
    root.attributes("-topmost", True)
    root.after(500, lambda: root.attributes("-topmost", False))
    root.focus_force()
    root.mainloop()
    link.close()
    rx.close()


if __name__ == "__main__":
    main()
