#!/usr/bin/env python3
"""Safe first test: take off / land over MQTT. No glove, no LLM, no movement.

  python3 scripts/safe_takeoff.py             # real
  python3 scripts/safe_takeoff.py --dry-run   # nothing is sent, just prints what would be

- Velocity is ALWAYS zero (sent at 10 Hz as a heartbeat). This script can never command motion.
- "takeoff" is sent once, only after the checklist box is ticked and the broker is connected.
  The box un-ticks itself after 30 s and after every takeoff.
- LAND (or Esc) is always available and also sends zero velocity.
- Keep the DJI remote in your hand: flipping to RC control is your real emergency stop.
"""
import argparse
import json
import time
import tkinter as tk

BG, CARD, FG, DIM = "#14171c", "#1e232b", "#f2f4f8", "#8b94a3"
GREEN, RED, AMBER = "#2ecc71", "#e5484d", "#f5a524"
ARM_TIMEOUT_S = 30


class Link:
    def __init__(self, a):
        self.a, self.connected = a, a.dry_run
        self.client = None
        if not a.dry_run:
            import paho.mqtt.client as mqtt
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            self.client.on_connect = lambda c, u, f, rc, p=None: setattr(self, "connected", rc == 0)
            self.client.on_disconnect = lambda *x: setattr(self, "connected", False)
            self.client.connect_async(a.host, a.port, keepalive=10)
            self.client.loop_start()

    def pub(self, topic, payload, qos=0):
        if self.a.dry_run:
            print(f"[dry-run] {topic} {payload}")
        elif self.connected:
            self.client.publish(topic, payload, qos=qos)

    def zero_vel(self):
        self.pub(self.a.vel_topic, json.dumps({"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}))

    def close_silent(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()

    def close(self):
        self.zero_vel()
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--host", default="192.168.4.3")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--vel-topic", default="drone/cmd_vel")
    p.add_argument("--cmd-topic", default="drone/command")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    link = Link(a)

    root = tk.Tk()
    root.title("Safe takeoff test" + ("  (DRY RUN)" if a.dry_run else ""))
    root.configure(bg=BG)
    root.geometry("460x520")
    f = "DejaVu Sans"

    def lbl(text="", size=11, fg=FG, bold=False, **kw):
        w = tk.Label(root, text=text, bg=BG, fg=fg, font=(f, size, "bold" if bold else "normal"), **kw)
        w.pack(fill="x", padx=24)
        return w

    def btn(text, color, cmd):
        b = tk.Button(root, text=text, command=cmd, bg=color, fg="white", activebackground=color,
                      activeforeground="white", disabledforeground="#6b7280", relief="flat", bd=0,
                      font=(f, 16, "bold"), cursor="hand2", pady=16)
        b.pack(fill="x", padx=24, pady=6)
        return b

    lbl("", 6)
    status = lbl("", 14, bold=True, pady=10)
    lbl(f"{a.host}:{a.port}   ·   velocity locked at 0", 9, DIM)
    lbl("", 6)

    armed = tk.BooleanVar(value=False)
    armed_at = [0.0]
    check = tk.Checkbutton(root, text="Area clear, nobody near, remote in my hand", variable=armed,
                           bg=BG, fg=FG, selectcolor=CARD, activebackground=BG, activeforeground=FG,
                           font=(f, 12), anchor="w", command=lambda: armed_at.__setitem__(0, time.time()))
    check.pack(fill="x", padx=24, pady=10)

    takeoff_b = btn("TAKE OFF", GREEN, lambda: takeoff())
    btn("LAND   (Esc)", RED, lambda: land())
    log = lbl("", 10, DIM, pady=10, wraplength=400)

    def takeoff():
        if not (armed.get() and link.connected):
            return
        link.pub(a.cmd_topic, "takeoff", qos=1)
        armed.set(False)
        log.config(text=f"{time.strftime('%H:%M:%S')}  sent: takeoff")

    def land():
        link.zero_vel()
        link.pub(a.cmd_topic, "land", qos=1)
        armed.set(False)
        log.config(text=f"{time.strftime('%H:%M:%S')}  sent: land")

    def tick():
        link.zero_vel()
        if armed.get() and time.time() - armed_at[0] > ARM_TIMEOUT_S:
            armed.set(False)
        ok = link.connected
        status.config(text="CONNECTED" if ok else "NOT CONNECTED", fg=GREEN if ok else RED)
        takeoff_b.config(state="normal" if armed.get() and ok else "disabled", bg=GREEN if armed.get() and ok else "#3a414d")
        root.after(100, tick)

    root.bind("<Escape>", lambda _e: land())
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    tick()
    root.mainloop()
    link.close()


if __name__ == "__main__":
    main()
