"""Pilot + AI velocity mixer that runs on the drone's onboard computer (scripts/ai_mux.py).

The pilot (glove) always has priority. The AI may only take over while ALL of these hold:
  - AI assist is enabled,
  - the pilot is actively pushing forward (fist, flat hand, land, glove loss -> AI is out immediately),
  - the AI message is fresh, confident, its camera->decision pipeline was fast, and the last
    AGREE_N AI messages agree on the same action.
When it takes over it runs a short manoeuvre: brake and BACK OFF (into the space the drone just flew through),
then slide to the free side (avoid_left / avoid_right) or hover (stop, e.g. a person: never steer around people).
Never faster than the caps below, never up/down or turning. Letting go of forward ends it at once. Anything else =
the pilot's own (capped) command passes through. "stop" acts on a single confident message (conservative: react
early); avoid needs agree_n agreeing messages with a near obstacle.
Agreeing messages only count if they arrive within max_agree_gap_s of each other (an old message plus a new
one is not agreement). stop_only=True turns avoid_left/right into a hover: the camera only looks forward, so a
sideways slide moves the drone where nothing has checked.
"""
from __future__ import annotations

from dataclasses import dataclass

ACTIONS = ("continue", "avoid_left", "avoid_right", "stop")


@dataclass
class Limits:
    max_fwd: float = 0.4        # m/s, cap on the pilot's forward/back (defence in depth; laptop caps too)
    max_side: float = 0.4       # m/s, cap on AI sideways slide (= pilot forward cap)
    max_vz: float = 0.15
    max_yaw: float = 0.35
    pilot_fresh_s: float = 0.3  # pilot command older than this = no pilot (mux hovers, then goes silent)
    ai_fresh_s: float = 1.0     # AI message older than this (local receive time) = ignored
    max_pipeline_s: float = 2.0  # AI frame age + model latency above this = ignored
    min_conf: float = 0.8
    stop_min_conf: float = 0.6  # "stop" (person / blocked) acts on ONE message at this confidence: react early
    agree_n: int = 2
    push_vx: float = 0.05       # pilot forward speed that counts as "pushing forward"
    max_agree_gap_s: float = 6.0  # agreeing AI messages further apart than this don't count as agreement
    stop_only: bool = False     # True: after backing off, hover instead of sliding sideways
    back_speed: float = 0.25    # m/s backwards right after an obstacle / person is detected
    back_s: float = 2.0         # s of backing off (~0.5 m)
    avoid_s: float = 1.5        # s of sideways slide after backing off (avoid_left/right only)
    hold_s: float = 1.0         # s of hovering after backing off (stop, or stop_only)
    avoid_distances: tuple = ("near",)  # avoid_left/right only counts when the VLM says the obstacle is this close


def _clamp(v, lim):
    return max(-lim, min(lim, float(v)))


class Mixer:
    def __init__(self, lim: Limits | None = None):
        self.lim = lim or Limits()
        self.ai, self.ai_rx = None, -1e9
        self.streak_action, self.streak = None, 0
        self.man = None  # running manoeuvre: {"phase": backoff|avoid|hold, "t0", "vy"}

    def on_ai(self, msg: dict, now: float):
        """Call once per new AI message. Invalid / unsure messages break the agreement streak."""
        L = self.lim
        try:
            need = L.stop_min_conf if msg.get("action") == "stop" else L.min_conf
            ok = (msg.get("action") in ACTIONS and float(msg.get("conf", 0)) >= need
                  and (msg.get("action") not in ("avoid_left", "avoid_right")
                       or msg.get("distance") in L.avoid_distances)
                  and float(msg.get("latency", 99)) + float(msg.get("frame_age", 99)) <= L.max_pipeline_s)
        except (TypeError, ValueError):
            ok = False
        action = msg.get("action") if ok else None
        if now - self.ai_rx > L.max_agree_gap_s:  # too long since the previous message: start over
            self.streak_action, self.streak = None, 0
        if action == self.streak_action:
            self.streak += 1
        else:
            self.streak_action, self.streak = action, 1
        self.ai, self.ai_rx = msg, now

    def ai_age(self, now):
        return now - self.ai_rx

    def step(self, pilot: dict | None, pilot_age: float, now: float, enabled: bool):
        """-> (cmd dict or None if no pilot, mode). mode: no_pilot | pilot | backoff | avoid_left | avoid_right | stop."""
        L = self.lim
        zero = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}
        if pilot is None or pilot_age > L.pilot_fresh_s:
            self.man = None
            return None, "no_pilot"
        p = {"vx": _clamp(pilot.get("vx", 0.0), L.max_fwd), "vy": 0.0,
             "vz": _clamp(pilot.get("vz", 0.0), L.max_vz), "yaw_rate": _clamp(pilot.get("yaw_rate", 0.0), L.max_yaw)}
        pushing = p["vx"] > L.push_vx
        if self.man and (not enabled or not pushing):
            self.man = None  # pilot let go of forward / AI switched off: the pilot has control again
        if self.man is None and enabled and pushing and self.ai_age(now) <= L.ai_fresh_s:
            a = self.streak_action
            if (a == "stop" and self.streak >= 1) or (a in ("avoid_left", "avoid_right") and self.streak >= L.agree_n):
                side = 0.0 if (a == "stop" or L.stop_only) else (-1.0 if a == "avoid_left" else 1.0)
                speed = min(abs(float(self.ai.get("vy", L.max_side)) or L.max_side), L.max_side)
                self.man = {"phase": "backoff", "t0": now, "vy": side * speed}
                self.streak = 0  # the next manoeuvre needs fresh decisions
        m = self.man
        if m is None:
            return p, "pilot"
        if m["phase"] == "backoff":
            if now - m["t0"] < L.back_s:
                return {**zero, "vx": -L.back_speed}, "backoff"
            m.update(phase="avoid" if m["vy"] else "hold", t0=now)
        if m["phase"] == "avoid":
            if now - m["t0"] < L.avoid_s:
                return {**zero, "vy": m["vy"]}, "avoid_left" if m["vy"] < 0 else "avoid_right"
        elif now - m["t0"] < L.hold_s:
            return zero, "stop"
        self.man = None
        return p, "pilot"
