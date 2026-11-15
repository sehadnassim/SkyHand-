"""Executes small, validated flight primitives (produced by gestures or the LLM copilot).

Closed-loop on position when the backend provides odometry, otherwise open-loop (timed velocity).
A glove FIST always aborts the plan (handled in app.py).
"""
import math
from dataclasses import dataclass, field

import numpy as np

from .mapping import VelCmd
from .shapes import SHAPES

CRUISE = 0.7       # m/s
YAW_SPEED = 0.8    # rad/s
MAX_STEP = 8.0     # m per single move
GEOFENCE_R = 15.0  # m from takeoff point
MAX_ALT = 5.0      # m

OPS = {
    "takeoff": {},
    "land": {},
    "hover": {"s": 2.0},
    "move": {"forward": 0.0, "right": 0.0, "up": 0.0},       # body frame, meters
    "move_world": {"x": 0.0, "y": 0.0, "z": 0.0},            # start frame, meters
    "yaw": {"deg": 90.0},                                    # + = turn right
    "orbit": {"radius": 2.0, "deg": 360.0},                  # centre = radius ahead; + deg = fly to the right
    "goto_pointed": {"distance": 3.0},                       # along the operator's index-finger ray
    "goto_waypoint": {"id": 1},
    "trace_sketch": {"size": 2.0},
    "path": {"size": 3.0},  # + "shape" from shapes.SHAPES; the pilot's hand drives along it (ride.py)                           # fly the last air-drawn shape in the vertical plane
}


def sanitize(plan) -> list:
    """Drop unknown ops, fill defaults, clamp numbers. Never trust LLM output."""
    out = []
    for step in plan if isinstance(plan, list) else []:
        if not isinstance(step, dict) or step.get("op") not in OPS:
            continue
        s = {"op": step["op"]}
        if s["op"] == "path":
            shape = str(step.get("shape", "")).lower().replace(" ", "").replace("-", "").replace("_", "")
            shape = {"figureeight": "figure8", "eight": "figure8", "8": "figure8"}.get(shape, shape)
            if shape not in SHAPES:
                continue
            s["shape"] = shape
            if step.get("mirror"):
                s["mirror"] = True
        for k, d in OPS[step["op"]].items():
            try:
                s[k] = float(step.get(k, d))
            except (TypeError, ValueError):
                s[k] = d
        for k in ("forward", "right", "up", "x", "y", "z", "distance"):
            if k in s:
                s[k] = float(np.clip(s[k], -MAX_STEP, MAX_STEP))
        if "radius" in s:
            s["radius"] = float(np.clip(s["radius"], 0.5, 6.0))
        if "s" in s:
            s["s"] = float(np.clip(s["s"], 0.0, 30.0))
        if s["op"] == "path":
            s["size"] = float(np.clip(s["size"], 0.5, 6.0))
        out.append(s)
    return out[:30]


@dataclass
class Context:
    point_yaw: float = 0.0
    point_pitch: float = 0.0
    waypoints: dict = field(default_factory=dict)  # id -> (x, y, z)
    sketch: list = field(default_factory=list)     # [(right, up)] normalized, starts at (0, 0)


def _body_to_world(yaw, f, r):
    return f * math.cos(yaw) - r * math.sin(yaw), f * math.sin(yaw) + r * math.cos(yaw)


def _world_to_body(yaw, x, y):
    return x * math.cos(yaw) + y * math.sin(yaw), -x * math.sin(yaw) + y * math.cos(yaw)


class Executor:
    def __init__(self, drone):
        self.drone = drone
        self.queue: list = []
        self.current = None
        self.status = "idle"

    @property
    def busy(self):
        return self.current is not None or bool(self.queue)

    def load(self, plan: list, ctx: Context):
        self.queue = self._expand(sanitize(plan), ctx)
        self.current = None

    def abort(self):
        self.queue, self.current, self.status = [], None, "aborted"

    def _expand(self, plan, ctx):
        """Turn high-level ops into low-level segments: ('vel', cmd, seconds) or ('goto', x, y, z, yaw) etc."""
        segs = []
        for s in plan:
            op = s["op"]
            if op in ("takeoff", "land"):
                segs.append((op,))
            elif op == "hover":
                segs.append(("hold", s["s"]))
            elif op == "move":
                segs.append(("rel", s["forward"], s["right"], s["up"]))
            elif op == "move_world":
                segs.append(("abs_delta", s["x"], s["y"], s["z"]))
            elif op == "yaw":
                segs.append(("yaw", math.radians(s["deg"])))
            elif op == "orbit":
                ang = math.radians(s["deg"])
                v = CRUISE
                t = abs(ang) * s["radius"] / v
                sgn = 1 if ang >= 0 else -1
                segs.append(("vel", VelCmd(vy=sgn * v, yaw_rate=-sgn * v / s["radius"]), t))
            elif op == "goto_pointed":
                d, py, pp = s["distance"], ctx.point_yaw, ctx.point_pitch
                segs.append(("abs_delta", d * math.cos(pp) * math.cos(py), d * math.cos(pp) * math.sin(py),
                             d * math.sin(pp)))
            elif op == "goto_waypoint":
                wp = ctx.waypoints.get(int(s["id"]))
                if wp:
                    segs.append(("abs", *wp))
            elif op == "path":
                segs.append(("path", s["shape"], s["size"], bool(s.get("mirror"))))
            elif op == "trace_sketch" and ctx.sketch:
                pts = np.array(ctx.sketch) * s["size"]
                for (r0, u0), (r1, u1) in zip(pts, pts[1:]):
                    segs.append(("rel", 0.0, r1 - r0, u1 - u0))
        return segs

    # ---- per-tick ----
    def step(self, dt: float) -> VelCmd | None:
        """Returns a velocity command while executing, None when idle."""
        st = self.drone.state()
        if self.current is None:
            if not self.queue:
                if self.status == "running":
                    self.status = "done"
                return None
            self.current = self._start(self.queue.pop(0), st)
            self.status = "running"
        cmd, done = self._run(self.current, st, dt)
        if done:
            self.current = None
        return cmd

    def _start(self, seg, st):
        kind = seg[0]
        if kind == "takeoff":
            self.drone.takeoff()
            return {"kind": "hold", "t": 3.0}  # give the backend time to climb
        if kind == "land":
            self.drone.land()
            return {"kind": "hold", "t": 4.0}  # don't fight the auto-landing with direct commands
        if kind == "hold":
            return {"kind": "hold", "t": seg[1]}
        if kind == "path":  # the app's PathRide takes over until the pilot reaches the end
            return {"kind": "path", "shape": seg[1], "size": seg[2], "mirror": seg[3], "t": 0.0}
        if kind == "vel":
            return {"kind": "vel", "cmd": seg[1], "t": seg[2]}
        if kind == "yaw":
            if st:
                return {"kind": "yaw_to", "yaw": st["yaw"] + seg[1]}
            return {"kind": "vel", "cmd": VelCmd(yaw_rate=math.copysign(YAW_SPEED, seg[1])),
                    "t": abs(seg[1]) / YAW_SPEED}
        if kind in ("rel", "abs_delta", "abs"):
            if kind == "rel":
                f, r, u = seg[1:]
                dist = math.sqrt(f * f + r * r + u * u)
                if not st:
                    if dist < 1e-3:
                        return {"kind": "hold", "t": 0.0}
                    k = CRUISE / dist
                    return {"kind": "vel", "cmd": VelCmd(f * k, r * k, u * k), "t": dist / CRUISE}
                dx, dy = _body_to_world(st["yaw"], f, r)
                tgt = (st["x"] + dx, st["y"] + dy, st["z"] + u)
            elif kind == "abs_delta":
                if not st:
                    return {"kind": "hold", "t": 0.0}  # world moves need odometry
                tgt = (st["x"] + seg[1], st["y"] + seg[2], st["z"] + seg[3])
            else:
                if not st:
                    return {"kind": "hold", "t": 0.0}
                tgt = seg[1:]
            return {"kind": "goto", "tgt": self._fence(tgt), "t": 0.0}
        return {"kind": "hold", "t": 0.0}

    @staticmethod
    def _fence(t):
        x, y, z = t
        r = math.hypot(x, y)
        if r > GEOFENCE_R:
            x, y = x * GEOFENCE_R / r, y * GEOFENCE_R / r
        return x, y, float(np.clip(z, 0.5, MAX_ALT))

    def _run(self, c, st, dt):
        c["t_elapsed"] = c.get("t_elapsed", 0.0) + dt
        k = c["kind"]
        if k == "hold":
            return VelCmd(), c["t_elapsed"] >= c["t"]
        if k == "path":
            return VelCmd(), False
        if k == "vel":
            return c["cmd"], c["t_elapsed"] >= c["t"]
        if k == "yaw_to":
            err = (c["yaw"] - st["yaw"] + math.pi) % (2 * math.pi) - math.pi
            return VelCmd(yaw_rate=float(np.clip(2.0 * err, -YAW_SPEED, YAW_SPEED))), abs(err) < 0.05
        if k == "goto":
            ex, ey, ez = (c["tgt"][0] - st["x"], c["tgt"][1] - st["y"], c["tgt"][2] - st["z"])
            dist = math.sqrt(ex * ex + ey * ey + ez * ez)
            if dist < 0.12 or c["t_elapsed"] > 40:
                return VelCmd(), True
            v = min(CRUISE, 1.5 * dist) / dist
            f, r = _world_to_body(st["yaw"], ex * v, ey * v)
            return VelCmd(f, r, ez * v), False
        return VelCmd(), True
