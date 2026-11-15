"""Hand drives the drone along an exact path (the LLM picked the shape, the hand sets speed and direction).

throttle in [-1, 1]: + = forward along the path, - = back along it, 0 = hold on the path.
shift in [-1, 1]: moves the whole path sideways (left/right of the start heading), live.
"""
import math

import numpy as np

from .mapping import VelCmd
from .planner import GEOFENCE_R, MAX_ALT
from . import shapes

SPEED = 1.0       # m/s along the path at full throttle
SHIFT_SPEED = 0.5  # m/s sideways shift at full roll
KP = 1.5          # pull back onto the path
MAX_V = 2.0
LEAD = 0.35        # s, about the drone's velocity lag
LEASH = 0.6       # m: the path point waits for the drone if it is further away than this


class PathRide:
    def __init__(self, shape: str, size: float, st: dict, mirror: bool = False):
        self.shape, self.size = shape, size
        self.yaw0 = st["yaw"]
        pts = shapes.to_world(shapes.make(shape, size, mirror), st["x"], st["y"], st["z"], self.yaw0)
        pts[:, 2] = np.clip(pts[:, 2], 0.5, MAX_ALT)
        r = np.hypot(pts[:, 0], pts[:, 1])
        k = np.minimum(1.0, GEOFENCE_R / np.maximum(r, 1e-9))
        pts[:, 0] *= k
        pts[:, 1] *= k
        self.pts = pts
        self.s_at = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        self.length = float(self.s_at[-1])
        self.s = 0.0
        self.offset = 0.0
        self.err = 0.0
        self.done = False

    @property
    def right(self):
        return np.array([-math.sin(self.yaw0), math.cos(self.yaw0), 0.0])

    def point(self, s):
        return np.array([np.interp(s, self.s_at, self.pts[:, i]) for i in range(3)]) + self.offset * self.right

    def world_path(self):
        return self.pts + self.offset * self.right

    @property
    def progress(self):
        return self.s / self.length if self.length else 1.0

    def step(self, st: dict, throttle: float, shift: float, dt: float) -> VelCmd:
        pos = np.array([st["x"], st["y"], st["z"]])
        self.offset += float(np.clip(shift, -1, 1)) * SHIFT_SPEED * dt
        self.err = float(np.linalg.norm(self.point(self.s) - pos))
        v_s = float(np.clip(throttle, -1, 1)) * SPEED
        if self.err > LEASH:  # let the drone catch up before moving on
            v_s *= max(0.0, 1.0 - (self.err - LEASH) / LEASH)
        s_new = float(np.clip(self.s + v_s * dt, 0.0, self.length))
        v_s = (s_new - self.s) / dt if dt > 0 else 0.0  # 0 at the ends of the path
        self.s = s_new
        if self.s >= self.length - 1e-6 and self.err < 0.25:
            self.done = True
        # the drone's velocity lags about LEAD s, so feed forward the path direction that far ahead
        s_ff = float(np.clip(self.s + v_s * LEAD, 0.0, self.length))
        ds = 0.1
        tan = self.point(min(s_ff + ds, self.length)) - self.point(max(s_ff - ds, 0.0))
        tan = tan / (np.linalg.norm(tan) + 1e-9)
        v = tan * v_s + KP * (self.point(self.s) - pos)
        n = np.linalg.norm(v)
        if n > MAX_V:
            v *= MAX_V / n
        yaw_rate = 0.0
        if np.hypot(tan[0], tan[1]) > 0.5 and abs(v_s) > 0.05:  # face the way the path goes
            want = math.atan2(tan[1], tan[0])
            e = (want - st["yaw"] + math.pi) % (2 * math.pi) - math.pi
            yaw_rate = float(np.clip(2.0 * e, -1.0, 1.0))
        c, s_ = math.cos(st["yaw"]), math.sin(st["yaw"])
        return VelCmd(v[0] * c + v[1] * s_, -v[0] * s_ + v[1] * c, v[2], yaw_rate)
