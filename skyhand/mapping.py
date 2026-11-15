"""Hand pose -> body-frame velocity command ("your hand is the drone")."""
from dataclasses import dataclass

import numpy as np


@dataclass
class VelCmd:
    vx: float = 0.0       # m/s forward (body)
    vy: float = 0.0       # m/s right (body)
    vz: float = 0.0       # m/s up
    yaw_rate: float = 0.0  # rad/s, + = turn right (clockwise seen from above)

    def as_list(self):
        return [self.vx, self.vy, self.vz, self.yaw_rate]

    def scaled(self, k: float) -> "VelCmd":
        return VelCmd(self.vx * k, self.vy * k, self.vz * k, self.yaw_rate * k)


@dataclass
class MapConfig:
    max_xy: float = 1.5        # m/s
    max_z: float = 0.8         # m/s
    max_yaw_rate: float = 1.0  # rad/s
    tilt_full: float = np.radians(35)  # hand tilt giving full speed
    yaw_full: float = np.radians(45)
    height_full: float = 0.25  # m of hand travel for full climb rate
    deadzone: float = 0.15     # fraction of full scale
    expo: float = 1.6          # >1 = finer control near center
    vertical_source: str = "thumb"  # "thumb" (glove only) | "height" (needs Coil Pro / suit)
    smoothing: float = 0.3     # EMA factor per tick (0 = none)
    axis_lock: bool = False    # one motion at a time: the strongest axis wins until the hand is flat again
    lock_start: float = 0.35   # fraction of full scale to start a motion
    lock_margin: float = 1.6   # ...and it must beat the 2nd strongest axis by this factor
    lock_release: float = 0.2  # all axes below this = hand flat again -> unlock
    lock_axes: tuple = ("vx", "vy", "yaw_rate", "vz")  # axes allowed in axis_lock mode


def _shape(x: float, full: float, cfg: MapConfig) -> float:
    u = float(np.clip(x / full, -1.0, 1.0))
    a = abs(u)
    if a < cfg.deadzone:
        return 0.0
    a = (a - cfg.deadzone) / (1.0 - cfg.deadzone)
    return float(np.sign(u) * a ** cfg.expo)


class HandToVelocity:
    def __init__(self, cfg: MapConfig | None = None):
        self.cfg = cfg or MapConfig()
        self._last = VelCmd()
        self.axis = None  # locked axis in axis_lock mode

    def _lock(self, hs, cmd):
        c = self.cfg
        t = hs.thumb
        thumb = (-t - 0.2) / 0.3 + c.lock_start if t < -0.2 else (t - 0.3) / 0.3 + c.lock_start if t > 0.3 else 0.0
        mag = {"vx": abs(hs.pitch) / c.tilt_full, "vy": abs(hs.roll) / c.tilt_full,
               "yaw_rate": abs(hs.yaw) / c.yaw_full, "vz": thumb}
        mag = {k: v for k, v in mag.items() if k in c.lock_axes}
        if max(mag.values()) < c.lock_release:
            self.axis = None
        if self.axis is None:
            (a1, m1), (_, m2) = (sorted(mag.items(), key=lambda kv: -kv[1]) + [("", 0.0)])[:2]
            if m1 >= c.lock_start and m1 >= c.lock_margin * m2:
                self.axis = a1
        keep = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}
        if self.axis:
            keep[self.axis] = getattr(cmd, self.axis)
        return VelCmd(**keep)

    def turn(self, hs) -> VelCmd:
        """--simple turn gesture: roll the hand right/left = turn right/left, nothing else."""
        c = self.cfg
        return VelCmd(yaw_rate=_shape(hs.roll, c.tilt_full, c) * c.max_yaw_rate)

    def reset(self):
        self._last = VelCmd()
        self.axis = None

    def __call__(self, hs) -> VelCmd:
        c = self.cfg
        # fingers tip down (negative pitch) = nose down = forward, like a quadrotor
        vx = -_shape(hs.pitch, c.tilt_full, c) * c.max_xy
        vy = _shape(hs.roll, c.tilt_full, c) * c.max_xy
        yaw = _shape(hs.yaw, c.yaw_full, c) * c.max_yaw_rate
        if c.vertical_source == "height":
            vz = _shape(hs.height, c.height_full, c) * c.max_z
        else:
            # relative to the relaxed calibration thumb: stuck out = climb, tucked into palm = descend
            t = hs.thumb
            vz = c.max_z if t < -0.2 else -c.max_z if t > 0.3 else 0.0
        cmd = VelCmd(vx, vy, vz, yaw)
        if c.axis_lock:
            cmd = self._lock(hs, cmd)
        k = c.smoothing
        out = VelCmd(*[k * p + (1 - k) * n for p, n in zip(self._last.as_list(), cmd.as_list())])
        self._last = out
        return out
