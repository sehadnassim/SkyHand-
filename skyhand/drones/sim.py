"""Kinematic quadrotor simulator (first-order velocity response) for testing without hardware."""
import math
import threading

import numpy as np

from ..mapping import VelCmd
from . import Drone


class SimDrone(Drone):
    name = "sim"

    def __init__(self, tau: float = 0.35, takeoff_alt: float = 1.2):
        self.tau, self.takeoff_alt = tau, takeoff_alt
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.yaw = 0.0
        self.yaw_rate = 0.0
        self.cmd = VelCmd()
        self.airborne = False
        self._target_alt = None
        self.trail = []
        self._lock = threading.Lock()

    def send_velocity(self, cmd: VelCmd):
        self.cmd = cmd

    def takeoff(self):
        self.airborne, self._target_alt = True, self.takeoff_alt

    def land(self):
        self._target_alt = 0.0

    def tick(self, dt: float):
        with self._lock:
            if not self.airborne:
                self.vel[:] = 0
                return
            c = self.cmd
            fwd = np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0])
            right = np.array([-math.sin(self.yaw), math.cos(self.yaw), 0.0])
            want = fwd * c.vx + right * c.vy + np.array([0, 0, c.vz])
            want_yaw = c.yaw_rate
            if self._target_alt is not None:  # automatic takeoff / landing
                want = np.array([0, 0, np.clip(2.0 * (self._target_alt - self.pos[2]), -0.8, 0.8)])
                want_yaw = 0.0
                if abs(self._target_alt - self.pos[2]) < 0.03:
                    if self._target_alt == 0.0:
                        self.airborne = False
                    self._target_alt = None
            a = min(1.0, dt / self.tau)
            self.vel += a * (want - self.vel)
            self.yaw_rate += a * (want_yaw - self.yaw_rate)
            self.pos += self.vel * dt
            self.pos[2] = max(0.0, self.pos[2])
            self.yaw = (self.yaw + self.yaw_rate * dt + math.pi) % (2 * math.pi) - math.pi
            self.trail.append(self.pos.copy())
            self.trail = self.trail[-3000:]

    def state(self):
        with self._lock:
            x, y, z = self.pos
            vx, vy, vz = self.vel
            return {"x": x, "y": y, "z": z, "yaw": self.yaw, "airborne": self.airborne,
                    "vx": vx, "vy": vy, "vz": vz,
                    "busy": self._target_alt is not None}
