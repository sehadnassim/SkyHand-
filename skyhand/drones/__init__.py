"""Drone backend interface.

Frame convention used everywhere in skyhand ("start frame"):
  x = forward at takeoff, y = right, z = up, yaw + = clockwise (turn right), rad.
Velocity commands are body frame: vx forward, vy right, vz up, yaw_rate + = turn right.
"""
from abc import ABC, abstractmethod

from ..mapping import VelCmd


class Drone(ABC):
    name = "base"

    @abstractmethod
    def send_velocity(self, cmd: VelCmd): ...

    @abstractmethod
    def takeoff(self): ...

    @abstractmethod
    def land(self): ...

    def state(self) -> dict | None:
        """{'x','y','z','yaw','airborne'} if the backend has odometry, else None (open-loop)."""
        return None

    def tick(self, dt: float):
        """Called every control loop iteration (used by the simulator)."""

    def close(self):
        pass


def make_drone(kind: str, **kw) -> Drone:
    if kind == "sim":
        from .sim import SimDrone
        return SimDrone()
    if kind == "dji":
        # Small DJI drones (Mini 3/4 Pro, Neo, ...) have no Python/onboard SDK. We stream velocities over
        # UDP to a phone app (DJI MSDK v5 Virtual Stick) or to your existing velocity code.
        from .udp import UdpDrone
        return UdpDrone(**kw)
    if kind == "mqtt":
        from .mqtt import MqttDrone
        return MqttDrone(**kw)
    if kind == "ros2":
        from .ros2 import Ros2Drone
        return Ros2Drone(**kw)
    if kind == "udp":
        from .udp import UdpDrone
        return UdpDrone(**kw)
    raise ValueError(f"unknown drone backend {kind}")
