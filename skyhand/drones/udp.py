"""Generic UDP JSON bridge: the easiest way to hook into an existing DJI velocity-control script.

Sends one datagram per control tick:
  {"type": "vel", "vx": .., "vy": .., "vz": .., "yaw_rate": ..}   (body frame, y right, yaw + = right)
  {"type": "takeoff"} / {"type": "land"}
Optionally listens for state on `state_port`: {"x","y","z","yaw"} in the same convention.
The receiver gets one JSON velocity command per packet.
"""
import json
import socket
import threading

from ..mapping import VelCmd
from . import Drone


class UdpDrone(Drone):
    name = "udp"

    def __init__(self, host="127.0.0.1", port=15000, state_port=0, **_):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._state = None
        if state_port:
            self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.rx.bind(("0.0.0.0", state_port))
            threading.Thread(target=self._rx_loop, daemon=True).start()

    def _rx_loop(self):
        while True:
            try:
                data, _ = self.rx.recvfrom(4096)
                s = json.loads(data)
                s.setdefault("airborne", s.get("z", 0) > 0.2)
                self._state = s
            except (OSError, ValueError):
                continue

    def _tx(self, msg):
        self.sock.sendto(json.dumps(msg).encode(), self.addr)

    def send_velocity(self, cmd: VelCmd):
        self._tx({"type": "vel", "vx": cmd.vx, "vy": cmd.vy, "vz": cmd.vz, "yaw_rate": cmd.yaw_rate})

    def takeoff(self):
        self._tx({"type": "takeoff"})

    def land(self):
        self._tx({"type": "land"})

    def state(self):
        return self._state

    def close(self):
        self.send_velocity(VelCmd())
