"""Receiver for Rokoko Studio "Custom streaming" (JSON v3, optionally LZ4).

In Rokoko Studio: Livestreaming -> Custom -> enable, set IP of this machine,
port 14043 (default), data format JSON (v3). LZ4 is supported if `lz4` is installed.

Packet layout (v3):
  {"version": 3, "scene": {"actors": [{"name": "...",
     "body": {"rightHand": {"position": {x,y,z}, "rotation": {x,y,z,w}}, ...}}]}}
Rotations/positions are global, Y-up, meters.
"""
import json
import socket
import threading
import time
from dataclasses import dataclass, field

import numpy as np

FINGERS = ("Thumb", "Index", "Middle", "Ring", "Little")
JOINTS = ("Proximal", "Medial", "Distal")


@dataclass
class Bone:
    position: np.ndarray
    rotation: np.ndarray  # (w, x, y, z)


@dataclass
class HandFrame:
    side: str  # "right" | "left"
    bones: dict = field(default_factory=dict)  # name without side prefix, e.g. "Hand", "IndexDistal"
    stamp: float = 0.0
    raw: dict | None = None  # IMU etc. from scripts/sdk_bridge.py (skyhand/sdk_raw.py), None from Rokoko Studio


def _decode(raw: bytes) -> dict:
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        import lz4.frame  # only needed when Studio sends LZ4

        return json.loads(lz4.frame.decompress(raw).decode("utf-8"))


def parse_packet(data: dict, side: str = "right", actor_index: int = 0) -> HandFrame | None:
    actors = data.get("scene", {}).get("actors", [])
    if len(actors) <= actor_index:
        return None
    actor = actors[actor_index]
    body = actor["body"] if data.get("version", 3) >= 3 else actor
    prefix = side
    names = ["Hand"] + [f"{f}{j}" for f in FINGERS for j in JOINTS]
    frame = HandFrame(side=side, stamp=time.time())
    for name in names:
        b = body.get(prefix + name)
        if b is None:
            continue
        p, r = b.get("position", {}), b["rotation"]
        frame.bones[name] = Bone(
            position=np.array([p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)]),
            rotation=np.array([r["w"], r["x"], r["y"], r["z"]]),
        )
    frame.raw = (data.get("skyhand_raw") or {}).get(side)
    return frame if "Hand" in frame.bones else None


class RokokoReceiver:
    """Background UDP listener keeping the latest hand frame."""

    def __init__(self, port: int = 14043, side: str = "right", actor_index: int = 0, host: str = "0.0.0.0"):
        self.side, self.actor_index = side, actor_index
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.settimeout(0.5)
        self._latest: HandFrame | None = None
        self._lock = threading.Lock()
        self._running = True
        self.packets = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._running:
            try:
                raw, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                frame = parse_packet(_decode(raw), self.side, self.actor_index)
            except (KeyError, ValueError, ImportError):
                continue
            if frame:
                with self._lock:
                    self._latest = frame
                    self.packets += 1

    def latest(self, max_age: float = 0.3) -> HandFrame | None:
        with self._lock:
            f = self._latest
        return f if f and time.time() - f.stamp <= max_age else None

    def close(self):
        self._running = False
        self._sock.close()
