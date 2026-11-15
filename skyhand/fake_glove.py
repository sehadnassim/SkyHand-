"""Synthetic Rokoko glove: builds Rokoko Studio JSON v3 packets from (pitch, roll, yaw, curls)."""
import json
import math
import socket

import numpy as np

from . import quat as Q
from .rokoko import FINGERS, JOINTS

FWD = np.array([0.0, 0.0, 1.0])
UP = np.array([0.0, 1.0, 0.0])
RIGHT = np.cross(FWD, UP)
RANGE = {"Thumb": 1.4, "Index": 3.4, "Middle": 3.4, "Ring": 3.4, "Little": 3.4}
OFFS = {"Thumb": -0.05, "Index": -0.02, "Middle": 0.0, "Ring": 0.02, "Little": 0.04}  # along RIGHT

SHAPES = {  # finger curls 0..1 in order Thumb, Index, Middle, Ring, Little
    "open": (0.4, 0, 0, 0, 0),
    "climb": (0.0, 0, 0, 0, 0),
    "descend": (0.9, 0, 0, 0, 0),
    "fist": (0.8, 1, 1, 1, 1),
    "point": (0.8, 0, 1, 1, 1),
    "peace": (0.8, 0, 0, 1, 1),
    "rock": (0.8, 0, 1, 1, 0),
    "thumbs_up": (0.0, 1, 1, 1, 1),
    "pinky": (0.8, 1, 1, 1, 0),
}


def hand_quat(pitch, roll, yaw):
    qy = Q.from_axis_angle(UP, -yaw)       # + yaw turns fingers toward RIGHT
    qp = Q.from_axis_angle(RIGHT, pitch)   # + pitch lifts fingers
    qr = Q.from_axis_angle(FWD, roll)      # + roll tilts back of the hand toward RIGHT
    return Q.normalize(Q.mul(qy, Q.mul(qp, qr)))


def build_packet(pitch=0.0, roll=0.0, yaw=0.0, curls=SHAPES["open"], side="right", hand_pos=(0.0, 1.2, 0.0),
                 point_pitch=None, point_yaw=None):
    qh = hand_quat(pitch, roll, yaw)
    hp = np.asarray(hand_pos, dtype=float)
    body = {}

    def put(name, pos, q):
        body[side + name] = {"position": dict(zip("xyz", map(float, pos))),
                             "rotation": dict(zip("wxyz", map(float, q)))}

    put("Hand", hp, qh)
    for f, c in zip(FINGERS, curls):
        theta = c * RANGE[f] / 3.0
        base = qh
        if f == "Index" and point_yaw is not None:  # independent index ray (for drawing)
            base = hand_quat(point_pitch or 0.0, 0.0, point_yaw)
        # the thumb starts near the wrist, so its tip is ~8 cm from the index tip like a real hand (not a pinch)
        pos = hp + Q.rotate(qh, FWD * (0.02 if f == "Thumb" else 0.09) + RIGHT * OFFS[f])
        for k, j in enumerate(JOINTS, start=1):
            q = Q.mul(base, Q.from_axis_angle(-RIGHT, theta * k))
            put(f + j, pos, q)
            pos = pos + Q.rotate(q, FWD * 0.03)
    return {"version": 3, "fps": 60, "scene": {"timestamp": 0, "actors": [{"name": "fake", "body": body}]}}


class FakeGlove:
    def __init__(self, host="127.0.0.1", port=14043):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, **kw):
        self.sock.sendto(json.dumps(build_packet(**kw)).encode(), self.addr)


def circle_drawing(t, period=4.0, r=0.35):
    a = 2 * math.pi * t / period
    return {"point_yaw": r * math.sin(a), "point_pitch": r * (1 - math.cos(a))}
