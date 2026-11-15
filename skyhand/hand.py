"""Turn raw glove bones into control features: hand tilt, finger curls, pointing ray.

All angles are relative to a calibration pose:
  hand flat, palm down, fingers straight and pointing "forward" (toward the drone's nose direction).
Using bone positions to find the forward direction makes this independent of Rokoko's local bone axes.
"""
from dataclasses import dataclass

import numpy as np

from . import quat as Q
from .rokoko import FINGERS, JOINTS, HandFrame

UP = np.array([0.0, 1.0, 0.0])  # Rokoko world is Y-up
DEFAULT_CURL_RANGE = {"Thumb": 1.4, "Index": 3.4, "Middle": 3.4, "Ring": 3.4, "Little": 3.4}


@dataclass
class HandState:
    pitch: float  # rad, + = fingers up
    roll: float   # rad, + = hand rolled to the right
    yaw: float    # rad, + = hand turned to the right
    curls: dict   # finger -> 0 (straight) .. 1 (fully curled)
    point_yaw: float    # rad, index finger ray relative to calibration forward
    point_pitch: float  # rad
    height: float       # m, hand height relative to calibration (needs Coil/suit for real meaning)
    thumb: float = 0.0  # signed thumb bend vs calibration: < 0 = stuck out further, > 0 = tucked in


def _finger_bend(frame: HandFrame, finger: str) -> float:
    chain = ["Hand"] + [finger + j for j in JOINTS]
    total = 0.0
    for a, b in zip(chain, chain[1:]):
        if a in frame.bones and b in frame.bones:
            total += Q.angle_between(frame.bones[a].rotation, frame.bones[b].rotation)
    return total


class HandModel:
    def __init__(self, curl_range: dict | None = None, local_forward=(0.0, 0.0, 1.0)):
        self.curl_range = dict(curl_range or DEFAULT_CURL_RANGE)
        self.local_forward = np.asarray(local_forward, dtype=float)
        self.calibrated = False

    def calibrate(self, frame: HandFrame):
        hand = frame.bones["Hand"]
        self.q0 = Q.normalize(hand.rotation)
        self.q0_index = Q.normalize(frame.bones.get("IndexProximal", hand).rotation)
        self.h0 = float(hand.position[1])
        fwd = None
        tip = frame.bones.get("MiddleProximal") or frame.bones.get("IndexProximal")
        if tip is not None:
            d = tip.position - hand.position
            if np.linalg.norm(d) > 1e-4:
                fwd = d
        if fwd is None:  # no usable positions -> trust configured local axis
            fwd = Q.rotate(self.q0, self.local_forward)
        fwd = fwd - UP * np.dot(fwd, UP)
        self.fwd0 = fwd / np.linalg.norm(fwd)
        self.right0 = np.cross(self.fwd0, UP)
        self.bend0 = {f: _finger_bend(frame, f) for f in FINGERS}
        self.calibrated = True

    def calibrate_fist(self, frame: HandFrame):
        for f in FINGERS:
            self.curl_range[f] = max(0.3, _finger_bend(frame, f) - self.bend0[f])

    def _angles(self, fwd, back=None):
        pitch = float(np.arcsin(np.clip(fwd @ UP, -1, 1)))
        yaw = float(np.arctan2(fwd @ self.right0, fwd @ self.fwd0))
        roll = 0.0 if back is None else float(np.arcsin(np.clip(back @ self.right0, -1, 1)))
        return pitch, roll, yaw

    def update(self, frame: HandFrame) -> HandState:
        if not self.calibrated:
            self.calibrate(frame)
        hand = frame.bones["Hand"]
        dq = Q.mul(Q.normalize(hand.rotation), Q.conj(self.q0))
        pitch, roll, yaw = self._angles(Q.rotate(dq, self.fwd0), Q.rotate(dq, UP))

        idx = frame.bones.get("IndexProximal", hand)
        dqi = Q.mul(Q.normalize(idx.rotation), Q.conj(self.q0_index))
        p_pitch, _, p_yaw = self._angles(Q.rotate(dqi, self.fwd0))

        curls = {
            f: float(np.clip((_finger_bend(frame, f) - self.bend0[f]) / self.curl_range[f], 0.0, 1.0))
            for f in FINGERS
        }
        thumb = (_finger_bend(frame, "Thumb") - self.bend0["Thumb"]) / self.curl_range["Thumb"]
        return HandState(pitch, roll, yaw, curls, p_yaw, p_pitch, float(hand.position[1]) - self.h0, thumb)
