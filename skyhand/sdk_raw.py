"""Everything else the Smartglove sends, besides the solved skeleton (raw Rokoko Device SDK driver stream).

Per glove, ~333 packets/s, sensors: hub (back of the hand), arm, and one per finger. For each:
  ANGULAR_VELOCITY     gyro, rad/s, sensor frame          -> flicks, shakes
  PROPER_ACCELERATION  accelerometer, in g (1.0 at rest)     -> taps
  TRANSFORM            sensor position (m) + rotation, relative to the hub (electromagnetic tracking)
  CUSTOM relative_emf_residual   EMF tracking residual (higher = less trustworthy)
  STATUS_FLAGS         sensor status bits
  ORIENTATION (hub, LTP_NED)     hub rotation in a north-east-down world frame

RawCollector keeps the latest values plus the PEAKS since the last snapshot (a 5 ms tap would be missed if we
only looked at every 2nd packet), and snapshot() returns a JSON-able dict per hand side.
"""
import threading

import numpy as np

G = 1.0  # the accelerometer reports in g (measured: |acc| = 1.00 at rest)
PARTS = {"hub": "hub", "arm": "arm", "thumb_finger": "thumb", "index_finger": "index", "middle_finger": "middle",
         "ring_finger": "ring", "little_finger": "pinky"}


def part_of(target_frame: str | None):
    t = (target_frame or "").lower()
    for k, v in PARTS.items():
        if t.startswith(k):
            return v
    return None


def side_of(definition) -> str | None:
    """'right'/'left' from the device definition, None if the device doesn't say."""
    try:
        from solver_core.smartglove import detect_is_right, device_info_str
        if device_info_str(definition, "hand") is None and "is_right_glove" not in str(definition):
            return None
        return "right" if detect_is_right(definition) else "left"
    except Exception:
        return None


class RawCollector:
    def __init__(self):
        self.lock = threading.Lock()
        self.dev_side = {}
        self.state = {}  # side or device key -> dict

    def _blank(self):
        return {"gyro": {}, "acc": {}, "gyro_peak": {}, "acc_peak": {}, "pos": {}, "emf": {}, "flags": {}}

    def update(self, dev, payload, layout, slots, read_slot):
        """`slots`: the slots of the group that this payload belongs to."""
        if dev not in self.dev_side:
            self.dev_side[dev] = side_of(getattr(layout, "definition", None))
        key = self.dev_side[dev] or f"dev{dev}"
        with self.lock:
            s = self.state.setdefault(key, self._blank())
            for slot in slots:
                e = slot.entry or {}
                part, m = part_of(e.get("target_frame")), e.get("measure_type")
                if part is None:
                    continue
                try:
                    v = [float(x) for x in read_slot(slot, payload)]
                except Exception:
                    continue
                if m == "ANGULAR_VELOCITY" and len(v) == 3:
                    s["gyro"][part] = v
                    s["gyro_peak"][part] = max(s["gyro_peak"].get(part, 0.0), float(np.linalg.norm(v)))
                elif m == "PROPER_ACCELERATION" and len(v) == 3:
                    s["acc"][part] = v
                    jolt = abs(float(np.linalg.norm(v)) - G)  # 0 when only gravity
                    s["acc_peak"][part] = max(s["acc_peak"].get(part, 0.0), jolt)
                elif m == "TRANSFORM" and len(v) == 7 and (e.get("reference_frame") or "").lower() == "hub":
                    s["pos"][part] = v[:3]
                elif m == "CUSTOM" and e.get("custom_label") == "relative_emf_residual" and v:
                    s["emf"][part] = v[0]
                elif m == "STATUS_FLAGS" and v and e.get("reference_frame") is None:
                    s["flags"][part] = int(v[0])

    def snapshot(self, sides_present=("right",)) -> dict:
        """{'right': {...}} with peaks since the last call. Devices without a known side are given the
        hand(s) seen in the skeleton packet."""
        with self.lock:
            out = {}
            unknown = [k for k in self.state if k.startswith("dev")]
            for k, s in self.state.items():
                side = k if not k.startswith("dev") else None
                if side is None and len(unknown) == 1 and len(sides_present) == 1:
                    side = sides_present[0]
                if side:
                    out[side] = {kk: dict(vv) for kk, vv in s.items()}
                s["gyro_peak"], s["acc_peak"] = {}, {}
            return out
