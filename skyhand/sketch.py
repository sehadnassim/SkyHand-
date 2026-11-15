"""Air-drawing: the index-finger ray acts like a laser pointer on a virtual wall.
Works with the glove IMUs only (no position tracking needed). The drawn shape is
simplified and later flown by the drone in its vertical plane, scaled up."""
import math

import numpy as np


class Sketch:
    def __init__(self, wall_dist: float = 1.0):
        self.wall = wall_dist
        self.raw: list = []
        self.recording = False

    def start(self):
        self.raw, self.recording = [], True

    def add(self, point_yaw: float, point_pitch: float):
        if self.recording:
            self.raw.append((self.wall * math.tan(np.clip(point_yaw, -1.2, 1.2)),
                             self.wall * math.tan(np.clip(point_pitch, -1.2, 1.2))))

    def stop(self, eps: float = 0.03, max_pts: int = 40) -> list:
        """Returns the shape normalized to unit size (max extent 1), starting at (0, 0)."""
        self.recording = False
        if len(self.raw) < 3:
            return []
        p = np.array(self.raw)
        p -= p[0]
        ext = np.ptp(p, axis=0).max()
        if ext < 1e-3:
            return []
        p /= ext
        p = rdp(p, eps)
        if len(p) > max_pts:
            p = p[np.linspace(0, len(p) - 1, max_pts).astype(int)]
        return [tuple(map(float, q)) for q in p]


def rdp(pts: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker polyline simplification."""
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    n = np.linalg.norm(ab)
    if n < 1e-9:
        d = np.linalg.norm(pts - a, axis=1)
    else:
        d = np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / n
    i = int(np.argmax(d))
    if d[i] > eps:
        return np.vstack([rdp(pts[: i + 1], eps)[:-1], rdp(pts[i:], eps)])
    return np.vstack([a, b])
