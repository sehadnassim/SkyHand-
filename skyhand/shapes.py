"""Exact flight shapes. The LLM only picks the name and size; the geometry comes from here.

Every shape starts at (0, 0, 0) and is in the drone's frame at the start: x forward, y right, z up (m).
Points are spaced evenly (STEP m) so the ride can move along them at a given speed.
"""
import numpy as np

STEP = 0.05
SHAPES = ("circle", "figure8", "square", "triangle", "zigzag", "spiral", "heart")


def _poly(corners):
    c = np.asarray(corners, float)
    return np.vstack([np.linspace(a, b, 50, endpoint=False) for a, b in zip(c, c[1:])] + [c[-1:]])


def _raw(shape: str, size: float) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, 600)
    z0 = np.zeros_like(t)
    if shape == "circle":  # size = diameter, curves to the right
        r = size / 2
        return np.c_[r * np.sin(t), r - r * np.cos(t), z0]
    if shape == "figure8":  # size = length; crosses itself at the start point
        return np.c_[size / 2 * np.sin(t), size / 4 * np.sin(2 * t), z0]
    if shape == "square":
        s = size
        return _poly([(0, 0, 0), (s, 0, 0), (s, s, 0), (0, s, 0), (0, 0, 0)])
    if shape == "triangle":
        s = size
        return _poly([(0, 0, 0), (s, 0, 0), (s / 2, s * np.sqrt(3) / 2, 0), (0, 0, 0)])
    if shape == "zigzag":  # size = width; 2 x size long, ends ahead of the start
        n = 4
        x = np.linspace(0, 2 * size, n + 2)
        y = np.r_[0, [(size / 2) * (-1) ** i for i in range(n)], 0]
        return _poly(np.c_[x, y, np.zeros(n + 2)])
    if shape == "spiral":  # size = diameter, 2 turns, climbs size / 2
        tt = np.linspace(0, 4 * np.pi, 1200)
        r = size / 2
        return np.c_[r * np.sin(tt), r - r * np.cos(tt), tt / (4 * np.pi) * size / 2]
    if shape == "heart":  # vertical, in front of the pilot: start at the bottom tip, size = height
        tt = np.linspace(np.pi, 3 * np.pi, 800)
        right = 16 * np.sin(tt) ** 3
        up = 13 * np.cos(tt) - 5 * np.cos(2 * tt) - 2 * np.cos(3 * tt) - np.cos(4 * tt)
        k = size / (up.max() - up.min())
        return np.c_[np.zeros_like(tt), right * k, (up - up[0]) * k]
    raise ValueError(f"unknown shape {shape!r}")


def make(shape: str, size: float = 2.0, mirror: bool = False) -> np.ndarray:
    """(N, 3) points, STEP apart along the path. mirror = flip left/right."""
    p = _raw(shape, float(np.clip(size, 0.5, 6.0)))
    p = p - p[0]
    if mirror:
        p[:, 1] = -p[:, 1]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    u = np.arange(0, s[-1], STEP)
    u = np.r_[u, s[-1]] if s[-1] - u[-1] > 1e-6 else u
    return np.c_[[np.interp(u, s, p[:, i]) for i in range(3)]].T


def to_world(pts: np.ndarray, x0, y0, z0, yaw0) -> np.ndarray:
    """Shape frame (forward, right, up) at the start pose -> world (x, y, z), same yaw convention as planner."""
    c, s = np.cos(yaw0), np.sin(yaw0)
    f, r, u = pts[:, 0], pts[:, 1], pts[:, 2]
    return np.c_[x0 + f * c - r * s, y0 + f * s + r * c, z0 + u]
