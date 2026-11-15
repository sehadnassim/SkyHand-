"""Minimal quaternion helpers. Convention: numpy array (w, x, y, z)."""
import numpy as np


def normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def rotate(q, v):
    qv = np.array([0.0, *v])
    return mul(mul(q, qv), conj(q))[1:]


def from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), *(axis * s)])


def angle_between(a, b):
    """Rotation angle (rad) of the relative rotation between two orientations."""
    d = abs(float(np.dot(normalize(a), normalize(b))))
    return 2.0 * np.arccos(min(1.0, d))
