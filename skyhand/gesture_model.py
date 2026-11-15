"""Per-pilot gesture classifier: a tiny numpy neural network on finger joint rotations.

Features are each finger bone's rotation relative to its parent bone (hand -> proximal -> medial -> distal),
as rotation vectors: 15 bones x 3 = 45 numbers. They don't change when the whole hand tilts or turns,
so the same gesture is recognized in any hand orientation and no flat-hand calibration is needed.
Train with `./fly.sh --train` (scripts/train_gestures.py); SkyHand uses it via --gesture-model.

With the SDK bridge's raw data (frame.raw) we add 39 IMU numbers: each sensor's gravity direction from its
accelerometer (7 x 3) and each finger sensor's position relative to the hub (6 x 3, tracked independently
of the skeleton solver). A model trained with them (GestureNet.imu) needs them at run time too.
"""
import numpy as np

from . import quat as Q
from .gestures import G
from .rokoko import FINGERS, JOINTS, HandFrame

N_FEATURES = len(FINGERS) * len(JOINTS) * 3
IMU_PARTS = ("hub", "arm", "thumb", "index", "middle", "ring", "pinky")
N_IMU = len(IMU_PARTS) * 3 + (len(IMU_PARTS) - 1) * 3


def imu_features(frame: HandFrame) -> np.ndarray | None:
    raw = getattr(frame, "raw", None) or {}
    acc, pos = raw.get("acc") or {}, raw.get("pos") or {}
    if not all(k in acc for k in IMU_PARTS) or not all(k in pos for k in IMU_PARTS[1:]):
        return None
    out = [np.asarray(acc[k], float) / (np.linalg.norm(acc[k]) + 1e-6) for k in IMU_PARTS]
    out += [np.asarray(pos[k][:3], float) * 10.0 for k in IMU_PARTS[1:]]  # m -> dm, ~unit scale
    return np.concatenate(out)


def _rotvec(q):
    q = Q.normalize(q)
    if q[0] < 0:  # q and -q are the same rotation
        q = -q
    s = np.linalg.norm(q[1:])
    return np.zeros(3) if s < 1e-9 else q[1:] / s * 2.0 * np.arctan2(s, q[0])


def features(frame: HandFrame, imu: bool = False) -> np.ndarray | None:
    b = frame.bones
    out = []
    for f in FINGERS:
        parent = "Hand"
        for j in JOINTS:
            name = f + j
            if name not in b or parent not in b:
                return None
            out.append(_rotvec(Q.mul(Q.conj(Q.normalize(b[parent].rotation)), Q.normalize(b[name].rotation))))
            parent = name
    x = np.concatenate(out)
    if imu:
        xi = imu_features(frame)
        return None if xi is None else np.concatenate([x, xi])
    return x


class GestureNet:
    """features -> standardize -> 32 tanh units -> softmax over gesture classes."""

    def __init__(self, classes, hidden=32, seed=0, n_in=N_FEATURES):
        self.classes = list(classes)
        self.imu = n_in > N_FEATURES
        rng = np.random.default_rng(seed)
        k = len(self.classes)
        self.W1 = rng.normal(0, 1 / np.sqrt(n_in), (n_in, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 1 / np.sqrt(hidden), (hidden, k))
        self.b2 = np.zeros(k)
        self.mu, self.sd = np.zeros(n_in), np.ones(n_in)

    def _forward(self, X):
        h = np.tanh((X - self.mu) / self.sd @ self.W1 + self.b1)
        z = h @ self.W2 + self.b2
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return h, p / p.sum(axis=1, keepdims=True)

    def fit(self, X, labels, epochs=400, lr=0.01, l2=1e-3, noise=0.05, seed=0):
        X = np.asarray(X, float)
        y = np.array([self.classes.index(c) for c in labels])
        self.mu, self.sd = X.mean(0), X.std(0) + 0.05
        Y = np.eye(len(self.classes))[y]
        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        rng = np.random.default_rng(seed)
        for t in range(1, epochs + 1):
            Xn = X + rng.normal(0, noise, X.shape)  # small jitter = robustness to sensor noise
            h, p = self._forward(Xn)
            d2 = (p - Y) / len(X)
            dh = d2 @ self.W2.T * (1 - h ** 2)
            Xs = (Xn - self.mu) / self.sd
            grads = [Xs.T @ dh + l2 * self.W1, dh.sum(0), h.T @ d2 + l2 * self.W2, d2.sum(0)]
            for i, (p_, g) in enumerate(zip(params, grads)):  # Adam
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                p_ -= lr * (m[i] / (1 - 0.9 ** t)) / (np.sqrt(v[i] / (1 - 0.999 ** t)) + 1e-8)
        return self

    def predict_proba(self, X):
        return self._forward(np.atleast_2d(np.asarray(X, float)))[1]

    def accuracy(self, X, labels):
        pred = self.predict_proba(X).argmax(1)
        return float(np.mean([self.classes[i] == c for i, c in zip(pred, labels)]))

    def save(self, path):
        np.savez(path, classes=np.array(self.classes), W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 mu=self.mu, sd=self.sd)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        net = cls([str(c) for c in d["classes"]], hidden=d["W1"].shape[1], n_in=d["W1"].shape[0])
        for k in ("W1", "b1", "W2", "b2", "mu", "sd"):
            setattr(net, k, d[k])
        return net


class GestureModel:
    """Frame -> (gesture, confidence). Unsure predictions become NONE, which makes the drone hover."""

    def __init__(self, path, min_conf=0.7):
        self.net = GestureNet.load(path)
        self.min_conf = min_conf

    def __call__(self, frame: HandFrame):
        x = features(frame, self.net.imu)
        if x is None:  # IMU model without raw data (e.g. Studio): no guess = hover
            return G.NONE, 0.0
        p = self.net.predict_proba(x)[0]
        i = int(p.argmax())
        name = self.net.classes[i]
        g = G(name) if name in G._value2member_map_ else G.NONE  # relax, or a class the app has no action for
        return (g if p[i] >= self.min_conf else G.NONE), float(p[i])
