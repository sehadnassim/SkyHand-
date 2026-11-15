"""What to record, and which gestures to keep.

Recording covers every candidate gesture in every hand pose (CONDITIONS). Afterwards:
  1. each pose is left out once and predicted by a model trained on the others (honest check:
     does a gesture still work in a pose the model never saw),
  2. frames the model is sure are something else are dropped (transitions, wrong shape),
  3. gestures that get confused with each other are dropped, keeping the ones the app needs,
  4. weak (pose, gesture) cells are what `--add` asks for next time.
"""
import numpy as np

from .gesture_model import GestureNet

# name, finger pattern (thumb index middle ring little; o = straight, c = curled), how to make it
CANDIDATES = [
    ("open", "ooooo", "OPEN hand: all fingers straight and together"),
    ("fist", "ccccc", "FIST: all fingers curled tight, thumb over them"),
    ("thumbs_up", "occcc", "THUMBS-UP: fist with the thumb sticking up"),
    ("peace", "coocc", "PEACE: index + middle straight, others curled"),
    ("point", "coccc", "POINT: only the index finger straight"),
    ("pinky", "cccco", "PINKY: fist with only the little finger straight"),
    ("rock", "cocco", "ROCK: index + little straight, others curled"),
    ("call", "occco", "CALL ME: thumb + little straight, others curled"),
    ("gun", "ooccc", "L / GUN: thumb + index straight, others curled"),
    ("three", "coooc", "THREE: index + middle + ring straight"),
    ("relax", "-----", "RELAXED: hand loose, fingers half bent, like resting (= no command)"),
]
REQUIRED = ("open", "fist", "thumbs_up", "peace")  # what the app uses in simple mode
REJECT = "relax"  # trained as a class, but means "no gesture" (hover)

CONDITIONS = [
    ("flat", "palm down, hand still in front of you"),
    ("tilt", "palm down, slowly tilt forward/back and roll left/right"),
    ("side", "hand on its side (palm facing left)"),
    ("up", "palm up"),
    ("high_low", "move the arm slowly up high, then down low"),
    ("moving", "move the arm around like you're flying (sweeps, turns)"),
]
QUICK = ("flat", "tilt")


def oof_proba(X, labels, groups, classes, epochs=250):
    """Out-of-fold class probabilities: every group (pose) is predicted by a model that never saw it."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    P = np.zeros((len(X), len(classes)))
    for g in np.unique(groups):
        te = groups == g
        tr_classes = sorted(set(labels[~te]))
        if not te.any() or len(tr_classes) < 2:
            continue
        net = GestureNet(tr_classes, n_in=X.shape[1]).fit(X[~te], labels[~te], epochs=epochs)
        p = net.predict_proba(X[te])
        for j, c in enumerate(tr_classes):
            P[te, classes.index(c)] = p[:, j]
    return P


def clean(labels, P, classes, min_true=0.1, max_drop=0.15):
    """Mask of frames to keep. Drops frames where the true class got < min_true out of fold (usually the
    hand moving into the shape or the wrong shape), at most max_drop of each class so a hard gesture
    isn't silently deleted."""
    labels = np.asarray(labels)
    keep = np.ones(len(labels), bool)
    for c in classes:
        idx = np.where(labels == c)[0]
        pt = P[idx, classes.index(c)]
        bad = idx[pt < min_true]
        worst = idx[np.argsort(pt)[: int(max_drop * len(idx))]]
        keep[np.intersect1d(bad, worst)] = False
    return keep


def confusion(labels, P, classes):
    """C[i, j] = share of class i frames predicted as j."""
    labels = np.asarray(labels)
    pred = P.argmax(1)
    C = np.zeros((len(classes), len(classes)))
    for i, c in enumerate(classes):
        m = labels == c
        if m.any():
            C[i] = np.bincount(pred[m], minlength=len(classes)) / m.sum()
    return C


def pick(C, classes, required=REQUIRED, max_mix=0.03):
    """Keep required gestures, then add candidates, least confused first, while every kept pair mixes
    up less than max_mix (both directions added). Returns (kept, {dropped: (look-alike, mix)})."""
    mix = C + C.T
    np.fill_diagonal(mix, 0)
    kept = [c for c in required if c in classes] + ([REJECT] if REJECT in classes else [])
    dropped = {}
    rest = [c for c in classes if c not in kept]
    rest.sort(key=lambda c: max(mix[classes.index(c), classes.index(k)] for k in kept))
    for c in rest:
        i = classes.index(c)
        worst = max(kept, key=lambda k: mix[i, classes.index(k)])
        m = mix[i, classes.index(worst)]
        if m > max_mix:
            dropped[c] = (worst, float(m))
        else:
            kept.append(c)
    return kept, dropped


def weak_cells(labels, groups, P, classes, min_acc=0.95, min_frames=40):
    """(pose, gesture) cells to record again: low out-of-fold accuracy or too few frames."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    pred = P.argmax(1)
    out = []
    for g in [c for c, _ in CONDITIONS]:
        for c in classes:
            m = (labels == c) & (groups == g)
            if m.sum() < min_frames:
                out.append((g, c, None))
                continue
            acc = float(np.mean(pred[m] == classes.index(c)))
            if acc < min_acc:
                out.append((g, c, acc))
    return out


def pca_2d(X):
    """Standardize, then SVD -> first two principal components (for the cluster plot)."""
    Z = (X - X.mean(0)) / (X.std(0) + 1e-6)
    _, s, Vt = np.linalg.svd(Z, full_matrices=False)
    return Z @ Vt[:2].T, float((s[:2] ** 2).sum() / (s ** 2).sum())
