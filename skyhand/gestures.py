"""Static hand-shape gestures from finger curls, with debouncing and hold-to-trigger."""
import time
from enum import Enum

OPEN_T, CLOSED_T = 0.35, 0.6


class G(str, Enum):
    NONE = "none"
    OPEN = "open"            # flat hand: fly (direct control armed)
    FIST = "fist"            # brake / hover in place (dead-man)
    POINT = "point"          # index only: point at target / draw sketch
    PEACE = "peace"          # index + middle: mark waypoint / snapshot
    THUMB = "thumbs_up"      # thumb only: hold -> take off
    PINKY = "pinky"          # little only: hold -> land
    ROCK = "rock"            # index + little: hold -> toggle sketch recording / replay


# finger order: Thumb, Index, Middle, Ring, Little. "o" = open, "c" = closed, "*" = don't care
PATTERNS = [
    (G.OPEN, "*oooo"),  # thumb is free: it drives climb/descend in direct mode
    (G.FIST, "*cccc"),
    (G.POINT, "*occc"),
    (G.PEACE, "*oocc"),
    (G.ROCK, "*occo"),
    (G.THUMB, "occcc"),
    (G.PINKY, "*ccco"),
]
# FIST must not swallow THUMB/POINT etc -> check more specific patterns first
_ORDER = [G.THUMB, G.POINT, G.PEACE, G.ROCK, G.PINKY, G.OPEN, G.FIST]
_PAT = dict(PATTERNS)
FINGER_ORDER = ("Thumb", "Index", "Middle", "Ring", "Little")


def classify(curls: dict) -> G:
    state = "".join("o" if curls[f] < OPEN_T else "c" if curls[f] > CLOSED_T else "?" for f in FINGER_ORDER)
    for g in _ORDER:
        if all(p == "*" or p == s for p, s in zip(_PAT[g], state)):
            return g
    return G.NONE


class GestureTracker:
    """Debounces gestures and emits one-shot 'held' events."""

    def __init__(self, stable_s: float = 0.12, hold_s: float = 1.0):
        self.stable_s, self.hold_s = stable_s, hold_s
        self.current = G.NONE
        self._cand, self._cand_t = G.NONE, 0.0
        self._since = time.time()
        self._fired = False

    def update(self, curls: dict, now: float | None = None, gesture: G | None = None):
        """Returns (stable_gesture, held_event_or_None). `gesture` overrides the curl rules (learned model)."""
        now = time.time() if now is None else now
        g = classify(curls) if gesture is None else gesture
        if g != self._cand:
            self._cand, self._cand_t = g, now
        if self._cand != self.current and now - self._cand_t >= self.stable_s:
            self.current, self._since, self._fired = self._cand, now, False
        event = None
        if not self._fired and self.current not in (G.NONE, G.OPEN, G.FIST) and now - self._since >= self.hold_s:
            self._fired, event = True, self.current
        return self.current, event
