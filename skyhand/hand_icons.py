"""Pictogram cards showing the right-hand pose for each flight command and what the drone does.

Views: TOP = looking down on the back of the hand (fingers point up = forward),
SIDE = seen from the right (fingers point right = forward), REAR = from behind the wrist.
"""
from matplotlib import transforms
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SKIN, EDGE, DRONE = "#f1c27d", "#8d5524", "tab:blue"
ON, OFF = "#b6f2b0", "#f7f7f7"


def _box(ax, x, y, w, h, tf, r=0.06, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=kw.get("fc", SKIN), ec=EDGE, lw=1.2, transform=tf))


def _arrow(ax, a, b, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>,head_width=6,head_length=8", lw=3,
                                 color=DRONE, connectionstyle=f"arc3,rad={rad}"))


def top_hand(ax, rot=0.0, fingers=(1, 1, 1, 1), thumb="side"):
    """fingers = index..little extended flags; thumb = side | out | in."""
    tf = transforms.Affine2D().rotate_deg(rot) + ax.transData
    lengths = (0.55, 0.62, 0.55, 0.42)
    for x, L, ext in zip((-0.26, -0.09, 0.08, 0.24), lengths, fingers):
        _box(ax, x - 0.07, 0.05, 0.14, L if ext else 0.14, tf)
    angle, L = {"side": (125, 0.38), "out": (175, 0.5), "in": (35, 0.32)}[thumb]
    ttf = transforms.Affine2D().rotate_deg(angle).translate(-0.3, -0.32) + tf
    if thumb != "in":
        _box(ax, 0, -0.07, L, 0.14, ttf)
    _box(ax, -0.36, -0.55, 0.72, 0.68, tf, r=0.14)
    if thumb == "in":
        _box(ax, 0, -0.07, L, 0.14, ttf, fc="#e0a96d")


def thumbs_up(ax):
    tf = ax.transData
    _box(ax, -0.18, 0.05, 0.16, 0.6, tf)
    _box(ax, -0.4, -0.5, 0.75, 0.6, tf, r=0.15)
    for y in (-0.38, -0.22, -0.06):
        ax.plot([-0.05, 0.33], [y, y], color=EDGE, lw=1)


def side_hand(ax, pitch=0.0):
    ax.plot([-0.8, 0.8], [0, 0], ":", color="gray", lw=1)
    tf = transforms.Affine2D().rotate_deg(pitch) + ax.transData
    _box(ax, -0.6, -0.09, 1.15, 0.18, tf, r=0.08)
    ax.text(0.62, 0, "fingers", fontsize=6, color=EDGE, va="center",
            transform=transforms.Affine2D().translate(0, 0.12) + tf)


def rear_hand(ax, roll=0.0):
    ax.plot([-0.8, 0.8], [0, 0], ":", color="gray", lw=1)
    tf = transforms.Affine2D().rotate_deg(roll) + ax.transData
    _box(ax, -0.55, -0.08, 1.1, 0.16, tf, r=0.07)
    _box(ax, -0.62, -0.2, 0.14, 0.2, tf, r=0.05)  # thumb on the left (right hand, palm down)
    ax.text(-0.68, 0.18, "thumb", fontsize=6, color=EDGE, transform=tf)


def _intent(st):
    return st.get("intent") or st["cmd"]


def _cmd(st, name, sign=1, th=0.1):
    return sign * getattr(_intent(st), name) > th


# (title, how, draw function, "is this command active right now?")
CARDS = [
    ("TAKE OFF", "thumbs-up, hold 1 s", lambda ax: (thumbs_up(ax), _arrow(ax, (0.7, -0.6), (0.7, 0.6))),
     lambda s: s["gesture"] == "thumbs_up"),
    ("FORWARD", "open hand, fingers DOWN", lambda ax: (side_hand(ax, -30), _arrow(ax, (-0.6, -0.8), (0.7, -0.8))),
     lambda s: _cmd(s, "vx")),
    ("BACKWARD", "open hand, fingers UP", lambda ax: (side_hand(ax, 30), _arrow(ax, (0.6, -0.8), (-0.7, -0.8))),
     lambda s: _cmd(s, "vx", -1)),
    ("SLIDE RIGHT", "roll hand right (left = mirror)",
     lambda ax: (rear_hand(ax, -25), _arrow(ax, (-0.6, -0.8), (0.7, -0.8))), lambda s: abs(_intent(s).vy) > 0.1),
    ("TURN RIGHT", "rotate hand right (left = mirror)",
     lambda ax: (top_hand(ax, -30), _arrow(ax, (-0.55, -0.85), (0.6, -0.65), rad=0.35)),
     lambda s: abs(_intent(s).yaw_rate) > 0.1),
    ("CLIMB", "open hand, thumb OUT",
     lambda ax: (top_hand(ax, thumb="out"), _arrow(ax, (0.75, -0.6), (0.75, 0.6))), lambda s: _cmd(s, "vz")),
    ("DESCEND", "open hand, thumb IN palm",
     lambda ax: (top_hand(ax, thumb="in"), _arrow(ax, (0.75, 0.6), (0.75, -0.6))), lambda s: _cmd(s, "vz", -1)),
    ("STOP / HOVER", "make a FIST (always works)", lambda ax: top_hand(ax, fingers=(0, 0, 0, 0), thumb="in"),
     lambda s: s["gesture"] == "fist"),
    ("LAND", "only pinky out, hold 1 s",
     lambda ax: (top_hand(ax, fingers=(0, 0, 0, 1), thumb="in"), _arrow(ax, (0.75, 0.6), (0.75, -0.6))),
     lambda s: s["gesture"] == "pinky"),
]


DOWN_CARD = ("DOWN A BIT", "point, finger angled DOWN",
             lambda ax: (top_hand(ax, fingers=(1, 0, 0, 0), thumb="in"), _arrow(ax, (0.75, 0.6), (0.75, -0.6))),
             lambda s: _cmd(s, "vz", -1))
LAND_CARD = ("LAND", "fist or pinky, hold 1 s",
             lambda ax: (top_hand(ax, fingers=(0, 0, 0, 0), thumb="in"), _arrow(ax, (0.75, 0.6), (0.75, -0.6))),
             lambda s: s["gesture"] in ("fist", "pinky"))
UP_CARD = ("UP A BIT", "point, finger angled UP",
           lambda ax: (top_hand(ax, fingers=(1, 0, 0, 0), thumb="in"), _arrow(ax, (0.75, -0.6), (0.75, 0.6))),
           lambda s: _cmd(s, "vz"))
SAFE_TITLES = ("TAKE OFF", "FORWARD", "BACKWARD", "TURN RIGHT")

SIMPLE_TITLES = ("TAKE OFF", "FORWARD", "BACKWARD", "TURN RIGHT", "STOP / HOVER")


def make_cards(fig, subgrid, simple=False, safe=False):
    """Draw the 3x3 card grid into subgrid; returns update(status) that highlights active cards."""
    axes = []
    cards = CARDS
    if safe:  # safe GUI: take off, forward/back, turn (peace), point = down a bit, land
        cards = [c for c in CARDS if c[0] in SAFE_TITLES]
        cards[3] = ("TURN RIGHT", "PEACE sign, roll hand right (left = mirror)",
                    lambda ax: (top_hand(ax, fingers=(1, 1, 0, 0), thumb="in"),
                                _arrow(ax, (-0.55, -0.85), (0.6, -0.65), rad=0.35)), cards[3][3])
        cards[0] = ("TAKE OFF", "thumbs-up, hold 1 s", *cards[0][2:])
        cards += [UP_CARD, DOWN_CARD, LAND_CARD]
    elif simple:  # --simple: 3 hand shapes, tilt only; thumbs-up toggles take off / land
        cards = [c for c in CARDS if c[0] in SIMPLE_TITLES]
        cards[0] = ("TAKE OFF / LAND", "thumbs-up, hold 1 s", *cards[0][2:])
        cards[3] = ("TURN RIGHT", "PEACE sign, roll hand right (left = mirror)",
                    lambda ax: (top_hand(ax, fingers=(1, 1, 0, 0), thumb="in"),
                                _arrow(ax, (-0.55, -0.85), (0.6, -0.65), rad=0.35)), cards[3][3])
    for i, (title, how, draw, _) in enumerate(cards):
        ax = fig.add_subplot(subgrid[i // 3, i % 3])
        ax.set_xlim(-1, 1); ax.set_ylim(-1.15, 1.0); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#cccccc")
        ax.set_facecolor(OFF)
        ax.set_title(title, fontsize=9, weight="bold", pad=2)
        ax.set_xlabel(how, fontsize=7, labelpad=2)
        draw(ax)
        axes.append(ax)

    def update(status):
        for ax, (*_, active) in zip(axes, cards):
            try:
                on = active(status)
            except (AttributeError, KeyError, TypeError):
                on = False
            ax.set_facecolor(ON if on else OFF)

    return update
