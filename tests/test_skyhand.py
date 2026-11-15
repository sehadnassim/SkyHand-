import numpy as np
import math

import pytest

from skyhand import llm
from skyhand.drones.sim import SimDrone
from skyhand.fake_glove import SHAPES, build_packet
from skyhand.gestures import G, GestureTracker, classify
from skyhand.hand import HandModel
from skyhand.mapping import HandToVelocity, MapConfig
from skyhand.planner import Context, Executor, sanitize
from skyhand.rokoko import parse_packet
from skyhand.sketch import Sketch


def _model():
    m = HandModel()
    m.calibrate(parse_packet(build_packet()))
    return m


@pytest.mark.parametrize("kw,idx,expected", [
    ({"pitch": math.radians(-30)}, 0, -30), ({"roll": math.radians(20)}, 1, 20), ({"yaw": math.radians(40)}, 2, 40),
])
def test_hand_angles_from_rokoko_packet(kw, idx, expected):
    hs = _model().update(parse_packet(build_packet(**kw)))
    angles = [math.degrees(a) for a in (hs.pitch, hs.roll, hs.yaw)]
    assert angles[idx] == pytest.approx(expected, abs=1.0)
    assert sum(abs(a) for i, a in enumerate(angles) if i != idx) < 1.0


def test_calibration_is_independent_of_operator_heading():
    m = HandModel()
    m.calibrate(parse_packet(build_packet(yaw=math.radians(120))))
    hs = m.update(parse_packet(build_packet(yaw=math.radians(120), pitch=math.radians(-25))))
    assert math.degrees(hs.pitch) == pytest.approx(-25, abs=1)
    assert abs(hs.yaw) < 0.02


@pytest.mark.parametrize("shape", [s for s in SHAPES if s not in ("climb", "descend")])
def test_gestures(shape):
    hs = _model().update(parse_packet(build_packet(curls=SHAPES[shape])))
    assert classify(hs.curls).value == ("open" if shape == "open" else shape)


def test_hold_event_fires_once():
    tr = GestureTracker(stable_s=0.1, hold_s=1.0)
    curls = dict(zip(("Thumb", "Index", "Middle", "Ring", "Little"), (0, 1, 1, 1, 1)))
    events = [tr.update(curls, now=t / 10)[1] for t in range(30)]
    assert [e for e in events if e] == [G.THUMB]


def test_mapping_tilt_and_thumb():
    m = _model()
    f = HandToVelocity(MapConfig(smoothing=0.0))
    cmd = f(m.update(parse_packet(build_packet(pitch=math.radians(-35), curls=SHAPES["climb"]))))
    assert cmd.vx == pytest.approx(1.5) and cmd.vz > 0 and abs(cmd.vy) < 1e-6
    assert f(m.update(parse_packet(build_packet(curls=SHAPES["descend"])))).vz < 0
    assert f(m.update(parse_packet(build_packet(pitch=math.radians(3))))).vx == 0.0  # deadzone


def _fly(ex, drone, seconds):
    for _ in range(int(seconds / 0.05)):
        cmd = ex.step(0.05)
        if cmd is None:
            return
        drone.send_velocity(cmd)
        drone.tick(0.05)


def test_executor_takeoff_then_point_closed_loop():
    d = SimDrone()
    ex = Executor(d)
    ex.load([{"op": "takeoff"}, {"op": "goto_pointed", "distance": 3}], Context(point_yaw=math.radians(90)))
    _fly(ex, d, 30)
    s = d.state()
    assert not ex.busy and s["y"] == pytest.approx(3.0, abs=0.2) and abs(s["x"]) < 0.2


def test_sanitize_rejects_garbage_and_clamps():
    plan = sanitize([{"op": "rm -rf"}, {"op": "move", "forward": 500}, "x", {"op": "orbit", "radius": "abc"}])
    assert plan == [{"op": "move", "forward": 8.0, "right": 0.0, "up": 0.0}, {"op": "orbit", "radius": 2.0, "deg": 360.0}]


def test_sketch_circle_is_simplified_and_normalized():
    sk = Sketch()
    sk.start()
    for i in range(200):
        a = 2 * math.pi * i / 199
        sk.add(0.3 * math.sin(a), 0.3 * (1 - math.cos(a)))
    pts = sk.stop()
    assert 6 < len(pts) <= 40 and pts[0] == (0.0, 0.0)


def test_llm_parses_nim_reply(monkeypatch):
    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": '```json\n{"say": "orbiting", "plan": [{"op": "orbit"}]}\n```'}}]}

    sent = {}
    monkeypatch.setenv("NVIDIA_API_KEY", "test")
    monkeypatch.setattr(llm, "ENABLED", True)
    monkeypatch.setattr(llm, "BACKEND", "nvidia")
    monkeypatch.setattr(llm.requests, "post", lambda url, **kw: sent.update(kw) or R())
    out = llm.ask_llm("circle around that tree", Context(), None, "open")
    assert out["plan"] == [{"op": "orbit"}] and "circle around that tree" in sent["json"]["messages"][1]["content"]


def test_llm_offline_fallback(monkeypatch):
    def down(*a, **k):
        raise llm.requests.ConnectionError()
    monkeypatch.setattr(llm.requests, "post", down)  # local LLM not running
    assert llm.ask_llm("fly a square of 3", Context(), None, "open")["plan"][0] == {"op": "path", "shape": "square", "size": 3.0}


def _sdk_hand(axis, deg):
    """SDK NED joints: hand rotated `deg` about NED `axis`; fingers point north (+x) at rest."""
    import numpy as np

    from skyhand import quat as Q
    s, c = math.sin(math.radians(deg) / 2), math.cos(math.radians(deg) / 2)
    q = [0.0, 0.0, 0.0]
    q[axis] = s
    fwd = Q.rotate(np.array([c, *q]), np.array([0.08, 0.0, 0.0]))
    hand = (0.0, 0.0, 0.0, *q, c)
    finger = (*fwd, *q, c)
    return {"right_hand": hand, "right_finger2_proximal": finger, "right_finger3_proximal": finger}


@pytest.mark.parametrize("axis,deg,idx,expected", [
    (2, 40, 2, 40),    # turn clockwise seen from above (about NED down) -> yaw right
    (1, 30, 0, 30),    # nose up about NED east -> pitch up
    (0, 20, 1, 20),    # roll right about NED north -> roll right
])
def test_sdk_joints_convert_to_skyhand_angles(axis, deg, idx, expected):
    from skyhand.sdk_convert import joints_to_studio_packet
    m = HandModel()
    m.calibrate(parse_packet(joints_to_studio_packet(_sdk_hand(2, 0))))
    s = m.update(parse_packet(joints_to_studio_packet(_sdk_hand(axis, deg))))
    assert math.degrees((s.pitch, s.roll, s.yaw)[idx]) == pytest.approx(expected, abs=1)


def test_gesture_net_learns_poses_independent_of_hand_orientation(tmp_path):
    import numpy as np

    from skyhand.gesture_model import GestureModel, GestureNet, features

    rng = np.random.default_rng(1)
    names = ["open", "fist", "thumbs_up", "pinky", "point", "peace", "rock"]

    def sample(name, max_tilt):
        curls = np.clip(np.array(SHAPES[name]) + rng.normal(0, 0.05, 5), 0, 1)
        p, r, y = rng.uniform(-max_tilt, max_tilt, 3)
        return features(parse_packet(build_packet(p, r, y, curls=tuple(curls))))

    X = [sample(n, 0.3) for n in names for _ in range(40)]
    labels = [n for n in names for _ in range(40)]
    net = GestureNet(names).fit(X, labels)
    # unseen, much larger hand tilts/turns: rotation-relative features keep the pose recognizable
    assert net.accuracy([sample(n, 1.2) for n in names for _ in range(10)],
                        [n for n in names for _ in range(10)]) > 0.95
    net.save(tmp_path / "g.npz")
    g, conf = GestureModel(tmp_path / "g.npz")(parse_packet(build_packet(0.8, -0.5, 1.0, curls=SHAPES["thumbs_up"])))
    assert g == G.THUMB and conf > 0.7


def test_gesture_net_with_imu_features(tmp_path):
    from skyhand.gesture_model import IMU_PARTS, N_FEATURES, N_IMU, GestureModel, GestureNet, features

    def frames(name, spread):  # hand held in different tilts, same finger shape
        rng = np.random.default_rng(0)
        return [parse_packet(build_packet(*rng.uniform(-spread, spread, 3), curls=SHAPES[name])) for _ in range(40)]

    def with_raw(f, bend):  # finger sensors swing down/back when the fingers fold
        f.raw = {"acc": {k: [0.0, 0.0, -1.0] for k in IMU_PARTS},
                 "pos": {k: [0.06, -0.03 * bend, 0.02 * (1 - bend)] for k in IMU_PARTS[1:]}}
        return f
    X, y = [], []
    for name, bend in (("open", 0.0), ("fist", 1.0)):
        for f in frames(name, 0.5):
            X.append(features(with_raw(f, bend), imu=True)); y.append(name)
    assert X[0].shape == (N_FEATURES + N_IMU,)
    GestureNet(["open", "fist"], n_in=len(X[0])).fit(X, y).save(tmp_path / "m.npz")
    m = GestureModel(str(tmp_path / "m.npz"))
    assert m.net.imu
    assert m(with_raw(frames("fist", 0.1)[0], 1.0))[0].value == "fist"
    assert m(frames("fist", 0.1)[0])[0].value == "none"  # no raw data -> no guess -> hover


def test_axis_lock_one_motion_at_a_time():
    from types import SimpleNamespace as HS
    from skyhand.mapping import HandToVelocity, MapConfig
    import math
    m = HandToVelocity(MapConfig(axis_lock=True, smoothing=0.0))
    d = math.radians
    # sloppy forward tilt: 25 deg pitch + 8 deg roll + 10 deg yaw -> ONLY forward
    c = m(HS(pitch=d(-25), roll=d(8), yaw=d(10), thumb=0.0))
    assert c.vx > 0 and c.vy == 0 and c.yaw_rate == 0 and c.vz == 0
    # while locked, rolling more doesn't add sideways
    c = m(HS(pitch=d(-20), roll=d(20), yaw=0.0, thumb=0.0))
    assert c.vx > 0 and c.vy == 0
    # back to flat -> stop and unlock; then a clean twist turns
    assert m(HS(pitch=0.0, roll=0.0, yaw=0.0, thumb=0.0)).as_list() == [0, 0, 0, 0]
    c = m(HS(pitch=0.0, roll=d(3), yaw=d(30), thumb=0.0))
    assert c.yaw_rate > 0 and c.vx == 0 and c.vy == 0
    m.reset()
    # two axes about equal = ambiguous -> nothing moves
    assert m(HS(pitch=d(-20), roll=d(19), yaw=0.0, thumb=0.0)).as_list() == [0, 0, 0, 0]
    # thumb out alone -> climb only
    assert m(HS(pitch=0.0, roll=0.0, yaw=0.0, thumb=-0.5)).vz > 0


def test_simple_mode_is_tilt_only():
    from types import SimpleNamespace as HS
    from skyhand.mapping import HandToVelocity, MapConfig
    import math
    m = HandToVelocity(MapConfig(axis_lock=True, smoothing=0.0, lock_axes=("vx",)))
    # open hand: only forward/back; roll, twist and thumb do nothing
    assert m(HS(pitch=0.0, roll=math.radians(30), yaw=math.radians(40), thumb=-0.6)).as_list() == [0, 0, 0, 0]
    assert m(HS(pitch=math.radians(-25), roll=math.radians(10), yaw=0.0, thumb=0.0)).vx > 0
    # peace sign: roll right = turn right, and nothing else
    t = m.turn(HS(pitch=math.radians(-25), roll=math.radians(25), yaw=0.0, thumb=0.0))
    assert t.yaw_rate > 0 and t.vx == 0 and t.vy == 0 and t.vz == 0
    assert m.turn(HS(pitch=0.0, roll=math.radians(-25), yaw=0.0, thumb=0.0)).yaw_rate < 0


def test_llm_plan_waits_for_thumbs_up():
    import math as _m
    from types import SimpleNamespace
    from skyhand.app import SkyHand
    from skyhand.gestures import G as _G

    app = SkyHand.__new__(SkyHand)
    app.status, app.pending, app.paused = {}, None, False
    app.args = SimpleNamespace(simple=True, override_deg=20.0)
    loaded = []
    app.exec = SimpleNamespace(load=lambda p, c: loaded.append(p))
    app.ctx = None
    app.drone = SimpleNamespace(state=lambda: {"z": 1.5, "airborne": True})
    app._propose({"say": "up", "plan": [{"op": "move", "up": 50}, {"op": "fly_to_moon"}]}, 0.0)
    assert loaded == [] and len(app.pending) == 1 and app.pending[0]["up"] <= 8  # clamped, bad op dropped
    app._on_event(_G.THUMB, None)  # thumbs up = approve (not land, although airborne)
    assert len(loaded) == 1 and loaded[0][0]["op"] == "move" and app.pending is None
    flat = SimpleNamespace(pitch=0.05, roll=0.0)
    tilted = SimpleNamespace(pitch=_m.radians(-30), roll=0.0)
    assert not app._hand_active(flat) and app._hand_active(tilted)
    app.paused = True  # fist paused the plan -> thumbs up continues it (does not land)
    app._on_event(_G.THUMB, None)
    assert app.paused is False and len(loaded) == 1


def test_plan_refinements_by_rules():
    sq = [{"op": "move", "forward": 2.0}, {"op": "yaw", "deg": 90.0}]
    big = llm.refine("make it bigger", sq)["plan"]
    assert big[0]["forward"] == 4.0 and big[1]["deg"] == 90.0
    hi = llm.refine("do it again 1 m higher", sq)["plan"]
    assert hi[0] == {"op": "move", "up": 1.0} and hi[1:] == sq
    assert llm.refine("other way", sq)["plan"][1]["deg"] == -90.0
    assert llm.refine("fly a 3 m square", sq) is None and llm.refine("again", None) is None


def test_small_llm_copied_number_is_fixed():
    out = llm._use_their_number("fly a 3 meter square", {"say": "1 m square", "plan": [
        {"op": "move", "forward": 1}, {"op": "yaw", "deg": 90}, {"op": "move", "forward": -1}]})
    assert [s.get("forward") for s in out["plan"]] == [3.0, None, -3.0] and out["say"] == "3 m square"


def test_small_llm_missed_shape_is_fixed():
    out = llm._use_their_shapes("take off, fly a 3 m square then land", {"say": "", "plan": [
        {"op": "move", "forward": 3}, {"op": "move", "forward": -3}]})
    assert out["plan"] == [{"op": "takeoff"}, {"op": "path", "shape": "square", "size": 3.0}, {"op": "land"}]
    keep = {"say": "", "plan": [{"op": "orbit", "radius": 2}]}
    assert llm._use_their_shapes("circle around me", keep)["plan"] == [{"op": "orbit", "radius": 2}]
    assert llm._shapes_in("go up 8 m and do a figure 8") == ["figure8"]


def test_other_way_mirrors_a_shape():
    from skyhand import shapes
    plan = sanitize(llm.refine("other way", [{"op": "path", "shape": "heart", "size": 2}])["plan"])
    assert plan[0]["mirror"] is True
    a, b = shapes.make("figure8", 3), shapes.make("figure8", 3, mirror=True)
    assert np.allclose(a[:, 1], -b[:, 1]) and np.allclose(a[:, 0], b[:, 0])


def test_shapes_are_exact():
    from skyhand import shapes
    for name in shapes.SHAPES:
        p = shapes.make(name, 3.0)
        assert np.allclose(p[0], 0) and np.all(np.linalg.norm(np.diff(p, axis=0), axis=1) <= shapes.STEP + 1e-6)
    f8 = shapes.make("figure8", 4.0)
    mid = f8[len(f8) // 2]
    assert np.linalg.norm(mid) < 0.06 and np.ptp(f8[:, 0]) > 3.9 and np.ptp(f8[:, 1]) > 1.9  # crosses at start
    assert np.allclose(shapes.make("square", 2.0)[-1], 0, atol=1e-6)  # closed


def test_path_step_sanitized():
    from skyhand.planner import sanitize
    out = sanitize([{"op": "path", "shape": "Figure-8", "size": 50}, {"op": "path", "shape": "dragon"}])
    assert out == [{"op": "path", "shape": "figure8", "size": 6.0}]


def _ride_sim(ride, st, throttle, secs, dt=1 / 30):
    for _ in range(int(secs / dt)):
        c = ride.step(st, throttle, 0.0, dt)
        cy, sy = np.cos(st["yaw"]), np.sin(st["yaw"])
        st["x"] += (c.vx * cy - c.vy * sy) * dt
        st["y"] += (c.vx * sy + c.vy * cy) * dt
        st["z"] += c.vz * dt
        st["yaw"] += c.yaw_rate * dt
        if ride.done:
            break
    return st


def test_hand_drives_along_path():
    from skyhand.ride import PathRide
    st = {"x": 0.0, "y": 0.0, "z": 1.5, "yaw": 0.3}
    ride = PathRide("figure8", 3.0, st)
    _ride_sim(ride, st, 0.0, 2.0)
    assert ride.s == 0.0  # flat hand = hold
    _ride_sim(ride, st, 1.0, 3.0)
    s_mid = ride.s
    assert 1.5 < s_mid < ride.length and ride.err < 0.35  # follows the path
    _ride_sim(ride, st, -1.0, 1.0)
    assert ride.s < s_mid  # tilt back = reverse
    _ride_sim(ride, st, 1.0, 30.0)
    assert ride.done


def test_gesture_select_drops_lookalikes_and_bad_frames():
    from skyhand import gesture_select as S
    rng = np.random.default_rng(0)
    centers = {"open": 0, "fist": 4, "thumbs_up": 8, "peace": 12, "point": 16, "pinky": 16.3, "relax": 20}
    X, y, g = [], [], []
    for pose in ("flat", "tilt", "side"):
        for c, m in centers.items():
            X.append(m + rng.normal(0, 0.5, (40, 6))); y += [c] * 40; g += [pose] * 40
    X = np.vstack(X); y = np.array(y)
    y[5] = "peace"  # one mislabeled open frame
    classes = sorted(set(y))
    P = S.oof_proba(X, y, np.array(g), classes, epochs=150)
    keep = S.clean(y, P, classes)
    assert not keep[5] and keep.mean() > 0.97
    kept, dropped = S.pick(S.confusion(y[keep], P[keep], classes), classes)
    assert set(S.REQUIRED) <= set(kept) and "relax" in kept
    assert ("point" in dropped) != ("pinky" in dropped)  # one of the two look-alikes goes


def test_unknown_model_class_means_no_gesture(tmp_path):
    from skyhand.gesture_model import GestureModel, GestureNet
    from skyhand.gestures import G
    net = GestureNet(["open", "relax"], n_in=3)
    net.b2[:] = [-5, 5]
    net.save(tmp_path / "m.npz")
    gm = GestureModel(str(tmp_path / "m.npz"))
    gm.net.imu = False
    import skyhand.gesture_model as M
    orig = M.features
    M.features = lambda f, imu=False: np.zeros(3)
    try:
        assert gm(None)[0] == G.NONE
    finally:
        M.features = orig
