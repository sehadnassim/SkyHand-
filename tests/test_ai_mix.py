from skyhand.ai_mix import Limits, Mixer

FWD = {"vx": 0.3, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}
ZERO = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}
L = Limits()


def ai(action, conf=0.9, vy=0.3, latency=0.5, frame_age=0.2, distance="near"):
    return {"action": action, "conf": conf, "vy": vy, "latency": latency, "frame_age": frame_age,
            "distance": distance}


def engaged(mixer, action, now=10.0, n=2, **kw):
    for i in range(n):
        mixer.on_ai(ai(action, **kw), now - 0.1 * (n - i))
    return mixer.step(FWD, 0.0, now, True)


def test_avoid_backs_off_then_slides_after_agreeing_messages():
    m = Mixer()
    assert m.step(FWD, 0.0, 10.0, True)[1] == "pilot"
    m.on_ai(ai("avoid_left"), 9.9)
    assert m.step(FWD, 0.0, 10.0, True)[1] == "pilot"  # one avoid message is not enough
    m.on_ai(ai("avoid_left"), 9.95)
    cmd, mode = m.step(FWD, 0.0, 10.0, True)
    assert mode == "backoff" and cmd == {**ZERO, "vx": -L.back_speed}
    cmd, mode = m.step(FWD, 0.0, 10.0 + L.back_s + 0.01, True)
    assert mode == "avoid_left" and cmd == {**ZERO, "vy": -0.3}
    cmd, mode = m.step(FWD, 0.0, 10.0 + L.back_s + L.avoid_s + 0.02, True)
    assert mode == "pilot" and cmd == FWD  # manoeuvre over: the pilot again
    m2 = Mixer()
    engaged(m2, "avoid_right")
    assert m2.step(FWD, 0.0, 10.0 + L.back_s + 0.01, True) == ({**ZERO, "vy": 0.3}, "avoid_right")


def test_stop_reacts_on_one_message_backs_off_then_hovers():
    m = Mixer()
    m.on_ai(ai("stop", conf=0.65), 9.9)  # a single, moderately confident stop is enough
    assert m.step(FWD, 0.0, 10.0, True) == ({**ZERO, "vx": -L.back_speed}, "backoff")
    assert m.step(FWD, 0.0, 10.0 + L.back_s + 0.01, True) == (ZERO, "stop")  # never steer around a person
    assert m.step(FWD, 0.0, 10.0 + L.back_s + L.hold_s + 0.02, True)[1] == "pilot"
    m = Mixer()
    m.on_ai(ai("stop", conf=0.5), 9.9)
    assert m.step(FWD, 0.0, 10.0, True)[1] == "pilot"


def test_letting_go_of_forward_or_disabling_ends_the_manoeuvre():
    m = Mixer()
    engaged(m, "avoid_left")
    hover = dict(ZERO)
    assert m.step(hover, 0.0, 10.1, True) == (hover, "pilot")
    assert m.step(FWD, 0.0, 10.2, True)[1] == "pilot"  # needs fresh decisions to start again
    m = Mixer()
    engaged(m, "stop")
    assert m.step(FWD, 0.0, 10.1, False) == (FWD, "pilot")


def test_ai_ignored_unless_pilot_pushes_forward_and_enabled():
    m = Mixer()
    m.on_ai(ai("avoid_left"), 9.8)
    m.on_ai(ai("avoid_left"), 9.9)
    assert m.step(ZERO, 0.0, 10.0, True) == (ZERO, "pilot")
    assert m.step({**FWD, "vx": -0.3}, 0.0, 10.0, True)[1] == "pilot"
    assert m.step(FWD, 0.0, 10.0, False) == (FWD, "pilot")


def test_unsure_stale_or_slow_ai_is_ignored():
    assert engaged(Mixer(), "avoid_left", conf=0.5)[1] == "pilot"
    assert engaged(Mixer(), "avoid_left", latency=1.5, frame_age=1.0)[1] == "pilot"
    assert engaged(Mixer(), "fly_up")[1] == "pilot"
    m = Mixer()
    m.on_ai(ai("avoid_left"), 9.8)
    m.on_ai(ai("avoid_left"), 9.9)
    assert m.step(FWD, 0.0, 9.9 + L.ai_fresh_s + 0.1, True)[1] == "pilot"
    m = Mixer()
    m.on_ai(ai("avoid_left"), 9.9)
    m.on_ai(ai("avoid_right"), 9.95)  # disagreement resets the streak
    assert m.step(FWD, 0.0, 10.0, True)[1] == "pilot"


def test_caps_and_pilot_loss():
    m = Mixer()
    engaged(m, "avoid_right", vy=5.0)
    cmd, _ = m.step(FWD, 0.0, 10.0 + L.back_s + 0.01, True)
    assert cmd["vy"] == L.max_side
    m = Mixer()
    cmd, mode = m.step({"vx": 3.0, "vy": 2.0, "vz": 1.0, "yaw_rate": 2.0}, 0.0, 10.0, True)
    assert cmd == {"vx": 0.4, "vy": 0.0, "vz": 0.15, "yaw_rate": 0.35} and mode == "pilot"
    assert m.step(FWD, 0.5, 10.0, True) == (None, "no_pilot")
    assert m.step(None, 0.0, 10.0, True) == (None, "no_pilot")
    m = Mixer()
    engaged(m, "stop")
    assert m.step(None, 0.0, 10.1, True) == (None, "no_pilot")  # glove lost mid-manoeuvre: mux hovers
    assert m.man is None


def test_agreement_needs_messages_close_in_time():
    m = Mixer()
    m.on_ai(ai("avoid_left"), 1.0)
    m.on_ai(ai("avoid_left"), 1.0 + L.max_agree_gap_s + 1)  # old + new is not agreement
    assert m.step(FWD, 0.0, 1.0 + L.max_agree_gap_s + 1.1, True)[1] == "pilot"


def test_hover_only_backs_off_then_hovers_instead_of_sliding():
    m = Mixer(Limits(stop_only=True))
    assert engaged(m, "avoid_left")[1] == "backoff"
    assert m.step(FWD, 0.0, 10.0 + L.back_s + 0.01, True) == (ZERO, "stop")


def test_avoid_only_when_obstacle_is_near():
    assert engaged(Mixer(), "avoid_left", distance="mid")[1] == "pilot"
    assert engaged(Mixer(), "avoid_right", distance="far")[1] == "pilot"
    assert engaged(Mixer(), "avoid_left", distance="near")[1] == "backoff"
