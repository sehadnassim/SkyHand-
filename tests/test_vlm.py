from skyhand.vlm import parse_avoid


def test_avoid_side_follows_wider_gap_not_model_guess():
    r = parse_avoid('{"action": "avoid_right", "span": [0.45, 0.75], "confidence": 0.9}')
    assert r["action"] == "avoid_left"
    assert parse_avoid('{"action": "avoid_left", "span": [0.2, 0.5]}')["action"] == "avoid_right"
    assert parse_avoid('{"action": "avoid_right", "span": [400, 750]}')["action"] == "avoid_left"  # 0..1000
    assert parse_avoid('{"action": "avoid_left", "span": "bad"}')["action"] == "avoid_left"
    assert parse_avoid('{"action": "stop", "span": [0.4, 0.6]}')["action"] == "stop"
