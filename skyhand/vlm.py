"""Vision-language model client (Nemotron 3.5 Super VL, OpenAI-compatible endpoint). Advisory only.

It looks at a forward camera frame and picks one of: continue, avoid_left, avoid_right, stop. It never sends
anything to the drone itself; scripts/vlm_service.py publishes the advice and the drone's ai_mux decides.
Needs VERDA_INFERENCE_KEY in the environment; never put the key in the repo.
"""
import base64
import json
import os
import re

import requests

from .ai_mix import ACTIONS

BASE_URL = os.environ.get("VERDA_INFERENCE_URL", "https://containers.datacrunch.io/nemotron-3-5-super-vl-b300/v1")
MODEL = os.environ.get("VERDA_INFERENCE_MODEL", "nemotron-3.5-super-vl")


def jpeg_b64(frame, max_side=640, quality=70):
    """OpenCV BGR frame -> downscaled JPEG, base64."""
    import cv2
    h, w = frame.shape[:2]
    k = max_side / max(h, w)
    if k < 1:
        frame = cv2.resize(frame, (int(w * k), int(h * k)))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("jpeg encode failed")
    return base64.b64encode(buf.tobytes()).decode()


AVOID_SYSTEM = (
    "You look through the forward camera of a slow drone that the pilot is flying straight ahead. "
    "Decide if it is safe to keep going forward. Reply with ONE JSON object and nothing else, with keys: "
    '"obstacle" (short name or ""), "distance" ("near", "mid", "far" or "none"), '
    '"free_side" ("left", "right" or "none": the side with more open space to pass the obstacle), '
    '"span" ([left, right] horizontal edges of the main obstacle, 0 = image left edge, 1000 = right edge, or []), '
    '"action" ("continue", "avoid_left", "avoid_right" or "stop"), "reason" (max 10 words), "confidence" (0..1). '
    "Distance: near = within about 1 m, the obstacle fills at least HALF of the image height (if in doubt, it is "
    "mid, not near); mid = 1-4 m; far = further. Rules: a real person anywhere ahead -> stop. People printed "
    "on a banner, poster, "
    "sign or screen are NOT people: treat the banner as an obstacle. Only a NEAR obstacle in the path matters: "
    "near with a free side -> avoid to that side, near blocking the whole view -> stop. An obstacle at mid or far "
    "distance -> continue (the pilot handles it). Nothing in the path -> continue."
)


def _span(v):
    """[left, right] in 0..1000 (as asked) or 0..1 (model sometimes does that) -> fractions 0..1."""
    try:
        x0, x1 = (float(x) for x in v)
    except (TypeError, ValueError):
        return []
    k = 1000.0 if max(x0, x1) > 1.0 else 1.0
    x0, x1 = max(0.0, min(1.0, x0 / k)), max(0.0, min(1.0, x1 / k))
    return [x0, x1] if x1 > x0 else []


def parse_avoid(text):
    """Malformed or unknown -> action 'stop' with confidence 0 (callers must treat that as 'no advice')."""
    out = {"obstacle": "", "distance": "none", "free_side": "none", "action": "stop", "reason": "", "confidence": 0.0}
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return out
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return out
    if d.get("action") not in ACTIONS:
        return out
    out.update(obstacle=str(d.get("obstacle", ""))[:60], reason=str(d.get("reason", ""))[:120], action=d["action"])
    out["distance"] = d.get("distance") if d.get("distance") in ("near", "mid", "far", "none") else "none"
    out["free_side"] = d.get("free_side") if d.get("free_side") in ("left", "right", "none") else "none"
    out["span"] = _span(d.get("span"))
    if out["action"] in ("avoid_left", "avoid_right") and out["span"]:
        # the model finds the obstacle well but picks the side poorly: pass on the wider gap
        x0, x1 = out["span"]
        side = "left" if x0 >= 1.0 - x1 else "right"
        if out["action"] != "avoid_" + side:
            out["action"], out["free_side"] = "avoid_" + side, side
            out["reason"] = f"{out['obstacle'] or 'obstacle'} ahead, more space on the {side}"
    try:
        out["confidence"] = max(0.0, min(1.0, float(d.get("confidence", 0.0))))
    except (TypeError, ValueError):
        pass
    return out


def avoid(image_b64, goal="", timeout=10.0, key=None):
    """One obstacle check on a forward camera frame. Advisory: the caller decides what (if anything) to do."""
    key = key or os.environ.get("VERDA_INFERENCE_KEY")
    if not key:
        raise RuntimeError("VERDA_INFERENCE_KEY is not set")
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": AVOID_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": f"Pilot goal: {goal or '(none)'}"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_b64}},
            ]},
        ],
        "max_tokens": 200,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = requests.post(BASE_URL + "/chat/completions", json=body, timeout=timeout,
                      headers={"Authorization": "Bearer " + key})
    r.raise_for_status()
    return parse_avoid(r.json()["choices"][0]["message"].get("content") or "")
