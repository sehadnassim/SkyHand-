"""LLM copilot: natural language + live glove context -> validated flight plan.

Default: a small LOCAL model through Ollama (SKYHAND_LLM_MODEL, default qwen2.5:1.5b; no internet, no key).
SKYHAND_LLM=nvidia uses NVIDIA NIM instead (NVIDIA_API_KEY). If the LLM is down, a rule-based parser answers.
The plan is only a proposal: planner.sanitize() drops unknown ops and clamps distances/altitude, and the app
waits for a thumbs-up before flying it (fist = cancel / abort).
"""
import json
import math
import os
import re

import requests

from .planner import OPS
from .shapes import SHAPES

URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1") + "/chat/completions"
MODEL = os.environ.get("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")

SYSTEM = f"""You are SkyHand, the copilot of a small DJI drone that an operator flies with a Rokoko motion-capture glove.
Turn the operator's request into a SHORT flight plan made only of these primitives (args with defaults):
{json.dumps(OPS)}
Conventions: body frame forward/right/up in meters; yaw deg + = turn right; orbit centre is `radius` m ahead of the drone.
"goto_pointed" flies along the ray the operator's index finger is pointing at right now (use it for "there", "that", "over there").
"trace_sketch" flies the shape the operator last drew in the air; "goto_waypoint" goes to a waypoint the operator marked.
Compose shapes from moves (e.g. a 2 m square = 4 moves + yaws, a zig-zag, a spiral = orbit + up).
Be conservative: keep moves <= 8 m, altitude <= 5 m. If the request is unsafe or unclear, return an empty plan and explain.
Reply with JSON only: {{"say": "<one short sentence for the operator>", "plan": [{{"op": "...", ...}}]}}"""


ENABLED = os.environ.get("SKYHAND_ENABLE_LLM") == "1"  # LLMs are off by default; typed requests use the rule parser
BACKEND = os.environ.get("SKYHAND_LLM", "ollama")
OLLAMA = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
LOCAL_MODEL = os.environ.get("SKYHAND_LLM_MODEL", "qwen2.5:1.5b")

# small models need a short prompt, a few examples and a fixed JSON shape
LOCAL_SYSTEM = """You turn a drone operator's request into a short flight plan. Reply with JSON only.
Steps you may use:
  takeoff | land | hover {s} | move {forward, right, up} (meters, negative = back/left/down)
  yaw {deg} (+ = turn right, - = turn left) | orbit {radius, deg} | goto_pointed {distance} (where the finger points)
  goto_waypoint {id}
  path {shape, size} (size in m) shapes: circle, figure8, square, triangle, zigzag, spiral, heart
Use path for any shape or figure. Keep moves <= 8 m. Unclear or unsafe -> empty plan.
Examples:
"go up 2 meters" -> {"say": "climbing 2 m", "plan": [{"op": "move", "up": 2}]}
"turn left" -> {"say": "turning left", "plan": [{"op": "yaw", "deg": -90}]}
"fly 3 m back then land" -> {"say": "back 3 m, then landing", "plan": [{"op": "move", "forward": -3}, {"op": "land"}]}
"turn around" -> {"say": "turning around", "plan": [{"op": "yaw", "deg": 180}]}
"fly a figure 8" -> {"say": "figure 8, 3 m", "plan": [{"op": "path", "shape": "figure8", "size": 3}]}
"draw a heart then land" -> {"say": "heart, then landing", "plan": [{"op": "path", "shape": "heart", "size": 3}, {"op": "land"}]}
"circle around here" -> {"say": "orbiting", "plan": [{"op": "orbit", "radius": 2, "deg": 360}]}
"go over there" -> {"say": "going where you point", "plan": [{"op": "goto_pointed", "distance": 3}]}"""

SCHEMA = {"type": "object", "required": ["say", "plan"], "properties": {
    "say": {"type": "string"},
    "plan": {"type": "array", "items": {"type": "object", "required": ["op"], "properties": {
        "op": {"type": "string", "enum": [k for k in OPS if k not in ("move_world", "trace_sketch")]},
        "forward": {"type": "number"}, "right": {"type": "number"}, "up": {"type": "number"},
        "deg": {"type": "number"}, "radius": {"type": "number"}, "distance": {"type": "number"},
        "s": {"type": "number"}, "id": {"type": "integer"},
        "shape": {"type": "string", "enum": list(SHAPES)}, "size": {"type": "number"}}}}}}


def _local_context(state) -> str:
    if not state:
        return "Drone: unknown"
    return f"Drone: {'flying' if state.get('airborne') else 'on the ground'}, altitude {state['z']:.1f} m"


def ask_local(request: str, state=None, timeout: float = 20.0, model: str | None = None) -> dict:
    msgs = [{"role": "system", "content": LOCAL_SYSTEM},
            {"role": "user", "content": f"{_local_context(state)}\nOperator: {request}"}]
    r = requests.post(f"{OLLAMA}/api/chat", timeout=timeout, json={
        "model": model or LOCAL_MODEL, "messages": msgs, "stream": False, "format": SCHEMA,
        "options": {"temperature": 0, "num_predict": 500}, "keep_alive": "30m"})
    r.raise_for_status()
    out = json.loads(r.json()["message"]["content"])
    if not isinstance(out.get("plan"), list):
        raise ValueError("no plan")
    return _use_their_shapes(request, _use_their_number(request, out))


def _use_their_number(request: str, out: dict) -> dict:
    """Small models copy numbers from the prompt examples ("3 m square" -> 1 m). If the request has one
    distance and the plan never uses it, put it in."""
    nums = {float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*(?:m\b|meter|metre)", request.lower())}
    if len(nums) != 1:
        return out
    n = nums.pop()
    keys = ("forward", "right", "up", "radius", "distance", "size")
    steps = [st for st in out["plan"] if isinstance(st, dict)]
    used = {abs(float(st[k])) for st in steps for k in keys if isinstance(st.get(k), (int, float))}
    if used and n not in used:
        for st in steps:
            for k in keys:
                if isinstance(st.get(k), (int, float)) and st[k]:
                    st[k] = math.copysign(n, st[k])
        out["say"] = re.sub(r"\d+(?:\.\d+)?\s*m\b", f"{n:g} m", out.get("say", ""))
    return out


SHAPE_WORDS = (("figure", "figure8"), ("eight", "figure8"), ("8", "figure8"), ("heart", "heart"),
               ("spiral", "spiral"), ("zig", "zigzag"), ("square", "square"), ("triangle", "triangle"),
               ("circle", "circle"))


def _shapes_in(text: str) -> list:
    """Shape names in the order they appear in the text ("orbit"/"around" means orbit, not a circle path)."""
    t = text.lower()
    hits = []
    for w, sh in SHAPE_WORDS:
        if w == "8":
            ms = [m.start() for m in re.finditer(r"(?<![\d.])8(?!\s*(?:m\b|meter|metre|\d))", t)]
        else:
            ms = [m.start() for m in re.finditer(w, t)]
        if sh == "circle" and ("orbit" in t or "around" in t):
            ms = []
        hits += [(i, sh) for i in ms]
    out = []
    for _, sh in sorted(hits):
        if not out or out[-1] != sh:
            out.append(sh)
    return out


def _use_their_shapes(request: str, out: dict) -> dict:
    """The 1.5B model sometimes writes moves for "square"/"triangle" or drops a shape from a longer
    mission. Shape words are easy to spot, so if the plan misses one, rebuild the shape part here."""
    want = _shapes_in(request)
    steps = [st for st in out.get("plan", []) if isinstance(st, dict)]
    got = [st.get("shape") for st in steps if st.get("op") == "path"]
    if not want or got == want:
        return out
    t = request.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:m\b|meter|metre)", t)
    size = float(m.group(1)) if m else 4.5 if ("big" in t or "large" in t) else 1.5 if "small" in t else 3.0
    sizes = [st.get("size") for st in steps if st.get("op") == "path"]
    if not m and sizes and all(x == sizes[0] for x in sizes) and isinstance(sizes[0], (int, float)):
        size = sizes[0]
    plan = [{"op": "takeoff"}] if "take" in t and "off" in t else []
    plan += [{"op": "path", "shape": sh, "size": size} for sh in want]
    if re.search(r"\bland\b", t):
        plan.append({"op": "land"})
    out["plan"] = plan
    out["say"] = ", ".join(sh for sh in want) + f" ({size:g} m)"
    return out


def _context_text(ctx, state, gesture) -> str:
    s = "unknown (no odometry)" if not state else \
        f"x={state['x']:.1f} y={state['y']:.1f} z={state['z']:.1f} yaw={math.degrees(state['yaw']):.0f}deg airborne={state['airborne']}"
    return (f"Drone state (start frame x fwd, y right, z up): {s}\n"
            f"Operator index finger pointing: yaw={math.degrees(ctx.point_yaw):.0f}deg pitch={math.degrees(ctx.point_pitch):.0f}deg\n"
            f"Current hand gesture: {gesture}\n"
            f"Marked waypoints: {sorted(ctx.waypoints)}\n"
            f"Air sketch available: {bool(ctx.sketch)} ({len(ctx.sketch)} points)")


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON in reply")
    return json.loads(m.group(0))


def refine(request: str, plan) -> dict | None:
    """Edits of the last plan, done by rules (a 1.5B model is unreliable at editing): bigger/smaller,
    again, higher/lower [N m], other way. Returns None if the text is not an edit."""
    if not plan:
        return None
    t = request.lower()
    n = re.search(r"(\d+(?:\.\d+)?)\s*(?:m\b|meter)", t)
    n = float(n.group(1)) if n else 1.0
    out, said = [dict(s) for s in plan], []
    if re.search(r"\b(bigger|larger|wider)\b", t):
        k, said = 2.0, ["twice as big"]
    elif re.search(r"\b(smaller|tighter)\b", t):
        k, said = 0.5, ["half the size"]
    else:
        k = 1.0
    for s in out:
        for key in ("forward", "right", "up", "radius", "distance", "size"):
            if key in s:
                s[key] = s[key] * k
    if re.search(r"\b(other way|reverse|opposite|mirror)\b", t):
        for s in out:
            for key in ("deg", "right"):
                if key in s:
                    s[key] = -s[key]
            if s.get("op") == "path":
                s["mirror"] = not s.get("mirror", False)
        said.append("mirrored")
    if re.search(r"\b(higher|up)\b", t) and re.search(r"\b(again|it|same|plan)\b", t):
        out.insert(0, {"op": "move", "up": n})
        said.append(f"{n:g} m higher")
    elif re.search(r"\b(lower|down)\b", t) and re.search(r"\b(again|it|same|plan)\b", t):
        out.insert(0, {"op": "move", "up": -n})
        said.append(f"{n:g} m lower")
    if not said and not re.search(r"\b(again|repeat|once more)\b", t):
        return None
    return {"say": "same plan" + (", " + ", ".join(said) if said else " again"), "plan": out}


def ask_llm(request: str, ctx, state, gesture, timeout: float = 20.0, plan=None) -> dict:
    edit = refine(request, plan)
    if edit:
        return edit
    if not ENABLED:
        return fallback(request)
    if BACKEND != "nvidia":
        try:
            return ask_local(request, state, timeout)
        except (requests.RequestException, ValueError, KeyError) as e:
            out = fallback(request)
            out["say"] = f"(local LLM unavailable: {type(e).__name__}) " + out["say"]
            return out
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        return fallback(request)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": _context_text(ctx, state, gesture) + f"\n\nOperator says: {request}"}]
    try:
        r = requests.post(URL, timeout=timeout, headers={"Authorization": f"Bearer {key}"},
                          json={"model": MODEL, "messages": msgs, "temperature": 0.1, "max_tokens": 600})
        r.raise_for_status()
        return _extract_json(r.json()["choices"][0]["message"]["content"])
    except (requests.RequestException, ValueError, KeyError) as e:
        out = fallback(request)
        out["say"] = f"(LLM unavailable: {type(e).__name__}) " + out["say"]
        return out


def fallback(text: str) -> dict:
    """Tiny offline parser: 'takeoff', 'land', 'forward 2', 'left 1', 'up 1', 'turn 90', 'orbit 2', 'square 2',
    'go there', 'draw', 'waypoint 1'."""
    t = text.lower()
    num = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", t)]
    n = num[0] if num else None
    plan = []
    if "take" in t and "off" in t:
        plan.append({"op": "takeoff"})
    elif "land" in t:
        plan.append({"op": "land"})
    elif any(w in t for w in ("square", "figure", "eight", "heart", "spiral", "zigzag", "zig zag", "triangle",
                              "circle")) and "around" not in t and "orbit" not in t:
        shape = next(sh for w, sh in (("square", "square"), ("figure", "figure8"), ("eight", "figure8"),
                                      ("heart", "heart"), ("spiral", "spiral"), ("zig", "zigzag"),
                                      ("triangle", "triangle"), ("circle", "circle")) if w in t)
        plan.append({"op": "path", "shape": shape, "size": n or 3.0})
    elif "orbit" in t or "circle" in t:
        plan.append({"op": "orbit", "radius": n or 2.0, "deg": 360})
    elif "there" in t or "point" in t:
        plan.append({"op": "goto_pointed", "distance": n or 3.0})
    elif "draw" in t or "sketch" in t or "shape" in t:
        plan.append({"op": "trace_sketch", "size": n or 2.0})
    elif "waypoint" in t or "back" in t:
        plan.append({"op": "goto_waypoint", "id": int(n or 1)})
    elif "turn" in t or "yaw" in t or "rotate" in t:
        plan.append({"op": "yaw", "deg": (-1 if "left" in t else 1) * (n or 90)})
    else:
        for word, key, sgn in (("forward", "forward", 1), ("back", "forward", -1), ("right", "right", 1),
                               ("left", "right", -1), ("up", "up", 1), ("down", "up", -1)):
            if word in t:
                plan.append({"op": "move", key: sgn * (n or 1.0)})
                break
    return {"say": "ok (offline parser)" if plan else "sorry, I did not understand", "plan": plan}
