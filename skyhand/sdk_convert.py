"""Convert Rokoko Device SDK solved-hand joints into a Rokoko Studio JSON v3 packet.

`rkk-hand-solver --emit world` publishes per-joint TRANSFORMs (x, y, z, qx, qy, qz, qw)
in a NED frame (x north, y east, z down) under names like `right_finger2_proximal`.
SkyHand expects Studio's Y-up frame and bone names like `rightIndexProximal`.

NED -> Y-up uses the proper rotation (x, y, z) -> (-y, -z, x): north becomes +Z (forward),
down becomes -Y, and SkyHand's `right = forward x up` still points east.
"""

# SDK finger index -> Studio finger name. The thumb has no medial joint in the SDK,
# so its metacarpal/proximal/distal fill Studio's Proximal/Medial/Distal.
FINGER_JOINTS = {
    1: ("Thumb", ("metacarpal", "proximal", "distal")),
    2: ("Index", ("proximal", "medial", "distal")),
    3: ("Middle", ("proximal", "medial", "distal")),
    4: ("Ring", ("proximal", "medial", "distal")),
    5: ("Little", ("proximal", "medial", "distal")),
}
# The SDK hand solver bends fingers less than Studio: ~1.5 rad from flat hand to fist (measured).
SDK_FINGER_RANGE = 1.5
STUDIO_JOINTS = ("Proximal", "Medial", "Distal")


def _bone_map(side: str) -> dict:
    m = {f"{side}_hand": f"{side}Hand"}
    for idx, (finger, sdk_joints) in FINGER_JOINTS.items():
        for sdk_j, studio_j in zip(sdk_joints, STUDIO_JOINTS):
            m[f"{side}_finger{idx}_{sdk_j}"] = f"{side}{finger}{studio_j}"
    return m


BONE_MAP = {**_bone_map("right"), **_bone_map("left")}


def ned_transform_to_studio(values) -> dict:
    x, y, z, qx, qy, qz, qw = values
    return {
        "position": {"x": -y, "y": -z, "z": x},
        "rotation": {"x": -qy, "y": -qz, "z": qx, "w": qw},
    }


def joints_to_studio_packet(joints: dict, actor: str = "SDK") -> dict | None:
    """`joints`: SDK target_frame -> 7 floats. Returns a Studio v3 packet, or None if no hand."""
    body = {}
    for name, values in joints.items():
        studio = BONE_MAP.get(name)
        if studio is not None:
            body[studio] = ned_transform_to_studio(values)
    if "rightHand" not in body and "leftHand" not in body:
        return None
    return {"version": 3, "scene": {"actors": [{"name": actor, "body": body}]}}
