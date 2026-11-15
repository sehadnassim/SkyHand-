#!/usr/bin/env python3
"""Runs on the DRONE's onboard computer: mixes the pilot's glove velocity with the cloud AI's avoidance.

  laptop (glove)  --skyhand/pilot_vel-->  [local broker]  --> ai_mux --> drone/cmd_vel --> drone
  cloud VLM       --skyhand/ai_cmd----->  [cloud broker, over 4G] --^
  ai_mux --> skyhand/ai_status, skyhand/ai_view (camera thumbnail) --> laptop GUI (display only)

  python3 scripts/ai_mux.py --local-host 192.168.4.2 --cloud-host <cloud-ip>      # real
  python3 scripts/ai_mux.py ... --out-topic skyhand/shadow_vel                     # shadow test: drone not driven

Takeoff/land (drone/command) never go through here. Safety rules: see skyhand/ai_mix.py. If the pilot stream
stops, the mux sends a short zero-velocity burst (hover) and then goes silent. If 4G drops, the AI goes
stale and the pilot's command simply passes through. Cloud login: SKYHAND_CLOUD_USER / SKYHAND_CLOUD_PASS.
"""
import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paho.mqtt.client as mqtt  # noqa: E402

from skyhand.ai_mix import Limits, Mixer  # noqa: E402

RATE_HZ = 20
SLEW = 0.4  # m/s^2 on the mixed output, so AI take-over / hand-back is smooth (looser than the laptop's 0.2)
SLEW_AI = 1.5  # m/s^2 during an AI manoeuvre: brake hard. At 0.4 m/s^2 a 0.4 m/s drone needs ~1 s / 0.2 m to stop
ZERO_BURST = 5


def client(name, host, port, subs, on_msg, user=None, pw=None):
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"{name}-{int(time.time())}")
    if user:
        c.username_pw_set(user, pw)
    c.on_connect = lambda cl, *_: [cl.subscribe(t, qos=0) for t in subs]
    c.on_message = on_msg
    c.reconnect_delay_set(1, 5)
    c.connect_async(host, port, keepalive=10)
    c.loop_start()
    return c


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--local-host", default="192.168.4.2", help="broker the laptop and the drone use")
    p.add_argument("--local-port", type=int, default=1883)
    p.add_argument("--cloud-host", required=True, help="broker the cloud VLM service publishes to")
    p.add_argument("--cloud-port", type=int, default=1883)
    p.add_argument("--pilot-topic", default="skyhand/pilot_vel")
    p.add_argument("--out-topic", default="drone/cmd_vel")
    p.add_argument("--stop-only", action="store_true",
                   help="after backing off, the AI hovers instead of sliding sideways (no blind sideways moves)")
    a = p.parse_args()

    lock = threading.Lock()
    mixer = Mixer(Limits(stop_only=a.stop_only))
    st = {"pilot": None, "pilot_t": -1e9, "enabled": False, "view": None}

    def on_local(_c, _u, m):
        with lock:
            if m.topic == a.pilot_topic:
                try:
                    st["pilot"], st["pilot_t"] = json.loads(m.payload), time.time()
                except ValueError:
                    st["pilot"] = None
            elif m.topic == "skyhand/ai_enable":
                st["enabled"] = m.payload.strip() == b"1"

    def on_cloud(_c, _u, m):
        if m.topic == "skyhand/cam_view":  # live preview: straight through to the pilot window
            local.publish("skyhand/ai_view", m.payload)
            return
        try:
            msg = json.loads(m.payload)
        except ValueError:
            return
        with lock:
            mixer.on_ai(msg, time.time())
            st["view"] = msg

    local = client("ai-mux-local", a.local_host, a.local_port, [a.pilot_topic, "skyhand/ai_enable"], on_local)
    cloud = client("ai-mux-cloud", a.cloud_host, a.cloud_port, ["skyhand/ai_cmd", "skyhand/cam_view"], on_cloud,
                   os.environ.get("SKYHAND_CLOUD_USER"), os.environ.get("SKYHAND_CLOUD_PASS"))
    print(f"[mux] local {a.local_host}  cloud {a.cloud_host}  out {a.out_topic}"
          f"  AI may: back off + {'hover' if a.stop_only else 'hover or slide sideways'}", flush=True)

    last = {"vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0}
    burst, n = 0, 0
    dt = 1.0 / RATE_HZ
    try:
        while True:
            t0 = time.time()
            with lock:
                cmd, mode = mixer.step(st["pilot"], t0 - st["pilot_t"], t0, st["enabled"])
                view, st["view"] = st["view"], None
                ai, ai_age, enabled = mixer.ai, mixer.ai_age(t0), st["enabled"]
            if cmd is None:  # no pilot: hover burst, then silence (never fight another publisher)
                last = {k: 0.0 for k in last}
                if burst < ZERO_BURST:
                    local.publish(a.out_topic, json.dumps(last))
                    burst += 1
            else:
                burst = 0
                slew = SLEW_AI if mode in ("backoff", "avoid_left", "avoid_right", "stop") else SLEW
                last = {k: last[k] + max(-slew * dt, min(slew * dt, cmd[k] - last[k])) for k in last}
                local.publish(a.out_topic, json.dumps({k: round(v, 3) for k, v in last.items()}))
            n += 1
            if n % 4 == 0:  # 5 Hz status for the laptop GUI
                local.publish("skyhand/ai_status", json.dumps({
                    "mode": mode, "enabled": enabled, "ai_age": round(min(ai_age, 99.0), 2),
                    "action": (ai or {}).get("action", ""), "conf": (ai or {}).get("conf", 0),
                    "reason": (ai or {}).get("reason", ""), "latency": (ai or {}).get("latency", 0),
                    "obstacle": (ai or {}).get("obstacle", ""), "distance": (ai or {}).get("distance", ""),
                    "interval": (ai or {}).get("interval", 1.0), "act_window": mixer.lim.ai_fresh_s,
                    "stop_only": a.stop_only,
                    "out": {k: round(v, 3) for k, v in last.items()}}))
            if view and view.get("thumb"):
                local.publish("skyhand/ai_view", view["thumb"])
            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        local.loop_stop()
        cloud.loop_stop()


if __name__ == "__main__":
    main()
