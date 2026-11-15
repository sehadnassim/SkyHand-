#!/usr/bin/env python3
"""Real glove -> SkyHand without Rokoko Studio, using the native Rokoko Device SDK.

    rokoko-sdk --auto-stream-usb        # terminal 1: SDK driver (glove on USB)
    python3 scripts/sdk_bridge.py       # terminal 2: solver + bridge -> UDP 14043
    python3 -m skyhand --drone sim --viz   # terminal 3

Starts `rkk-hand-solver --no-driver --emit world` unless one is already serving on
--solver-port, reads the solved hand over RGMP v2 and re-sends it as Rokoko Studio
JSON v3 to SkyHand's UDP port. It also reads the raw driver stream (gyro, accelerometer, sensor positions,
tracking quality, status) and adds it to each packet as "skyhand_raw" (see skyhand/sdk_raw.py).
"""
import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from skyhand.sdk_convert import joints_to_studio_packet  # noqa: E402
from skyhand.sdk_raw import RawCollector  # noqa: E402

SDK_HOME = os.environ.get("ROKOKO_SDK_HOME", os.path.expanduser("~/.local/share/rokoko-device-sdk"))
sys.path.insert(0, os.path.join(SDK_HOME, "examples", "python"))
try:
    from rgmp_core.client import RgmpClient  # noqa: E402
    from rgmp_core.layout import read_slot  # noqa: E402
except ImportError:
    sys.exit(f"Rokoko SDK examples not found under {SDK_HOME}; install the SDK or set ROKOKO_SDK_HOME")


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_solver(port: int):
    exe = shutil.which("rkk-hand-solver") or os.path.join(SDK_HOME, "bin", "rkk-hand-solver")
    proc = subprocess.Popen([exe, "--no-driver", "--emit", "world", "--listen", f"127.0.0.1:{port}"],
                            start_new_session=True)  # Ctrl+Z in the terminal must not freeze it
    for _ in range(50):
        if port_open(port):
            return proc
        if proc.poll() is not None:
            sys.exit("rkk-hand-solver exited; is the driver running (rokoko-sdk --auto-stream-usb)?")
        time.sleep(0.1)
    proc.terminate()
    sys.exit("rkk-hand-solver did not start serving")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver-port", type=int, default=12277)
    ap.add_argument("--host", default="127.0.0.1", help="where SkyHand runs")
    ap.add_argument("--port", type=int, default=14043, help="SkyHand --port")
    ap.add_argument("--driver-port", type=int, default=12276, help="raw SDK driver (IMU data); 0 = off")
    a = ap.parse_args()
    raw = RawCollector()

    def raw_loop():
        while True:
            try:
                with RgmpClient("127.0.0.1", a.driver_port) as client:
                    client.on_data = lambda dev, gid, _ts, payload, layout: raw.update(
                        dev, payload, layout, layout.groups.get(gid, []), read_slot)
                    print(f"[sdk_bridge] raw IMU rgmp://127.0.0.1:{a.driver_port}", flush=True)
                    client.run()
            except (OSError, ConnectionError):
                time.sleep(1)

    if a.driver_port:
        import threading
        threading.Thread(target=raw_loop, daemon=True).start()

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # so `finally` stops the solver
    solver = None if port_open(a.solver_port) else start_solver(a.solver_port)
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sent, t0 = 0, time.time()

    def on_data(_dev, _gid, _ts, payload, layout):
        nonlocal sent, t0
        joints = {}
        for slots in layout.groups.values():
            for slot in slots:
                entry = slot.entry or {}
                if entry.get("measure_type") == "TRANSFORM" and slot.count == 7:
                    try:
                        joints[entry.get("target_frame")] = read_slot(slot, payload)
                    except Exception:
                        pass
        packet = joints_to_studio_packet(joints)
        if packet and a.driver_port:
            body = packet["scene"]["actors"][0]["body"]
            sides = tuple(sd for sd in ("right", "left") if sd + "Hand" in body)
            packet["skyhand_raw"] = raw.snapshot(sides)
        if packet:
            out.sendto(json.dumps(packet).encode(), (a.host, a.port))
            sent += 1
        if time.time() - t0 >= 2.0:
            print(f"[sdk_bridge] {sent / (time.time() - t0):5.1f} Hz -> {a.host}:{a.port}", flush=True)
            sent, t0 = 0, time.time()

    try:
        while True:
            try:
                with RgmpClient("127.0.0.1", a.solver_port) as client:
                    client.on_data = on_data
                    print(f"[sdk_bridge] solved hand rgmp://127.0.0.1:{a.solver_port} -> udp://{a.host}:{a.port}")
                    client.run()
            except (OSError, ConnectionError) as e:
                print(f"[sdk_bridge] {e}; retrying in 1 s")
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if solver:
            solver.terminate()


if __name__ == "__main__":
    main()
