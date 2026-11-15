#!/usr/bin/env python3
"""CLOUD side: watches the drone's camera stream and publishes AI avoidance commands (skyhand/ai_cmd).

  export VERDA_INFERENCE_KEY=...     # never commit it
  python3 scripts/vlm_service.py --source rtsp://localhost:8554/drone --mqtt-host localhost

Back-to-back: newest frame -> VLM obstacle check (~0.5 s) -> {"action", "vy", "conf", ...} + a small thumbnail of
the exact frame it judged. The drone's ai_mux decides if/when that is allowed to move the drone.
A failed call publishes nothing, so the mux sees the AI as stale and the pilot keeps full control.
--source: RTSP/RTMP/http URL, a video file (played in real time, looped) or a webcam index (0).
--dry-run prints instead of publishing.
"""
import argparse
import base64
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The drone's RTMP stream carries tiny picture-less packets; with default probing OpenCV takes ~30 s to open it.
# Smaller probe = ~5 s. Also mute the per-packet 'missing picture' decoder spam.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "analyzeduration;1000000|probesize;500000")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
import cv2  # noqa: E402

from skyhand import vlm  # noqa: E402

AI_SIDE_SPEED = 0.4  # m/s requested for avoid_left/right; the mux caps it again
STREAM_LOST_S = 3.0  # no good frame for this long = stream lost, reconnect
PREVIEW_HZ = 2.0     # live camera thumbnails (skyhand/cam_view) for the pilot window


class LatestFrame:
    """Reads the stream continuously so we always analyse the newest frame, not a buffered old one."""

    def __init__(self, source):
        self.src = int(source) if str(source).isdigit() else source
        self.is_file = isinstance(self.src, str) and os.path.isfile(self.src)
        self.frame, self.t = None, 0.0
        self.stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self.stop:
            cap = cv2.VideoCapture(self.src)
            period = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 25.0) if self.is_file else 0.0
            last_ok = time.time()
            while not self.stop:
                ok, f = cap.read()
                if not ok:
                    # The drone's stream has picture-less packets that make read() fail now and then. Reconnecting
                    # on every failure restarted the stream from old buffered frames every few seconds (stale,
                    # laggy video). Only treat it as lost after STREAM_LOST_S without a single good frame.
                    if self.is_file or not cap.isOpened() or time.time() - last_ok > STREAM_LOST_S:
                        break
                    time.sleep(0.005)
                    continue
                last_ok = time.time()
                self.frame, self.t = f, last_ok
                if period:
                    time.sleep(period)  # play files in real time
            cap.release()
            if not self.is_file:
                time.sleep(1.0)  # lost stream: retry

    def get(self):
        return self.frame, (time.time() - self.t) if self.frame is not None else 1e9


def thumb_b64(frame, side=240):
    h, w = frame.shape[:2]
    k = side / max(h, w)
    ok, buf = cv2.imencode(".jpg", cv2.resize(frame, (int(w * k), int(h * k))), [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--mqtt-host", default="localhost")
    p.add_argument("--mqtt-port", type=int, default=1883)
    p.add_argument("--min-interval", type=float, default=0.3, help="s between the starts of two VLM calls")
    p.add_argument("--workers", type=int, default=1,
                   help="VLM calls allowed in flight at once (2 = a new decision about twice as often)")
    p.add_argument("--max-frame-age", type=float, default=1.0, help="skip frames older than this (s)")
    p.add_argument("--goal", default="", help="optional context, e.g. 'fly to the red car'")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    client = None
    if not a.dry_run:
        import paho.mqtt.client as mqtt
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"vlm-service-{int(time.time())}")
        if os.environ.get("SKYHAND_CLOUD_USER"):
            client.username_pw_set(os.environ["SKYHAND_CLOUD_USER"], os.environ.get("SKYHAND_CLOUD_PASS"))
        client.reconnect_delay_set(1, 5)
        client.connect_async(a.mqtt_host, a.mqtt_port, keepalive=15)
        client.loop_start()

    src = LatestFrame(a.source)

    def preview():  # live camera thumbnail for the pilot's window, independent of the (slower) VLM calls
        while not src.stop:
            frame, age = src.get()
            if client and frame is not None and age < 1.0:
                client.publish("skyhand/cam_view", thumb_b64(frame), qos=0)
            time.sleep(1.0 / PREVIEW_HZ)
    threading.Thread(target=preview, daemon=True).start()
    print(f"[vlm] source={a.source} broker={'(dry-run)' if a.dry_run else a.mqtt_host}", flush=True)
    st = {"seq": 0, "next": 0.0, "last_frame_t": 0.0}
    lock = threading.Lock()

    def worker():
        # Several workers may have calls in flight at once; call starts are spaced by --min-interval and a result
        # is only published if its frame is newer than the last published one (never an out-of-order, older view).
        while not src.stop:
            with lock:
                now = time.time()
                wait = st["next"] - now
                if wait <= 0:
                    st["next"] = now + a.min_interval
            if wait > 0:
                time.sleep(wait)
                continue
            t0 = time.time()
            frame, age = src.get()
            frame_t = src.t
            if frame is None or age > a.max_frame_age:
                time.sleep(0.2)
                continue
            try:
                r = vlm.avoid(vlm.jpeg_b64(frame), a.goal)
            except Exception as e:  # noqa: BLE001 - any failure = say nothing; the mux sees the AI as stale
                print(f"[vlm] call failed: {type(e).__name__}: {e}", flush=True)
                time.sleep(1.0)
                continue
            with lock:
                if frame_t <= st["last_frame_t"]:
                    continue  # a newer view was already published
                st["last_frame_t"] = frame_t
                st["seq"] += 1
                seq = st["seq"]
            side = {"avoid_left": -AI_SIDE_SPEED, "avoid_right": AI_SIDE_SPEED}.get(r["action"], 0.0)
            msg = {"seq": seq, "action": r["action"], "vx": 0.0, "vy": side, "conf": r["confidence"],
                   "distance": r["distance"], "obstacle": r["obstacle"], "reason": r["reason"],
                   "latency": round(time.time() - t0, 2), "frame_age": round(age, 2),
                   "interval": a.min_interval}
            print(f"[vlm] {json.dumps(msg)}", flush=True)
            if client:
                client.publish("skyhand/ai_cmd", json.dumps({**msg, "thumb": thumb_b64(frame)}), qos=0)

    for _ in range(max(1, a.workers)):
        threading.Thread(target=worker, daemon=True).start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        src.stop = True
        if client:
            client.loop_stop()


if __name__ == "__main__":
    main()
