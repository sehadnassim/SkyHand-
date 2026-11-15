"""SkyHand main loop: Rokoko glove -> gestures / direct velocity / LLM copilot -> drone.

Priority each tick:  FIST (abort + hover)  >  running plan  >  OPEN hand direct flight  >  hover.
"""
import argparse
import math
import queue
import sys
import threading
import time

from .drones import make_drone
from .gestures import G, GestureTracker
from .hand import DEFAULT_CURL_RANGE, HandModel
from .llm import ask_llm
from . import shapes
from .mapping import HandToVelocity, MapConfig, VelCmd, _shape
from .ride import PathRide
from .planner import Context, Executor, sanitize
from .recorder import Recorder
from .rokoko import FINGERS, RokokoReceiver
from .sketch import Sketch

HELP = """Gestures (right hand):
  open hand  -> direct flight: tilt fingers down = forward, roll = sideways, turn = yaw,
                thumb out = climb, thumb tucked = descend
  fist       -> brake & hover (also aborts any plan)            [dead-man]
  thumbs-up (hold 1s) -> take off        pinky (hold 1s) -> land
  point (hold 1s)     -> fly 3 m where you point
  peace (hold 1s)     -> mark waypoint here
  rock (hold 1s)   -> start / stop air drawing (draw with the index finger); on stop the drone flies it
Keyboard: type a sentence + Enter for the LLM copilot | c = recalibrate (hand flat, fingers fwd) | f = calibrate fist | q = quit"""


# --simple: the 4 best-separated shapes in the pilot data (peace is >10 cluster widths from open/fist/thumbs up;
# point and pinky only ~4 from fist); everything else hovers. Open hand tilt = forward/back only,
# peace sign rolled right/left = turn right/left.
SIMPLE = (G.OPEN, G.FIST, G.THUMB, G.PEACE)  # thumbs up toggles take off / land, peace = turn


class SkyHand:
    def __init__(self, args):
        self.args = args
        self.rx = RokokoReceiver(port=args.port, side=args.side, actor_index=args.actor)
        kw = {}
        if args.drone in ("udp", "dji"):
            kw = dict(host=args.udp_host, port=args.udp_port, state_port=args.state_port)
        elif args.drone == "mqtt":
            kw = dict(host=args.mqtt_host, port=args.mqtt_port, vel_topic=args.mqtt_vel_topic,
                      cmd_topic=args.mqtt_cmd_topic)
        elif args.drone == "ros2":
            kw = dict(vel_topic=args.vel_topic, odom_topic=args.odom_topic)
        self.drone = make_drone(args.drone, **kw)
        rng = None
        if args.finger_range:
            rng = {**DEFAULT_CURL_RANGE, **{f: args.finger_range for f in FINGERS if f != "Thumb"}}
        self.hand = HandModel(rng)
        self.gest = GestureTracker(hold_s=args.hold)
        self.gmodel = None
        if args.gesture_model:
            from .gesture_model import GestureModel
            self.gmodel = GestureModel(args.gesture_model, min_conf=args.min_conf)
            print(f"[skyhand] using learned gesture model {args.gesture_model}")
        self.map = HandToVelocity(MapConfig(vertical_source=args.vertical, max_xy=args.max_speed,
                                              axis_lock=args.simple))
        if args.simple:
            self.map.cfg.lock_axes = ("vx",)
        self.exec = Executor(self.drone)
        self.ctx = Context()
        self.sketch = Sketch()
        self.rec = Recorder(args.record_dir) if args.record else None
        self.inbox: queue.Queue = queue.Queue()
        self.llm_busy = False
        self.pending = None  # LLM plan waiting for the pilot's thumbs-up
        self.pending_t = 0.0
        self.last_plan = None  # what the LLM refines ("bigger", "again but higher")
        self.paused = False    # fist during a plan = pause (plan kept), thumbs up = continue
        self.ride = None       # PathRide while the hand drives along an LLM-picked shape
        self._fist_t = 0.0
        self._prev_g = G.NONE
        self.running = True
        self.status = {"mode": "waiting for glove", "gesture": "-", "cmd": VelCmd(), "msg": "", "hs": None}
        self._first_frame_t = None

    # ---------- user input ----------
    def _stdin_loop(self):
        for line in sys.stdin:
            self.inbox.put(line.strip())

    def _handle_text(self, text):
        if not text:
            return
        if text == "q":
            self.running = False
        elif text == "c":
            self._recal = True
        elif text == "f":
            self._fistcal = True
        elif not self.llm_busy:
            self.llm_busy = True
            self.say(f"thinking about: {text!r}")

            def work():
                try:
                    res = ask_llm(text, self.ctx, self.drone.state(), self.status["gesture"],
                                  plan=self.last_plan)
                    self.inbox.put(("plan", res))
                finally:
                    self.llm_busy = False
            threading.Thread(target=work, daemon=True).start()

    def say(self, msg):
        self.status["msg"] = msg
        print(f"[skyhand] {msg}", flush=True)

    # ---------- human in the loop ----------
    def _propose(self, res, now):
        """The LLM only proposes. Nothing flies until the pilot shows a thumbs-up."""
        plan = sanitize(res.get("plan", []))
        if not plan:
            self.say(f"copilot: {res.get('say', '')} (no plan)")
            return
        self.pending, self.pending_t, self.last_plan = plan, now, plan
        self.say(f"copilot: {res.get('say', '')}  plan={[s['op'] for s in plan]}  "
                 f"-> THUMBS UP to fly, FIST to cancel")

    def _ride_tick(self, g, hs, dt):
        """While the plan is at a `path` step, the hand drives along the shape. None = not riding."""
        cur = self.exec.current
        st = self.drone.state()
        if not cur or cur.get("kind") != "path" or not st:
            self.ride = None
            return None
        if self.ride is None or self.ride.seg is not cur:
            self.ride = PathRide(cur["shape"], cur["size"], st, cur.get("mirror", False))
            self.ride.seg = cur
            self.say(f"{cur['shape']} {cur['size']:g} m: tilt forward = go, back = reverse, flat = hold, "
                     f"peace + roll = shift sideways, fist = pause")
        cfg = self.map.cfg
        throttle = -_shape(hs.pitch, cfg.tilt_full, cfg) if g == G.OPEN else 0.0
        shift = _shape(hs.roll, cfg.tilt_full, cfg) if g == G.PEACE else 0.0
        cmd = self.ride.step(st, throttle, shift, dt)
        if self.ride.done:
            self.exec.current = None  # next plan step (e.g. land) can run
            self.say(f"{self.ride.shape} done")
            self.ride = None
            return VelCmd()
        return cmd

    def path_preview(self):
        """(world points, current point or None) for the viz: the ride, or the shape of a plan waiting for thumbs up."""
        if self.ride is not None:
            return self.ride.world_path(), self.ride.point(self.ride.s)
        st = self.drone.state()
        for step in self.pending or []:
            if step["op"] == "path" and st:
                return shapes.to_world(shapes.make(step["shape"], step["size"], step.get("mirror", False)), st["x"], st["y"], st["z"],
                                       st["yaw"]), None
        return None, None

    def _hand_active(self, hs) -> bool:
        return max(abs(hs.pitch), abs(hs.roll)) > math.radians(self.args.override_deg)

    def _approve(self):
        self.exec.load(self.pending, self.ctx)
        self.pending, self.paused = None, False
        self.say("plan approved - flying it (fist = abort, tilt your hand = take over)")

    def _act(self, name, now):
        """Button actions from the GUI (same effects as the gestures)."""
        if name == "approve" and self.pending:
            self._approve()
        elif name == "cancel" and self.pending:
            self.pending = None
            self.say("plan cancelled")
        elif name in ("takeoff", "land"):
            self.pending = None
            self.exec.load([{"op": name}], self.ctx)
            self.say("taking off" if name == "takeoff" else "landing")
        elif name == "stop":
            self.exec.abort()
            self.pending, self.paused = None, False
            self.map.reset()
            self.say("stopped - hovering")
        elif name == "resume" and self.paused:
            self.paused = False
            self.say("continuing the plan")

    # ---------- gesture events ----------
    def _on_event(self, ev, hs):
        st = self.drone.state()
        if ev == G.THUMB and self.paused:
            self.paused = False
            self.say("continuing the plan")
        elif ev == G.THUMB and self.pending:
            self._approve()
        elif ev == G.THUMB and self.args.simple and st and st.get("airborne", st.get("z", 0) > 0.3):
            self.exec.load([{"op": "land"}], self.ctx)
            self.say("landing")
        elif ev == G.THUMB:
            self.exec.load([{"op": "takeoff"}], self.ctx)
            self.say("take off")
        elif ev == G.PINKY:
            self.exec.load([{"op": "land"}], self.ctx)
            self.say("landing")
        elif ev == G.ROCK:
            if self.sketch.recording:
                self.ctx.sketch = self.sketch.stop()
                if self.ctx.sketch:
                    self.exec.load([{"op": "trace_sketch", "size": self.args.sketch_size}], self.ctx)
                    self.say(f"flying your drawing ({len(self.ctx.sketch)} pts)")
                else:
                    self.say("drawing too small, ignored")
            else:
                self.sketch.start()
                self.say("drawing... point with the index finger, rock again to finish")
        elif ev == G.POINT and not self.sketch.recording:
            self.exec.load([{"op": "goto_pointed", "distance": self.args.point_dist}], self.ctx)
            self.say(f"going where you point (yaw {math.degrees(hs.point_yaw):.0f}°, "
                     f"pitch {math.degrees(hs.point_pitch):.0f}°)")
        elif ev == G.PEACE:
            if st:
                wid = len(self.ctx.waypoints) + 1
                self.ctx.waypoints[wid] = (st["x"], st["y"], st["z"])
                self.say(f"waypoint {wid} marked")
            else:
                self.say("waypoints need drone odometry")

    # ---------- main loop ----------
    def control_loop(self):
        self._recal = self._fistcal = False
        dt, last = 1.0 / self.args.rate, time.time()
        while self.running:
            t0 = time.time()
            real_dt, last = t0 - last, t0
            while not self.inbox.empty():
                item = self.inbox.get()
                if isinstance(item, tuple) and item[0] == "act":
                    self._act(item[1], t0)
                elif isinstance(item, tuple):
                    self._propose(item[1], t0)
                else:
                    self._handle_text(item)
            frame = self.rx.latest()
            cmd = VelCmd()
            hs, g = None, G.NONE
            if frame is None:
                self.status["mode"] = "NO GLOVE -> hover"
                if self.exec.busy and self.args.abort_on_glove_loss:
                    self.exec.abort()
                    self.say("glove lost - plan aborted, hovering")
            else:
                self._first_frame_t = self._first_frame_t or t0
                if not self.hand.calibrated and t0 - self._first_frame_t < 1.5:
                    self.status["mode"] = "calibrating: hold hand flat, fingers forward"
                    self._tick(cmd, real_dt)
                    continue
                if self._recal or not self.hand.calibrated:
                    self.hand.calibrate(frame)
                    self.map.reset()
                    self._recal = False
                    self.say("calibrated")
                if self._fistcal:
                    self.hand.calibrate_fist(frame)
                    self._fistcal = False
                    self.say("fist calibrated")
                hs = self.hand.update(frame)
                self.ctx.point_yaw, self.ctx.point_pitch = hs.point_yaw, hs.point_pitch
                learned = None
                if self.gmodel is not None:
                    learned, self.status["conf"] = self.gmodel(frame)
                g, ev = self.gest.update(hs.curls, t0, learned)
                if self.args.simple and g not in SIMPLE:  # ambiguous shapes (point/peace/rock) -> hover
                    g, ev = G.NONE, None
                if g == G.POINT:
                    self.sketch.add(hs.point_yaw, hs.point_pitch)
                if self.pending and t0 - self.pending_t > self.args.plan_timeout:
                    self.pending = None
                    self.say("plan not approved in time - dropped")
                if g == G.FIST:
                    if self._prev_g != G.FIST:
                        self._fist_t = t0
                    if self.pending:
                        self.pending = None
                        self.say("plan cancelled")
                    if self.exec.busy and not self.paused:
                        self.paused = True
                        self.say(f"PAUSED - thumbs up = continue, hold fist {self.args.abort_hold:.0f} s = abort")
                    elif self.paused and t0 - self._fist_t >= self.args.abort_hold:
                        self.exec.abort()
                        self.paused = False
                        self.say("plan ABORTED - hovering (thumbs up = land)")
                    self.map.reset()
                    self.status["mode"] = "PAUSED (fist)" if self.paused else "BRAKE (fist)"
                else:
                    if ev and not (self.args.simple and ev == G.PEACE):  # peace = turn, not a waypoint
                        self._on_event(ev, hs)
                    riding = (self.exec.current or {}).get("kind") == "path"
                    if self.exec.busy and g == G.OPEN and self._hand_active(hs) and not riding:
                        self.exec.abort()  # the pilot's hand always wins over the plan
                        self.paused = False
                        self.say("hand override - plan stopped, you have control")
                    plan_cmd = None if self.paused else self._ride_tick(g, hs, real_dt)
                    if plan_cmd is None and not self.paused:
                        plan_cmd = self.exec.step(real_dt)
                    if plan_cmd is not None:
                        cmd = plan_cmd
                        self.status["mode"] = (f"PATH {self.ride.shape} {self.ride.progress:.0%}: tilt = go / back, "
                                               f"flat = hold, peace+roll = shift, fist = pause") if self.ride \
                            else "AUTO plan (fist = abort)"
                    elif g == G.OPEN:
                        cmd = self.map(hs)
                        self.status["mode"] = "DIRECT flight"
                    elif g == G.PEACE and self.args.simple:
                        cmd = self.map.turn(hs)
                        self.status["mode"] = "TURN (peace, roll hand)"
                    else:
                        self.map.reset()
                        self.status["mode"] = "drawing" if self.sketch.recording else "hover"
            if frame is None and self.exec.busy and not self.paused and not self.args.abort_on_glove_loss:
                cmd = self.exec.step(real_dt) or VelCmd()
            self._prev_g = g
            if self.paused:
                self.status["mode"] = "PAUSED: thumbs up = continue, hold fist = abort, tilt = take over"
            if self.pending:
                self.status["mode"] = f"PLAN READY {[st['op'] for st in self.pending]}: thumbs up = fly, fist = cancel"
            self.status.update(gesture=g.value, hs=hs)
            self._tick(cmd, real_dt)
            if self.rec:
                self.rec.log(gesture=g.value, mode=self.status["mode"], cmd=cmd.as_list(), hand=hs,
                             drone=self.drone.state(), sketching=self.sketch.recording)
            time.sleep(max(0.0, dt - (time.time() - t0)))

    def _tick(self, cmd, dt):
        intent = cmd  # what the hand asks for, even while the drone is still on the ground
        st = self.drone.state()
        if st is not None and not st.get("airborne", True):
            cmd = VelCmd()
        self.status["intent"] = intent
        self.status["cmd"] = cmd
        self.drone.send_velocity(cmd)
        self.drone.tick(dt)
        if not self.args.quiet:
            c = cmd
            hs = self.status["hs"]
            tilt = f"p{math.degrees(hs.pitch):+4.0f} r{math.degrees(hs.roll):+4.0f} y{math.degrees(hs.yaw):+4.0f}" if hs else ""
            sys.stderr.write(f"\r{self.status['mode'][:28]:28s} {self.status['gesture']:9s} {tilt} "
                             f"cmd vx{c.vx:+.2f} vy{c.vy:+.2f} vz{c.vz:+.2f} wz{c.yaw_rate:+.2f}   ")

    def run(self):
        print(HELP)
        import signal

        def quit_on_ctrl_z(*_):  # Ctrl+Z would leave a frozen sim holding the glove port
            self.running = False
            if self.args.viz:
                import matplotlib.pyplot as plt
                plt.close("all")
            else:
                raise KeyboardInterrupt
        signal.signal(signal.SIGTSTP, quit_on_ctrl_z)
        threading.Thread(target=self._stdin_loop, daemon=True).start()
        if self.args.viz:
            from .viz import run_viz
            threading.Thread(target=self.control_loop, daemon=True).start()
            run_viz(self)
            self.running = False
        else:
            try:
                self.control_loop()
            except KeyboardInterrupt:
                pass
        self.shutdown()

    def shutdown(self):
        self.running = False
        self.drone.send_velocity(VelCmd())
        self.drone.close()
        self.rx.close()
        if self.rec:
            self.rec.close()
            print(f"\nrecording saved to {self.rec.path}")


def main(argv=None):
    p = argparse.ArgumentParser(description="SkyHand: fly a drone with a Rokoko glove + LLM copilot")
    p.add_argument("--drone", default="sim", choices=["sim", "dji", "udp", "ros2", "mqtt"])
    p.add_argument("--port", type=int, default=14043, help="Rokoko Studio custom streaming UDP port")
    p.add_argument("--side", default="right", choices=["right", "left"])
    p.add_argument("--actor", type=int, default=0)
    p.add_argument("--rate", type=float, default=30.0, help="control loop Hz")
    p.add_argument("--hold", type=float, default=1.0, help="seconds to hold a command gesture")
    p.add_argument("--vertical", default="thumb", choices=["thumb", "height"])
    p.add_argument("--finger-range", type=float, default=None,
                   help="finger bend [rad] from calibration to full fist (Studio ~3.4, Device SDK ~1.5)")
    p.add_argument("--gesture-model", default=None, help="learned gesture model (.npz from ./fly.sh --train)")
    p.add_argument("--min-conf", type=float, default=0.7, help="below this model confidence -> no gesture (hover)")
    p.add_argument("--simple", action="store_true",
                   help="open hand tilt = forward/back, peace + roll = turn, fist = stop, thumbs up = take off / land")
    p.add_argument("--max-speed", type=float, default=1.5)
    p.add_argument("--point-dist", type=float, default=3.0)
    p.add_argument("--sketch-size", type=float, default=2.0)
    p.add_argument("--udp-host", default="127.0.0.1", help="dji/udp: phone or bridge IP")
    p.add_argument("--udp-port", type=int, default=15000)
    p.add_argument("--state-port", type=int, default=15001, help="dji/udp: port we receive drone state on (0=off)")
    p.add_argument("--mqtt-host", default="192.168.4.3")
    p.add_argument("--mqtt-port", type=int, default=1883)
    p.add_argument("--mqtt-vel-topic", default="drone/cmd_vel")
    p.add_argument("--mqtt-cmd-topic", default="drone/command")
    p.add_argument("--vel-topic", default="/drone/cmd_vel")
    p.add_argument("--odom-topic", default="")
    p.add_argument("--abort-on-glove-loss", action="store_true", default=True)
    p.add_argument("--viz", action="store_true", help="live 3D plot (sim or any backend with odometry)")
    p.add_argument("--record", action="store_true", help="log demonstrations to JSONL")
    p.add_argument("--record-dir", default="recordings")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--abort-hold", type=float, default=2.0, help="s of fist (while paused) that aborts a plan")
    p.add_argument("--plan-timeout", type=float, default=20.0, help="s to approve an LLM plan with thumbs up")
    p.add_argument("--override-deg", type=float, default=20.0, help="hand tilt that takes over from a plan")
    SkyHand(p.parse_args(argv)).run()


if __name__ == "__main__":
    main()
