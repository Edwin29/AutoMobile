"""Run a bench controller (default hyb2m) on the car's Raspberry Pi, or in a software-only
stand-in (--sim) that uses the bench renderer and vehicle model instead of camera/motors.

    python -m bench.pi_run                       # DRY on the Pi: camera + decisions, motors untouched
    python -m bench.pi_run --drive               # real motors
    python -m bench.pi_run --sim --scenario inner_loop   # laptop: same loop, fake camera / motors
    python -m bench.pi_run --pose 0 -60 0 ...    # start somewhere else (x y yaw; yaw 0 = +y, + = clockwise)

Orchestrator commands (same Pi; UDP JSON or plain words to 127.0.0.1:<--cmd-port>, default 8765):
    {"cmd": "direction", "move": "right"}   or   right / left / straight
    {"cmd": "stop", "reason": "stop_sign"}   or   stop      (motors zeroed at once by the receiver)
    {"cmd": "resume", "reason": "stop_sign"} or   resume    (releases only that reason's hold;
                                                             the car moves when no hold is left)
    {"cmd": "resume_all"}                    or   resume_all
    optional "command_id": a repeated id is ignored (UDP re-sends)
    plain words use reason "orchestrator"
    {"cmd": "quit"}                          or   quit      (ends the run, motors zero)
  python -m bench.pi_cmd stop     sends one from a terminal.
Events go to 127.0.0.1:<--event-port> (default 8766) as JSON lines (junction decisions,
stop/resume acknowledgements) and into the run log.

Stops: Ctrl+C / SIGTERM / SSH hang-up / any board button / quit command / --max-seconds.
A 0.5 s watchdog zeroes the motors if the loop hangs. The camera is upright after a 180 deg
rotation (as all bench controllers expect: the team CNN's --flip180, v4 rotate180=False).
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import json
import queue
import signal
import socket
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

WORDS = {"left": ("direction", "left"), "right": ("direction", "right"), "straight": ("direction", "straight"),
         "stop": ("stop", None), "resume": ("resume", None), "resume_all": ("resume_all", None),
         "quit": ("quit", None)}


def parse_command(raw):
    """-> dict(kind, arg, reason, command_id) or None."""
    text = raw.decode("utf-8", "replace").strip()
    try:
        m = json.loads(text)
    except json.JSONDecodeError:
        m = None
    if isinstance(m, dict):
        cmd, reason, cid = m.get("cmd"), str(m.get("reason") or "orchestrator"), m.get("command_id")
        if cmd == "direction" and m.get("move") in ("left", "right", "straight"):
            return dict(kind="direction", arg=m["move"], reason=reason, command_id=cid)
        if cmd in ("left", "right", "straight"):
            return dict(kind="direction", arg=cmd, reason=reason, command_id=cid)
        if cmd in ("stop", "resume", "resume_all", "quit"):
            return dict(kind=cmd, arg=None, reason=reason, command_id=cid)
        return None
    w = WORDS.get(text.lower())
    return None if w is None else dict(kind=w[0], arg=w[1], reason="orchestrator", command_id=None)


class StopFlag:
    def __init__(self):
        self.reason, self.event = None, threading.Event()

    def set(self, reason):
        if self.reason is None:
            self.reason = reason
        self.event.set()


class Motors:
    """probe.Motors (verified wiring, signed duty = reverse supported) + watchdog thread
    that also polls the board buttons. hold() zeroes at once (orchestrator stop)."""

    def __init__(self, probe_dir, cap, stop, watchdog_s=0.5):
        import fcntl
        sys.path.insert(0, str(probe_dir))
        from probe import Motors as ProbeMotors
        self.lockfile = open("/tmp/autodrive-power-probe.lock", "a")
        fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.m = ProbeMotors(1000)
        self.cap, self.stop, self.watchdog_s = cap, stop, watchdog_s
        self.lock, self.held = threading.Lock(), False
        self.last_write = time.monotonic()
        self.clock, self.trips, self.tripped = None, [], False   # watchdog zero writes (boot time)
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._guard, daemon=True)
        self.thread.start()

    def _guard(self):
        while not self.done.wait(0.02):
            if self.m.pressed():
                self.stop.set("board_button")
            with self.lock:
                if self.stop.event.is_set() or time.monotonic() - self.last_write > self.watchdog_s:
                    self.m.set(0.0, 0.0)
                    if not self.tripped and self.clock is not None:   # the controller must know
                        self.trips.append(self.clock())
                    self.tripped = True

    def hold(self, on):
        with self.lock:
            self.held = on
            if on:
                self.m.set(0.0, 0.0)

    def write(self, pwm):
        with self.lock:
            if self.stop.event.is_set() or self.held:
                self.m.set(0.0, 0.0)
                self.last_write = time.monotonic()
                return [0.0, 0.0]
            p = np.clip(np.asarray(pwm, float), -self.cap, self.cap)
            self.m.set(float(p[0]), float(p[1]))
            self.last_write, self.tripped = time.monotonic(), False
            return p.tolist()

    def drain_trips(self):
        with self.lock:
            out, self.trips = self.trips, []
            return out

    def close(self):
        self.done.set()
        self.thread.join(timeout=1)
        with self.lock:
            try:
                self.m.set(0.0, 0.0)
            finally:
                self.m.close()
                self.lockfile.close()


class CommandChannel:
    """UDP receiver thread -> queue. A stop zeroes the motors in the receiver itself, so the
    stop does not wait for the next camera frame."""

    def __init__(self, port, motors, clock):
        self.q, self.motors, self.clock = queue.Queue(), motors, clock
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.settimeout(0.2)
        self.alive, self.seen = True, set()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.alive:
            try:
                raw, _ = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            c = parse_command(raw)
            if c is None:
                continue
            if c["command_id"] is not None:
                if c["command_id"] in self.seen:          # UDP re-send of a command already taken
                    continue
                self.seen.add(c["command_id"])
            if c["kind"] == "stop" and self.motors is not None:
                self.motors.hold(True)
            self.q.put((self.clock(), c))

    def drain(self):
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                return out

    def close(self):
        self.alive = False
        self.sock.close()


class FrameSaver:
    """JPEG writing off the control loop (GPT review: inline saving slowed the loop).
    A full queue drops the frame and counts it instead of waiting."""

    def __init__(self, depth=8):
        self.q, self.dropped, self.saved = queue.Queue(maxsize=depth), 0, 0
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def put(self, path, image):
        try:
            self.q.put_nowait((path, image))
        except queue.Full:
            self.dropped += 1

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            cv2.imwrite(str(item[0]), item[1], [cv2.IMWRITE_JPEG_QUALITY, 85])
            self.saved += 1

    def close(self):
        self.q.put(None)
        self.t.join(timeout=5)


class Events:
    def __init__(self, port, log):
        self.addr = ("127.0.0.1", port) if port else None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if port else None
        self.log = log

    def send(self, **e):
        self.log.write(json.dumps(dict(event=True, **e), default=float) + "\n")
        if self.sock:
            try:
                self.sock.sendto(json.dumps(e, default=float).encode(), self.addr)
            except OSError:
                pass


# ---- software stand-in for camera + motors (laptop / desktop) -------------------------
class SimRig:
    """Bench renderer + vehicle in place of the Pi camera and motors; time is simulated.
    The scripted orchestrator of bench.mission talks to the run through the real UDP port."""

    def __init__(self, args, port):
        from .mission import SCENARIOS, Orchestrator, MissionScorer
        from .vehicle import Vehicle
        from .run import vehicle_from_args
        from .world import Camera, Renderer, Track
        from .disturb import Disturbances, combine
        self.track, self.camera = Track(), Camera()
        self.renderer = Renderer(self.track, self.camera)
        self.params = vehicle_from_args(args.vehicle, None)
        self.hint = start_pose(args, self.track, self.params)
        self.car = Vehicle(self.params, self.hint.copy())
        self.orch = Orchestrator(SCENARIOS[args.scenario], self.track, self.params.front_cm)
        self.scorer = MissionScorer(self.track, self.car.footprint(), args.direction, self.hint, self.params.front_cm)
        effects = [e for e in args.effects.split(",") if e]
        self.dist = Disturbances(combine(effects), args.direction, args.seed)
        self.dist.arm(self.renderer)
        self.realcam, self.history = None, [(0.0, self.hint.copy())]
        self.t, self.pwm, self.held = 0.0, np.zeros(2), False
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.port = port
        self.seed = args.seed

    def clock(self):
        return self.t

    def advance(self, target):
        while self.t < target - 1e-10 and self.scorer.end is None:
            dt = min(0.02, target - self.t)
            self.car.step(self.t, dt)
            self.t += dt
            self.scorer.sample(self.t, dt, self.car.pose, self.pwm, self.held)
            self.orch.watch(self.t, self.car.pose)
            self.history.append((self.t, self.car.pose.copy()))
            if len(self.history) > 400:
                del self.history[:200]
        for when, method, arg in [p for p in self.orch.pending if p[0] <= self.t]:
            self.orch.pending.remove((when, method, arg))
            msg = {"push_direction": {"cmd": "direction", "move": arg}, "stop": {"cmd": "stop"},
                   "resume": {"cmd": "resume"}}[method]
            self.sock.sendto(json.dumps(msg).encode(), ("127.0.0.1", self.port))
            self.orch.log.append(dict(t=self.t, method=method, arg=arg))

    def capture(self):
        self.advance(self.t + 1 / 30)
        if self.dist.p.get("realcam"):
            from .realcam import RealCam, pose_from_history
            if self.realcam is None:
                self.realcam = RealCam(self.renderer, self.camera, self.dist.p["realcam"], self.seed)
            frame = self.realcam.frame(self.t, pose_from_history(self.history), raw=True)
        else:
            frame = self.renderer.render_raw(self.car.pose)
        image, _ = self.dist.image(frame, self.t, self.renderer, self.history, raw=True)
        return cv2.rotate(image, cv2.ROTATE_180), self.t           # as the Pi camera delivers it (upside down)

    def write(self, pwm, done):
        self.advance(done)
        self.pwm = np.asarray(pwm, float)
        self.car.command(done, self.pwm)
        return self.pwm.tolist()


ARMS = {"N": (0.0, 1.0), "E": (1.0, 0.0), "S": (0.0, -1.0), "W": (-1.0, 0.0)}


def start_pose(args, track, params):
    """--pose x y yaw_deg, or --start junction:arm:cm (that far out on the arm, facing the
    junction; e.g. center:S:60 = 60 cm south of the crossroad heading north), or the
    start zone between the red lines."""
    if args.pose:
        return np.array([args.pose[0], args.pose[1], np.radians(args.pose[2])])
    if args.start:
        j, arm, cm = args.start.split(":")
        c, a = np.asarray(track.data["junctions"][j]["xy"], float), np.asarray(ARMS[arm.upper()])
        xy = c + float(cm) * a
        return np.array([xy[0], xy[1], float(np.arctan2(-a[0], -a[1]))])
    return track.start_pose(args.direction, params.front_cm, params.rear_cm)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controller", default="hyb2m")
    ap.add_argument("--direction", choices=["cw", "ccw"], default="cw")
    ap.add_argument("--pose", nargs=3, type=float, help="start x y yaw_deg (yaw 0 = +y / north, + = clockwise)")
    ap.add_argument("--start", help="junction:arm:cm, e.g. center:S:60 (60 cm south of the crossroad, facing it)")
    ap.add_argument("--stop-after-junctions", type=int, help="end the run once this many junctions are passed")
    ap.add_argument("--drive", action="store_true", help="enable motor output (Pi only)")
    ap.add_argument("--sim", action="store_true", help="software stand-in for camera and motors")
    ap.add_argument("--scenario", default="lap_defaults", help="--sim: scripted orchestrator (bench.mission SCENARIOS)")
    ap.add_argument("--vehicle", default="field_estimate", help="--sim: vehicle preset")
    ap.add_argument("--effects", default="", help="--sim: camera effects, e.g. realcam")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--pivot-duty", type=float, help="override the pivot wheel duty (default 0.40)")
    ap.add_argument("--stop-lead", type=float, help="stop this many seconds of travel early at junction stop points (default 0.12)")
    ap.add_argument("--creep-duty", type=float, help="split turn's short straight move duty (default 0.36)")
    ap.add_argument("--cnn-mixer", choices=["team", "v4"], default="team",
                    help="CNN command -> wheels: the team mixer (default) or the v4 allocator (keeps the "
                         "CNN's curvature when the inner wheel would fall below the start duty)")
    ap.add_argument("--turn-duty", type=float, help="override the slow mean duty used near junctions (default 0.34)")
    ap.add_argument("--inline-map", action="store_true", help="v4 map in the control loop (default: own process)")
    ap.add_argument("--script", help="scripted signs / lights from the car's map pose: a preset name "
                                     "(see --list-scripts) or tokens, e.g. \"dir:south:E:right, red:S:4\"")
    ap.add_argument("--list-scripts", action="store_true", help="show script presets and pink line indices")
    ap.add_argument("--max-seconds", type=float, default=600.0)
    ap.add_argument("--save-every", type=int, default=3, help="save every Nth upright frame (0 = none)")
    ap.add_argument("--cmd-port", type=int, default=8765)
    ap.add_argument("--event-port", type=int, default=8766)
    ap.add_argument("--probe-dir", default="/home/pi/straight30")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "pi_runs"))
    args = ap.parse_args()
    if args.list_scripts:
        from .pi_script import describe
        from .world import Track as _Track
        print(describe(_Track().data))
        return
    if args.drive and args.sim:
        ap.error("--drive and --sim are exclusive")
    os.environ.setdefault("BENCH_CNN_SPEED", "1.0")

    stop = StopFlag()
    for sig, name in ((signal.SIGINT, "ctrl_c"), (signal.SIGTERM, "sigterm")) + \
            (((signal.SIGHUP, "ssh_hangup"),) if hasattr(signal, "SIGHUP") else ()):
        signal.signal(sig, lambda *_, n=name: stop.set(n))
    cv2.setNumThreads(1)

    from .controllers import CONTROLLERS
    from .junction import load_plan_cache
    from .run import timing_profile
    mode = "sim" if args.sim else ("drive" if args.drive else "dry")
    out = Path(args.out) / f"{mode}_{args.controller}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)

    rig = SimRig(args, args.cmd_port) if args.sim else None
    if rig is None:
        from .world import Track
        from .vehicle import PRESETS
        hint = start_pose(args, Track(), PRESETS["field_estimate"])
    else:
        hint = rig.hint
    print("preparing controller (motors untouched)...", flush=True)
    ctrl = CONTROLLERS[args.controller](args.direction, hint, timing_profile("cnn_pi"))
    # The controller's motion model must integrate the PWM really written (dry run: zero;
    # stop / watchdog: zero; clipping), not its proposal (GPT review 2026-11-11).
    defer = hasattr(ctrl, "on_applied")
    if defer:
        ctrl.defer_apply = True
    if not args.inline_map and hasattr(ctrl, "use_map_process"):
        print("starting the map process...", flush=True)
        ctrl.use_map_process(hint)
    if args.pivot_duty is not None:
        ctrl.PIVOT_DUTY = float(args.pivot_duty)
    if hasattr(ctrl, "CNN_MIXER"):
        ctrl.CNN_MIXER = args.cnn_mixer
    if args.stop_lead is not None:
        ctrl.STOP_LEAD_S = float(args.stop_lead)
    if args.creep_duty is not None:
        ctrl.CREEP_DUTY = float(args.creep_duty)
    if args.turn_duty is not None:
        ctrl.TURN_DUTY = float(args.turn_duty)
    has_api = hasattr(ctrl, "push_direction")
    if has_api and load_plan_cache() is None:
        print("WARNING: no junction plan cache (python -m bench.junction --build-plans); junctions will stall", flush=True)
    (out / "manifest.json").write_text(json.dumps(dict(
        mode=mode, controller=args.controller, source=getattr(ctrl, "source", None), direction=args.direction,
        start_hint=hint.tolist(), args=vars(args), pivot_duty=getattr(ctrl, "PIVOT_DUTY", None),
        turn_duty=getattr(ctrl, "TURN_DUTY", None), stop_lead_s=getattr(ctrl, "STOP_LEAD_S", None),
        cnn_mixer=getattr(ctrl, "CNN_MIXER", None),
        creep_duty=getattr(ctrl, "CREEP_DUTY", None),
        stop_policy="ctrl_c / sigterm / ssh_hangup / board_button / quit command / max_seconds"), indent=2, default=float))

    camera = motors = channel = saver = None
    last_est = last_sxy = None
    script = None
    if args.script is not None:
        from .pi_script import ScriptOrchestrator, parse
        from .world import Track as _Track
        script = ScriptOrchestrator(parse(args.script, args.direction), _Track().data)
        print("script:", [{k: v for k, v in e.items() if k != "done"} for e in script.events], flush=True)
    started, count, decisions_sent = False, 0, 0
    lat_est = 0.03
    try:
        if rig is not None:
            clock = rig.clock
        else:
            from picamera2 import Picamera2
            camera = Picamera2()
            camera.configure(camera.create_video_configuration(
                main={"size": (640, 480), "format": "RGB888"}, queue=False, buffer_count=4, controls={"FrameRate": 30}))
            if args.drive:
                motors = Motors(args.probe_dir, 0.66, stop)
            camera.start()
            started = True
            time.sleep(1.0)
            origin = time.clock_gettime(time.CLOCK_BOOTTIME) - 0.1
            clock = lambda: time.clock_gettime(time.CLOCK_BOOTTIME) - origin
            if motors is not None:
                motors.clock = clock
        channel = CommandChannel(args.cmd_port, motors, clock)
        print(f"{mode.upper()} - commands on udp 127.0.0.1:{args.cmd_port}, stop with Ctrl+C / board button", flush=True)
        saver = FrameSaver()
        with (out / "steps.jsonl").open("w", buffering=1) as log:
            events = Events(args.event_port, log)
            t_start = None
            while not stop.event.is_set():
                if rig is not None:
                    image, capture = rig.capture()
                    if rig.scorer.end:
                        stop.set("sim_" + rig.scorer.end)
                        break
                else:
                    request = camera.capture_request()
                    try:
                        image = request.make_array("main")
                        meta = request.get_metadata()
                    finally:
                        request.release()
                    capture = meta["SensorTimestamp"] / 1e9 - origin
                t_start = capture if t_start is None else t_start
                if capture - t_start > args.max_seconds:
                    stop.set("max_seconds")
                    break
                upright = cv2.rotate(image, cv2.ROTATE_180)
                now = clock()
                if motors is not None and defer:               # watchdog zeroed the motors meanwhile
                    for t_trip in motors.drain_trips():
                        ctrl.on_applied(t_trip, np.zeros(2))
                        events.send(t=now, type="watchdog_zero", at=t_trip)
                incoming = channel.drain()
                if script is not None:                          # scripted signs / lights from the map pose
                    for kind, arg, reason in script.tick(now, last_est, last_sxy):
                        if kind == "stop" and motors is not None:
                            motors.hold(True)
                        incoming.append((now, dict(kind=kind, arg=arg, reason=reason, command_id=None, source="script")))
                for t_cmd, c in incoming:
                    kind, arg, reason = c["kind"], c["arg"], c["reason"]
                    ack = dict(cmd=kind, reason=reason, command_id=c["command_id"], received=t_cmd,
                               source=c.get("source", "udp"))
                    if kind == "quit":
                        stop.set("quit_command")
                    elif not has_api:
                        events.send(t=now, type="ignored", why="controller has no orchestrator API", **ack)
                    elif kind == "direction":
                        ctrl.push_direction(now, arg)
                        events.send(t=now, type="ack", move=arg, **ack)
                    elif kind == "stop":
                        if motors is not None and defer:       # the receiver already wrote zero at t_cmd
                            ctrl.on_applied(t_cmd, np.zeros(2))
                        ctrl.stop(now, reason)
                        events.send(t=now, type="ack", holds=sorted(ctrl.intr.holds), **ack)
                    elif kind in ("resume", "resume_all"):
                        if kind == "resume":
                            ctrl.resume(now, reason)
                        else:
                            ctrl.resume_all(now)
                        if motors is not None and not ctrl.intr.held:   # moves only when no hold is left
                            motors.hold(False)
                        events.send(t=now, type="ack", holds=sorted(ctrl.intr.holds), **ack)
                if stop.event.is_set():
                    break
                tic = time.perf_counter()
                done_est = (capture + 0.036) if rig is not None else (now + lat_est)
                try:
                    pwm, info = ctrl.step(upright, capture, done_est)
                except Exception as exc:                       # one bad frame is not a stop condition
                    pwm, info = np.zeros(2), {"reason": "frame_error", "error": f"{type(exc).__name__}: {exc}"}
                step_ms = (time.perf_counter() - tic) * 1e3
                if rig is not None:
                    applied = rig.write(pwm, done_est)
                    write_at = done_est
                else:
                    applied = motors.write(pwm) if motors else [0.0, 0.0]   # dry: nothing moves
                    write_at = clock()
                    lat_est = 0.8 * lat_est + 0.2 * (write_at - now)
                if defer:                                      # the motion model gets the real output
                    ctrl.on_applied(write_at, applied)
                if has_api and args.stop_after_junctions and len(ctrl.decisions) >= args.stop_after_junctions                         and not ctrl.active and ctrl.visit is None:
                    stop.set("junctions_done")
                if has_api:
                    for dec in ctrl.decisions[decisions_sent:]:
                        events.send(t=write_at, type="junction_decision", decision=dec)
                    decisions_sent = len(ctrl.decisions)
                last_est, last_sxy = info.get("est"), info.get("sxy")
                rec = dict(index=count, capture=capture, write=write_at, step_ms=step_ms,
                           sensor_to_write_ms=(write_at - capture) * 1e3, pwm=np.asarray(pwm).tolist(),
                           applied=applied, **info)
                if rig is not None:
                    rec["truth"] = rig.car.pose.tolist()
                log.write(json.dumps(rec, default=float) + "\n")
                if args.save_every and count % args.save_every == 0:
                    saver.put(out / f"{count:05d}.jpg", upright)
                if count % 15 == 0:
                    m = info.get("mission") or {}
                    print(json.dumps({"f": count, "t": round(capture, 1), "mode": info.get("reason"),
                                      "phase": info.get("phase"), "pwm": [round(v, 2) for v in np.asarray(pwm)],
                                      "w": round(float(info.get("w_map") or 0), 2), "sxy": round(float(info.get("sxy") or 0), 2),
                                      "last": m.get("last"), "queue": m.get("queue"), "held": m.get("held"),
                                      "ms": round(step_ms)}), flush=True)
                count += 1
    except Exception as exc:
        stop.set(f"error:{type(exc).__name__}:{exc}")
        import traceback
        traceback.print_exc()
    finally:
        if motors:
            motors.close()
        if getattr(ctrl, "worker", None) is not None:
            ctrl.worker.close()
        if channel:
            channel.close()
        if saver is not None:
            saver.close()
        if started:
            camera.stop()
        if camera is not None:
            camera.close()
        summary = dict(output=str(out), stop_reason=stop.reason or "loop_exit", frames=count, mode=mode,
                       script=None if script is None else dict(spec=args.script, fired=script.log,
                                                               not_fired=[{k: v for k, v in e.items() if k != "done"}
                                                                          for e in script.events if not e["done"]]),
                       direction_changes=getattr(ctrl, "direction_changes", None),
                       frames_saved=getattr(saver, "saved", 0), frames_dropped=getattr(saver, "dropped", 0),
                       decisions=getattr(ctrl, "decisions", None),
                       policy_log=getattr(getattr(ctrl, "policy", None), "log", None),
                       interrupts=getattr(getattr(ctrl, "intr", None), "log", None))
        if rig is not None:
            s = rig.scorer.summary(rig.scorer.end or "stopped")
            summary.update(sim=dict(outcome=s["outcome"], min_margin_cm=s["min_margin_cm"], passages=s["passages"],
                                    stop_lines=s["stop_lines"], orchestrator=rig.orch.log))
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
        print(json.dumps({k: summary[k] for k in ("output", "stop_reason", "frames")} |
                         ({"sim": summary["sim"]["outcome"], "passages": [f"{p['junction']}:{p['came_from']}->{p['exit_arm']}"
                                                                          for p in summary["sim"]["passages"]]}
                          if rig is not None else {}), default=float), flush=True)


if __name__ == "__main__":
    main()
