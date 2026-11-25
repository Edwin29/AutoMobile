"""drive_v4 on the Pi. Default is DRY (camera + decisions, GPIO untouched).

    python -m drive_v4.run_pi                 # dry: no motor output
    python -m drive_v4.run_pi --drive         # real motors

The run ends ONLY on a stop interrupt:
  * Ctrl+C (SIGINT), SIGTERM, or SSH disconnect (SIGHUP)
  * any board button (pins 5/6/13/19 via probe.Motors.pressed)
There is no lap-complete or planner stop. If the road is lost the output is held
at zero while the camera keeps looking; it resumes by itself when the road returns.
A 0.5s command watchdog zeroes the motors if this loop ever hangs.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import json
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .core import Driver, load_config


class StopFlag:
    def __init__(self):
        self.reason = None
        self.event = threading.Event()

    def set(self, reason):
        if self.reason is None:
            self.reason = reason
        self.event.set()


class Motors:
    """probe.Motors + watchdog thread that also polls the board buttons."""

    def __init__(self, probe_dir, cap, stop, watchdog_s=0.5):
        import fcntl
        sys.path.insert(0, str(probe_dir))
        from probe import Motors as ProbeMotors  # verified wiring from earlier field tests
        self.lockfile = open("/tmp/autodrive-power-probe.lock", "a")
        fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.m = ProbeMotors(1000)
        self.cap, self.stop, self.watchdog_s = cap, stop, watchdog_s
        self.lock = threading.Lock()
        self.last_write = time.monotonic()
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

    def write(self, pwm):
        with self.lock:
            if self.stop.event.is_set():
                self.m.set(0.0, 0.0)
                return [0.0, 0.0]
            p = np.clip(np.asarray(pwm, float), 0.0, self.cap)
            self.m.set(float(p[0]), float(p[1]))
            self.last_write = time.monotonic()
            return p.tolist()

    def close(self):
        self.done.set()
        self.thread.join(timeout=1)
        with self.lock:
            try:
                self.m.set(0.0, 0.0)
            finally:
                self.m.close()
                self.lockfile.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drive", action="store_true", help="enable GPIO motor output")
    ap.add_argument("--direction", choices=["ccw", "cw"])
    ap.add_argument("--pose", nargs=3, type=float, help="initial x y yaw (default: right_red start region)")
    ap.add_argument("--resume", action="store_true",
                    help="start from where the latest live run ended (its final map pose)")
    ap.add_argument("--cruise", type=float, help="override cruise duty")
    ap.add_argument("--save-every", type=int, default=1, help="save every Nth raw frame (0 = none)")
    ap.add_argument("--probe-dir", default="/home/pi/straight30")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "runs"))
    args = ap.parse_args()

    stop = StopFlag()
    for sig, name in ((signal.SIGINT, "ctrl_c"), (signal.SIGTERM, "sigterm"), (signal.SIGHUP, "ssh_hangup")):
        signal.signal(sig, lambda *_, n=name: stop.set(n))
    cv2.setNumThreads(1)

    cfg = load_config()
    if args.cruise is not None:
        cfg["motor"]["cruise_duty"] = float(args.cruise)
    out = Path(args.out) / (("live_" if args.drive else "dry_") + time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    if args.resume and args.pose is None:
        previous = sorted(Path(args.out).glob("live_*/summary.json"))
        final = [json.loads(p.read_text()) for p in previous]
        final = [f for f in final if f.get("final_pose")]
        if not final:
            ap.error("--resume: no previous live run with a saved final pose")
        args.pose = final[-1]["final_pose"]
        print(f"resume from {final[-1]['output']} pose {[round(v, 1) for v in args.pose]}", flush=True)
    print("preparing map/route (motors untouched)...", flush=True)
    driver = Driver(cfg, direction=args.direction, pose=args.pose)
    (out / "manifest.json").write_text(json.dumps({
        "mode": "drive" if args.drive else "dry", "direction": driver.direction,
        "pose": driver.map.est.primary.mean.tolist(), "config": cfg,
        "stop_policy": "interrupt_only (ctrl_c / sigterm / ssh_hangup / board_button)"}, indent=2, ensure_ascii=False))

    from picamera2 import Picamera2
    camera = Picamera2()
    motors = None
    count = 0
    started = False
    try:
        camera.configure(camera.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"}, queue=False, buffer_count=4, controls={"FrameRate": 30}))
        if args.drive:
            motors = Motors(args.probe_dir, cfg["motor"]["pwm_max"], stop)
        camera.start()
        started = True
        time.sleep(1.0)
        origin = time.clock_gettime(time.CLOCK_BOOTTIME) - 0.1
        print(("DRIVE" if motors else "DRY (no GPIO)") + " - stop with Ctrl+C or a board button", flush=True)
        with (out / "steps.jsonl").open("w", buffering=1) as log:
            while not stop.event.is_set():
                request = camera.capture_request()
                try:
                    image = request.make_array("main")
                    meta = request.get_metadata()
                finally:
                    request.release()
                capture = meta["SensorTimestamp"] / 1e9 - origin
                now = time.clock_gettime(time.CLOCK_BOOTTIME) - origin
                try:
                    pwm, rec = driver.step(image, capture, now, rotate180=True)
                except Exception as exc:  # one bad frame is not a stop condition
                    pwm = np.zeros(2)
                    rec = {"reason": "frame_error", "error": f"{type(exc).__name__}: {exc}", "source": "none",
                           "w_map": 0.0, "corner": 0.0, "k": 0.0, "pwm": [0.0, 0.0], "map_status": "error",
                           "timings_ms": {"total": 0.0}}
                if stop.event.is_set():
                    break
                applied = motors.write(pwm) if motors else [0.0, 0.0]
                write_at = time.clock_gettime(time.CLOCK_BOOTTIME) - origin
                # Dry runs feed the proposal to the model so decisions evolve as if driving
                # would (the pose then is a prediction, not where the car is).
                driver.applied(write_at, applied if motors else pwm)
                rec.update(index=count, applied_pwm=applied, write_s=write_at,
                           sensor_to_write_ms=(write_at - capture) * 1e3)
                log.write(json.dumps(rec, default=float) + "\n")
                if args.save_every and count % args.save_every == 0:
                    cv2.imwrite(str(out / f"{count:04d}_raw.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if count % 10 == 0:
                    print(json.dumps({"f": count, "reason": rec["reason"], "src": rec["source"],
                                      "w": round(rec["w_map"], 2), "corner": round(rec["corner"], 2),
                                      "k": round(rec["k"], 3), "pwm": [round(v, 2) for v in rec["pwm"]],
                                      "map": rec["map_status"], "ms": round(rec["timings_ms"]["total"])}), flush=True)
                count += 1
    except Exception as exc:  # report, but motors still go to zero below
        stop.set(f"error:{type(exc).__name__}:{exc}")
    finally:
        if motors:
            motors.close()
        if started:
            camera.stop()
        camera.close()
        final_pose = driver.map.pose_at(driver.last_now) if driver.last_now is not None else None
        summary = {"output": str(out), "stop_reason": stop.reason or "loop_exit", "frames": count,
                   "drive": bool(args.drive), "final_output": [0, 0],
                   "final_pose": None if final_pose is None else [float(v) for v in final_pose],
                   "final_map_weight": driver.map.weight()[0], "final_map_trust": driver.trust}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
