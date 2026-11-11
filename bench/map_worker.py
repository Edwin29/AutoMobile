"""v4 map matching in a separate process (the "other core" of hyb2), for the Pi.

Measured on the Pi 2026-11-11 (bench.pi_bench, 300 saved field frames): CNN 4.9 ms, the
v4 map step inline 56 ms median, p95 249 ms, max 415 ms. Inline, the steering command
would freeze that long. Here the worker process owns its own v4 Driver:
  main -> worker (one ordered queue): every applied command, route changes, and a frame
          whenever the worker is idle
  worker -> main: after each frame, the estimator snapshot (v3 Estimator.snapshot) and
          the MapAdvisor / Driver fields the controllers read; main copies them into its
          own Driver, whose map then predicts forward with the commands it already has.
The bench keeps the inline map (deterministic); bench.pi_run turns this on.
"""
import multiprocessing as mp
import sys
import time

import numpy as np

MAP_FIELDS = ("t", "last_obs_t", "dist_at_obs", "w_at_obs", "last_reloc", "status", "diag")
DRIVER_FIELDS = ("trust", "bias", "bias_n", "bias_time")
MOTOR_FIELDS = ("gain_left", "gain_right", "floor_raise")


def snapshot(d):
    return dict(est=d.map.est.snapshot(), map={k: getattr(d.map, k) for k in MAP_FIELDS},
                drv={k: getattr(d, k) for k in DRIVER_FIELDS}, motor={k: getattr(d.motor, k) for k in MOTOR_FIELDS})


def restore(d, s):
    d.map.est.restore(s["est"])
    for k, v in s["map"].items():
        setattr(d.map, k, v)
    for k, v in s["drv"].items():
        setattr(d, k, v)
    for k, v in s["motor"].items():
        setattr(d.motor, k, v)


def set_route(d, pts, outer):
    if pts is None:
        d.route = d.map.route = outer[0]
        d.map._curvature_at = outer[1]
        return
    from drive_v3.map_model import Route
    rt = Route("junction", np.asarray(pts, float))
    d.route = d.map.route = rt
    d.map._curvature_at = lambda s: np.interp(np.clip(s, 0, rt.s[-1]), rt.s, rt.curvature)


def _worker(vendor, direction, pose, inq, outq):
    sys.path.insert(0, vendor)
    import cv2
    cv2.setNumThreads(1)
    from drive_v4.core import Driver, load_config
    d = Driver(load_config(), direction=direction, pose=np.asarray(pose, float), clock="bench")
    outer = (d.route, d.map._curvature_at)
    outq.put(("ready", None))
    while True:
        msg = inq.get()
        kind = msg[0]
        if kind == "quit":
            return
        if kind == "applied":
            d.applied(msg[1], msg[2])
        elif kind == "signed":                      # reverse wheels: v4's motor model is forward-only
            _, t, v, om, raw, pwm = msg
            d.pwm = np.asarray(pwm, float).copy()
            d.map.command(t, v, om)
            d.raw_cmds.append((float(t), raw))
            d.body_cmds.append((float(t), v, om))
        elif kind == "route":
            set_route(d, msg[1], outer)
        elif kind == "frame":
            _, image, capture = msg
            tic = time.perf_counter()
            try:
                _, rec = d.step(image, capture, capture, rotate180=False, rectified=False)
                light = dict(reason=rec.get("reason"), map_status=rec.get("map_status"), w_map=rec.get("w_map"))
            except Exception as exc:                # one bad frame is not fatal
                light = dict(error=f"{type(exc).__name__}: {exc}")
            outq.put(("result", dict(snap=snapshot(d), rec=light, capture=capture,
                                     ms=(time.perf_counter() - tic) * 1e3)))


class MapWorker:
    def __init__(self, vendor, direction, pose):
        ctx = mp.get_context("spawn")
        self.inq, self.outq = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=_worker, args=(str(vendor), direction, list(map(float, pose)), self.inq, self.outq),
                                daemon=True)
        self.proc.start()
        kind, _ = self.outq.get(timeout=120)          # wait until its Driver is built
        self.busy = False

    def send(self, msg):
        self.inq.put(msg)

    def submit(self, image, capture):
        self.inq.put(("frame", image, float(capture)))
        self.busy = True

    def poll(self):
        """Latest finished result, or None."""
        last = None
        while True:
            try:
                kind, payload = self.outq.get_nowait()
            except Exception:
                return last
            if kind == "result":
                last = payload
                self.busy = False

    def close(self):
        try:
            self.inq.put(("quit",))
            self.proc.join(timeout=2)
        finally:
            if self.proc.is_alive():
                self.proc.terminate()
