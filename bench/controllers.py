"""Thin adapters. The bench calls only:  step(image, capture_s, done_s) -> pwm.

A driver receives a rectified camera image, the capture time, the time its command
will be applied, and a start hint (the start-zone centre, same for every driver).
It never receives truth or vehicle parameters.

Each driver runs in its own process because both trees ship a package named
drive_v3 (original vs. v3-recovery).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AUTOMOBILE = HERE.parent                      # drive_v3 (original) + drive_v4
VENDOR_V3R = HERE / "_vendor" / "v3r_8b41dd9"  # v3-recovery, frozen at commit 8b41dd9
# drive_v4 frozen by `python -m bench.pin_v4` (code+config hash in the folder name)
VENDOR_V4 = HERE / "_vendor" / (HERE / "_vendor" / "V4_PIN.txt").read_text(encoding="utf-8").strip()
TRACK_JSON = HERE / "world" / "track.json"


class V4:
    name = "v4"
    source = f"drive_v4 pinned {VENDOR_V4.name}"

    def __init__(self, direction, start_hint, timing):
        sys.path.insert(0, str(VENDOR_V4))
        from drive_v4.core import Driver, load_config
        self.driver = Driver(load_config(), direction=direction, pose=np.asarray(start_hint, float), clock="bench")

    def step(self, image, capture_s, done_s):
        pwm, rec = self.driver.step(image, capture_s, done_s, rotate180=False, rectified=True)
        self.driver.applied(done_s, pwm)
        g = rec.get("turn_gain") or {}
        est = self.driver.map.pose_at(done_s)          # v4's belief (map pose): logged, never scored
        m = rec.get("map") or {}
        return np.asarray(pwm, float), dict(reason=rec["reason"], source=rec["source"], w_map=rec["w_map"],
                                            map=rec["map_status"], gain_l=g.get("left"), gain_r=g.get("right"),
                                            est=[float(v) for v in est], ms=rec["timings_ms"]["total"],
                                            k=rec.get("k"), k_map=rec.get("k_map"), k_local=rec.get("k_local"),
                                            w_corner=rec.get("w_corner"), k_ff=m.get("k_ff"), k_head=m.get("k_head"),
                                            k_lat=m.get("k_lat"), bias=m.get("bias_cm"), mix=rec.get("mix"),
                                            push=(rec.get("guard") or {}).get("push_k"))


class V3R:
    name = "v3r"
    source = "v3-recovery commit 8b41dd9 (recovery_rollout policy)"

    def __init__(self, direction, start_hint, timing):
        sys.path.insert(0, str(VENDOR_V3R))
        from drive_v3.compare_recovery import settings
        from drive_v3.map_model import TrackMap
        from drive_v3.runtime import Runtime
        cfg = settings("recovery")
        # Same timing contract as its own map_delay_experiment.experiment_settings().
        period, worst = timing["controller_period_s"], timing["max_latency_s"]
        cfg["controller"]["watchdog_s"] = period + worst + 0.1
        cfg["planner"].update(replan_period_s=period, rollout_dt_s=period, exact_rollout_horizon=True,
                              path_valid_s=period + worst + 0.1)
        track = TrackMap(TRACK_JSON)
        self.rt = Runtime(np.asarray(start_hint, float), track.outer_route(direction), cfg, clock_id="bench", track=track)

    def step(self, image, capture_s, done_s):
        rec, _, _, _ = self.rt.step(image, capture_s, done_s, rectified=True)
        est = self.rt.estimator.primary.mean            # v3r's belief: logged, never scored
        return np.asarray(rec["pwm"], float), dict(reason=rec["path_reason"], control=rec["control_reason"],
                                                   map=(rec.get("estimate") or {}).get("status"),
                                                   est=[float(v) for v in est], ms=rec.get("total_ms"))


VENDOR_CNN = HERE / "_vendor" / "tdt_models_2026-11-01"


class TeamCNN:
    """Team CNN lane policy (gawbi/track-digital-twin, Release models-2026-11-01).

    Same contract as its run_vehicle.py: the raw 640x480 camera frame (upright after
    --flip180, fisheye NOT undistorted) -> bottom half -> YUV -> 3x3 blur -> 200x66 ->
    ONNX -> [linear, angular] in [-1, 1] -> x speed_scale 0.5 -> DutyMixer from
    configs/real_car.json (min_duty 0.175, max_duty 0.66, turn_mix 0.72), signed duty.
    Its realtime loop stops on results older than 0.25 s; the bench latency is 36 ms.
    Inference uses OpenCV's ONNX reader (onnxruntime if installed gives the same graph).
    """
    raw_camera = True
    model = "lane-real-C5b"
    # The team runs --speed-scale 0.5; with its DutyMixer that puts straight-road duty
    # at ~0.27-0.31, at/below the start threshold (bench 0.30, team note: <0.33 stops).
    # BENCH_CNN_SPEED overrides it (1.0 gives ~0.36-0.40, close to v4's 0.36 cruise).
    speed_scale = float(__import__("os").environ.get("BENCH_CNN_SPEED", "0.5"))

    def __init__(self, direction, start_hint, timing):
        import json
        import cv2
        self.cv2 = cv2
        folder = VENDOR_CNN / "models" / self.model
        self.meta = json.loads((folder / "policy.json").read_text(encoding="utf-8"))
        if self.meta["preprocess"] != "rgb_bottom_half_yuv_blur3_200x66_chw_div255":
            raise ValueError("unsupported preprocessing")
        # read bytes first: OpenCV cannot open paths with non-ASCII characters (Korean user folder)
        self.net = cv2.dnn.readNetFromONNX(np.frombuffer((folder / "policy.onnx").read_bytes(), np.uint8))
        mix = json.loads((VENDOR_CNN / "configs" / "real_car.json").read_text(encoding="utf-8"))["duty_mixer"]
        self.mix = mix
        self.source = f"team CNN {self.model} (track-digital-twin models-2026-11-01), speed_scale {self.speed_scale}"

    def _preprocess(self, rgb):
        cv2 = self.cv2
        image = rgb[rgb.shape[0] // 2:]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        image = cv2.resize(image, (200, 66))
        return np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)[None] / 255.0

    def _side(self, value, trim):
        m = self.mix
        value = max(-1.0, min(1.0, value))
        if abs(value) <= m["deadband"]:
            return 0.0
        magnitude = m["min_duty"] + (m["max_duty"] - m["min_duty"]) * abs(value)
        return max(-1.0, min(1.0, float(np.copysign(magnitude, value)) * trim))

    def step(self, image, capture_s, done_s):
        rgb = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)
        self.net.setInput(self._preprocess(rgb))
        out = self.net.forward()[0]
        linear, angular = (float(np.clip(out[0], -1, 1)) * self.speed_scale,
                           float(np.clip(out[1], -1, 1)) * self.speed_scale)
        left, right = linear - self.mix["turn_mix"] * angular, linear + self.mix["turn_mix"] * angular
        scale = max(1.0, abs(left), abs(right))
        pwm = np.array([self._side(left / scale, self.mix["trim_left"]), self._side(right / scale, self.mix["trim_right"])])
        return pwm, dict(reason="drive", linear=linear, angular=angular)


def _cnn(name):
    return type(f"TeamCNN_{name}", (TeamCNN,), {"model": name})


CONTROLLERS = {"v4": V4, "v3r": V3R, "cnnC5b": _cnn("lane-real-C5b"), "cnnA5b": _cnn("lane-v5b"),
               "cnnB": _cnn("lane-real-B"), "cnnC": _cnn("lane-real-C"), "cnnA": _cnn("lane-v4b")}


class HybridCNN:
    """CNN drives; the v4 map only steps in when needed (2026-11-11, improvement of the
    cnn_map_hybrid_lab prototype).

    Findings that shaped it (that prototype's logs): its map weight w*corner was > 0 on
    almost every frame (median 0.33, > 0.5 on 40 %), so the v4 allocator and the map's
    corner timing replaced the CNN even where the CNN alone was best (outer lap 18/18
    clean vs 15/18), and the full v4 pipeline ran on every frame (~30 ms of 32 ms).

      * every frame: the CNN command, unchanged (its own duty mixer and speed)
      * the v4 map runs on every MAP_EVERY-th frame as a separate worker (on the Pi:
        another core); its result is used only MAP_LATENCY_S after that frame was taken.
        In between, its pose is predicted from the commands actually sent.
      * the map changes the command only
          - when the image is unusable (flat: white-out / black-out): drive the map's
            curvature for at most MAP_ONLY_S, then hold briefly, then stop
          - through the v4 departure guard: a small curvature push added to the CNN
            command as a wheel difference, keeping the CNN's mean duty
    """
    raw_camera = True
    MAP_EVERY = 3            # 30 fps camera -> map at 10 Hz
    CNN_MIXER = "team"       # "team": the CNN's own duty mixer; "v4": through the v4 allocator
    CNN_TOP = 0.66           # the team mixer's max duty (keep its speed range)
    MAP_LATENCY_S = 0.070    # v4 sensor->result median on the Pi (live 2026-11-10)
    FLAT_STD = 8.0           # grey-level std of the floor half below this = no picture ...
    FLAT_WHITE, FLAT_BLACK = 200.0, 35.0   # ... if also white-out / black-out (road V ~74)
    MAP_ONLY_S = 1.5         # same as v4's coast_turn_s
    HOLD_S = 0.3

    def __init__(self, direction, start_hint, timing):
        sys.path.insert(0, str(VENDOR_V4))
        from drive_v4.core import Driver, load_config
        self.cnn = CONTROLLERS["cnnA5b"](direction, start_hint, timing)
        self.cnn.speed_scale = 1.0     # as in all hyb2 results; independent of BENCH_CNN_SPEED
        self.driver = Driver(load_config(), direction=direction, pose=np.asarray(start_hint, float), clock="bench")
        self.cfg = self.driver.cfg
        self.n = 0
        self.pending = []          # (ready_s, record) of map runs not yet usable
        self.rec = None            # latest usable map record
        self.flat_since = None
        self.last_cnn = np.zeros(2)
        self.worker, self.last_submit, self.worker_ms = None, -10, None
        # On the car the motion model must integrate what the motors really got (dry run:
        # zero; orchestrator stop / watchdog: zero; clipping), not the proposal: the runner
        # sets defer_apply and calls on_applied() after each write (GPT review 2026-11-11).
        self.defer_apply = False
        # "v4": re-map the CNN's wheel command through the v4 allocator (field 2026-11-11: in
        # 54-59 % of corner frames the team mixer asked the inner wheel for less than the
        # 0.30 start duty, so it stopped and started; the allocator keeps the CNN's curvature
        # by raising both wheels or dithering the inner wheel between 0 and the floor).
        from drive_v4.core import MotorModel
        self.cnn_alloc = MotorModel(dict(self.cfg["motor"]))
        # The departure guard's clearance grid is built lazily on first use: ~0.6 s on the
        # Pi, which landed in the middle of a junction approach. Build it now.
        self.driver.track.clearance_fast(np.zeros((1, 2)))
        self.source = f"CNN A5b + v4 map on demand ({VENDOR_V4.name}), map every {self.MAP_EVERY} frames"

    def _remap_cnn(self, pwm):
        """Same curvature and mean duty as the team mixer's command, realised by the v4
        allocator: identical while the inner wheel is above the floor; below it the ratio
        is kept by raising both wheels (outer capped at CNN_TOP), or the inner wheel is
        dithered between 0 and the floor so the AVERAGE curvature matches."""
        m, a = self.driver.motor, self.cnn_alloc
        p = np.asarray(pwm, float)
        mean = float(np.mean(p))
        if mean < 0.05:                          # (near) stop: nothing to re-map
            return p
        c = m.c
        speed = p * c["cm_s_per_duty"]           # intended wheel speeds, no start dead band
        v = float(speed.mean())
        raw = float(speed[0] - speed[1]) / c["track_cm"]
        k = raw * m.yaw_gain(raw) / max(v, 1e-6)
        a.gain_left, a.gain_right, a.floor_raise = m.gain_left, m.gain_right, m.floor_raise
        out, _ = a.mix(mean, k, self.CNN_TOP)
        return np.asarray(out, float)

    def on_applied(self, t, pwm):
        """The runner reports the PWM that was actually written at time t."""
        self.driver.applied(t, np.asarray(pwm, float))

    def use_map_process(self, pose):
        """Pi: run the v4 map in a separate process (bench/map_worker.py). Every applied
        command and route change is mirrored to it."""
        from .map_worker import MapWorker
        d = self.driver
        self.worker = MapWorker(VENDOR_V4, d.direction, pose)
        orig = d.applied

        def applied(t, pwm):
            orig(t, pwm)
            self.worker.send(("applied", float(t), np.asarray(pwm, float).copy()))
        d.applied = applied

    def _map_now(self, done):
        while self.pending and self.pending[0][0] <= done + 1e-9:
            self.rec = self.pending.pop(0)[1]
        d = self.driver
        w, _ = d.map.weight()
        if self.rec is None or w <= 0:
            return None, None, w
        pose = d.map.pose_at(done)
        adv, info = d.map.advice(pose, self.cfg["steer"])
        if adv is None:
            return None, None, 0.0
        return pose, (adv, info), w

    def step(self, image, capture_s, done_s):
        import time
        t0 = time.perf_counter()
        d, cm = self.driver, self.cfg["motor"]
        pwm_cnn, cinfo = self.cnn.step(image, capture_s, done_s)
        pwm_team = np.asarray(pwm_cnn, float)
        if self.CNN_MIXER == "v4":
            pwm_cnn = self._remap_cnn(pwm_team)
        t1 = time.perf_counter()
        map_ms = 0.0
        if self.worker is not None:                           # Pi: map in its own process
            res = self.worker.poll()
            if res is not None:
                from .map_worker import restore
                restore(d, res["snap"])
                self.rec, self.worker_ms = res["rec"], res["ms"]
            if not self.worker.busy and self.n - self.last_submit >= self.MAP_EVERY:
                self.worker.submit(image, capture_s)
                self.last_submit = self.n
        elif self.n % self.MAP_EVERY == 0:                    # bench: the map worker takes this frame
            tm = time.perf_counter()
            ready = capture_s + self.MAP_LATENCY_S
            _, rec = d.step(image, capture_s, ready, rotate180=False, rectified=False)
            self.pending.append((ready, rec))
            map_ms = (time.perf_counter() - tm) * 1e3
        self.n += 1
        floor_half = image[image.shape[0] // 2:][::4, ::4]
        # flat AND white/black: a plain stretch of road (inner roads, no line in view) is
        # also low-contrast in the clean render; it must not count as "no picture"
        # (2026-11-11, junction approach from the south held the car).
        mean = float(floor_half.mean())
        flat = float(floor_half.std()) < self.FLAT_STD and (mean >= self.FLAT_WHITE or mean <= self.FLAT_BLACK)
        if flat:
            self.flat_since = capture_s if self.flat_since is None else self.flat_since
        else:
            self.flat_since = None
        pose, adv, w = self._map_now(done_s)
        mode, push = "cnn", 0.0
        out = np.asarray(pwm_cnn, float).copy()
        if flat:
            since = capture_s - self.flat_since
            if adv is not None and w >= 0.5 and since <= self.MAP_ONLY_S:
                (k_ff, k_head, corner), info = adv
                k = float(np.clip(k_ff + k_head, -self.cfg["steer"]["k_max"], self.cfg["steer"]["k_max"]))
                k, g = d._departure_guard(pose, info, k)
                out, _ = d.motor.mix(cm["cruise_duty"], k, cm["corner_pwm_max"])
                mode = "map_only"
            elif since <= self.HOLD_S:
                out, mode = self.last_cnn.copy(), "hold_last"
            else:
                out, mode = np.zeros(2), "stop_no_picture"
        else:
            self.last_cnn = out.copy()
            if adv is not None and w >= 0.5 and d.trust >= 0.8 and d.bias_time >= self.cfg["guard"]["min_bias_s"]:
                v, om = d.motor.velocity(out)
                if v > 1.0:
                    _, g = d._departure_guard(pose, adv[1], om / v)
                    push = g["push_k"]
                    if push:
                        out = self._add_curvature(out, push)
                        mode = "cnn+guard"
            out, mode = self._xcheck(out, pwm_cnn, pose, adv, w, mode)
        diag = self._diag(pwm_cnn, done_s)
        if not self.defer_apply:          # bench: the command is applied as proposed
            d.applied(done_s, out)
        est = d.map.pose_at(done_s)
        return out, dict(reason=mode, w_map=w, push=push, flat=flat, linear=cinfo.get("linear"),
                         angular=cinfo.get("angular"), cnn_pwm=np.asarray(pwm_cnn).tolist(),
                         cnn_pwm_team=pwm_team.tolist(), cnn_mixer=self.CNN_MIXER,
                         est=[float(v) for v in est], ms=(time.perf_counter() - t0) * 1e3,
                         ms_cnn_path=(time.perf_counter() - t0) * 1e3 - map_ms, ms_cnn=(t1 - t0) * 1e3, ms_map=map_ms,
                         ms_map_worker=self.worker_ms,
                         **diag)

    def _add_curvature(self, out, dk):
        """Add curvature dk as a wheel difference, keeping the mean duty (the CNN's speed)."""
        d, cm = self.driver, self.cfg["motor"]
        mean = float(np.mean(out))
        a = dk * cm["track_cm"] / (2 * d.motor.yaw_gain(dk))
        return np.clip(out + mean * a * np.array([1.0, -1.0]), 0.0, cm["pwm_max"] + 0.06)

    def _xcheck(self, out, pwm_cnn, pose, adv, w, mode):
        return out, mode                       # hyb2: no cross-check

    def _diag(self, pwm_cnn, done_s):
        """Log only (no effect on the command): CNN curvature vs the curvature the map
        would ask for at its own (possibly predicted) pose, whatever its weight."""
        d = self.driver
        v, om = d.motor.velocity(np.asarray(pwm_cnn, float))
        w, sxy = d.map.weight()
        out = dict(k_cnn=float(om / max(v, 1.0)), v_cnn=float(v), sxy=float(sxy), w_raw=float(w), map_status=d.map.status,
                   obs_age=(None if d.map.last_obs_t is None or d.map.t is None else float(d.map.t - d.map.last_obs_t)))
        if self.rec is not None:
            adv, info = d.map.advice(d.map.pose_at(done_s), self.cfg["steer"])
            if adv is not None:
                out.update(k_mapf=float(adv[0] + adv[1]), k_ff=float(adv[0]), corner=float(adv[2]),
                           arc=float(info["route_arc_cm"]), lat=float(info["lateral_cm"]))
        return out


CONTROLLERS["hyb2"] = HybridCNN


class HybridXCheck(HybridCNN):
    """hyb2 + CNN x map cross-check (plan 3, 2026-11-11).

    Design data (J_xcheck seeds 41/43, hyb2 with logging, truth used ONLY offline):
      * the map's weight w does not tell a right map from a wrong one in harsh light
        (w median 0.90 when the map was right, 0.56 when it was > 6 cm off), but its
        position spread sxy does: sxy < 0.5 cm -> right 100 %, 0.5-1 -> 84 %,
        1-2 -> 43 %, > 2 -> 15 %.
      * so the map's authority q comes from sxy, not w. It may act even when w has
        faded (dead reckoning after the last match is often still good), and it is
        ignored when w is high but sxy says the fit is loose.
      * cross-check: while CNN and map curvature agree (|dk| <= DK0) the CNN drives
        alone. When they disagree and q > 0 the command is pulled part of the way
        toward the map (wheel difference, CNN speed kept). Lateral position stays
        with the CNN: the map advice is along-track only (k_ff + k_head).
    """
    SXY_GOOD, SXY_BAD = 0.7, 1.3    # cm: q = 1 below, 0 above
    MAX_AGE_S = 5.0                 # no match for this long -> q = 0
    DK0 = 0.02                      # 1/cm: disagreement the CNN is allowed alone
    BETA = 0.6                      # share of the excess disagreement corrected
    DK_MAX = 0.06                   # 1/cm: largest correction
    TAU_S = 0.1                     # correction low-pass

    def __init__(self, direction, start_hint, timing):
        super().__init__(direction, start_hint, timing)
        self.corr, self.last_t, self.q = 0.0, None, 0.0
        self.source = self.source.replace("on demand", "on demand + sxy cross-check")

    def _quality(self):
        m = self.driver.map
        _, sxy = m.weight()
        age = 1e9 if m.last_obs_t is None or m.t is None else m.t - m.last_obs_t
        if age > self.MAX_AGE_S:
            return 0.0
        return float(np.clip((self.SXY_BAD - sxy) / (self.SXY_BAD - self.SXY_GOOD), 0.0, 1.0))

    def _map_now(self, done):
        """Same as hyb2, but the gate value returned in place of w is q (from sxy)."""
        while self.pending and self.pending[0][0] <= done + 1e-9:
            self.rec = self.pending.pop(0)[1]
        d = self.driver
        self.q = q = self._quality() if self.rec is not None else 0.0
        if q <= 0:
            return None, None, q
        pose = d.map.pose_at(done)
        adv, info = d.map.advice(pose, self.cfg["steer"])
        if adv is None:
            return None, None, 0.0
        return pose, (adv, info), q

    def _xcheck(self, out, pwm_cnn, pose, adv, q, mode):
        d = self.driver
        dt = 1 / 30 if self.last_t is None else max(1e-3, d.map.t - self.last_t if d.map.t else 1 / 30)
        self.last_t = d.map.t
        target = 0.0
        if adv is not None and q > 0:
            v, om = d.motor.velocity(np.asarray(pwm_cnn, float))
            if v > 1.0:
                (k_ff, k_head, _), _ = adv
                dk = (k_ff + k_head) - om / v
                excess = abs(dk) - self.DK0
                if excess > 0:
                    target = float(np.clip(q * self.BETA * excess * np.sign(dk), -self.DK_MAX, self.DK_MAX))
        a = np.exp(-dt / self.TAU_S)
        self.corr = a * self.corr + (1 - a) * target
        if abs(self.corr) > 1e-4:
            out = self._add_curvature(out, self.corr)
            mode = mode + "+xcheck"
        return out, mode

    def _diag(self, pwm_cnn, done_s):
        out = super()._diag(pwm_cnn, done_s)
        out.update(q=self.q, xcorr=self.corr)
        return out


CONTROLLERS["hyb3"] = HybridXCheck


# Junction movement controller (hyb2j) lives in its own module.
from .controllers_junction import HybridJunction  # noqa: E402,F401  (registers "hyb2j")
from .controllers_mission import HybridMission  # noqa: E402,F401  (registers "hyb2m")
