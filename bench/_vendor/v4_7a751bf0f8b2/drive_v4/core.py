"""drive_v4 core: the local lane follower always drives; the map only advises.

No GPIO and no camera here, so this module is safe to import in tests.
Conventions follow drive_v3: body x right / y forward (cm), world x east / y north,
yaw clockwise from north, pose origin = axle midpoint.

Only the caller's stop interrupt ends a run. Inside this module nothing "fails":
missing evidence lowers the map weight, and a fully lost road only zeroes the
output while observation continues.
"""
import json
import math
import time
from pathlib import Path

import numpy as np

from drive_v3.camera import CameraGeometry
from drive_v3.contracts import FrameStamp, PathDecision
from drive_v3.estimation import Belief, Estimator
from drive_v3.geometry import advance, footprint, path_geometry, to_body, to_world, wrap
from drive_v3.local_corridor import LocalCorridorPlanner
from drive_v3.map_model import HERE as V3, TrackMap
from drive_v3.observation import Observer
from drive_v3.settings import load_settings
from drive_v3.start_region import right_red_prior

from .route import with_lane_offsets

HERE = Path(__file__).resolve().parent
# Road geometry fixed to photo measurements (see build_map.py). v3 map is untouched.
MAP_PATH = HERE / "config/map_v4.json"


def load_config(path=HERE / "config.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------
# Motors
# ----------------------------------------------------------------------------
class MotorModel:
    """PWM <-> body motion with an explicit start deadband.

    Field 2026-11-10: all four wheels start at 0.30. v3 assumed motion from 0.12,
    so the estimator believed the car moved while it stood still.
    """

    def __init__(self, c):
        self.c = c
        self.dither = 0.0  # sigma-delta accumulator for tight-turn inner wheel
        # Turn efficiency per direction (v2 "output compensation" mix, 1/output gain).
        # Left = omega < 0 (counter-clockwise), right = omega > 0.
        self.gain_left = float(c["yaw_gain_left"])
        self.gain_right = float(c["yaw_gain_right"])

    def yaw_gain(self, signed):
        return self.gain_left if signed < 0 else self.gain_right

    def raw_omega(self, pwm):
        """Ideal differential-drive yaw rate of the applied PWM (gain 1, deadband applied)."""
        p = np.asarray(pwm, float)
        speed = np.where(p >= self.c["start_duty"], p, 0.0) * self.c["cm_s_per_duty"]
        return float((speed[0] - speed[1]) / self.c["track_cm"])

    def velocity(self, pwm):
        p = np.asarray(pwm, float)
        effective = np.where(p >= self.c["start_duty"], p, 0.0)
        speed = effective * self.c["cm_s_per_duty"]
        raw = (speed[0] - speed[1]) / self.c["track_cm"]
        return float(speed.mean()), float(raw * self.yaw_gain(raw))

    def curvature_limits(self, top=None, signed=-1.0):
        floor, T, g = self.c["pwm_floor"], self.c["track_cm"], self.yaw_gain(signed)
        top = self.c["pwm_max"] if top is None else top
        k_floor = 2 * g / T * (top - floor) / (top + floor)   # outer=top, inner=floor
        k_zero = 2 * g / T                                    # outer=top, inner=0
        return k_floor, k_zero

    def mix(self, mean_duty, k, top=None):
        """Wheel duties for curvature k (1/cm, + = right). Returns (pwm, note).

        The turn efficiency for that direction (field: ~0.6, skid steer) means a turn
        needs 1/gain times the wheel-speed difference of an ideal differential drive."""
        floor, T, g = self.c["pwm_floor"], self.c["track_cm"], self.yaw_gain(k)
        top = self.c["pwm_max"] if top is None else top
        a = float(np.clip(k * T / (2 * g), -0.999, 0.999))
        pwm = mean_duty * np.array([1 + a, 1 - a])
        low = pwm.min()
        if low < floor:                       # raise both, keep the ratio (= keep curvature)
            pwm *= floor / max(low, 1e-9)
        if pwm.max() > top:                   # then cap the outer wheel, still keeping the ratio
            pwm *= top / pwm.max()
        if pwm.min() >= floor - 1e-9:
            self.dither = 0.0
            return pwm, "ratio"
        # Tight turn: floor..max cannot reach k. Alternate inner between floor and 0
        # so the time-averaged curvature matches; never leave it inside the deadband.
        k_floor, k_zero = self.curvature_limits(top, k)
        fraction = float(np.clip((abs(k) - k_floor) / (k_zero - k_floor), 0, 1))
        self.dither += fraction
        inner_zero = self.dither >= 0.5
        if inner_zero:
            self.dither -= 1.0
        inner = 0.0 if inner_zero else floor
        out = np.array([top, inner]) if k > 0 else np.array([inner, top])
        return out, "dither_zero" if inner_zero else "dither_floor"


class TurnGainLearner:
    """Online turn-efficiency learning, ported from v2 feedback_driver.Feedback.

    Compares the map-observed yaw change with the ideal (gain 1) yaw of the applied
    PWM over windows of one turn direction, and moves that direction's gain toward
    the ratio. Guards kept from v2: never mix left/right evidence, need enough turn
    in the window, reject inconsistent windows, bounded and rate-limited updates.
    Evidence in v4 = consecutive map matches (not relocalisations) with high trust.
    """

    def __init__(self, motor, c):
        self.motor, self.c = motor, c
        self.window = []          # (raw_deg, measured_deg, dt)
        self.prev = None          # (t, yaw) of the last usable map match
        self.events = []

    def reset(self):
        self.window.clear()
        self.prev = None

    def observe(self, t, yaw, raw_deg, usable):
        """raw_deg: ideal yaw of applied commands since the previous call's time."""
        event = None
        if not self.c["enabled"] or not usable:
            self.reset()
            return None
        if self.prev is not None:
            dt = t - self.prev[0]
            measured = math.degrees(wrap(yaw - self.prev[1]))
            if dt <= 0 or dt > 0.5:
                self.window.clear()
            elif abs(raw_deg) < self.c["min_step_deg"]:
                self.window.clear()                      # straight: no turn evidence
            else:
                if self.window and self.window[-1][0] * raw_deg <= 0:
                    self.window.clear()                  # direction changed
                self.window.append((raw_deg, measured, dt))
                event = self._try_update()
        self.prev = (t, yaw)
        return event

    def _try_update(self):
        c = self.c
        values = np.asarray(self.window)
        raw, measured, elapsed = values.sum(axis=0)
        if abs(raw) < c["min_raw_deg"] or len(values) < c["min_samples"]:
            if elapsed > c["max_elapsed_s"]:
                self.window.clear()
            return None
        self.window.clear()
        ratio = measured / raw
        fit_error = float(np.max(np.abs(np.cumsum(values[:, 1] - ratio * values[:, 0]))))
        lo, hi = c["ratio_range"]
        if not lo < ratio < hi or fit_error > c["max_fit_err_deg"] or elapsed > c["max_elapsed_s"]:
            return {"rejected": True, "ratio": float(ratio), "fit_error_deg": fit_error}
        left = raw < 0
        old = self.motor.gain_left if left else self.motor.gain_right
        b0, b1 = c["bounds"]
        new = float(np.clip(old + c["rate"] * (ratio - old), max(b0, old - c["max_step"]), min(b1, old + c["max_step"])))
        if left:
            self.motor.gain_left = new
        else:
            self.motor.gain_right = new
        event = {"side": "left" if left else "right", "raw_deg": float(raw), "observed_deg": float(measured),
                 "ratio": float(ratio), "before": old, "after": new}
        self.events.append(event)
        return event


# ----------------------------------------------------------------------------
# Map advisor
# ----------------------------------------------------------------------------
class MotionScaledEstimator(Estimator):
    """v3 estimator, but process noise grows with motion, not with waiting.

    In v3 a parked car's covariance kept growing (motion_sigma_per_s * dt), so the
    longer it waited the less feasible every plan became.
    """

    def predict(self, v, omega, dt):
        if not np.isfinite([v, omega, dt]).all() or not 0 <= dt <= 2.0:
            raise ValueError("Invalid prediction interval")
        c = self.cfg
        for h in self.hypotheses:
            yaw = h.mean[2] + omega * dt / 2
            F = np.eye(3)
            F[:2, 2] = v * dt * np.sinc(omega * dt / (2 * np.pi)) * np.array([np.cos(yaw), -np.sin(yaw)])
            q = (np.square(c["v4_sigma_per_cm"]) * abs(v) * dt
                 + np.square(c["v4_sigma_per_rad"]) * abs(omega) * dt
                 + np.square(c["v4_sigma_idle_per_s"]) * dt)
            h.mean = advance(h.mean, v, omega, dt)
            h.P = F @ h.P @ F.T + np.diag(q)
        self.motion_distance += abs(v) * dt


class MapAdvisor:
    """Pose from map matching + a confidence weight w in [0, 1].

    The estimator lives at capture time; commands are replayed piecewise so the
    pose can be projected to 'now' for steering. Failing to match never stops
    the car; it only fades w.
    """

    def __init__(self, track, route, est_cfg, pose, sigma, c):
        cfg = dict(est_cfg, initial_sigma=list(sigma), v4_sigma_per_cm=c["sigma_per_cm"],
                   v4_sigma_per_rad=c["sigma_per_rad"], v4_sigma_idle_per_s=c["sigma_idle_per_s"])
        self.est = MotionScaledEstimator(track, pose, cfg)
        self.track, self.route, self.c = track, route, c
        self.t = None
        self.commands = [(-1e9, 0.0, 0.0)]
        self.last_obs_t = None
        self.dist_at_obs = 0.0
        self.w_at_obs = c["prior_weight"]  # start prior is a placement region, not a match
        self.last_reloc = -1e9
        self.status = "prior"
        self.diag = {}

    # --- command history -------------------------------------------------
    def command(self, t, v, omega):
        self.commands.append((float(t), float(v), float(omega)))
        if self.t is not None:  # keep one command at or before the estimator time
            keep = max(i for i, cmd in enumerate(self.commands) if cmd[0] <= self.t)
            self.commands = self.commands[keep:]

    def _command_at(self, t):
        cmd = self.commands[0]
        for c in self.commands:
            if c[0] <= t:
                cmd = c
        return cmd[1], cmd[2]

    def _segments(self, a, b):
        cuts = [a] + [c[0] for c in self.commands if a < c[0] < b] + [b]
        for x, y in zip(cuts, cuts[1:]):
            v, w = self._command_at(x)
            yield x, y, v, w

    def _predict_to(self, t):
        if self.t is None:
            self.t = t
            return
        if t <= self.t:
            return
        for x, y, v, w in self._segments(self.t, t):
            while y - x > 1.0:
                self.est.predict(v, w, 1.0)
                x += 1.0
            if y > x:
                self.est.predict(v, w, y - x)
        self.t = t

    # --- observation -----------------------------------------------------
    def observe(self, observation, capture_s, on_lane=True):
        """on_lane: the camera follower sees a lane right now. Route relocalization
        is only accepted then; otherwise clutter (a desk, a person) can fit the map."""
        self._predict_to(capture_s)
        matched = self.est.observe(observation)
        self.diag = dict(self.est.last_diagnostics)
        if not matched and on_lane:
            since = capture_s - (self.last_obs_t if self.last_obs_t is not None else -1e9)
            if since >= self.c["reloc_after_s"] and capture_s - self.last_reloc >= self.c["reloc_period_s"]:
                self.last_reloc = capture_s
                matched = self._relocalize(observation)
        if matched:
            self.last_obs_t = capture_s
            self.dist_at_obs = self.est.motion_distance
            self.w_at_obs = 1.0
            if self.status != "relocalized":
                self.status = "observed"
        else:
            self.status = "prediction_only"
        return matched

    def _relocalize(self, observation):
        """Route-constrained search: the car is on the road, roughly along the route."""
        prior = self.est.primary
        s0 = self.route.s[self.route.nearest(prior.mean[:2])]
        best = None
        for ds in self.c["reloc_arcs_cm"]:
            xy = self.route.sample(s0 + ds)[0]
            yaw = float(self.route.heading[self.route.nearest(xy)])
            seed = np.array([xy[0], xy[1], yaw])
            seeded = Belief(seed.copy(), np.diag([3.0 ** 2, 3.0 ** 2, 0.08 ** 2]), associations={})
            try:
                fitted = self.est._fit(observation, seeded, seed.copy())
            except Exception:
                fitted = None
            if fitted is None:
                continue
            belief, diag = fitted
            if diag["rms_cm"] > self.c["reloc_max_rms_cm"]:
                continue
            score = belief.score + 0.01 * abs(ds)  # prefer the predicted progress
            if best is None or score < best[0]:
                best = (score, belief, diag, ds)
        if best is None:
            return False
        _, belief, diag, ds = best
        self.est.hypotheses = [belief]
        self.est.last_diagnostics = {"status": "observed", **diag, "relocalized_route_ds_cm": ds}
        self.diag = dict(self.est.last_diagnostics)
        self.status = "relocalized"
        return True

    # --- advice ----------------------------------------------------------
    def pose_at(self, now):
        pose = self.est.primary.mean.copy()
        if self.t is not None and now > self.t:
            for x, y, v, w in self._segments(self.t, now):
                pose = advance(pose, v, w, y - x)
        return pose

    def weight(self):
        c = self.c
        travelled = self.est.motion_distance - self.dist_at_obs
        w = self.w_at_obs * max(0.0, 1 - travelled / c["fade_cm"])
        if self.last_obs_t is not None and self.t is not None:
            w *= max(0.0, 1 - (self.t - self.last_obs_t) / c["fade_s"])
        sxy = math.sqrt(max(np.diag(self.est.primary.P)[:2]))
        w *= float(np.clip((c["sigma_zero_cm"] - sxy) / (c["sigma_zero_cm"] - c["sigma_full_cm"]), 0, 1))
        return float(w), sxy

    def _curvature_at(self, arc):
        r = self.route
        return np.interp(np.asarray(arc) % r.s[-1], r.s, r.curvature)

    def advice(self, pose, sc):
        """Map advice that does NOT use the map's lateral position.

        Field replay (live_20261110_182520): on a straight the map pose sat 4-5cm
        sideways of reality, so pursuit onto the route asked for R~17cm turns.
        The map is used only for *where along the track* we are:
          k_ff    route curvature at the axle (+ small preview): turn when the AXLE
                  reaches the corner, not when the camera first sees it
          k_head  heading alignment to the route tangent
          corner  0..1, how much of the camera view ahead is corner
        Lateral position is left to the camera follower.
        """
        r = self.route
        i = r.nearest(pose[:2])
        offset = float(np.linalg.norm(r.points[i] - pose[:2]))
        if offset > self.c["max_route_offset_cm"]:
            return None, {"route_offset_cm": offset, "map_reject": "far_from_route"}
        s = float(r.s[i])
        # Turn-in: average route curvature over a window ahead of the axle, so the
        # turn starts earlier and ramps instead of stepping (field: turn-in was late,
        # then sharp). Window covers actuation delay + slow yaw build-up.
        win = s + sc["preview_start_cm"] + np.arange(0, sc["preview_len_cm"] + 1e-9, 1.0)
        k_ff = float(np.mean(self._curvature_at(win)))
        psi = float(wrap(r.heading[i] - pose[2]))
        k_head = 2 * math.sin(psi) / sc["heading_lookahead_cm"]
        ahead = self._curvature_at(s + np.arange(0, sc["corner_window_cm"], 1.0))
        corner = float(np.clip(np.max(np.abs(ahead)) / sc["corner_k_full"], 0, 1))
        normal = np.array([math.cos(r.heading[i]), -math.sin(r.heading[i])])  # route right
        lateral = float((pose[:2] - r.points[i]) @ normal)                      # + = right of route
        lane = float(getattr(r, "lane_offset", np.zeros(len(r.points)))[i])     # lane centre vs route
        return (k_ff, k_head, corner), {"route_offset_cm": offset, "route_arc_cm": s, "route_index": i,
                                        "k_ff": k_ff, "k_head": k_head, "heading_err_rad": psi, "corner": corner,
                                        "lateral_cm": lateral, "lane_centre_cm": lane,
                                        "normal": normal.tolist()}


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def pursuit_on_path(points, lookahead):
    """Curvature to the point at arc length `lookahead` along a body-frame path."""
    p = np.asarray(points, float)
    if len(p) < 2:
        return None
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if s[-1] < 2.0:
        return None
    a = min(lookahead, s[-1])
    q = np.array([np.interp(a, s, p[:, 0]), np.interp(a, s, p[:, 1])])
    if q[1] <= 1.0:
        return None
    return float(2 * q[0] / float(q @ q))


def camera_lane_offset(points):
    """Car offset from the camera-observed lane centre (cm, + = car right of centre).

    LocalCorridorPlanner paths are 12 synthetic join points from the car followed by
    the observed corridor centre; extrapolate the observed centre line back to y=0.
    Lane-centre position is insensitive to a symmetric camera scale error."""
    p = np.asarray(points, float)[12:]
    if len(p) < 4:
        return None
    q = p[:8]
    b, a = np.polyfit(q[:, 1], q[:, 0], 1)
    return float(-a)


class Driver:
    def __init__(self, cfg=None, direction=None, pose=None, sigma=None, clock="boottime_relative"):
        self.cfg = cfg or load_config()
        self.direction = direction or self.cfg["direction"]
        v3, _ = load_settings()
        self.v3 = v3
        self.camera = CameraGeometry(V3 / "config/camera_legacy_track.json")
        self.observer = Observer(self.camera, V3 / "config/observation.json")
        self.observer.mark_cfg = v3["marks"]
        self.track = TrackMap(MAP_PATH)
        self.route = with_lane_offsets(self.track, self.track.outer_route(self.direction))
        self.rim = footprint(v3["vehicle"])
        self.bias = 0.0          # map lateral - camera lateral, learned on straights (cm)
        self.bias_n = 0
        if pose is None:
            pose, prior_sigma, _ = right_red_prior(self.track, v3["vehicle"], self.direction)
            sigma = sigma or prior_sigma
        sigma = sigma or [2.0, 2.0, math.radians(8)]
        self.map = MapAdvisor(self.track, self.route, v3["estimator"], np.asarray(pose, float), sigma, self.cfg["map"])
        self.local = LocalCorridorPlanner(self.camera, self.direction)
        self.motor = MotorModel(self.cfg["motor"])
        self.learner = TurnGainLearner(self.motor, self.cfg["yaw_learn"])
        self.raw_cmds = [(-1e9, 0.0)]  # (time, ideal yaw rate of applied PWM)
        self.body_cmds = [(-1e9, 0.0, 0.0)]  # (time, v, omega) predicted from applied PWM
        self.learn_event = None
        self.clock = clock
        self.sequence = 0
        self.pwm = np.zeros(2)
        self.k = 0.0
        self.last_seen = None
        self.last_lane = None
        self.lane_streak = 0
        self.trust = 1.0
        self.last_now = None

    def _departure_guard(self, pose, info, k_cmd):
        """Soft lane-keeping: predict the footprint ~0.4s ahead on the commanded arc
        (bias-corrected pose) and steer away from a boundary that gets close.
        Never stops the car; only adds curvature."""
        g = self.cfg["guard"]
        corrected = pose.copy()
        corrected[:2] -= self.bias * np.asarray(info["normal"])
        v, _ = self.motor.velocity(self.motor.mix(self.cfg["motor"]["cruise_duty"], k_cmd)[0])
        v = max(v, 3.0)
        left, right = self.rim[:, 0] < 0, self.rim[:, 0] > 0
        cl0 = self.track.clearance_fast(to_world(self.rim, corrected))
        now_worst = min(float(cl0[left].min()), float(cl0[right].min()))
        worst = (np.inf, np.inf)
        for t in np.arange(g["step_s"], g["horizon_s"] + 1e-9, g["step_s"]):
            p = advance(corrected, v, v * k_cmd, t)
            cl = self.track.clearance_fast(to_world(self.rim, p))
            worst = (min(worst[0], float(cl[left].min())), min(worst[1], float(cl[right].min())))
        push = 0.0
        # Act only when the commanded arc brings a boundary CLOSER (not merely near),
        # and never by more than max_push: the guard nudges, it does not steer.
        if min(worst) < g["clearance_cm"] and min(worst) < now_worst - g["approach_cm"]:
            depth = g["clearance_cm"] - min(worst)
            push = g["gain"] * depth * (1.0 if worst[0] < worst[1] else -1.0)  # left close -> turn right
            push = float(np.clip(push, -g["max_push"], g["max_push"]))
        return k_cmd + push, {"left_cm": worst[0], "right_cm": worst[1], "now_cm": now_worst, "push_k": push}

    def applied(self, t, pwm):
        """Call after the motors actually received `pwm` (also with zeros)."""
        self.pwm = np.asarray(pwm, float).copy()
        self.map.command(t, *self.motor.velocity(self.pwm))
        self.raw_cmds.append((float(t), self.motor.raw_omega(self.pwm)))
        self.body_cmds.append((float(t), *self.motor.velocity(self.pwm)))
        if len(self.body_cmds) > 200:
            self.body_cmds = self.body_cmds[-100:]
        if len(self.raw_cmds) > 200:
            self.raw_cmds = self.raw_cmds[-100:]

    def _motion_between(self, a, b):
        """Body-frame pose change from time a to b, integrating the applied-command history."""
        pose, cmds = np.zeros(3), self.body_cmds
        for i, (t, v, w) in enumerate(cmds):
            end = cmds[i + 1][0] if i + 1 < len(cmds) else b
            lo, hi = max(a, t), min(b, end)
            if hi > lo:
                pose = advance(pose, v, w, hi - lo)
        return pose

    def _local_now(self, debug, capture_s, now):
        """Camera lane at capture time, moved into the current body frame.

        drive_v3's LocalCorridorPlanner drops frames older than 0.4s and compensates
        with the last command only. Here the corridor is fitted in the capture frame
        and moved with the full applied-command history, so a slower loop (e.g.
        0.5s processing) still gets a lane instead of 'local_frame_stale'."""
        d = self.local.plan_local(debug, capture_s, capture_s, self.clock, 0.0, 0.0)
        if d.reason != "tracking":
            return d
        age = now - capture_s
        if age > self.cfg["lost"]["max_frame_age_s"]:
            return self.local.stop(now, self.clock, "local_frame_stale")
        observed = to_body(np.asarray(d.points_body_cm, float)[12:], self._motion_between(capture_s, now))
        observed = observed[observed[:, 1] > 1.0]
        if len(observed) < 4:
            return self.local.stop(now, self.clock, "local_frame_passed")
        q = observed[0]
        t = np.linspace(0.0, 1.0, 12, endpoint=False)
        join = np.c_[q[0] * (3 * t * t - 2 * t * t * t), q[1] * t]
        path = np.vstack((join, observed))
        start_arc = float(np.linalg.norm(np.diff(join, axis=0), axis=1).sum() + np.linalg.norm(q - join[-1]))
        return PathDecision(path.tolist(), d.speed_limit_cm_s, now + 0.4, self.clock, "tracking",
                            lookahead_cm=start_arc + 5.0)

    def _raw_yaw(self, a, b):
        """Integral of the ideal (gain 1) yaw rate of applied commands over [a, b]."""
        total, cmds = 0.0, self.raw_cmds
        for i, (t, w) in enumerate(cmds):
            end = cmds[i + 1][0] if i + 1 < len(cmds) else b
            lo, hi = max(a, t), min(b, end)
            if hi > lo:
                total += w * (hi - lo)
        return total

    def step(self, image, capture_s, now, rotate180=True, rectified=False):
        t0 = time.perf_counter()
        c, cm = self.cfg, self.cfg["motor"]
        stamp = FrameStamp(self.sequence, capture_s, self.clock)
        self.sequence += 1
        observation, debug = self.observer.process(image, stamp, rectified=rectified, rotate180=rotate180)
        t1 = time.perf_counter()

        v, omega = self.motor.velocity(self.pwm)
        local = self._local_now(debug, capture_s, now)
        self.last_local = local  # for visualisation
        k_local = None
        if local.reason == "tracking":
            look = max(c["steer"]["local_lookahead_cm"], local.lookahead_cm or 0.0)
            k_local = pursuit_on_path(local.points_body_cm, look)
        # A lane must persist a few frames before it counts (clutter off-track can
        # look like a lane for one frame). While driving, short gaps need no re-confirm.
        self.lane_streak = self.lane_streak + 1 if k_local is not None else 0
        recently_on_lane = self.last_lane is not None and now - self.last_lane <= c["lost"]["coast_s"]
        if k_local is not None and not (self.lane_streak >= c["lost"]["lane_confirm_frames"] or recently_on_lane):
            k_local = None
        if k_local is not None:
            self.last_lane = now

        t2 = time.perf_counter()
        matched = self.map.observe(observation, capture_s, on_lane=k_local is not None)
        # Turn-efficiency learning (v2 output feedback): only consecutive ordinary
        # matches with a trusted map; a relocalisation jump is not a yaw measurement.
        usable = (matched and self.map.status == "observed" and self.trust >= 0.8
                  and self.map.weight()[0] >= 0.5)
        prev_t = self.learner.prev[0] if self.learner.prev is not None else None
        raw_deg = math.degrees(self._raw_yaw(prev_t, capture_s)) if prev_t is not None else 0.0
        self.learn_event = self.learner.observe(capture_s, float(self.map.est.primary.mean[2]), raw_deg, usable)
        t3 = time.perf_counter()

        pose = self.map.pose_at(now)
        w, sxy = self.map.weight()
        k_map, corner, map_info = None, 0.0, {}
        if w > 0:
            adv, map_info = self.map.advice(pose, c["steer"])
            if adv is None:
                w = 0.0
            else:
                k_ff, k_head, corner = adv
                k_map = k_ff + k_head

        # ---- error absorption: learn (map lateral - camera lateral) on straights --
        # The camera sees the car's offset from the real lane centre; the map pose
        # gives an offset from the map's lane centre. Their difference is a bias
        # (camera projection + map geometry, field: 1-5cm) that varies slowly, so
        # learn it where both are reliable and remove it where the camera is weak.
        sc = c["steer"]
        cam_lat = camera_lane_offset(local.points_body_cm) if k_local is not None else None
        if k_map is not None:
            map_lat = map_info["lateral_cm"] - map_info["lane_centre_cm"]   # car vs lane centre, map view
            # Learn only on a clear straight: during a turn (or its exit) the camera's
            # lane-centre extrapolation and the map heading are both off (sim: bias
            # swung 2.2 -> -0.7cm in 1.5s at a corner exit and misled the guard).
            if (cam_lat is not None and corner < sc["bias_learn_max_corner"] and w >= 0.5 and self.trust > 0.8
                    and abs(map_info["heading_err_rad"]) < sc["bias_learn_max_heading_rad"]
                    and abs(k_local) < sc["bias_learn_max_k"] and abs(map_info["k_ff"]) < sc["bias_learn_max_k"]):
                a = sc["bias_alpha"] if self.bias_n > 10 else 1.0 / (self.bias_n + 1)
                self.bias = float(np.clip((1 - a) * self.bias + a * (map_lat - cam_lat),
                                          -sc["bias_max_cm"], sc["bias_max_cm"]))
                self.bias_n += 1
            lat_corrected = map_info["lateral_cm"] - self.bias              # vs route, bias removed
            k_lat = -2 * lat_corrected / sc["lat_lookahead_cm"] ** 2 if self.bias_n >= 5 else 0.0
            k_map = k_map + k_lat
            map_info.update(cam_lat_cm=cam_lat, map_lat_cm=map_lat, bias_cm=self.bias, k_lat=k_lat)

        # ---- map trust: camera vs map disagreement ---------------------------
        # Field live_20261110_200105: started at the 2nd corner while the prior
        # said "start zone"; the route search matched a look-alike corner and the
        # map then kept 'matching' the wrong straight. At the next corner the
        # camera said left (-0.044) and the map said right (+0.052). Opposite
        # turn directions are evidence the map is somewhere else.
        mc = c["map"]
        if k_map is not None and k_local is not None and max(abs(k_map), abs(k_local)) > mc["disagree_min_k"]:
            if k_map * k_local < 0 and abs(k_map - k_local) > mc["disagree_k"]:
                self.trust *= mc["disagree_decay"]
            else:
                self.trust = min(1.0, self.trust + mc["trust_recover"])
        else:
            self.trust = min(1.0, self.trust + mc["trust_recover"] / 3)
        if k_map is not None:
            w *= self.trust

        # ---- fuse -----------------------------------------------------------
        # Straights: the camera follower drives (it sees the real lane centre).
        # Near corners the follower turns too early (camera ~12cm ahead of the
        # axle + pursuit chord = the v1 inside cut), so the map's corner timing
        # takes over in proportion to w * corner.
        wc = w * corner
        if k_map is not None and k_local is not None:
            k_cmd, source = wc * k_map + (1 - wc) * k_local, "fused"
        elif k_local is not None:
            k_cmd, source = k_local, "local"
        elif (k_map is not None and self.map.last_obs_t is not None and w >= c["map"]["map_only_min_w"]
              and self.last_lane is not None and now - self.last_lane <= c["map"]["map_only_after_lane_s"]):
            # Camera lost the lane (typical mid-corner): drive on a map that has
            # actually matched, and only shortly after a real lane was seen. The
            # start prior alone, or a car lifted off the track, never drives blind.
            k_cmd, source = k_map, "map"
        else:
            k_cmd, source = None, "none"

        guard = None
        if (k_cmd is not None and k_map is not None and w >= 0.5 and self.trust >= 0.8
                and self.bias_n >= c["guard"]["min_bias_samples"]):
            k_cmd, guard = self._departure_guard(pose, map_info, k_cmd)

        if k_cmd is not None:
            self.last_seen = now
            k_cmd = float(np.clip(k_cmd, -c["steer"]["k_max"], c["steer"]["k_max"]))
            a = c["steer"]["k_smooth"]
            self.k = a * self.k + (1 - a) * k_cmd
            reason = "drive"
        elif self.last_seen is not None and now - self.last_seen <= (
                c["lost"]["coast_turn_s"] if abs(self.k) > c["lost"]["turning_k"] else c["lost"]["coast_s"]):
            # Brief gap: keep the last curvature. Mid-corner the camera often loses
            # the lane, so an ongoing turn is held a little longer than a straight.
            reason = "coast"
        else:
            reason = "road_lost_hold"  # zero output, keep observing; not an exit

        if reason == "road_lost_hold":
            target, mix_note = np.zeros(2), "zero"
            self.motor.dither = 0.0
        else:
            # Corners: lower the outer-wheel ceiling (slower, and the inner wheel
            # dithers rather than the car speeding up to keep it above the floor).
            top = cm["corner_pwm_max"] if wc > 0.5 else cm["pwm_max"]
            target, mix_note = self.motor.mix(cm["cruise_duty"], self.k, top)

        # Slew: limit changes, but jump straight over the deadband when starting.
        dt = 0.1 if self.last_now is None else float(np.clip(now - self.last_now, 0.0, 0.5))
        self.last_now = now
        step_limit = cm["slew_per_s"] * max(dt, 1e-3)
        out = target.copy()
        for i in range(2):
            if target[i] == 0.0 or self.pwm[i] < cm["pwm_floor"] - 1e-9 or mix_note.startswith("dither"):
                continue  # stop instantly, start at the floor, dither toggles freely
            out[i] = self.pwm[i] + np.clip(target[i] - self.pwm[i], -step_limit, step_limit)
            out[i] = max(out[i], cm["pwm_floor"])
        out = np.clip(out, 0.0, cm["pwm_max"])

        record = {
            "seq": stamp.sequence, "capture_s": capture_s, "now_s": now,
            "reason": reason, "source": source, "w_map": w, "map_trust": self.trust, "corner": corner, "w_corner": wc, "k_map": k_map, "k_local": k_local,
            "k": self.k, "mix": mix_note, "pwm": out.tolist(),
            "turn_gain": {"left": self.motor.gain_left, "right": self.motor.gain_right, "event": self.learn_event},
            "map_status": self.map.status, "matched": bool(matched), "sigma_xy_cm": sxy,
            "pose_now": pose.tolist(), "estimate": {k: self.map.diag.get(k) for k in
                                                  ("status", "rms_cm", "inlier_fraction", "relocalized_route_ds_cm")},
            "map": map_info, "guard": guard, "local_reason": local.reason,
            "local": {k: self.local.last_diagnostics.get(k) for k in ("paired_rows", "width_inferred_rows", "fit_rms_cm", "support_cm")},
            "fragments": len(observation.fragments),
            "timings_ms": {"observe": (t1 - t0) * 1e3, "local": (t2 - t1) * 1e3, "map": (t3 - t2) * 1e3,
                           "total": (time.perf_counter() - t0) * 1e3},
        }
        return out, record
