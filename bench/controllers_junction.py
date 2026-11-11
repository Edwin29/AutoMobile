"""hyb2j: hyb2 + one junction movement along an open map route (2026-11-11).

Port of cnn_map_hybrid_lab Hybrid's junction branch (GPT): the CNN has no direction
input, so map route pursuit steers through the junction and the CNN takes over on the
exit road. Phases: APPROACH (hyb2) -> TRAVERSE (route) -> EXIT (CNN again).

Changes against the prototype, from bench/junction.py runs (K_junction, 88 runs, first
port: left 17/22 correct exits but 21/22 touched a line, 4 held for ever):
  * turn method from the car's ability: with the inner wheel stopped the tightest radius
    is track / (2 * turn gain), ~10.8 cm at gain 0.6. An arc is used only if one with
    radius >= ARC_FACTOR x that keeps a static margin >= ARC_MIN_MARGIN; otherwise the
    car pivots in place at the junction (inner wheel backwards). The central crossroad
    turns need radius <= 6-8 cm: forward-only arcs cannot fit there.
  * slower through the turn (mean duty TURN_DUTY): less lag-driven overshoot
  * pose jump gate: in TRAVERSE the map pose is accepted only while it agrees with the
    pose predicted from the sent commands (a 34 cm route relocalization during a
    composite-disturbance turn steered the car wrong, then held it for ever)
  * a lost fix holds the car for at most HOLD_MAX_S, then it continues on the predicted pose
  * EXIT needs a tight map (sxy < SXY_EXIT) for EXIT_FRAMES frames, and is reversible
  * reverse commands: v4's motion model treats negative duty as standstill, so the map
    is told the signed motion here (the pinned v4 is not changed)
"""
import numpy as np

from .controllers import CONTROLLERS, HybridCNN


class HybridJunction(HybridCNN):
    takes_route = True
    LOOKAHEAD_CM = 12.0
    TRAVERSE_ARC_CM = 12.0
    EXIT_LEFT_CM, EXIT_HEADING_DEG = 18.0, 12.0
    SXY_EXIT, EXIT_FRAMES, SXY_HOLD, HOLD_MAX_S = 1.0, 3, 2.0, 1.0
    BACK_HEADING_DEG, BACK_LATERAL_CM = 25.0, 6.0
    ARC_FACTOR, ARC_MIN_MARGIN = 1.25, 0.5
    TURN_DUTY = 0.34
    PIVOT_DUTY, PIVOT_DONE_DEG, SETTLE_S, PIVOT_MIN_MARGIN = 0.40, 5.0, 0.3, 0.5
    STOP_LEAD_S = 0.25        # stop this much travel time before P (latency + motor lag)
    GATE_CM, GATE_DEG, GATE_FRAMES = 5.0, 15.0, 15

    def __init__(self, direction, start_hint, timing, route=None, plans=None):
        super().__init__(direction, start_hint, timing)
        from drive_v3.map_model import Route
        from drive_v4.core import MotorModel
        self.Route = Route
        self.actuator = MotorModel(dict(self.cfg["motor"]))
        self.last_done, self.last_out, self.alpha = None, np.zeros(2), 0.0
        self.active = False
        if route is not None:
            self.begin(route, plans)

    def begin(self, route, plans):
        """Start one junction movement (scene mode: at construction; mission mode: when the
        car reaches a junction)."""
        self.plans, self.choice, self.active = plans, None, True
        self._set_route(np.asarray(route, float))
        self.phase, self.sub, self.exit_count, self.corr = "APPROACH", None, 0, 0.0
        self.tp, self.disagree, self.hold_since, self.sub_since = None, 0, None, None
        self.source = self.source + ", junction route (ability-chosen arc / pivot, gated pose, reversible exit)"

    def _set_route(self, pts):
        rt = self.Route("junction", np.asarray(pts, float))
        d = self.driver
        self.route = d.route = d.map.route = rt
        d.map._curvature_at = lambda s: np.interp(np.clip(s, 0, rt.s[-1]), rt.s, rt.curvature)

    # --- signed motion (v4's model is forward-only) ---------------------------------
    def _signed_velocity(self, pwm):
        m = self.driver.motor
        p = np.asarray(pwm, float)
        speed = np.sign(p) * np.where(np.abs(p) >= m.start, np.abs(p), 0.0) * m.c["cm_s_per_duty"]
        raw = (speed[0] - speed[1]) / m.c["track_cm"]
        return float(speed.mean()), float(raw * m.yaw_gain(raw)), float(raw)

    def _apply(self, done, pwm):
        d = self.driver
        if np.all(np.asarray(pwm) >= 0):
            d.applied(done, pwm)
            return
        v, om, raw = self._signed_velocity(pwm)
        d.pwm = np.asarray(pwm, float).copy()
        d.map.command(done, v, om)
        d.raw_cmds.append((float(done), raw))
        d.body_cmds.append((float(done), v, om))

    # --- choice of the turn -----------------------------------------------------------
    def _choose(self):
        if self.plans is None or self.choice is not None:
            return
        m = self.driver.motor
        sign = 1.0 if self.plans["exit"] == "right" else -1.0
        r_min = m.c["track_cm"] / (2 * m.yaw_gain(sign))
        ok = [a for a in self.plans["arcs"] if a[0] >= self.ARC_FACTOR * r_min and a[1] >= self.ARC_MIN_MARGIN]
        pv = self.plans["pivot"]
        # Pivot first wherever it fits (course TA: "go straight in, turn 90 deg, go out" is
        # fine; field: pivoting works on the car, with some wheel slip / body slide).
        if pv["margin_cm"] >= self.PIVOT_MIN_MARGIN or not ok or pv["margin_cm"] >= max(a[1] for a in ok):
            ok = []
        if ok:
            r, margin, pts = min(ok, key=lambda a: a[0])
            self._set_route(pts)
            self.choice = dict(kind="arc", radius_cm=r, margin_cm=margin, r_min_cm=r_min)
        else:
            pv = self.plans["pivot"]
            self._set_route(pv["pts_in"])
            self.choice = dict(kind="pivot", P=pv["P"], margin_cm=pv["margin_cm"], r_min_cm=r_min)
            self.sub = "IN"

    # --- pose with jump gate ------------------------------------------------------------
    def _gated_pose(self, done, dt, sxy):
        from drive_v3.geometry import wrap as wrap3
        mp = np.asarray(self.driver.map.pose_at(done), float)
        if self.tp is None:
            self.tp = mp
            return mp, False
        v, om, _ = self._signed_velocity(self.last_out)
        a = om * dt
        pred = self.tp.copy()
        pred[:2] += v * dt * np.array([np.sin(pred[2] + a / 2), np.cos(pred[2] + a / 2)])
        pred[2] = float(wrap3(pred[2] + a))
        agree = (np.linalg.norm(mp[:2] - pred[:2]) <= self.GATE_CM
                 and abs(float(wrap3(mp[2] - pred[2]))) <= np.radians(self.GATE_DEG))
        self.disagree = 0 if agree else self.disagree + 1
        if agree or (self.disagree >= self.GATE_FRAMES and sxy < self.SXY_EXIT):
            self.tp, self.disagree = mp, 0
            return mp, False
        self.tp = pred
        return pred, True

    def _pursue(self, pose, arc, info, mean):
        from drive_v3.geometry import to_body
        d, rt = self.driver, self.route
        q = to_body(rt.sample(arc + self.LOOKAHEAD_CM), pose)[0]
        k_route = float(2 * q[0] / max(float(q @ q), 1.0))
        v, om = d.motor.velocity(np.asarray(info["cnn_pwm"], float))
        k_cnn = om / max(v, 0.1)
        self.corr = self.alpha * self.corr + (1 - self.alpha) * (k_route - k_cnn)
        k = float(np.clip(k_cnn + self.corr, -0.16, 0.16))
        self.actuator.gain_left, self.actuator.gain_right = d.motor.gain_left, d.motor.gain_right
        self.actuator.floor_raise = d.motor.floor_raise
        if mean is None:
            mean = (self.TURN_DUTY if abs(k_route) > 0.02
                    else float(np.clip(np.mean(np.maximum(info["cnn_pwm"], 0)), 0.34, 0.48)))
        out, _ = self.actuator.mix(mean, k, 0.66)
        return out

    def step(self, image, capture_s, done_s):
        from drive_v3.geometry import wrap as wrap3
        d = self.driver
        out, info = super().step(image, capture_s, done_s)     # hyb2 command; feeds the map worker
        dt = 1 / 30 if self.last_done is None else done_s - self.last_done
        self.last_done = done_s
        self.alpha = float(np.exp(-dt / 0.075))
        if not self.active:
            self.last_out = np.asarray(out, float).copy()
            info.update(phase=None, sub=None)
            return np.asarray(out, float), info
        return self._movement(out, info, done_s, dt)

    def _movement(self, out, info, done_s, dt):
        from drive_v3.geometry import wrap as wrap3
        d = self.driver
        _, sxy = d.map.weight()
        if self.phase == "APPROACH":
            pose, gated = np.asarray(d.map.pose_at(done_s), float), False
            self.tp = pose
        else:
            pose, gated = self._gated_pose(done_s, dt, sxy)
        rt = self.route
        i = rt.nearest(pose[:2])
        arc = float(rt.s[i])
        if self.phase == "APPROACH" and arc >= self.TRAVERSE_ARC_CM:
            self.phase = "TRAVERSE"
            self._choose()
            rt = self.route
            i = rt.nearest(pose[:2])
            arc = float(rt.s[i])
        reason = info["reason"]
        was_traverse = self.phase == "TRAVERSE"
        if self.phase == "TRAVERSE":
            hold = sxy > self.SXY_HOLD
            self.hold_since = (done_s if self.hold_since is None else self.hold_since) if hold else None
            if hold and done_s - self.hold_since <= self.HOLD_MAX_S:
                out, reason = np.zeros(2), "traverse_hold_no_fix"
            elif self.sub == "IN":
                # pivot plan: the CNN keeps the lane (it does so better than the map in harsh
                # light: map lateral error 4-14 cm there); the map only says where to stop
                P = np.asarray(self.choice["P"])
                d_in = rt.points[-1] - rt.points[0]
                d_in = d_in / np.linalg.norm(d_in)
                v_now = self._signed_velocity(self.last_out)[0]
                if (P - pose[:2]) @ d_in <= 0.5 + max(v_now, 0.0) * self.STOP_LEAD_S:
                    self.sub, self.sub_since, out, reason = "SETTLE_IN", done_s, np.zeros(2), "pivot_settle"
                else:
                    reason = "pivot_in_cnn"                         # out = hyb2 (CNN) command
            elif self.sub in ("SETTLE_IN", "SETTLE_OUT"):
                out, reason = np.zeros(2), "pivot_settle"
                if done_s - self.sub_since >= self.SETTLE_S:
                    if self.sub == "SETTLE_IN":
                        self.sub = "ROTATE"
                    else:                                           # facing the exit road: CNN again
                        self.sub = "OUT"
                        self._set_route(self.plans["pivot"]["pts_out"])
                        self.phase = "EXIT"
            elif self.sub == "ROTATE":
                err = float(wrap3(self.plans["pivot"]["h_out"] - pose[2]))
                if abs(err) <= np.radians(self.PIVOT_DONE_DEG):
                    self.sub, self.sub_since, out, reason = "SETTLE_OUT", done_s, np.zeros(2), "pivot_settle"
                else:
                    s = 1.0 if err > 0 else -1.0                      # + = clockwise = right
                    out, reason = self.PIVOT_DUTY * np.array([s, -s]), "pivot_rotate"
            else:                                                     # arc, or the pivot OUT segment
                out = self._pursue(pose, arc, info, None)
                reason = "pivot_out" if self.sub == "OUT" else "traverse"
                end_err = abs(float(wrap3(pose[2] - rt.heading[-1])))
                ok = (arc >= rt.s[-1] - self.EXIT_LEFT_CM and end_err < np.radians(self.EXIT_HEADING_DEG)
                      and sxy < self.SXY_EXIT)
                self.exit_count = self.exit_count + 1 if ok else 0
                if self.exit_count >= self.EXIT_FRAMES:
                    self.phase = "EXIT"
            if gated:
                reason += "+pose_gated"
        if was_traverse:
            self._apply(done_s, out)                                  # tell the map what was really sent
        elif self.phase == "EXIT":
            head_err = abs(float(wrap3(pose[2] - rt.heading[i])))
            lateral = float(np.linalg.norm(rt.points[i] - pose[:2]))
            if sxy < self.SXY_EXIT and arc < rt.s[-1] - 5 and (
                    head_err > np.radians(self.BACK_HEADING_DEG) or lateral > self.BACK_LATERAL_CM):
                self.phase, self.exit_count = "TRAVERSE", 0
                reason += "+exit_revoked"
        if self.phase != "TRAVERSE":
            self.corr = 0.0
        self.last_out = np.asarray(out, float).copy()
        info.update(reason=reason, phase=self.phase, sub=self.sub, route_arc=arc, sxy_now=sxy,
                    est=[float(v) for v in pose])
        return np.asarray(out, float), info


CONTROLLERS["hyb2j"] = HybridJunction
