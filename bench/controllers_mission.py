"""hyb2m: a whole run with junctions (2026-11-11).

hyb2 on every road; at each junction the driving code itself notices the arrival from
its map pose (no trigger line exists before a junction), asks the junction policy
(drive_v4/junction_policy.py: queued orchestrator command if it fits, default
otherwise) and executes the movement with hyb2j's junction logic.

Orchestrator interface (called by the bench's scripted orchestrator, on the car by
the real one):
    push_direction(t, "left" | "right" | "straight")   one sign -> one queued command
    stop(t) / resume(t)                                 interrupt: zero output / continue
"""
import importlib.util
import json
import math

import numpy as np

from .controllers import AUTOMOBILE, CONTROLLERS, HybridCNN, VENDOR_V4
from .controllers_junction import HybridJunction


def _load_policy():
    # drive_v4/junction_policy.py from the working tree (pure logic; the pinned v4 copy
    # on sys.path predates it and shares the package name)
    spec = importlib.util.spec_from_file_location("junction_policy", AUTOMOBILE / "drive_v4" / "junction_policy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class HybridMission(HybridJunction):
    ENTER_CM = 34.0          # decide here: = hyb2j's commit point (route starts 45 cm out, TRAVERSE at 12)
    ARRIVE_LAT_CM, ARRIVE_HEAD_DEG, ARRIVE_SXY = 9.0, 35.0, 2.0
    LEAVE_CM = 40.0          # movement done once the car is this far out along the exit arm
    INNER_ROUTE_CM = 110.0

    def __init__(self, direction, start_hint, timing):
        super().__init__(direction, start_hint, timing)              # no route: inactive until a junction
        jp = _load_policy()
        self.ARMS = jp.ARMS
        m = json.loads((VENDOR_V4 / "drive_v4" / "config" / "map_v4.json").read_text(encoding="utf-8"))
        self.junctions = m["junctions"]
        self.policy, self.intr = jp.JunctionPolicy(self.junctions, direction), jp.Interrupt()
        d = self.driver
        self.outer_route, self.outer_curv = d.route, d.map._curvature_at
        self.visit, self.decision, self.decisions, self.plan_cache = None, None, [], {}
        from .vehicle import PRESETS, Vehicle
        self.rim = Vehicle(PRESETS["field_estimate"], np.zeros(3)).footprint()    # measured body (all presets)
        self.source = self.source + " + mission (junction policy, stop/resume interrupt)"

    # --- orchestrator API -----------------------------------------------------------
    def push_direction(self, t, move):
        self.policy.push(t, move)

    def stop(self, t):
        self.intr.stop(t)

    def resume(self, t):
        self.intr.resume(t)

    # --- helpers --------------------------------------------------------------------
    def _plans(self, j, arm, move):
        from .junction import make_route, turn_plans
        key = (j, arm, move)
        if key not in self.plan_cache:
            route, _ = make_route(j, arm, move, self.rim)
            self.plan_cache[key] = (route, turn_plans(j, arm, move, self.rim))
        return self.plan_cache[key]

    def _arrival(self, pose, sxy):
        if sxy > self.ARRIVE_SXY:
            return None
        for j, jd in self.junctions.items():
            rel = pose[:2] - np.asarray(jd["xy"], float)
            for arm in jd["arms"]:
                a = np.asarray(self.ARMS[arm])
                along, lat = float(rel @ a), abs(float(rel @ np.array([a[1], -a[0]])))
                head = math.atan2(-a[0], -a[1])                         # travelling toward the centre
                dh = abs((pose[2] - head + math.pi) % (2 * math.pi) - math.pi)
                if 0 < along <= self.ENTER_CM and lat < self.ARRIVE_LAT_CM and dh < math.radians(self.ARRIVE_HEAD_DEG):
                    return j, arm
        return None

    def _road_after(self, j, exit_arm):
        """Route for the road after the junction: the outer loop again, or an inner road line."""
        d = self.driver
        on_outer = j != "center" and self.policy.kind(j, exit_arm) == "outer_T"   # not the T's inner arm
        if on_outer:
            d.route = d.map.route = self.outer_route
            d.map._curvature_at = self.outer_curv
        else:
            c, a = np.asarray(self.junctions[j]["xy"], float), np.asarray(self.ARMS[exit_arm])
            self._set_route(c + np.arange(0.0, self.INNER_ROUTE_CM, 1.0)[:, None] * a)
        return on_outer

    # --- one step ---------------------------------------------------------------------
    def step(self, image, capture_s, done_s):
        d = self.driver
        if self.intr.held:
            out, info = HybridCNN.step(self, image, capture_s, done_s)    # map worker keeps observing
            dt = 1 / 30 if self.last_done is None else done_s - self.last_done
            self.last_done = done_s
            for k in ("sub_since", "hold_since"):                          # freeze the movement's timers
                if getattr(self, k, None) is not None:
                    setattr(self, k, getattr(self, k) + dt)
            z = np.zeros(2)
            self._apply(done_s, z)
            self.last_out = z
            info.update(reason="interrupt_stop", phase=getattr(self, "phase", None) if self.active else None,
                        sub=getattr(self, "sub", None) if self.active else None, mission=self._state())
            return z, info
        pose = np.asarray(d.map.pose_at(done_s), float)
        _, sxy = d.map.weight()
        if self.visit is not None and not self.active:
            if np.linalg.norm(pose[:2] - np.asarray(self.junctions[self.visit[0]]["xy"])) > self.LEAVE_CM + 15:
                self.visit = None
        if not self.active and self.visit is None:
            hit = self._arrival(pose, sxy)
            if hit is not None:
                self.visit = hit
                dec = self.policy.decide(done_s, *hit)
                self.decision = dec
                self.decisions.append(dict(t=done_s, junction=dec.junction, came_from=dec.came_from, move=dec.move,
                                           exit_arm=dec.exit_arm, source=dec.source, options=dec.options, note=dec.note))
                if dec.move != "straight":
                    route, plans = self._plans(dec.junction, dec.came_from, dec.move)
                    self.begin(route, plans)
        out, info = super().step(image, capture_s, done_s)
        if self.active and self.phase == "EXIT":
            j, ex = self.decision.junction, self.decision.exit_arm
            rel = np.asarray(d.map.pose_at(done_s)[:2]) - np.asarray(self.junctions[j]["xy"], float)
            if rel @ np.asarray(self.ARMS[ex]) >= self.LEAVE_CM:
                self.active = False
                self._road_after(j, ex)
        info.update(mission=self._state())
        return out, info

    def _state(self):
        dec = self.decision
        return dict(active=self.active, visit=self.visit, held=self.intr.held,
                    last=None if dec is None else f"{dec.junction}:{dec.came_from}->{dec.move}({dec.source})",
                    queue=[c["move"] for c in self.policy.queue])


CONTROLLERS["hyb2m"] = HybridMission
