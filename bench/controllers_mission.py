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
    # Which outer road the car is on, counted from corners (field 2026-11-11 17:52: the map
    # lost the car at a corner and relocalized one road behind; the track looks the same
    # on every side, so it stayed a quarter lap off and every junction was the wrong one).
    # A corner = the applied commands turn >= CORNER_DEG the lap way within CORNER_WINDOW_S;
    # one per CORNER_BLOCK_S (a 90 deg corner crossed the threshold twice). On the five
    # field runs this found every corner once, nothing else. If the map sits on another
    # road for SIDE_FIX_S, its pose is rotated about the track centre to the counted road.
    SIDES_CCW = ("E", "N", "W", "S")
    CORNER_DEG, CORNER_WINDOW_S, CORNER_BLOCK_S, SIDE_FIX_S = 45.0, 5.0, 8.0, 3.0

    def __init__(self, direction, start_hint, timing):
        super().__init__(direction, start_hint, timing)              # no route: inactive until a junction
        jp = _load_policy()
        self.ARMS = jp.ARMS
        m = json.loads((VENDOR_V4 / "drive_v4" / "config" / "map_v4.json").read_text(encoding="utf-8"))
        self.junctions = m["junctions"]
        self.policy, self.intr = jp.JunctionPolicy(self.junctions, direction), jp.Interrupt()
        d = self.driver
        self.outer_route, self.outer_curv = d.route, d.map._curvature_at
        # Both lap directions: a merge can turn the car the other way round the loop
        # (e.g. cw, then 'left' at a T merge = ccw from then on). The map's outer route and
        # the next merge default must follow the way the car really goes (field 2026-11-11).
        from drive_v4.route import with_lane_offsets
        other = "ccw" if direction == "cw" else "cw"
        self.outer_routes = {direction: d.route, other: with_lane_offsets(d.track, d.track.outer_route(other))}
        self.lap_dir = direction
        self.visit, self.decision, self.decisions, self.plan_cache = None, None, [], {}
        self.direction_changes = []
        self.side = self._side_of(np.asarray(start_hint, float)[:2])     # None: not on the outer loop
        self.turn_hist, self.corner_block, self.side_bad_since = [], -1e9, None
        self.corners, self.side_fixes = [], []
        self.start_route = self._inner_start_route(np.asarray(start_hint, float))
        if self.start_route is not None:      # started on an inner road (field 16:47: the outer
            self._set_route(self.start_route)  # route made the map relocalize onto the outer road)
        from .vehicle import PRESETS, Vehicle
        self.rim = Vehicle(PRESETS["field_estimate"], np.zeros(3)).footprint()    # measured body (all presets)
        from .junction import load_plan_cache                # load before driving, not at the first junction
        self.plan_cache = {k: v[:2] for k, v in (load_plan_cache() or {}).items()}
        self.source = self.source + " + mission (junction policy, stop/resume interrupt)"

    # --- orchestrator API -----------------------------------------------------------
    def push_direction(self, t, move):
        self.policy.push(t, move)

    def stop(self, t, reason="orchestrator"):
        self.intr.stop(t, reason)

    def resume(self, t, reason="orchestrator"):
        self.intr.resume(t, reason)

    def resume_all(self, t):
        self.intr.resume_all(t)

    # --- helpers --------------------------------------------------------------------
    def _plans(self, j, arm, move):
        """Precomputed (bench/world/junction_plans.pkl, python -m bench.junction --build-plans);
        computing on the fly costs seconds per junction (too slow while driving on the Pi)."""
        from .junction import load_plan_cache, make_route, turn_plans
        key = (j, arm, move)
        if not self.plan_cache:
            self.plan_cache = {k: v[:2] for k, v in (load_plan_cache() or {}).items()}
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

    @staticmethod
    def _inner_start_route(pose):
        """Centre line of the inner road the start pose is on (None if on the outer loop)."""
        x, y, yaw = pose
        a = np.array([math.sin(yaw), math.cos(yaw)])
        if abs(x) < 9.25 and abs(y) < 73.0 and abs(a[1]) > 0.7:          # inner vertical road
            p0, a = np.array([0.0, y]), np.array([0.0, math.copysign(1.0, a[1])])
        elif abs(y) < 10.5 and abs(x) < 74.0 and abs(a[0]) > 0.7:        # inner horizontal road
            p0, a = np.array([x, 0.0]), np.array([math.copysign(1.0, a[0]), 0.0])
        else:
            return None
        return p0 + np.arange(-20.0, 130.0, 1.0)[:, None] * a

    @staticmethod
    def _lap_dir_of(xy, a):
        """cw or ccw for travelling along the outer road at xy in direction a (map frame,
        north up: cw goes east on the north road, south on the east road, west on the
        south road, north on the west road)."""
        x, y = xy
        if abs(y) >= abs(x):
            cw = np.array([1.0, 0.0]) if y > 0 else np.array([-1.0, 0.0])
        else:
            cw = np.array([0.0, -1.0]) if x > 0 else np.array([0.0, 1.0])
        return "cw" if float(cw @ a) > 0 else "ccw"

    @staticmethod
    def _side_of(xy):
        x, y = float(xy[0]), float(xy[1])
        if abs(x) > 60 and abs(y) > 60:
            return None                                   # a corner: no side
        if abs(x) > 70:
            return "E" if x > 0 else "W"
        if abs(y) > 70:
            return "N" if y > 0 else "S"
        return None                                       # inside: inner roads

    def _rotate_map(self, q):
        """Rotate the map's belief by q quarter turns counter-clockwise about the track
        centre (the track is the same on every side up to a few cm)."""
        from .map_worker import rotate_estimator
        rotate_estimator(self.driver.map.est, q)
        if self.worker is not None:
            self.worker.send(("rotate", int(q)))

    def _track_side(self, done_s, pose):
        """Corner counting on the outer loop and the map side check."""
        dt = 0.0 if getattr(self, "_side_t", None) is None else done_s - self._side_t
        self._side_t = done_s
        if self.side is None or self.active or self.intr.held:
            self.turn_hist = []
            return
        om = self._signed_velocity(self.last_out)[1]
        self.turn_hist.append((done_s, om * dt))
        self.turn_hist = [h for h in self.turn_hist if h[0] > done_s - self.CORNER_WINDOW_S]
        sgn = 1.0 if self.lap_dir == "cw" else -1.0       # bench yaw: + = clockwise
        if done_s > self.corner_block and sgn * sum(h[1] for h in self.turn_hist) >= math.radians(self.CORNER_DEG):
            i = self.SIDES_CCW.index(self.side)
            self.side = self.SIDES_CCW[(i + (1 if self.lap_dir == "ccw" else -1)) % 4]
            self.corners.append(dict(t=done_s, side=self.side))
            self.turn_hist, self.corner_block, self.side_bad_since = [], done_s + self.CORNER_BLOCK_S, None
            return
        ms = self._side_of(pose[:2])
        if ms is None or ms == self.side:
            self.side_bad_since = None
            return
        self.side_bad_since = done_s if self.side_bad_since is None else self.side_bad_since
        if done_s - self.side_bad_since >= self.SIDE_FIX_S:
            q = (self.SIDES_CCW.index(self.side) - self.SIDES_CCW.index(ms)) % 4
            self._rotate_map(q)
            self.side_fixes.append(dict(t=done_s, map_side=ms, counted_side=self.side, quarter_turns_ccw=q,
                                        pose_before=[float(v) for v in pose]))
            self.side_bad_since = None

    def use_map_process(self, pose):
        super().use_map_process(pose)
        if self.start_route is not None:
            self.worker.send(("route", np.asarray(self.start_route, float)))

    def _road_after(self, j, exit_arm):
        """Route for the road after the junction: the outer loop again, or an inner road line."""
        d = self.driver
        on_outer = j != "center" and self.policy.kind(j, exit_arm) == "outer_T"   # not the T's inner arm
        self.side = {"east": "E", "north": "N", "west": "W", "south": "S"}.get(j) if on_outer else None
        self.turn_hist, self.side_bad_since = [], None
        if on_outer:
            new_dir = self._lap_dir_of(self.junctions[j]["xy"], np.asarray(self.ARMS[exit_arm]))
            if new_dir != self.lap_dir:
                self.direction_changes.append(dict(t=self.last_done, junction=j, exit_arm=exit_arm,
                                                   old=self.lap_dir, new=new_dir))
                self.lap_dir = new_dir
                self.policy.lap_direction = new_dir                  # merge defaults follow it
            d.route = d.map.route = self.outer_routes[new_dir]
            d.map._curvature_at = self.outer_curv
            if self.worker is not None:
                self.worker.send(("route_outer", new_dir))
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
        import time as _t
        tic = _t.perf_counter()
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
        ms_pre = (_t.perf_counter() - tic) * 1e3
        out, info = super().step(image, capture_s, done_s)
        info["ms_mission_pre"] = ms_pre
        if self.active and self.phase == "EXIT":
            j, ex = self.decision.junction, self.decision.exit_arm
            rel = np.asarray(d.map.pose_at(done_s)[:2]) - np.asarray(self.junctions[j]["xy"], float)
            if rel @ np.asarray(self.ARMS[ex]) >= self.LEAVE_CM:
                self.active = False
                self._road_after(j, ex)
        self._track_side(done_s, np.asarray(d.map.pose_at(done_s), float))
        info.update(mission=self._state())
        return out, info

    def _state(self):
        dec = self.decision
        return dict(active=self.active, visit=self.visit, held=self.intr.held, lap=self.lap_dir, side=self.side,
                    side_fixes=len(self.side_fixes),
                    last=None if dec is None else f"{dec.junction}:{dec.came_from}->{dec.move}({dec.source})",
                    queue=[c["move"] for c in self.policy.queue])


CONTROLLERS["hyb2m"] = HybridMission
