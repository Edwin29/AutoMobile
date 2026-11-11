"""Whole runs with junctions and orchestrator commands (2026-11-11).

    python -m bench.mission --controller hyb2m --direction cw --scenario inner_loop --out DIR

The car starts in the start zone, drives the outer loop and may be sent through the
inner roads by direction commands. A scripted orchestrator stands in for the vision
team + orchestrator: it watches TRUTH (as their camera would watch the floor and the
signs) and calls the controller's push_direction / stop / resume, with a vision delay.
The controller never sees truth. Scenarios = sign / light placements of one run.

Ends: back in the start zone heading the lap way after >= 300 cm (mission_complete),
a severe departure (> 12 cm over a line, or the whole body off the road), 5 s of zero
output while not held by the orchestrator, or the time limit.
"""
import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .disturb import Disturbances, combine
from .realcam import pose_from_history
from .run import CAMERA_PERIOD_S, OWN, timing_profile, vehicle_from_args
from .vehicle import VARIATIONS, Vehicle
from .world import Camera, Renderer, Track, to_world

ARMS = {"N": (0.0, 1.0), "E": (1.0, 0.0), "S": (0.0, -1.0), "W": (-1.0, 0.0)}
VISION_DELAY_S = 0.15

# Scenarios: what the signs / lights of one run say. cw starts southbound on the east road.
#   direction: sign seen when the car is `at_cm` before the junction centre on that arm
#   stop_sign: pink line index (track.json pink_lines); vision trigger when the line is
#              `trigger_cm` ahead of the front bumper; the orchestrator resumes after 3 s
#   red_light: at the crossroad arm; trigger `trigger_cm` before the crosswalk; green after hold_s
SCENARIOS = {
    "lap_defaults": [],                                              # no sign at all: outer lap, all straight
    "inner_loop": [dict(kind="direction", move="right", junction="south", arm="E", at_cm=60),
                   dict(kind="red_light", arm="S", trigger_cm=4, hold_s=4.0)],
    "inner_loop_west": [dict(kind="direction", move="right", junction="west", arm="S", at_cm=60),
                        dict(kind="red_light", arm="W", trigger_cm=4, hold_s=4.0)],
    "stop_sign": [dict(kind="stop_sign", pink=6, trigger_cm=20, hold_s=3.0)],
    "wrong_sign": [dict(kind="direction", move="left", junction="south", arm="E", at_cm=60)],   # impossible: default
}


def params_name(p):
    from .vehicle import PRESETS
    return next((k for k, v in PRESETS.items() if v == p), "custom car")


def heading_vec(yaw):
    return np.array([math.sin(yaw), math.cos(yaw)])


class Orchestrator:
    """Scripted stand-in for vision + orchestrator. Truth in, controller calls out."""

    def __init__(self, events, track, front_cm):
        self.events = [dict(e, done=False) for e in events]
        self.track, self.front = track, front_cm
        self.pending = []           # (time, method, arg)
        self.log = []

    def watch(self, t, pose):
        front = pose[:2] + self.front * heading_vec(pose[2])
        h = heading_vec(pose[2])
        for e in self.events:
            if e["done"]:
                continue
            if e["kind"] == "direction":
                c = np.asarray(self.track.data["junctions"][e["junction"]]["xy"], float)
                a = np.asarray(ARMS[e["arm"]])
                rel = pose[:2] - c
                if 0 < rel @ a <= e["at_cm"] and abs(rel @ np.array([a[1], -a[0]])) < 10 and h @ -a > 0.8:
                    e["done"] = True
                    self.pending.append((t + VISION_DELAY_S, "push_direction", e["move"]))
            elif e["kind"] == "stop_sign":
                a, b = (np.asarray(p, float) for p in self.track.data["pink_lines"][e["pink"]])
                mid = (a + b) / 2
                ahead = float((mid - front) @ h)
                across = abs(float((mid - front) @ np.array([h[1], -h[0]])))
                if 0 < ahead <= e["trigger_cm"] and across < 15:
                    e["done"], e["line"] = True, mid.tolist()
                    self.pending.append((t + VISION_DELAY_S, "stop", None))
                    self.pending.append((t + VISION_DELAY_S + e["hold_s"], "resume", None))
            elif e["kind"] == "red_light":
                arm = np.asarray(ARMS[e["arm"]])
                for x0, y0, _ in self.track.data["crosswalks"]:
                    cw = np.array([x0, y0], float)
                    if abs(cw @ np.array([arm[1], -arm[0]])) < 1 and cw @ arm > 0:        # the crosswalk on that arm
                        ahead = float((cw - front) @ h) - 3.0                            # near edge (stripes +-3 cm)
                        across = abs(float((cw - front) @ np.array([h[1], -h[0]])))      # same road, not beside it
                        if 0 < ahead <= e["trigger_cm"] and across < 12 and h @ -arm > 0.8:
                            e["done"], e["edge_xy"] = True, (cw + 3.0 * arm).tolist()
                            self.pending.append((t + VISION_DELAY_S, "stop", None))
                            self.pending.append((t + VISION_DELAY_S + e["hold_s"], "resume", None))

    def deliver(self, t, ctrl):
        due = [p for p in self.pending if p[0] <= t]
        self.pending = [p for p in self.pending if p[0] > t]
        for when, method, arg in due:
            getattr(ctrl, method)(t, arg) if arg is not None else getattr(ctrl, method)(t)
            self.log.append(dict(t=t, method=method, arg=arg))


class MissionScorer:
    def __init__(self, track, rim, direction, start_pose, front_cm):
        self.track, self.rim, self.dir, self.front = track, rim, direction, front_cm
        self.min_margin, self.outside_s, self.t, self.end, self.travel = np.inf, 0.0, 0.0, None, 0.0
        self.episode, self.episodes, self.inside_since = None, [], None
        self.last_xy, self.zero_run = np.asarray(start_pose[:2], float), 0.0
        self.passages, self.inside_j = [], None
        r = track.data["start_regions"]["right_red"]
        ys = sorted(float(np.mean(track.data["red_lines"][i], axis=0)[1]) for i in r["red_indices"])
        self.zone = (r["road_x_cm"], ys, r["heading_by_direction"][direction])
        self.line_checks = []        # (kind, xy point of the line, normal toward the car's side)

    def watch_line(self, kind, xy, travel_dir):
        self.line_checks.append(dict(kind=kind, xy=np.asarray(xy, float), dir=np.asarray(travel_dir, float),
                                     min_ahead_cm=np.inf, crossed_while_held=False))

    def sample(self, t, dt, pose, pwm, held):
        self.t = t
        m = float(self.track.clearance(to_world(self.rim, pose)).min())
        self.min_margin = min(self.min_margin, m)
        self.travel += float(np.linalg.norm(pose[:2] - self.last_xy))
        self.last_xy = pose[:2].copy()
        if m < 0:
            self.outside_s += dt
            self.inside_since = None
            if self.episode is None:
                self.episode = dict(start_s=t, xy=pose[:2].tolist(), min_cm=m)
            self.episode["min_cm"] = min(self.episode["min_cm"], m)
        elif self.episode is not None:
            self.inside_since = t if self.inside_since is None else self.inside_since
            if t - self.inside_since >= 1.0 - 1e-9:
                self.episode.update(end_s=t, recovered=True)
                self.episodes.append(self.episode)
                self.episode, self.inside_since = None, None
        front = pose[:2] + self.front * heading_vec(pose[2])
        for lc in self.line_checks:
            ahead = float((lc["xy"] - front) @ lc["dir"])
            if held:
                lc["min_ahead_cm"] = min(lc["min_ahead_cm"], ahead)
        # junction passages (truth): entry arm when coming within 22 cm, exit arm when leaving 30 cm
        for j, jd in self.track.data["junctions"].items():
            rel = pose[:2] - np.asarray(jd["xy"], float)
            dist = float(np.linalg.norm(rel))
            if self.inside_j is None and dist < 22:
                self.inside_j = dict(junction=j, came_from=self._arm(rel), t=t)
            elif self.inside_j is not None and self.inside_j["junction"] == j and dist > 30:
                self.passages.append(dict(self.inside_j, exit_arm=self._arm(rel), t_out=t))
                self.inside_j = None
        self.zero_run = self.zero_run + dt if (np.max(np.abs(pwm)) < 1e-8 and not held) else 0.0
        (x0, x1), (y0, y1), yaw = self.zone
        in_zone = x0 < pose[0] < x1 and y0 < pose[1] < y1 and math.cos(pose[2] - yaw) > 0.85
        d = self.track.clearance(to_world(self.rim, pose))
        if d.max() < 0 or m < -12:
            self.end = "severe_departure"
        elif in_zone and self.travel > 300 and self.episode is None:
            self.end = "mission_complete"
        elif self.zero_run >= 5.0:
            self.end = "held"

    @staticmethod
    def _arm(rel):
        return min(ARMS, key=lambda k: -(np.asarray(ARMS[k]) @ rel))

    def summary(self, outcome):
        if self.episode is not None:
            self.episode.update(end_s=self.t, recovered=False)
            self.episodes.append(self.episode)
        return dict(outcome=outcome, min_margin_cm=self.min_margin, outside_s=self.outside_s, simulated_s=self.t,
                    travel_cm=self.travel, episodes=self.episodes, passages=self.passages,
                    clean=outcome == "mission_complete" and self.min_margin >= 0,
                    stop_lines=[dict(kind=l["kind"], xy=l["xy"].tolist(), min_ahead_cm=l["min_ahead_cm"])
                                for l in self.line_checks])


def expected_passages(scenario, direction, track):
    """Junction sequence a correct run makes (from the policy's own rules, no driving)."""
    import importlib.util
    from .controllers import AUTOMOBILE
    spec = importlib.util.spec_from_file_location("jp", AUTOMOBILE / "drive_v4" / "junction_policy.py")
    jp = importlib.util.module_from_spec(spec); spec.loader.exec_module(jp)
    pol = jp.JunctionPolicy(track.data["junctions"], direction)
    cmds = {(e["junction"], e["arm"]): e["move"] for e in SCENARIOS[scenario] if e["kind"] == "direction"}
    # walk the road graph from the start zone
    if direction != "cw":
        raise NotImplementedError("expected sequence only written for cw so far")
    seq, j, arm, t = [], "south", "E", 0.0   # cw from the start zone: first junction = south T from the east
    for _ in range(12):
        if (j, arm) in cmds:                         # one sign, seen once
            pol.push(t, cmds.pop((j, arm)))
        dec = pol.decide(t, j, arm)
        seq.append((j, arm, dec.exit_arm))
        t += 1.0
        if j == "east" and dec.exit_arm == "S":      # southbound on the east road again: the start zone
            break
        j, arm = _next_junction(j, dec.exit_arm, track)
    return seq


def _next_junction(j, exit_arm, track):
    """Next junction after leaving j through exit_arm; outer corners turn the lap way (cw: right)."""
    J = track.data["junctions"]
    c, d = np.asarray(J[j]["xy"], float), np.asarray(ARMS[exit_arm])
    for _ in range(3):
        best = None
        for k, v in J.items():
            rel = np.asarray(v["xy"], float) - c
            along, lat = float(rel @ d), abs(float(rel @ np.array([d[1], -d[0]])))
            if along > 1 and lat < 2 and (best is None or along < best[0]):
                best = (along, k)
        if best is not None:
            return best[1], min(ARMS, key=lambda a: float(np.linalg.norm(np.asarray(ARMS[a]) + d)))
        # outer corner: move to the corner of the outer centre lines, turn right (cw)
        xs, ys = (-87.25, 87.25), (-85.25, 86.25)
        c = np.array([xs[int(d[0] > 0)] if d[0] else c[0], ys[int(d[1] > 0)] if d[1] else c[1]])
        d = np.array([d[1], -d[0]])
    raise RuntimeError("no junction found")


class Film:
    """Video for people: top-down map (truth body, map belief, trail, stop lines) next to
    the camera frame the controller saw, with the mission state. Every `every`-th frame
    at 30 fps (every=3 -> about 3x real time)."""

    def __init__(self, path, renderer, rim, every=3):
        self.tex = cv2.flip(renderer.base_texture, 0)           # rows = +y: flip so north is up
        self.ppcm, self.origin, self.n = renderer.ppcm, renderer.origin, self.tex.shape[0]
        self.rim, self.every, self.k, self.trail = rim, every, 0, []
        self.size = 540
        self.w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (self.size + 720, self.size))

    def px(self, xy):
        xy = np.asarray(xy, float)
        x = (xy[..., 0] - self.origin) * self.ppcm
        y = self.n - 1 - (xy[..., 1] - self.origin) * self.ppcm
        return np.stack([x, y], -1)

    def frame(self, t, truth, est, image, info, margin, held, lines, title):
        self.trail.append(truth[:2].copy())
        self.k += 1
        if self.k % self.every:
            return
        top = self.tex.copy()
        tr = self.px(np.array(self.trail)).astype(np.int32)
        if len(tr) > 1:
            cv2.polylines(top, [tr], False, (255, 255, 255), 2, cv2.LINE_AA)
        for xy in lines:
            cv2.circle(top, tuple(self.px(xy).astype(int)), 10, (60, 60, 255), 3, cv2.LINE_AA)
        body = self.px(to_world(self.rim, truth)).astype(np.int32)
        cv2.fillPoly(top, [body], (0, 200, 255) if margin >= 0 else (40, 40, 230))
        if est is not None:
            e = np.asarray(est, float)
            tip = self.px(e[:2] + 9 * heading_vec(e[2]))
            cv2.arrowedLine(top, tuple(self.px(e[:2]).astype(int)), tuple(tip.astype(int)), (255, 120, 0), 3,
                            cv2.LINE_AA, tipLength=0.4)
        top = cv2.resize(top, (self.size, self.size), interpolation=cv2.INTER_AREA)
        cam = cv2.resize(image, (720, 540))
        canvas = np.hstack([top, cam])
        m = info.get("mission") or {}
        rows = [title,
                f"t {t:5.1f} s   margin {margin:+5.2f} cm   {'HELD (orchestrator stop)' if held else ''}",
                f"mode {info.get('reason', '')}   phase {info.get('phase') or '-'} {info.get('sub') or ''}",
                f"last junction: {m.get('last') or '-'}   queue: {m.get('queue') or []}"]
        for i, r in enumerate(rows):
            for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
                cv2.putText(canvas, r, (self.size + 12, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, th, cv2.LINE_AA)
        legend = "body: yellow inside / red over a line   blue arrow: map belief   white: path"
        for col, th in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(canvas, legend, (10, self.size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, th, cv2.LINE_AA)
        self.w.write(canvas)

    def close(self):
        self.w.release()


def run(controller, params, effects, direction, scenario, timing, seed, seconds, out, variation="fixed", video=False):
    from .controllers import CONTROLLERS
    cv2.setNumThreads(1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    profile = timing_profile(OWN[controller] if timing == "own" else {"fast": "v4_pi", "slow": "v3r_laptop"}[timing])
    track, camera = Track(), Camera()
    renderer = Renderer(track, camera)
    rng = np.random.default_rng(seed)
    hint = track.start_pose(direction, params.front_cm, params.rear_cm)
    truth0 = hint + np.array([rng.uniform(-1, 1), rng.uniform(-2, 2), np.radians(rng.uniform(-3, 3))])
    var = replace(VARIATIONS[variation], seed=seed)
    car = Vehicle(params, truth0, var)
    scorer = MissionScorer(track, car.footprint(), direction, truth0, params.front_cm)
    orch = Orchestrator(SCENARIOS[scenario], track, params.front_cm)
    dist = Disturbances(combine(effects), direction, seed)
    dist.arm(renderer)
    ctrl = CONTROLLERS[controller](direction, hint, profile)
    has_api = hasattr(ctrl, "push_direction")
    t, last_pwm, history = 0.0, np.zeros(2), [(0.0, truth0.copy())]
    held = lambda: bool(has_api and ctrl.intr.held)

    def advance_to(target):
        nonlocal t
        while t < target - 1e-10 and scorer.end is None:
            dt = min(0.02, target - t)
            dist.update(t, scorer.travel, 60.0)
            car.step(t, dt, dist.speed_scale(t))
            t += dt
            scorer.sample(t, dt, car.pose, last_pwm, held())
            orch.watch(t, car.pose)
            for e in orch.events:                     # register stop lines once their trigger fired
                if e.get("done") and not e.get("scored"):
                    e["scored"] = True
                    if e["kind"] == "stop_sign":
                        scorer.watch_line("pink", e["line"], heading_vec(car.pose[2]))
                    if e["kind"] == "red_light":
                        scorer.watch_line("crosswalk", e["edge_xy"], heading_vec(car.pose[2]))
            history.append((t, car.pose.copy()))
            if len(history) > 400:
                del history[:200]

    capture, seq, wall, realcam = CAMERA_PERIOD_S, 0, time.perf_counter(), None
    title = f"{scenario}  {params_name(params)}  {'+'.join(effects) or 'sim render'}  seed {seed}"
    writer = Film(out / "film.mp4", renderer, car.footprint()) if video else None
    with (out / "steps.jsonl").open("w", encoding="utf-8") as log:
        while capture < seconds:
            advance_to(capture)
            if scorer.end:
                break
            raw = getattr(ctrl, "raw_camera", False)
            if dist.p.get("realcam"):
                if realcam is None:
                    from .realcam import RealCam
                    realcam = RealCam(renderer, camera, dist.p["realcam"], seed)
                frame = realcam.frame(capture, pose_from_history(history), raw=raw)
            else:
                frame = renderer.render_raw(car.pose) if raw else renderer.render(car.pose)
            image, flags = dist.image(frame, capture, renderer, history, raw=raw)
            delay = profile["latencies_s"][seq % len(profile["latencies_s"])]
            done = capture + delay
            advance_to(done)
            if scorer.end:
                break
            if has_api:
                orch.deliver(done, ctrl)
            pwm, info = ctrl.step(image, capture, done)
            if info.get("est") is not None:
                info["pose_err_cm"] = float(np.linalg.norm(np.asarray(info["est"][:2]) - car.pose[:2]))
            last_pwm = np.asarray(pwm, float)
            car.command(done, last_pwm)
            log.write(json.dumps(dict(t=done, pwm=last_pwm.tolist(), truth=car.pose.tolist(), flags=flags,
                                      **info), default=float) + "\n")
            if writer:
                m = float(track.clearance(to_world(car.footprint(), car.pose)).min())
                lines = [l["xy"] for l in scorer.line_checks]
                writer.frame(done, car.pose.copy(), info.get("est"), image, info, m, held(), lines, title)
            capture = max(capture + CAMERA_PERIOD_S, done)
            seq += 1
    if writer:
        writer.close()
    s = scorer.summary(scorer.end or "time_limit")
    got = [(p["junction"], p["came_from"], p["exit_arm"]) for p in s["passages"]]
    want = [tuple(x) for x in expected_passages(scenario, direction, track)] if direction == "cw" else None
    s.update(controller=controller, scenario=scenario, events=SCENARIOS[scenario], orchestrator_calls=orch.log,
             expected_passages=want, route_correct=(want is not None and got == want),
             decisions=getattr(ctrl, "decisions", None), policy_log=getattr(getattr(ctrl, "policy", None), "log", None),
             direction=direction, effects=list(effects), seed=seed, vehicle=params.to_dict(),
             variation=dict(name=variation, **var.to_dict()), frames=seq, wall_s=time.perf_counter() - wall,
             camera=dict(profile=realcam.profile, drawn={k: round(v, 3) for k, v in realcam.cfg.items()}) if realcam else None)
    (out / "summary.json").write_text(json.dumps(s, indent=2, default=float), encoding="utf-8")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controller", choices=sorted(OWN), default="hyb2m")
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="lap_defaults")
    ap.add_argument("--direction", choices=["cw", "ccw"], default="cw")
    ap.add_argument("--vehicle", default="field_estimate")
    ap.add_argument("--vehicle-json", type=json.loads)
    ap.add_argument("--variation", choices=sorted(VARIATIONS), default="fixed")
    ap.add_argument("--effects", default="")
    ap.add_argument("--timing", choices=["own", "fast", "slow"], default="own")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--expected", action="store_true", help="only print the expected junction sequence")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.expected:
        for sc in SCENARIOS:
            print(sc, expected_passages(sc, a.direction, Track()))
        return
    params = vehicle_from_args(a.vehicle, a.vehicle_json)
    s = run(a.controller, params, [e for e in a.effects.split(",") if e], a.direction, a.scenario, a.timing, a.seed,
            a.seconds, a.out, a.variation, a.video)
    print(json.dumps({k: s[k] for k in ("controller", "scenario", "outcome", "route_correct", "min_margin_cm",
                                         "outside_s", "simulated_s")} | {"passages": [
        f"{p['junction']}:{p['came_from']}->{p['exit_arm']}" for p in s["passages"]]}, default=float))


if __name__ == "__main__":
    main()
