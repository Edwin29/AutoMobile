"""Junction scenes in the common bench (2026-11-11).

    python -m bench.junction --controller hyb2j --junction east --came-from S --exit left --out DIR

A scene starts ENTRY_CM before a junction centre on the incoming road and ends when the
car has clearly left through an arm. The movement (straight / left / right) is given to
the controller as an open map route, generated offline from the known map only
(never from a simulated trajectory). Truth is used for physics, rendering and scoring.

Generalises cnn_map_hybrid_lab/routes.py + run.py (GPT, one T junction) to every
junction and movement of the track: road centre lines meet at the junction centre;
a turn is a tangent quarter circle whose radius and small lateral offsets are swept
for the largest whole-body clearance.
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
from .run import OWN, CAMERA_PERIOD_S, timing_profile, vehicle_from_args
from .vehicle import VARIATIONS, Vehicle, PRESETS
from .world import Camera, Renderer, Track, to_world, wrap

ARM = {"N": (0.0, 1.0), "E": (1.0, 0.0), "S": (0.0, -1.0), "W": (-1.0, 0.0)}
ENTRY_CM, EXIT_CM = 45.0, 50.0
MARGIN_TARGET_CM = 1.0
EXIT_REACHED_CM = 40.0      # car axle this far out along the exit arm, aligned -> left the junction


def heading_of(d):
    return math.atan2(d[0], d[1])           # bench yaw: 0 = +y, positive = clockwise


def arm_of(d):
    return min(ARM, key=lambda k: np.hypot(*(np.array(ARM[k]) - d)))


def movement(junction, came_from, exit_kind, track=None):
    """-> (centre, d_in, d_out, exit_arm). came_from: arm the car arrives on."""
    track = track or Track()
    j = track.data["junctions"][junction]
    if came_from not in j["arms"]:
        raise ValueError(f"{junction} has no {came_from} arm")
    d_in = -np.array(ARM[came_from])
    d_out = {"straight": d_in, "left": np.array([-d_in[1], d_in[0]]), "right": np.array([d_in[1], -d_in[0]])}[exit_kind]
    exit_arm = arm_of(d_out)
    if exit_arm not in j["arms"] or exit_arm == came_from:
        raise ValueError(f"{junction}: no {exit_kind} exit when coming from {came_from}")
    return np.array(j["xy"], float), d_in, d_out, exit_arm


def _poses(points):
    d = np.gradient(points, axis=0)
    return np.c_[points, np.arctan2(d[:, 0], d[:, 1])]


def route_clearance(points, rim, track):
    P = _poses(points)
    return float(min(track.clearance(to_world(rim, p)).min() for p in P[::2]))


def make_route(junction, came_from, exit_kind, rim, track=None, radius=None):
    """Open route (N x 2 points, 1 cm spacing) + how it was generated."""
    track = track or Track()
    c, d_in, d_out, exit_arm = movement(junction, came_from, exit_kind, track)
    start, end = c - ENTRY_CM * d_in, c + EXIT_CM * d_out
    if exit_kind == "straight":
        n = int(ENTRY_CM + EXIT_CM) + 1
        pts = np.linspace(start, end, n)
        return pts, dict(method="straight centre line", junction=junction, came_from=came_from, exit=exit_kind,
                         exit_arm=exit_arm, nominal_min_margin_cm=route_clearance(pts, rim, track))
    n_in = np.array([d_in[1], -d_in[0]])            # right of the incoming direction
    n_out = np.array([d_out[1], -d_out[0]])
    cands = []
    for radius in ([float(radius)] if radius else np.arange(6.0, 30.1, 1.0)):
        for off_in in np.arange(-2.0, 2.1, 1.0):
            for off_out in np.arange(-2.0, 2.1, 1.0):
                # incoming line: c + off_in*n_in + t*d_in ; outgoing: c + off_out*n_out + u*d_out
                a, b = c + off_in * n_in, c + off_out * n_out
                A = np.c_[d_in, -d_out]
                t, u = np.linalg.solve(A, b - a)
                X = a + t * d_in                          # corner of the two lines
                p0, p1 = X - radius * d_in, X + radius * d_out
                side = 1.0 if exit_kind == "right" else -1.0
                ctr = p0 + side * radius * n_in
                a0 = math.atan2(*(p0 - ctr)[::-1]); a1 = math.atan2(*(p1 - ctr)[::-1])
                da = wrap(a1 - a0)
                arc = ctr + radius * np.c_[np.cos(a0 + np.linspace(0, 1, max(8, int(abs(da) * radius))) * da),
                                           np.sin(a0 + np.linspace(0, 1, max(8, int(abs(da) * radius))) * da)]
                s_in, s_out = np.linalg.norm(p0 - (start + off_in * n_in)), np.linalg.norm(end + off_out * n_out - p1)
                if (p0 - start) @ d_in < 0 or (end - p1) @ d_out < 0:
                    continue                          # the arc would start before the entry / end after the exit
                seg_in = np.linspace(start + off_in * n_in, p0, max(2, int(s_in)))[:-1]
                seg_out = np.linspace(p1, end + off_out * n_out, max(2, int(s_out)))[1:]
                raw = np.vstack((seg_in, arc, seg_out))
                s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))]
                q = np.arange(0, s[-1], 1.0)
                pts = np.c_[np.interp(q, s, raw[:, 0]), np.interp(q, s, raw[:, 1])]
                cands.append((route_clearance(pts, rim, track), radius, pts, off_in, off_out))
    # Closed loop (east S->left, hyb2j, seeds 17/29) the static-best tightest radius ran
    # wide at the exit: r 6 -1.97 cm, 8 -0.65, 10 -0.24, 12 -0.22/+0.10, 14 -0.01/-0.03.
    # So take the LARGEST radius whose static whole-body margin is >= MARGIN_TARGET_CM
    # (or within 0.25 cm of the best one where no radius reaches it).
    best_margin = max(c[0] for c in cands)
    need = min(MARGIN_TARGET_CM, best_margin - 0.25)
    score, radius, pts, off_in, off_out = max((c for c in cands if c[0] >= need), key=lambda c: (c[1], c[0]))
    return pts, dict(method="offline tangent quarter-circle sweep (radius 6-30, offsets -2..2): largest radius with "
                            f"static margin >= {need:.2f} cm", junction=junction,
                     came_from=came_from, exit=exit_kind, exit_arm=exit_arm, radius_cm=float(radius),
                     offset_in_cm=float(off_in), offset_out_cm=float(off_out), nominal_min_margin_cm=score)


def pivot_plan(junction, came_from, exit_kind, rim, track=None):
    """Turn in place: drive the incoming centre line to a pivot point P, rotate about the
    axle midpoint (inner wheel backwards) to the exit heading, drive out. P is searched
    near the junction centre for the largest whole-body margin over the entire sweep."""
    track = track or Track()
    c, d_in, d_out, exit_arm = movement(junction, came_from, exit_kind, track)
    h_in, h_out = heading_of(d_in), heading_of(d_out)
    sweep = h_in + np.linspace(0, 1, 46) * wrap(h_out - h_in)
    best = None
    for a in np.arange(-4.0, 4.1, 1.0):            # along the incoming road (+ = further in)
        for b in np.arange(-4.0, 4.1, 1.0):        # along the exit road
            P = c + a * d_in + b * d_out
            m = min(float(track.clearance(to_world(rim, [*P, h])).min()) for h in sweep)
            # the drive in and out runs on the lines through P: a P off the road centre
            # costs margin there (K_junction 124: P 1 cm off -> 0.25 cm on 18.5 cm roads)
            p_in = np.linspace(c - ENTRY_CM * d_in, P, int(np.linalg.norm(P - (c - ENTRY_CM * d_in))) + 1)
            end = P + (EXIT_CM + (c - P) @ d_out) * d_out
            p_out = np.linspace(P, end, int(np.linalg.norm(end - P)) + 1)
            m_lines = min(route_clearance(p_in, rim, track), route_clearance(p_out, rim, track))
            score = min(m, m_lines)
            if best is None or score > best[0]:
                best = (score, m, m_lines, P, p_in, p_out)
    score, m, m_lines, P, p_in, p_out = best
    return dict(P=P.tolist(), margin_cm=score, sweep_margin_cm=m, lines_margin_cm=m_lines, h_in=h_in, h_out=h_out,
                pts_in=p_in, pts_out=p_out)


def turn_plans(junction, came_from, exit_kind, rim, track=None):
    """Everything a controller may choose from for one movement: the best arc per radius
    (offsets swept) and the pivot plan. The choice needs the car's turning ability, which
    only the controller knows (v4 learns its turn gain while driving)."""
    track = track or Track()
    if exit_kind == "straight":
        return None
    arcs = []
    for r in np.arange(6.0, 30.1, 2.0):
        pts, g = make_route(junction, came_from, exit_kind, rim, track, radius=r)
        arcs.append((float(r), g["nominal_min_margin_cm"], pts))
    return dict(arcs=arcs, pivot=pivot_plan(junction, came_from, exit_kind, rim, track), exit=exit_kind)


PLAN_CACHE = Path(__file__).resolve().parent / "world" / "junction_plans.pkl"


def build_plan_cache(path=PLAN_CACHE, rim=None):
    """Every movement's default route + turn plans, computed once (the Pi would need
    seconds per junction). Keyed by the track file hash and the body outline."""
    import hashlib
    import pickle
    track = Track()
    rim = Vehicle(PRESETS["field_estimate"], np.zeros(3)).footprint() if rim is None else rim
    plans = {}
    for jn, j in track.data["junctions"].items():
        for cf in j["arms"]:
            for ex in ("straight", "left", "right"):
                try:
                    route, gen = make_route(jn, cf, ex, rim, track)
                except ValueError:
                    continue
                plans[(jn, cf, ex)] = (route, turn_plans(jn, cf, ex, rim, track), gen)
                print(jn, cf, ex, "done", flush=True)
    meta = dict(track_sha256=hashlib.sha256((Path(__file__).resolve().parent / "world" / "track.json").read_bytes()).hexdigest(),
                rim_sha256=hashlib.sha256(np.ascontiguousarray(rim).tobytes()).hexdigest())
    Path(path).write_bytes(pickle.dumps(dict(meta=meta, plans=plans)))
    return path


def load_plan_cache(path=PLAN_CACHE):
    import hashlib
    import pickle
    if not Path(path).exists():
        return None
    d = pickle.loads(Path(path).read_bytes())
    track_hash = hashlib.sha256((Path(__file__).resolve().parent / "world" / "track.json").read_bytes()).hexdigest()
    return d["plans"] if d["meta"]["track_sha256"] == track_hash else None


class JunctionScorer:
    """Truth only. Ends on a clear exit through an arm (correct / wrong), a severe
    departure (> 12 cm over a line, or the whole body off the road), or 5 s of zero output."""

    def __init__(self, track, rim, junction, d_out, exit_arm, route):
        self.track, self.rim = track, rim
        self.centre = np.array(track.data["junctions"][junction]["xy"], float)
        self.arms = track.data["junctions"][junction]["arms"]
        self.exit_arm, self.route = exit_arm, route
        self.s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
        self.min_margin, self.outside_s, self.zero_run, self.end = np.inf, 0.0, 0.0, None
        self.episode, self.episodes, self.inside_since = None, [], None
        self.progress, self.observed_exit, self.t = 0.0, None, 0.0

    def sample(self, t, dt, pose, pwm):
        self.t = t
        d = self.track.clearance(to_world(self.rim, pose))
        m = float(d.min())
        self.min_margin = min(self.min_margin, m)
        self.progress = float(self.s[int(np.argmin(np.linalg.norm(self.route - pose[:2], axis=1)))])
        if m < 0:
            self.outside_s += dt
            self.inside_since = None
            if self.episode is None:
                self.episode = dict(start_s=t, start_progress_cm=self.progress, min_cm=m)
            self.episode["min_cm"] = min(self.episode["min_cm"], m)
        elif self.episode is not None:
            self.inside_since = t if self.inside_since is None else self.inside_since
            if t - self.inside_since >= 1.0 - 1e-9:
                self.episode.update(end_s=t, recovered=True)
                self.episodes.append(self.episode)
                self.episode, self.inside_since = None, None
        self.zero_run = self.zero_run + dt if np.max(np.abs(pwm)) < 1e-8 else 0.0
        rel = pose[:2] - self.centre
        for arm in ARM:
            a = np.array(ARM[arm])
            if rel @ a >= EXIT_REACHED_CM and abs(wrap(pose[2] - heading_of(a))) < math.radians(30) \
                    and abs(rel @ np.array([a[1], -a[0]])) < 15:
                self.observed_exit = arm
        if d.max() < 0 or m < -12:
            self.end = "severe_departure"
        elif self.observed_exit and self.episode is None:
            self.end = "correct_exit" if self.observed_exit == self.exit_arm else "wrong_exit"
        elif self.zero_run >= 5.0:
            self.end = "held"

    def summary(self, outcome):
        if self.episode is not None:
            self.episode.update(end_s=self.t, recovered=False)
            self.episodes.append(self.episode)
            self.episode = None
        return dict(outcome=outcome, observed_exit=self.observed_exit, expected_exit=self.exit_arm,
                    correct_exit=outcome == "correct_exit", clean=outcome == "correct_exit" and self.min_margin >= 0,
                    min_margin_cm=self.min_margin, outside_s=self.outside_s, simulated_s=self.t,
                    progress_cm=self.progress, episodes=self.episodes,
                    unrecovered_episodes=sum(not e["recovered"] for e in self.episodes))


def run(controller, params, effects, junction, came_from, exit_kind, timing, seed, seconds, out,
        variation="fixed", video=False, radius=None):
    from .controllers import CONTROLLERS
    cv2.setNumThreads(1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    profile = timing_profile(OWN[controller] if timing == "own" else {"fast": "v4_pi", "slow": "v3r_laptop"}[timing])
    track, camera = Track(), Camera()
    renderer = Renderer(track, camera)
    rim = Vehicle(params, np.zeros(3)).footprint()
    route, generation = make_route(junction, came_from, exit_kind, rim, track, radius)
    c, d_in, d_out, exit_arm = movement(junction, came_from, exit_kind, track)
    hint = np.array([*route[0], heading_of(d_in)])
    rng = np.random.default_rng(seed)
    truth0 = hint + np.array([*(rng.uniform(-1, 1) * np.array([d_in[1], -d_in[0]]) + rng.uniform(-2, 2) * d_in),
                              np.radians(rng.uniform(-3, 3))])
    var = replace(VARIATIONS[variation], seed=seed)
    car = Vehicle(params, truth0, var)
    scorer = JunctionScorer(track, car.footprint(), junction, d_out, exit_arm, route)
    dist = Disturbances(combine(effects), "ccw", seed)
    dist.arm(renderer)
    trigger = 25.0                         # disturbances start 25 cm into the scene (as the GPT T test)
    cls = CONTROLLERS[controller]
    if getattr(cls, "takes_route", False):
        ctrl = cls("ccw", hint, profile, route=route, plans=turn_plans(junction, came_from, exit_kind, rim, track))
    else:
        ctrl = cls("ccw", hint, profile)
    t, last_pwm, history = 0.0, np.zeros(2), [(0.0, truth0.copy())]
    writer = cv2.VideoWriter(str(out / "view.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (640, 480)) if video else None

    def advance_to(target):
        nonlocal t
        while t < target - 1e-10 and scorer.end is None:
            dt = min(0.02, target - t)
            dist.update(t, scorer.progress, trigger)
            car.step(t, dt, dist.speed_scale(t))
            t += dt
            scorer.sample(t, dt, car.pose, last_pwm)
            history.append((t, car.pose.copy()))
            if len(history) > 400:
                del history[:200]

    capture, seq, wall, pose_errors, realcam = CAMERA_PERIOD_S, 0, time.perf_counter(), [], None
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
            if realcam is not None:
                flags = flags + realcam.flags
            delay = profile["latencies_s"][seq % len(profile["latencies_s"])]
            done = capture + delay
            advance_to(done)
            if scorer.end:
                break
            pwm, info = ctrl.step(image, capture, done)
            if info.get("est") is not None:
                info["pose_err_cm"] = float(np.linalg.norm(np.asarray(info["est"][:2]) - car.pose[:2]))
                pose_errors.append(info["pose_err_cm"])
            last_pwm = np.asarray(pwm, float)
            car.command(done, last_pwm)
            log.write(json.dumps(dict(t=done, pwm=last_pwm.tolist(), truth=car.pose.tolist(), progress=scorer.progress,
                                      flags=flags, cam=realcam.stats if realcam else None, **info), default=float) + "\n")
            if writer:
                writer.write(image)
            capture = max(capture + CAMERA_PERIOD_S, done)
            seq += 1
    if writer:
        writer.release()
    s = scorer.summary(scorer.end or "time_limit")
    s.update(controller=controller, controller_source=getattr(ctrl, "source", controller), junction=junction,
             came_from=came_from, exit=exit_kind, route=generation, turn_choice=getattr(ctrl, "choice", None), effects=list(effects), seed=seed,
             timing=profile | {"latencies_s": None}, vehicle=params.to_dict(), frames=seq,
             variation=dict(name=variation, **var.to_dict()), initial_truth=truth0.tolist(), hint=hint.tolist(),
             pose_error_cm=dict(mean=float(np.mean(pose_errors)), p95=float(np.percentile(pose_errors, 95)),
                                max=float(np.max(pose_errors))) if pose_errors else None,
             camera=dict(profile=realcam.profile, drawn={k: round(v, 3) for k, v in realcam.cfg.items()}) if realcam else None,
             wall_s=time.perf_counter() - wall)
    (out / "summary.json").write_text(json.dumps(s, indent=2, default=float), encoding="utf-8")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controller", choices=sorted(OWN), required=True)
    ap.add_argument("--junction", default="east")
    ap.add_argument("--came-from", default="S", help="arm the car arrives on (N/E/S/W)")
    ap.add_argument("--exit", default="left", choices=["straight", "left", "right"])
    ap.add_argument("--vehicle", default="field_estimate")
    ap.add_argument("--vehicle-json", type=json.loads)
    ap.add_argument("--variation", choices=sorted(VARIATIONS), default="fixed")
    ap.add_argument("--effects", default="")
    ap.add_argument("--timing", choices=["own", "fast", "slow"], default="own")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--radius", type=float, help="fix the turn radius (cm) instead of sweeping it")
    ap.add_argument("--build-plans", action="store_true", help="precompute every movement into bench/world/junction_plans.pkl")
    ap.add_argument("--routes", action="store_true", help="only print every movement's generated route")
    ap.add_argument("--out")
    a = ap.parse_args()
    params = vehicle_from_args(a.vehicle, a.vehicle_json)
    if a.build_plans:
        print(build_plan_cache())
        return
    if a.routes:
        rim, track = Vehicle(params, np.zeros(3)).footprint(), Track()
        for jn, j in track.data["junctions"].items():
            for cf in j["arms"]:
                for ex in ("straight", "left", "right"):
                    try:
                        _, g = make_route(jn, cf, ex, rim, track)
                    except ValueError:
                        continue
                    print(jn, cf, ex, "->", g["exit_arm"], "margin %.2f" % g["nominal_min_margin_cm"],
                          "r", g.get("radius_cm"), "off", g.get("offset_in_cm"), g.get("offset_out_cm"))
        return
    s = run(a.controller, params, [e for e in a.effects.split(",") if e], a.junction, a.came_from, a.exit,
            a.timing, a.seed, a.seconds, a.out, a.variation, a.video, a.radius)
    print(json.dumps({k: s[k] for k in ("controller", "junction", "came_from", "exit", "outcome", "observed_exit",
                                         "min_margin_cm", "outside_s", "simulated_s")}, default=float))


if __name__ == "__main__":
    main()
