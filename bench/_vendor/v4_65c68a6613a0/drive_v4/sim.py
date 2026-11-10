"""Closed-loop simulation of drive_v4 on drive_v3's synthetic camera, with video.

Truth lives only here (plant + renderer); the driver sees rendered images and its
own applied PWM, exactly like on the Pi. Plant parameters are deliberately NOT
the driver's model parameters.

    python -m drive_v4.sim --scenario start        # normal start in the start zone
    python -m drive_v4.sim --scenario resume_wrong # restart at the NW corner, prior still says start zone
    python -m drive_v4.sim --scenario resume       # same place, started with --resume (correct prior)

Limits: flat raster world, no shadows/glare/3D, ideal rectified camera, simple
skid model. It shows decision logic, not the real tyre/motor behaviour.
"""
import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from drive_v3.camera import CameraGeometry
from drive_v3.geometry import advance, footprint, to_world, wrap
from drive_v3.map_model import HERE as V3, TrackMap
from drive_v3.simulation import Renderer
from drive_v3.start_region import right_red_prior
from drive_v3.settings import load_settings

from .core import MAP_PATH, Driver, load_config

SCENARIOS = {
    # true start arc on the ccw route, driver prior ('zone' = start-zone prior)
    "start": dict(true_arc=None, prior="zone"),
    "resume_wrong": dict(true_arc=628.0, prior="zone"),
    "resume": dict(true_arc=628.0, prior="truth"),
}


class Plant:
    """Independent truth model: deadband, per-direction yaw slip, one-frame delay.

    yaw_drift_to: turn efficiency moves linearly to this value over drift_s seconds
    (e.g. battery sag / surface change), to test online learning."""

    def __init__(self, deadband=0.28, cm_s_per_duty=16.8, track_cm=13.0, yaw_left=0.6, yaw_right=0.6,
                 left_bias=0.0, yaw_drift_to=None, drift_s=100.0):
        self.deadband, self.gain, self.track = deadband, cm_s_per_duty, track_cm
        self.yaw_left0, self.yaw_right0 = yaw_left, yaw_right
        self.left_bias, self.yaw_drift_to, self.drift_s = left_bias, yaw_drift_to, drift_s
        self.t = 0.0

    def efficiency(self, omega):
        base = self.yaw_left0 if omega < 0 else self.yaw_right0
        if self.yaw_drift_to is None:
            return base
        f = min(1.0, self.t / self.drift_s)
        return base + f * (self.yaw_drift_to - base)

    def step(self, pose, pwm, dt):
        self.t += dt
        wheel = np.where(np.asarray(pwm) >= self.deadband, pwm, 0.0) * self.gain
        v = wheel.mean()
        omega = (wheel[0] - wheel[1]) / self.track
        omega *= self.efficiency(omega)
        if v > 0:
            omega -= self.left_bias
        return advance(pose, v, omega, dt)


class TopView:
    def __init__(self, renderer, route, size=480):
        self.r, self.size = renderer, size
        tex = renderer.texture  # 1001x1001, origin -125cm, 4 px/cm, y down = +y world
        self.scale = size / tex.shape[0]
        base = cv2.flip(cv2.resize(tex, (size, size), interpolation=cv2.INTER_AREA), 0)  # north up
        pts = self.px(route.points)
        cv2.polylines(base, [pts], True, (120, 120, 120), 1, cv2.LINE_AA)
        self.base = base
        self.trail = []

    def px(self, xy):
        xy = np.asarray(xy, float)
        p = (xy - self.r.origin) * self.r.ppcm * self.scale
        p[:, 1] = self.size - 1 - p[:, 1]  # north up
        return np.rint(p).astype(np.int32).reshape(-1, 1, 2)

    def draw(self, truth, estimate, w):
        img = self.base.copy()
        self.trail.append(truth[:2].copy())
        if len(self.trail) > 1:
            cv2.polylines(img, [self.px(self.trail)], False, (255, 255, 255), 1, cv2.LINE_AA)
        box = np.array([[-8, -10], [8, -10], [8, 14.5], [-8, 14.5]], float)
        cv2.fillPoly(img, [self.px(to_world(box, truth))], (60, 200, 60))
        cv2.polylines(img, [self.px(to_world(box, truth))], True, (0, 90, 0), 1, cv2.LINE_AA)
        color = (0, 140, 255) if w > 0.3 else (0, 0, 255)
        cv2.polylines(img, [self.px(to_world(box, estimate))], True, color, 2, cv2.LINE_AA)
        nose = self.px(to_world([[0, 0], [0, 20]], truth)).reshape(-1, 2)
        cv2.arrowedLine(img, tuple(nose[0]), tuple(nose[1]), (0, 90, 0), 1, tipLength=.3)
        return img


# "before" = v4 as driven on 2026-11-10 (no turn-efficiency model, point preview,
# no lateral term in corners, no departure guard, no corner slow-down).
BEFORE = {"motor": {"yaw_gain_left": 1.0, "yaw_gain_right": 1.0, "corner_pwm_max": 0.6},
          "yaw_learn": {"enabled": False},
          "steer": {"preview_start_cm": 3.0, "preview_len_cm": 0.0, "lat_lookahead_cm": 1e6},
          "guard": {"clearance_cm": -100.0}}


def corner_margins(records, route):
    """Minimum body margin inside each corner (route |k| > 0.02), in driving order."""
    out, current = [], None
    for r in records:
        s = route.s[route.nearest(np.asarray(r["truth"][:2]))]
        k = abs(np.interp(s, route.s, route.curvature))
        if k > 0.02:
            current = r["margin_cm"] if current is None else min(current, r["margin_cm"])
        elif current is not None:
            out.append(round(current, 2)); current = None
    if current is not None:
        out.append(round(current, 2))
    return out


def run(scenario, out, seconds=150.0, fps=15.0, video=True, plant_kw=None, laps=1.0, variant="after",
        cam_scale=1.0, cam_offset=0.0, tag=None, override=None):
    cfg = load_config()
    for k, v in (override or {}).items():
        cfg[k].update(v)
    if variant == "before":
        for k, v in BEFORE.items():
            cfg[k].update(v)
    tag = tag or f"{scenario}_{variant}"
    v3, _ = load_settings()
    track = TrackMap(MAP_PATH)
    camera = CameraGeometry(V3 / "config/camera_legacy_track.json")
    renderer = Renderer(track, camera)
    # Camera projection error (field: 26cm roads measured 24.4-25.6cm, lateral bias
    # 1-5cm): the pixel the driver believes is at body x sees the world at
    # x*cam_scale + cam_offset.
    renderer.ground = renderer.ground * np.array([cam_scale, 1.0]) + np.array([cam_offset, 0.0])
    route = track.outer_route("ccw")
    spec = SCENARIOS[scenario]
    zone_pose, _, _ = right_red_prior(track, v3["vehicle"], "ccw")
    if spec["true_arc"] is None:
        truth = zone_pose.copy() + np.array([0.8, 1.5, math.radians(3)])  # hand placement error
    else:
        xy = route.sample(spec["true_arc"])[0]
        truth = np.r_[xy, route.heading[route.nearest(xy)]]
    prior = zone_pose if spec["prior"] == "zone" else truth + np.array([1.0, -1.0, 0.03])
    driver = Driver(cfg, direction="ccw", pose=prior, clock="sim")
    plant = Plant(**(plant_kw or {}))
    rim = footprint(v3["vehicle"])
    top = TopView(renderer, route)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    writer = None
    if video:
        writer = cv2.VideoWriter(str(out / f"{tag}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1120, 480))
    dt = 1.0 / fps
    applied = np.zeros(2)
    progress, last_s = 0.0, route.s[route.nearest(truth[:2])]
    records, min_margin, outcome = [], float("inf"), "time_limit"
    off_since = None
    start = time.perf_counter()
    with (out / f"{tag}.jsonl").open("w", encoding="utf-8") as log:
        for i in range(int(seconds * fps)):
            now = (i + 1) * dt
            truth = plant.step(truth, applied, dt)              # previous command acts during this frame
            image = renderer.render(truth)
            pwm, rec = driver.step(image, now - 0.03, now, rotate180=False, rectified=True)
            driver.applied(now, pwm)
            applied = pwm
            margin = float(track.clearance(to_world(rim, truth)).min())
            min_margin = min(min_margin, margin)
            s = route.s[route.nearest(truth[:2])]
            d = (s - last_s + route.s[-1] / 2) % route.s[-1] - route.s[-1] / 2
            progress += d if abs(d) < 20 else 0.0
            last_s = s
            est = driver.map.pose_at(now)
            rec.update(t=now, truth=truth.tolist(), margin_cm=margin, progress_cm=progress,
                       map_error_cm=float(np.linalg.norm(est[:2] - truth[:2])),
                       map_arc_error_cm=float(((driver.route.s[driver.route.nearest(est[:2])] - s + 323) % 646) - 323))
            log.write(json.dumps(rec, default=float) + "\n")
            records.append(rec)
            if writer:
                cam = image.copy()
                loc = getattr(driver, "last_local", None)
                if loc is not None and loc.reason == "tracking" and len(loc.points_body_cm) > 1:
                    # Draw only floor the camera can see. Points closer than ~16cm are
                    # under/behind the lens and project to meaningless pixels (they made
                    # the line jump across the image in the first videos).
                    pts = np.asarray(loc.points_body_cm)
                    pts = pts[pts[:, 1] >= camera.lens_ahead + 4.0]
                    if len(pts) > 1:
                        px = camera.body_to_rectified(pts)
                        cv2.polylines(cam, [px.astype(np.int32)], False, (255, 0, 255), 2)
                panel = np.zeros((480, 1120, 3), np.uint8)
                panel[:, :640] = cv2.resize(cam, (640, 480))
                panel[:, 640:] = top.draw(truth, est, rec["w_map"])
                lines = [f"{tag}  t={now:5.1f}s  progress {progress:5.0f}cm",
                         f"{rec['reason']} / {rec['source']}  map:{rec['map_status']}",
                         f"w={rec['w_map']:.2f} trust={rec['map_trust']:.2f} corner={rec['corner']:.2f}",
                         f"k={rec['k']:+.3f} local={rec['k_local'] if rec['k_local'] is None else round(rec['k_local'], 3)}"
                         f" map={rec['k_map'] if rec['k_map'] is None else round(rec['k_map'], 3)}",
                         f"PWM L{pwm[0]:.2f} R{pwm[1]:.2f}  margin {margin:+.1f}cm",
                         f"turn gain L{rec['turn_gain']['left']:.2f} R{rec['turn_gain']['right']:.2f}"
                         f" (true L{plant.efficiency(-1):.2f} R{plant.efficiency(1):.2f})",
                         f"map error {rec['map_error_cm']:.1f}cm (along {rec['map_arc_error_cm']:+.0f})"]
                for j, text in enumerate(lines):
                    cv2.putText(panel, text, (8, 22 + j * 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(panel, text, (8, 22 + j * 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(panel, "green=true car  orange=map estimate (red: low weight)  magenta=camera lane",
                            (648, 470), cv2.FONT_HERSHEY_SIMPLEX, .38, (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(panel)
            if margin < -8.0:   # simulator stop only: body fully off the road
                off_since = off_since or now
                if now - off_since > 1.0:
                    outcome = "left_road"
                    break
            else:
                off_since = None
            if progress >= laps * route.s[-1]:
                outcome = "lap_complete"
                break
    if writer:
        writer.release()
    reasons = {}
    for r in records:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    summary = dict(scenario=scenario, variant=variant, cam_scale=cam_scale, cam_offset=cam_offset, outcome=outcome, sim_s=records[-1]["t"], progress_cm=progress,
                   route_cm=float(route.s[-1]), min_margin_cm=min_margin,
                   frames_margin_below_0=sum(r["margin_cm"] < 0 for r in records),
                   frames_margin_below_minus3=sum(r["margin_cm"] < -3 for r in records),
                   reasons=reasons, wall_s=time.perf_counter() - start, corner_min_margin_cm=corner_margins(records, route),
                   plant=dict(plant.__dict__), turn_gain_end=records[-1]["turn_gain"],
                   turn_gain_updates=sum(1 for r in records if r["turn_gain"]["event"] and not r["turn_gain"]["event"].get("rejected")),
                   turn_gain_rejected=sum(1 for r in records if r["turn_gain"]["event"] and r["turn_gain"]["event"].get("rejected")), video=str(out / f"{tag}.mp4") if video else None)
    (out / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="start")
    ap.add_argument("--seconds", type=float, default=150)
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "sim_runs"))
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--variant", choices=["before", "after"], default="after")
    ap.add_argument("--override", type=json.loads, help='e.g. {"guard": {"gain": 0.02}}')
    ap.add_argument("--tag")
    ap.add_argument("--cam-scale", type=float, default=1.0)
    ap.add_argument("--cam-offset", type=float, default=0.0)
    ap.add_argument("--yaw-left", type=float, default=0.6, help="true left-turn efficiency")
    ap.add_argument("--yaw-right", type=float, default=0.6, help="true right-turn efficiency")
    ap.add_argument("--yaw-drift-to", type=float, help="true efficiency drifts to this over --drift-s")
    ap.add_argument("--drift-s", type=float, default=100.0)
    ap.add_argument("--deadband", type=float, default=0.28)
    ap.add_argument("--left-bias", type=float, default=0.0)
    a = ap.parse_args()
    run(a.scenario, a.out, a.seconds, a.fps, not a.no_video,
        dict(deadband=a.deadband, yaw_left=a.yaw_left, yaw_right=a.yaw_right, left_bias=a.left_bias,
             yaw_drift_to=a.yaw_drift_to, drift_s=a.drift_s),
        variant=a.variant, cam_scale=a.cam_scale, cam_offset=a.cam_offset, tag=a.tag, override=a.override)


if __name__ == "__main__":
    main()
