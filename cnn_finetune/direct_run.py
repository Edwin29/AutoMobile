"""Run one driver in the common bench.

    python -m bench.run --controller v4 --vehicle field_estimate --direction ccw --out DIR
    python -m bench.run --controller v3r --vehicle random:7 --effects glare,low_power --timing own --out DIR
    python -m bench.run --controller v4 --vehicle-json '{"turn_eff_left":0.5}' --out DIR

World: fixed (bench/world).  Vehicle: --vehicle preset | random:<seed> | --vehicle-json overrides.
Timing: own = each driver's measured processing profile; fast / slow = force one profile on both.
Loop: camera captures when the driver is free (30 fps max); the previous command keeps acting
while the frame is processed; the new command is applied at capture + latency.
"""
import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from bench.disturb import Disturbances, combine
from bench.realcam import pose_from_history
from bench.score import Scorer
from bench.vehicle import PRESETS, VARIATIONS, Vehicle, VehicleParams, random_params
from bench.world import Camera, Renderer, Track

HERE = Path(__file__).resolve().parents[1] / 'bench'
CAMERA_PERIOD_S = 1 / 30


def timing_profile(name):
    if name == "v4_pi":
        lat = [0.070]                               # Pi sensor->write median, v4 live 2026-11-10
        src = "drive_v4 measured on the Pi (sensor->write median 70 ms)"
    elif name == "cnn_pi":
        lat = [0.036]                               # team CNN on the Pi: capture->motor p50 34-38 ms
        src = "team CNN measured on the Pi (track-digital-twin README, p50 34-38 ms)"
    elif name == "v3r_laptop":
        trace = json.loads((HERE / "profiles/v3r_laptop_processing.json").read_text(encoding="utf-8"))
        lat = [v / 1000 + trace["assumed_sensor_delivery_s"] for v in trace["total_ms"]]
        src = "v3-recovery laptop processing trace + assumed 50 ms delivery (NOT a Pi measurement)"
    else:
        raise ValueError(name)
    worst = max(lat)
    return dict(name=name, latencies_s=lat, max_latency_s=worst, source=src,
                controller_period_s=0.2 if worst <= 0.2 else 0.6)


OWN = {"v4": "v4_pi", "v3r": "v3r_laptop", "cnnC5b": "cnn_pi", "cnnA5b": "cnn_pi", "cnnB": "cnn_pi",
       "cnnC": "cnn_pi", "cnnA": "cnn_pi", "hyb2": "cnn_pi", "hyb3": "cnn_pi", "hyb2j": "cnn_pi", "hyb2m": "cnn_pi"}   # hyb2/3: map on another core, CNN path only


def vehicle_from_args(vehicle, overrides):
    if vehicle.startswith("random:"):
        p = random_params(int(vehicle.split(":")[1]))
    else:
        p = PRESETS[vehicle]
    return replace(p, **(overrides or {}))


def run(controller, params, effects, direction, timing, seed, seconds, out, video=False, variation="fixed", variant=0):
    from cnn_finetune.direct_teacher import DirectTeacher as Teacher
    cv2.setNumThreads(1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    profile = timing_profile(OWN[controller] if timing == "own" else {"fast": "v4_pi", "slow": "v3r_laptop"}[timing])
    track, camera = Track(), Camera()
    renderer = Renderer(track, camera)
    rng = np.random.default_rng(seed)
    hint = track.start_pose(direction, params.front_cm, params.rear_cm)
    truth0 = hint + np.array([rng.uniform(-1, 1), rng.uniform(-2, 2), np.radians(rng.uniform(-3, 3))])  # hand placement
    var = replace(VARIATIONS[variation], seed=seed)          # how the car changes during the run
    car = Vehicle(params, truth0, var)
    scorer = Scorer(track, car.footprint(), camera.route, direction, truth0)
    dist = Disturbances(combine(effects), direction, seed)
    dist.arm(renderer)
    trigger = scorer.first_corner_progress - (30.0 if "seam" in effects else 4.0)
    if "trigger_cm" in dist.p:                 # e.g. white-out while already turning
        trigger = scorer.first_corner_progress + dist.p["trigger_cm"]
    ctrl = Teacher(direction, variant=variant)
    ctrl.sim_params = params
    ctrl.source = 'truth-pose teacher 18cm pursuit; simulation only'
    ctrl.raw_camera = True
    t, last_pwm, history = 0.0, np.zeros(2), [(0.0, truth0.copy())]
    outcome = "time_limit"
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

    capture, seq, wall, pose_errors = CAMERA_PERIOD_S, 0, time.perf_counter(), []
    realcam = None
    with (out / "steps.jsonl").open("w", encoding="utf-8") as log:
        while capture < seconds:
            advance_to(capture)
            if scorer.end:
                break
            raw = getattr(ctrl, "raw_camera", False)
            if dist.p.get("realcam"):                  # realistic camera (bench/realcam.py, GPU if present)
                if realcam is None:
                    from bench.realcam import RealCam
                    realcam = RealCam(renderer, camera, dist.p["realcam"], seed)
                frame = realcam.frame(capture, pose_from_history(history), raw=raw)
            else:
                frame = renderer.render_raw(car.pose) if raw else renderer.render(car.pose)
            image, flags = dist.image(frame, capture, renderer, history, raw=raw)
            if realcam is not None:
                flags = flags + realcam.flags
                info_cam = realcam.stats
            else:
                info_cam = None
            captured_pose = car.pose.copy()
            delay = profile["latencies_s"][seq % len(profile["latencies_s"])]
            done = capture + delay
            advance_to(done)                       # previous command acts while processing
            if scorer.end:
                break
            pwm, info = ctrl.step(captured_pose)
            if info.get("est") is not None:            # belief vs truth: analysis only
                info["pose_err_cm"] = float(np.linalg.norm(np.asarray(info["est"][:2]) - car.pose[:2]))
                pose_errors.append(info["pose_err_cm"])
            last_pwm = np.asarray(pwm, float)
            car.command(done, last_pwm)
            e = car.now_params
            log.write(json.dumps(dict(t=done, pwm=last_pwm.tolist(), truth=car.pose.tolist(),
                                      progress=scorer.progress, flags=flags, cam=info_cam,
                                      car=dict(turn_l=e.get("turn_left"), turn_r=e.get("turn_right"),
                                               max_duty=e.get("max_duty"),
                                               speed=float(np.mean(e["gain"])) if e else None),
                                      **info), default=float) + "\n")
            if writer:
                writer.write(image)
            capture = max(capture + CAMERA_PERIOD_S, done)
            seq += 1
    if writer:
        writer.release()
    s = scorer.summary(scorer.end or outcome)
    s.update(controller=controller, controller_source=ctrl.source, direction=direction, effects=list(effects),
             seed=seed, timing=profile | {"latencies_s": None}, vehicle=params.to_dict(), frames=seq,
             variation=dict(name=variation, **var.to_dict()), floor_patches=car.patches,
             pose_error_cm=dict(mean=float(np.mean(pose_errors)), p95=float(np.percentile(pose_errors, 95)),
                                max=float(np.max(pose_errors))) if pose_errors else None,
             camera=dict(profile=realcam.profile, device=str(realcam.dev),
                         drawn={k: round(v, 3) for k, v in realcam.cfg.items()}) if realcam else None,
             wall_s=time.perf_counter() - wall)
    (out / "summary.json").write_text(json.dumps(s, indent=2, default=float), encoding="utf-8")
    return s


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--variant', type=int, default=0)
    ap.add_argument('--vehicles', default='field_estimate,weak_left_turn,sluggish,unequal_sides')
    ap.add_argument('--directions', default='cw,ccw')
    ap.add_argument('--effects', default='realcam')
    a = ap.parse_args()
    summaries = []
    root = Path(a.out)
    for vehicle in a.vehicles.split(','):
        for effect in a.effects.split(','):
            for direction in a.directions.split(','):
                name = vehicle.replace(':','_')+'_'+effect+'_'+direction
                result = run('cnnA5b', vehicle_from_args(vehicle, None), [effect], direction, 'own', 41, 200, root/name, variant=a.variant)
                result['vehicle_name'] = vehicle
                summaries.append(result)
                print(json.dumps(dict(name=name, verified=result['verified_lap'], margin=result['min_margin_cm'], outcome=result['outcome'])), flush=True)
                (root/'matrix.json').write_text(json.dumps(summaries,indent=2), encoding='utf-8')
    clean = sum(s['verified_lap'] and s['min_margin_cm'] >= 0 for s in summaries)
    print(json.dumps(dict(clean=clean,total=len(summaries))),flush=True)
