"""Run drive_v4 inside the v3-recovery field_stress simulator (same plant, same
disturbances, same rendering, same scoring) for a like-for-like comparison.

The v3-recovery worktree is READ-ONLY here: its modules are imported, never edited.
Its own v3-recovery results are read from its run folders.

    python -m drive_v4.compare_v3r --scenario baseline --direction ccw --timing same --out DIR
    timing=same : v3r's 0.6s period + its laptop processing-delay trace (pure algorithm comparison)
    timing=own  : v4's measured Pi timing (~15 fps, ~70ms capture->command)
"""
import argparse
import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path

LAB = Path(__file__).resolve().parents[3] / "v3_recovery_lab"
sys.path.insert(0, str(LAB))                                  # drive_v3 + research = v3-recovery (read-only)
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))  # drive_v4

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from drive_v3.camera import CameraGeometry  # noqa: E402
from drive_v3.field_stress import SCENARIOS, STUDY, StressPlant, render_input  # noqa: E402
from drive_v3.geometry import footprint, wrap  # noqa: E402
from drive_v3.map_delay_experiment import EXPERIMENT, MAPS, completion, initial_for_map  # noqa: E402
from drive_v3.map_model import HERE, TrackMap  # noqa: E402
from drive_v3.settings import load_settings  # noqa: E402
from drive_v4.core import Driver, load_config  # noqa: E402
from research.recovery_comparison.summarize import corner_events  # noqa: E402


def run(out, scenario, direction, timing, seed=17, seconds=240.0):
    cv2.setNumThreads(1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    profile = copy.deepcopy(SCENARIOS[scenario])
    evidence = json.loads((STUDY / "inputs/field_evidence.json").read_text(encoding="utf-8"))
    track = TrackMap(MAPS["corrected"])           # identical geometry to drive_v4/config/map_v4.json
    route = track.outer_route(direction)
    initial, index = initial_for_map(track, route)
    if timing == "same":
        period = 0.6
        trace = json.loads((EXPERIMENT / "inputs/processing_delay.json").read_text(encoding="utf-8"))
        delays = [v / 1000 + trace["assumed_sensor_delivery_s"] for v in trace["total_ms"]]
    else:
        period, delays = 1 / 15, [0.070]
    camera = CameraGeometry(HERE / "config/camera_legacy_track.json")
    from drive_v3.simulation import Renderer
    renderer = Renderer(track, camera)
    plant = StressPlant(initial, track, route, footprint(load_settings()[0]["vehicle"]), index, profile,
                        direction, seed, (evidence["left_turn_proxy"], evidence["right_turn_proxy"]))
    if profile.get("seam"):
        y = -50.0 if direction == "ccw" else 50.0
        pixel = lambda p: tuple(np.rint((np.array(p) - renderer.origin) * renderer.ppcm).astype(int))
        cv2.line(renderer.texture, pixel([-103.0, y]), pixel([-65.0, y]), (180, 190, 190), 3)
    driver = Driver(load_config(), direction=direction, pose=initial, clock="field_stress_sim")
    capture, seq, records, zero_since, outcome = period, 0, [], None, "time_limit"
    start = time.perf_counter()
    with (out / "steps.jsonl").open("w", encoding="utf-8") as f:
        while capture < seconds:
            plant.advance_to(capture)
            if plant.end:
                outcome = plant.end
                break
            image, flags = render_input(renderer, plant, capture, profile, seed)
            done, next_capture, delay = completion(capture, period, seq, delays)
            if done > seconds:
                plant.advance_to(seconds)
                break
            plant.advance_to(done)                     # old command continues while processing
            if plant.end:
                outcome = plant.end
                break
            pwm, rec = driver.step(image, capture, done, rotate180=False, rectified=True)
            plant.pwm = np.asarray(pwm, float)
            driver.applied(done, pwm)
            est = driver.map.est.primary.mean
            rec.update(truth=plant.pose.tolist(), true_margin_cm=plant.rows[-1]["true_margin_cm"],
                       progress_cm=plant.progress, pose_error_cm=float(np.linalg.norm(plant.pose[:2] - est[:2])),
                       camera_flags=flags, event_age_s=plant.age(), modeled_latency_s=delay)
            rec.pop("pose_now", None)
            records.append(rec)
            f.write(json.dumps(rec, default=float) + "\n")
            if np.max(plant.pwm) < 1e-8:
                zero_since = done if zero_since is None else zero_since
                if done - zero_since >= 5.0:
                    outcome = "held"
                    break
            else:
                zero_since = None
            capture, seq = next_capture, seq + 1
    if plant.episode:
        plant.episode.update(end_s=plant.t, recovered=False)
        plant.episodes.append(plant.episode)
    corners = corner_events(plant.rows, direction)
    proper = sum(c["proper_exit"] for c in corners)
    summary = dict(controller="drive_v4", timing=timing, period_s=period, scenario=scenario, direction=direction, seed=seed,
                   outcome=outcome,
                   verified_lap=outcome == "lap_complete" and proper == 4 and not any(not e["recovered"] for e in plant.episodes),
                   simulated_s=plant.t, progress_cm=plant.progress, proper_corner_exits=proper,
                   minimum_true_margin_cm=plant.minimum, outside_s=plant.outside_s, stopped_s=plant.stopped_s,
                   penetration_episodes=plant.episodes, frames=len(records),
                   mean_pose_error_cm=float(np.mean([r["pose_error_cm"] for r in records])) if records else None,
                   reasons=dict(Counter(r["reason"] for r in records)),
                   turn_gain_end={"left": driver.motor.gain_left, "right": driver.motor.gain_right},
                   median_step_ms=float(np.median([r["timings_ms"]["total"] for r in records])) if records else None,
                   wall_s=time.perf_counter() - start)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("scenario", "direction", "timing", "outcome", "verified_lap",
                                                "proper_corner_exits", "minimum_true_margin_cm")}, default=float), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=list(SCENARIOS), required=True)
    ap.add_argument("--direction", choices=["ccw", "cw"], required=True)
    ap.add_argument("--timing", choices=["same", "own"], required=True)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--seconds", type=float, default=240.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.out, a.scenario, a.direction, a.timing, a.seed, a.seconds)


if __name__ == "__main__":
    main()
