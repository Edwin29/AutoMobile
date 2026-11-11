"""Processing time of a controller on saved camera frames (no camera, no GPIO).

    python -m bench.pi_bench --frames ~/drive_v4_trial/drive_v4/runs/live_20261110_195808 --rotate
    python -m bench.pi_bench --frames pi_runs/dry_hyb2m_... (already upright)

Feeds the frames in order at 30 fps timestamps; commands are fed back as if applied.
Reports CNN path, map worker and whole-step times (median / p95 / max).
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("BENCH_CNN_SPEED", "1.0")
import argparse
import glob
import json
import time

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="folder with *.jpg frames")
    ap.add_argument("--rotate", action="store_true", help="frames are as the camera delivers them (rotate 180)")
    ap.add_argument("--controller", default="hyb2m")
    ap.add_argument("--direction", default="cw")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--map-process", action="store_true", help="map in its own process; frames paced at 30 fps")
    a = ap.parse_args()
    cv2.setNumThreads(1)
    files = sorted(glob.glob(os.path.join(os.path.expanduser(a.frames), "*.jpg")))[: a.limit]
    if not files:
        raise SystemExit("no frames")
    from .controllers import CONTROLLERS
    from .run import timing_profile
    from .world import Track
    hint = Track().start_pose(a.direction)
    t0 = time.perf_counter()
    ctrl = CONTROLLERS[a.controller](a.direction, hint, timing_profile("cnn_pi"))
    if a.map_process:
        ctrl.use_map_process(hint)
    init_s = time.perf_counter() - t0
    rows = []
    wall0 = time.perf_counter()
    for i, f in enumerate(files):
        img = cv2.imread(f)
        if a.rotate:
            img = cv2.rotate(img, cv2.ROTATE_180)
        cap = (i + 1) / 30
        if a.map_process:                                   # real time: the worker runs beside the loop
            time.sleep(max(0.0, wall0 + cap - time.perf_counter()))
        tic = time.perf_counter()
        pwm, info = ctrl.step(img, cap, cap + 0.036)
        rows.append(((time.perf_counter() - tic) * 1e3, info.get("ms_cnn") or 0.0,
                     info.get("ms_map") or 0.0, info.get("ms_map_worker") or 0.0))
    R = np.array(rows)
    q = lambda x: dict(median=round(float(np.median(x)), 1), p95=round(float(np.percentile(x, 95)), 1),
                       max=round(float(x.max()), 1))
    mp = R[:, 2][R[:, 2] > 0]
    print(json.dumps(dict(frames=len(R), controller=a.controller, init_s=round(init_s, 2), step_ms=q(R[:, 0]),
                          cnn_ms=q(R[:, 1]), map_ms_when_run=q(mp) if len(mp) else None,
                          step_ms_without_map=q(R[:, 0][R[:, 2] == 0]),
                          map_worker_ms=q(R[:, 3][R[:, 3] > 0]) if (R[:, 3] > 0).any() else None), indent=1))
    if a.map_process:
        ctrl.worker.close()


if __name__ == "__main__":
    main()
