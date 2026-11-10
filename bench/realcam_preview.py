"""Side-by-side: real Pi frames vs the bench camera (clean / typical / harsh).

    python -m bench.realcam_preview --out preview.jpg [--speed] [--device cuda]
"""
import argparse
import time

import cv2
import numpy as np

from .realcam import ASSETS, RealCam, imread, pose_from_history
from .world import Camera, Renderer, Track


def drive_history(track, camera, t_end, direction="ccw"):
    """True poses along the outer centre line at 18 cm/s (20 ms steps)."""
    r = camera.route if direction == "cw" else camera.route[::-1]
    seg = np.linalg.norm(np.diff(np.vstack((r, r[:1])), axis=0), axis=1)
    arc = np.r_[0, np.cumsum(seg)]
    hist = []
    start = track.start_pose(direction)
    s0 = arc[int(np.argmin(np.sum((r - start[:2]) ** 2, 1)))]
    for t in np.arange(0, t_end + 1e-9, 0.02):
        s = (s0 + 18 * t) % arc[-1]
        i = int(np.searchsorted(arc, s) - 1) % len(r)
        a, b = r[i], r[(i + 1) % len(r)]
        f = (s - arc[i]) / max(seg[i], 1e-9)
        p = a + f * (b - a)
        yaw = np.arctan2(b[0] - a[0], b[1] - a[1])
        hist.append((t, np.array([p[0], p[1], yaw])))
    return hist


def label(img, text):
    img = cv2.resize(img, (320, 240))
    cv2.putText(img, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3)
    cv2.putText(img, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device")
    ap.add_argument("--speed", action="store_true", help="time 200 frames per profile")
    ap.add_argument("--times", default="1.0,6.0,11.0")
    a = ap.parse_args()
    track, cam = Track(), Camera()
    ren = Renderer(track, cam)
    times = [float(x) for x in a.times.split(",")]
    hist = drive_history(track, cam, max(times) + 1)
    at = pose_from_history(hist)
    real = [imread(p) for p in sorted(ASSETS.glob("bg_*.jpg"))[:len(times)]]
    rows = []
    cams = {name: RealCam(ren, cam, name, seed=3, device=a.device) for name in ("typical", "harsh")}
    for k, tq in enumerate(times):
        row = [label(real[k], "REAL (Pi log)"), label(ren.render_raw(at(tq)), f"bench clean t={tq:.0f}s")]
        for name, rc in cams.items():
            for t_ in np.arange(max(0.0, tq - 1.0), tq, 1 / 30):          # let exposure settle like a run
                rc.frame(t_, at)
            img = rc.frame(tq, at)
            row.append(label(img, f"{name} raw gain {rc.gain:.2f}"))
        row.append(label(cams["harsh"].frame(tq + 0.03, at, raw=False), "harsh rectified (v4 sees)"))
        rows.append(np.hstack(row))
    cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    if a.speed:
        for name, rc in cams.items():
            rc.frame(0, at)
            t0 = time.perf_counter()
            for i in range(200):
                rc.frame(i / 30, at, raw=bool(i % 2))
            print(name, rc.dev, f"{(time.perf_counter() - t0) / 200 * 1000:.1f} ms/frame")


if __name__ == "__main__":
    main()
