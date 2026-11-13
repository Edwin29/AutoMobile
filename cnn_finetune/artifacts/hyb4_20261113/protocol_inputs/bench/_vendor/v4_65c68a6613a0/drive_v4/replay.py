"""Replay saved field runs (NNNN_raw.jpg + steps.jsonl) through drive_v4. No GPIO.

The recorded actual_pwm is fed back as 'applied', because that is what the car
really did. v4's own proposal is logged next to it for comparison.

    python -m drive_v4.replay RUN_DIR [RUN_DIR ...] [--pose x y yaw] [--direction ccw]
"""
import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np

from .core import Driver


def replay(run, pose=None, direction=None, out=None):
    run = Path(run)
    rows = [json.loads(l) for l in (run / "steps.jsonl").open(encoding="utf-8") if l.strip()]
    rows = [r for r in rows if "event" not in r and "capture_s" in r]
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8")) if (run / "manifest.json").exists() else {}
    direction = direction or manifest.get("direction") or "ccw"
    if pose is None and manifest.get("pose") is not None:
        pose = manifest["pose"]
    drv = Driver(direction=direction, pose=pose, clock="replay")
    records = []
    for r in rows:
        image = cv2.imread(str(run / f"{r.get('index', len(records)):04d}_raw.jpg"))
        if image is None:
            continue
        pwm, rec = drv.step(image, r["capture_s"], r["now_s"], rotate180=True)
        actual = r.get("actual_pwm") or [0.0, 0.0]
        drv.applied(r["now_s"], actual)
        rec.update(v3_path_reason=r.get("path_reason"), v3_estimate=(r.get("estimate") or {}).get("status"),
                   actual_pwm=actual, phase=r.get("phase"))
        records.append(rec)
    if out:
        Path(out).write_text("\n".join(json.dumps(x, default=float) for x in records))
    return records


def summarize(name, records):
    n = len(records)
    count = lambda key: collections.Counter(r[key] for r in records).most_common()
    moving = [r for r in records if max(r["pwm"]) > 0]
    t = np.array([r["timings_ms"]["total"] for r in records])
    w = np.array([r["w_map"] for r in records])
    print(f"== {name}: {n} frames, v4 would output PWM on {len(moving)} ({100 * len(moving) / max(n, 1):.0f}%)")
    print("   reason ", count("reason"))
    print("   source ", count("source"))
    print("   map    ", count("map_status"))
    print("   local  ", count("local_reason"))
    print("   mix    ", count("mix"))
    print(f"   w_map mean {w.mean():.2f}  >0.5 on {(w > .5).mean() * 100:.0f}% | time ms median {np.median(t):.0f} p95 {np.percentile(t, 95):.0f}")
    v3 = collections.Counter(r["v3_path_reason"] for r in records).most_common(3)
    print("   v3 was ", v3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--pose", nargs=3, type=float)
    ap.add_argument("--direction", choices=["ccw", "cw"])
    ap.add_argument("--save", action="store_true", help="write v4_replay.jsonl inside each run dir")
    a = ap.parse_args()
    for run in a.runs:
        recs = replay(run, a.pose, a.direction, Path(run) / "v4_replay.jsonl" if a.save else None)
        summarize(Path(run).name, recs)


if __name__ == "__main__":
    main()
