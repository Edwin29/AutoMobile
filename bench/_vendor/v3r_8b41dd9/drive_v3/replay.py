"""Run offline image/video observations. Does not import camera or GPIO code."""
from pathlib import Path
import argparse
import csv
import json
import platform
import time

import cv2
import numpy as np

from .camera import CameraGeometry, sha256_file
from .contracts import FrameStamp
from .observation import Observer

HERE = Path(__file__).resolve().parent


def read_image(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def write_image(path, image):
    ok, encoded = cv2.imencode(Path(path).suffix, image)
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    encoded.tofile(path)


def overlay(frame, debug, geometry):
    left, right = debug["rectified"].copy(), debug["bev"].copy()
    colors = {"left": (255, 230, 40), "right": (80, 240, 255), "unknown": (170, 170, 170)}
    for f in frame.fragments:
        color = colors[f.side_hint]
        for canvas, pts in ((left, geometry.body_to_rectified(f.points_cm)),
                            (right, geometry.body_to_pixels(f.points_cm))):
            q = np.rint(pts).astype(np.int32)
            cv2.polylines(canvas, [q], f.closed, color, 2, cv2.LINE_AA)
            for j, reason in zip((0, -1), f.endpoint_reasons):
                cv2.circle(canvas, tuple(q[j]), 3, (0, 80, 255) if reason == "branch" else color, -1)
    by_id = {f.id: f for f in frame.fragments}
    for c in frame.connections:
        a = by_id[c.fragment_a].points_cm[0 if c.end_a == 0 else -1]
        b = by_id[c.fragment_b].points_cm[0 if c.end_b == 0 else -1]
        for canvas, pts in ((left, geometry.body_to_rectified([a, b])),
                            (right, geometry.body_to_pixels([a, b]))):
            for t in np.arange(0, 1, .18):
                u = pts[0]+t*(pts[1]-pts[0])
                v = pts[0]+min(t+.09, 1)*(pts[1]-pts[0])
                cv2.line(canvas, tuple(np.rint(u).astype(int)), tuple(np.rint(v).astype(int)), (255, 0, 255), 1)
    for mark in frame.landmarks:
        if mark.kind == "blue_circle_candidate":
            for canvas, p in ((left, geometry.body_to_rectified([mark.center_cm])[0]),
                              (right, geometry.body_to_pixels([mark.center_cm])[0])):
                cv2.drawMarker(canvas, tuple(np.rint(p).astype(int)), (255, 120, 30), cv2.MARKER_CROSS, 12, 2)
    right[~debug["valid"]] //= 3
    scale = left.shape[0]/right.shape[0]
    right = cv2.resize(right, (round(right.shape[1]*scale), left.shape[0]), interpolation=cv2.INTER_NEAREST)
    body = np.hstack((left, right))
    bar = np.zeros((65, body.shape[1], 3), np.uint8)
    text = f"OFFLINE {frame.stamp.sequence} | fragments={len(frame.fragments)} gaps={len(frame.connections)} | {frame.diagnostics['processing_ms']:.1f} ms"
    cv2.putText(bar, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .55, (240, 240, 240), 1)
    cv2.putText(bar, "cyan/yellow=side candidates; gray=unknown; magenta dashed=UNOBSERVED link; cross=blue candidate", (10, 49), cv2.FONT_HERSHEY_SIMPLEX, .46, (220, 220, 220), 1)
    return np.vstack((bar, body))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory, never overwritten")
    parser.add_argument("--glob", default="*.png", help="Relative glob for image directories")
    parser.add_argument("--camera", type=Path, default=HERE/"config/camera_legacy_track.json")
    parser.add_argument("--config", type=Path, default=HERE/"config/observation.json")
    parser.add_argument("--frame-metadata", type=Path, help="Capture frames.csv with file,sensor_ns")
    parser.add_argument("--rectified", action="store_true")
    parser.add_argument("--rotate180", action="store_true", help="Only for unrotated raw input")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="0 = all")
    parser.add_argument("--save-every", type=int, default=1)
    args = parser.parse_args(argv)
    if args.stride < 1 or args.limit < 0 or args.save_every < 1:
        parser.error("Invalid stride/limit/save-every")
    if not args.input.exists():
        parser.error("Input does not exist")
    if args.rectified and args.rotate180:
        parser.error("Do not rotate an already rectified input")
    g = CameraGeometry(args.camera)
    observer = Observer(g, args.config)
    metadata = {}
    if args.frame_metadata:
        with args.frame_metadata.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                metadata[(args.frame_metadata.parent / row["file"]).resolve()] = int(row["sensor_ns"])/1e9
    records = []
    video = args.input.is_file() and args.input.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")
    if not video:
        paths = sorted(args.input.glob(args.glob)) if args.input.is_dir() else [args.input]
        paths = paths[::args.stride]
        if args.limit:
            paths = paths[:args.limit]
        if not paths:
            parser.error("No matching input frames")
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    timings = []

    def inputs():
        if video:
            cap = cv2.VideoCapture(str(args.input))
            if not cap.isOpened():
                raise ValueError("Cannot open video")
            index, count, last_pts = 0, 0, -1.
            try:
                while True:
                    ok, img = cap.read()
                    if not ok:
                        break
                    pts = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                    if index % args.stride == 0:
                        valid_pts = np.isfinite(pts) and pts >= 0 and pts > last_pts
                        stamp = FrameStamp(index, float(pts) if valid_pts else None,
                                           "video_pts" if valid_pts else "unavailable")
                        yield img, stamp, {"file": args.input.name, "video_frame": index}
                        last_pts = pts
                        count += 1
                        if args.limit and count >= args.limit:
                            break
                    index += 1
            finally:
                cap.release()
        else:
            last_sensor = None
            for i, path in enumerate(paths):
                capture = metadata.get(path.resolve())
                if capture is not None:
                    if last_sensor is not None and capture <= last_sensor:
                        raise ValueError("Sensor timestamps must increase in replay order")
                    last_sensor = capture
                stamp = FrameStamp(i*args.stride, capture, "sensor_monotonic" if capture is not None else "unavailable")
                relative = str(path.relative_to(args.input)) if args.input.is_dir() else path.name
                yield read_image(path), stamp, {"file": relative, "sha256": sha256_file(path)}

    with (args.output/"observations.jsonl").open("w", encoding="utf-8") as stream:
        for i, (img, stamp, source) in enumerate(inputs()):
            frame, debug = observer.process(img, stamp, rectified=args.rectified, rotate180=args.rotate180)
            payload = frame.to_dict()
            payload["input"] = source
            stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False)+"\n")
            if i % args.save_every == 0:
                write_image(args.output/f"{i:05d}_overlay.jpg", overlay(frame, debug, g))
                np.savez_compressed(args.output/f"{i:05d}_masks.npz",
                                    **{key: val for key, val in debug.items() if val.dtype == bool})
            timings.append(frame.diagnostics["processing_ms"])
            records.append({"sequence": stamp.sequence, "file": source["file"],
                            "fragments": len(frame.fragments), "connections": len(frame.connections),
                            "blue_candidates": sum(m.kind == "blue_circle_candidate" for m in frame.landmarks),
                            "status": frame.status})
    if not records:
        raise ValueError("No decoded frames; no successful run")
    summary = {
        "schema": 1, "mode": "offline_observation_no_control", "frame_count": len(records),
        "median_observer_ms": float(np.median(timings)), "p95_observer_ms": float(np.percentile(timings, 95)),
        "elapsed_with_io_s": time.perf_counter()-start,
        "calibration_id": g.id, "calibration_asset_sha256": sha256_file(g.asset),
        "config_id": observer.config_id, "config": observer.cfg,
        "input_preprocessing": {"rectified": args.rectified, "rotate180": args.rotate180},
        "metadata_sha256": sha256_file(args.frame_metadata) if args.frame_metadata else None,
        "video_sha256": sha256_file(args.input) if video else None,
        "code_sha256": {str(p.relative_to(HERE)): sha256_file(p) for p in sorted(HERE.rglob("*.py"))},
        "environment": {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__},
        "records": records,
        "limits": ["No independent real-frame labels", "No pose/planning/motor/mission execution",
                   "Timing is laptop observer compute, excludes IO and is not Pi throughput",
                   "Circle/side candidates are unverified", "Pink/red/white patches are not grouped mission markers"],
    }
    (args.output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("mode", "frame_count", "median_observer_ms", "p95_observer_ms")}, indent=2))


if __name__ == "__main__":
    main()
