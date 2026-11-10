"""Reproduce the first observation smoke review on original local recordings.

Not an accuracy benchmark: no independently reviewed real-world labels are used.
"""
from pathlib import Path
import argparse
import json

import cv2
import numpy as np

from .camera import sha256_file
from .replay import main as replay, read_image, write_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path, help="Original AutoDrive 2 folder")
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    args = parser.parse_args()
    runs = ["ccw_edges_20261106_173559_edges", "ccw_edges_20261106_173933_edges",
            "ccw_v135_20261106_174607_edges", "ccw_forward127_20261106_184214_edges"]
    jobs = [(name, ["--input", str(args.workspace/"map_localization/outer_driver/field_runs"/name),
                    "--glob", "[0-9][0-9].png", "--save-every", "6"]) for name in runs]
    capture = args.workspace/"road_ai_v1/20261106_193448_316130"
    jobs.append(("capture_257", ["--input", str(capture/"take_001"), "--frame-metadata",
                                str(capture/"frames.csv"), "--save-every", "16"]))
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"kind": "real_images_offline_smoke_review_not_accuracy", "runs": []}
    previews = []
    for name, cli in jobs:
        destination = args.output/name
        replay(cli + ["--output", str(destination)])
        source = destination/"summary.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        summary["runs"].append({
            "name": name, "frames": data["frame_count"],
            "frames_with_boundary_fragments": sum(r["fragments"] > 0 for r in data["records"]),
            "frames_with_blue_candidates": sum(r["blue_candidates"] > 0 for r in data["records"]),
            "median_observer_ms": data["median_observer_ms"], "p95_observer_ms": data["p95_observer_ms"],
            "summary_sha256": sha256_file(source), "calibration_id": data["calibration_id"],
            "config_id": data["config_id"], "code_sha256": data["code_sha256"],
        })
        images = sorted(destination.glob("*_overlay.jpg"))
        selected = images[-1] if name != "capture_257" else images[min(2, len(images)-1)]
        preview = read_image(selected)
        preview = cv2.resize(preview, (1000, round(preview.shape[0]*1000/preview.shape[1])))
        title = np.zeros((28, preview.shape[1], 3), np.uint8)
        cv2.putText(title, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1)
        previews.append(np.vstack((title, preview)))
    summary["total_frames"] = sum(r["frames"] for r in summary["runs"])
    summary["limits"] = ["Counts measure candidate presence, not correctness", "No pose or control executed",
                          "Single-session archived calibration, extrinsic agreement unverified",
                          "CPU timings measured on laptop, not Raspberry Pi",
                          "Stop buffers are known regressions, not independent holdout data"]
    (args.output/"review_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_image(args.output/"contact_sheet.jpg", np.vstack(previews))
    print(f"Reviewed {summary['total_frames']} frames; no accuracy or driving claim.")


if __name__ == "__main__":
    main()
