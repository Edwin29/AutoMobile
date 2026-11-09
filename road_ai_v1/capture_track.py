"""Record upright, unrectified OV5647 frames for road segmentation. No GPIO."""
import argparse
import csv
import json
import math
import signal
import tempfile
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

SIZE = (640, 480)
FPS = 15
SCENES = ("t_mid_clear", "t_mid_sign", "t_approach", "t_exit", "seam_near", "seam_far")
ROOT = Path(__file__).resolve().parent / "captures"


def open_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), FPS, SIZE)
    if not writer.isOpened():
        raise RuntimeError("Could not open FFV1 video writer")
    return writer


def self_test():
    # Verify the actual Pi encoder/decoder preserves BGR values, orientation and frame count.
    rng = np.random.default_rng(41)
    source = [rng.integers(0, 256, (SIZE[1], SIZE[0], 3), dtype=np.uint8) for _ in range(4)]
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "test.avi"
        writer = open_video(path)
        try:
            for frame in source:
                writer.write(frame)
        finally:
            writer.release()
        reader = cv2.VideoCapture(str(path))
        try:
            for frame in source:
                ok, saved = reader.read()
                if not ok or not np.array_equal(frame, saved):
                    raise RuntimeError("FFV1 round-trip failed")
            if reader.read()[0]:
                raise RuntimeError("Unexpected extra frame")
        finally:
            reader.release()
    print("FFV1 lossless check passed (4 frames, no camera opened)")


def record(scene, seconds):
    from picamera2 import Picamera2
    folder = ROOT / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{scene}"
    folder.mkdir(parents=True)
    print(f"Output: {folder}", flush=True)
    summary = dict(scene=scene, requested_seconds=seconds, size=SIZE, requested_fps=FPS,
                   camera_format="RGB888", orientation="flipped 180 degrees", rectified=False,
                   codec="FFV1", status="recording")
    stamps = []
    try:
        with ExitStack() as stack:
            camera = Picamera2()
            stack.callback(camera.close)
            camera.configure(camera.create_video_configuration(
                main={"format":"RGB888", "size":SIZE}, controls={"FrameRate":FPS}, queue=False))
            writer = open_video(folder/"video.avi")
            stack.callback(writer.release)
            stream = stack.enter_context((folder/"frames.csv").open("x", newline=""))
            log = csv.writer(stream)
            log.writerow(("frame", "sensor_ns", "elapsed_s", "exposure_us", "analogue_gain",
                          "digital_gain", "colour_gain_r", "colour_gain_b"))
            camera.start()
            stack.callback(camera.stop)
            time.sleep(1.2)
            print(f"Recording {scene} for {seconds:g}s", flush=True)
            start = time.monotonic()
            while time.monotonic()-start < seconds:
                req = camera.capture_request()
                try:
                    frame = cv2.flip(req.make_array("main"), -1)
                    meta = req.get_metadata()
                finally:
                    req.release()
                stamp = int(meta["SensorTimestamp"])
                writer.write(frame)
                if not stamps and not cv2.imwrite(str(folder/"first.png"), frame):
                    raise OSError("Failed to save preview")
                gains = meta.get("ColourGains") or ("", "")
                log.writerow((len(stamps), stamp, (stamp-(stamps[0] if stamps else stamp))*1e-9,
                              meta.get("ExposureTime"), meta.get("AnalogueGain"),
                              meta.get("DigitalGain"), *gains))
                stamps.append(stamp)
                stream.flush()
        if not stamps:
            raise RuntimeError("No frames captured")
        summary["status"] = "completed"
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
    except Exception as error:
        summary.update(status="failed", error=str(error))
        raise
    finally:
        summary["frames"] = len(stamps)
        if len(stamps)>1:
            gaps = np.diff(stamps)*1e-9
            summary.update(sensor_span_s=float(gaps.sum()), measured_fps=float(1/gaps.mean()),
                           max_gap_ms=float(gaps.max()*1000))
        (folder/"capture.json").write_text(json.dumps(summary, indent=2))
        print(f"{summary['status']}: {len(stamps)} frames | {folder}", flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", nargs="?", choices=SCENES)
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not args.scene or not math.isfinite(args.seconds) or not 1<=args.seconds<=30:
            parser.error("Choose a scene and a duration between 1 and 30 seconds")
        def interrupt(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGHUP, interrupt)
        record(args.scene, args.seconds)
