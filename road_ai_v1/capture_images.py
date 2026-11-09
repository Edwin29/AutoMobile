"""Collect full-frame track images with SPACE in a Pi or SSH terminal."""
import csv
import os
import select
import shutil
import signal
import sys
import termios
import time
import tty
from datetime import datetime
from pathlib import Path

import cv2
from picamera2 import Picamera2

from capture_track import FPS, SIZE


ROOT = Path(__file__).resolve().parent / "images"
INTERVAL = 0.5
RESERVE = 256 * 1024**2


def run(camera, fd):
    folder = ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    take = count = 0
    active = False
    print(f"READY | SPACE start/pause | Q quit | {folder}", flush=True)
    with (folder / "frames.csv").open("x", newline="") as stream:
        log = csv.writer(stream)
        log.writerow(("file", "sensor_ns", "exposure_us", "analogue_gain", "colour_gains"))
        while True:
            timeout = max(0, due - time.monotonic()) if active else None
            if select.select([fd], [], [], timeout)[0]:
                key = os.read(fd, 1)
                if key in (b"", b"q", b"Q"):
                    break
                if key == b" ":
                    active = not active
                    if active:
                        take += 1
                        segment = folder / f"take_{take:03d}"
                        segment.mkdir()
                        due = time.monotonic()
                    print(f"{'RECORDING' if active else 'PAUSED'} | take {take} | {count} saved", flush=True)
                continue
            if shutil.disk_usage(folder).free < RESERVE:
                print("Stopped: less than 256 MiB free.", flush=True)
                break
            request = camera.capture_request()
            try:
                frame = cv2.flip(request.make_array("main"), -1)
                meta = request.get_metadata()
            finally:
                request.release()
            photo = segment / f"{count + 1:06d}.png"
            partial = segment / ".writing.png"
            if not cv2.imwrite(str(partial), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                raise OSError(f"Could not save {photo}")
            partial.replace(photo)
            log.writerow((photo.relative_to(folder), meta["SensorTimestamp"], meta.get("ExposureTime"),
                          meta.get("AnalogueGain"), meta.get("ColourGains")))
            stream.flush()
            count += 1
            if count == 1 or count % 10 == 0:
                print(f"Saved {count} | {photo.name}", flush=True)
            due += INTERVAL
            if due < time.monotonic():
                due = time.monotonic() + INTERVAL
    print(f"Finished | {count} photos | {folder}", flush=True)


def interrupt(*_):
    raise KeyboardInterrupt


def main():
    if not sys.stdin.isatty():
        sys.exit("Use an interactive terminal (ssh -t).")
    signal.signal(signal.SIGHUP, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    with Picamera2() as camera:
        camera.configure(camera.create_video_configuration(
            main={"format": "RGB888", "size": SIZE}, controls={"FrameRate": FPS}, queue=False))
        camera.start()
        time.sleep(1.2)
        fd = sys.stdin.fileno()
        original = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            run(camera, fd)
        finally:
            termios.tcsetattr(fd, termios.TCSANOW, original)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped.", flush=True)
