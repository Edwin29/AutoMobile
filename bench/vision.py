"""Sign / light detection for the mission run: YOLO11n (ncnn) in its OWN process (2026-11-14).

The camera stays with bench.pi_run (one owner; the driving loop must not wait for YOLO).
pi_run hands over its upright frame through shared memory whenever this process is idle;
this process runs YOLO, applies bench.vision_rules and sends the resulting orchestrator
commands to pi_run over the same UDP port as ./send.sh (127.0.0.1:8765). The horn is
driven here (gpiozero), because it is not a driving command.

On the car (started by pi_run):
    ./run_mission.sh --drive --direction cw --vision                       # model in ~/hybrid_mission/models/signs
    ./run_mission.sh --drive --direction cw --vision --horn-pin 18         # with a buzzer on GPIO18
    ./run_mission.sh --vision                               # dry: detections + commands, motors off
    ./run_mission.sh --drive --direction cw --vision --vision-view         # + live view: http://<Pi address>:5000

Replay saved frames (laptop or Pi, no camera, no motors; prints what would be sent):
    python -m bench.vision --replay pi_runs/drive_hyb2m_20261111_175226 --fps 10

Model folder: model.ncnn.param + model.ncnn.bin (Ultralytics NCNN export, imgsz 320,
input in0, output out0, 11 classes as below).
"""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

from . import vision_rules as R

CLASS_NAMES = ["cross", "dashline", "green", "horn", "left", "line",
               "red", "right", "slow", "stop", "straight"]
CLASS_THRESH = {"cross": 0.35, "dashline": 0.35, "green": 0.35, "horn": 0.65, "left": 0.35,
                "line": 0.35, "red": 0.35, "right": 0.93, "slow": 0.45, "stop": 0.45,
                "straight": 0.25}
THRESH = np.array([CLASS_THRESH[n] for n in CLASS_NAMES], np.float32)
INPUT = 320
NMS = 0.45
FRAME_SHAPE = (480, 640, 3)
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "signs"


# ---- detector -------------------------------------------------------------------------
class Detector:
    """Same pre/post-processing as camera_val_test_v2.py (letterbox, per-class thresholds,
    class-agnostic NMS). Input: BGR frame, already upright."""

    def __init__(self, model_dir, threads=2):
        import cv2
        import ncnn
        self.cv2, self.ncnn = cv2, ncnn
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = threads
        p, b = Path(model_dir) / "model.ncnn.param", Path(model_dir) / "model.ncnn.bin"
        if self.net.load_param(str(p)) != 0 or self.net.load_model(str(b)) != 0:
            raise FileNotFoundError(f"ncnn model not loaded from {model_dir}")

    def __call__(self, frame):
        cv2, ncnn = self.cv2, self.ncnn
        h, w = frame.shape[:2]
        s = min(INPUT / w, INPUT / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        px, py = (INPUT - nw) // 2, (INPUT - nh) // 2
        canvas = np.full((INPUT, INPUT, 3), 114, np.uint8)
        canvas[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        m = ncnn.Mat.from_pixels(canvas, ncnn.Mat.PixelType.PIXEL_BGR2RGB, INPUT, INPUT)
        m.substract_mean_normalize([0.0, 0.0, 0.0], [1 / 255.0] * 3)
        ex = self.net.create_extractor()
        ex.input("in0", m)
        _, out = ex.extract("out0")
        out = np.array(out)
        if out.ndim == 3:
            out = out[0]
        if out.shape[0] < out.shape[1]:
            out = out.T
        sc = out[:, 4:4 + len(CLASS_NAMES)]
        cid = np.argmax(sc, axis=1)
        conf = sc[np.arange(len(cid)), cid]
        keep = conf >= THRESH[cid]
        if not np.any(keep):
            return []
        out, conf, cid = out[keep], conf[keep], cid[keep]
        x1 = np.clip((out[:, 0] - out[:, 2] / 2 - px) / s, 0, w - 1)
        y1 = np.clip((out[:, 1] - out[:, 3] / 2 - py) / s, 0, h - 1)
        bw, bh = out[:, 2] / s, out[:, 3] / s
        boxes = np.stack([x1, y1, bw, bh], 1).astype(int).tolist()
        scores = conf.astype(float).tolist()
        idx = cv2.dnn.NMSBoxes(boxes, scores, float(THRESH.min()), NMS)
        return [dict(name=CLASS_NAMES[int(cid[i])], score=round(scores[i], 3), box=boxes[i])
                for i in np.array(idx).flatten()]


class NoDetector:
    """--vision-model none: pipeline test without ncnn (detects nothing)."""

    def __call__(self, frame):
        return []


def make_detector(model_dir, threads=2):
    return NoDetector() if str(model_dir).lower() == "none" else Detector(model_dir, threads)


# ---- live view (browser) ---------------------------------------------------------------
COLORS = {"cross": (255, 0, 0), "dashline": (255, 255, 0), "green": (0, 255, 0), "horn": (255, 0, 255),
          "left": (255, 128, 0), "line": (203, 102, 255), "red": (0, 0, 255), "right": (0, 165, 255),
          "slow": (0, 255, 255), "stop": (0, 0, 160), "straight": (0, 255, 128)}
PAGE = """<html><head><title>vision</title><style>body{margin:0;background:#1e1e1e;color:#ccc;
font-family:sans-serif;display:flex;flex-direction:column;align-items:center}img{max-width:100%}</style>
</head><body><img src="/stream"><p>detector frames only (about 8 fps), boxes + rule state</p></body></html>"""


class LiveView:
    """MJPEG over the standard library http.server (no Flask). The vision process encodes a
    picture only while a browser is watching, so an unwatched view costs nothing."""

    def __init__(self, port):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.cond, self.jpg, self.seq, self.clients = threading.Condition(), None, 0, 0
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if not self.path.startswith("/stream"):
                    body = PAGE.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                with view.cond:
                    view.clients += 1
                last = -1
                try:
                    while True:
                        with view.cond:
                            view.cond.wait_for(lambda: view.seq != last, timeout=2.0)
                            jpg, seq = view.jpg, view.seq
                        if jpg is None or seq == last:
                            continue
                        last = seq
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                except OSError:                       # browser closed
                    pass
                finally:
                    with view.cond:
                        view.clients -= 1

        self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def watching(self):
        return self.clients > 0

    def push(self, jpg):
        with self.cond:
            self.jpg, self.seq = jpg, self.seq + 1
            self.cond.notify_all()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def own_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))          # no packet is sent; picks the LAN interface
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "<Pi address>"


def annotate(frame, dets, st, ms, fps):
    import cv2
    img = frame.copy()
    h, w = img.shape[:2]
    y_mid = int(R.LINE_MID_RATIO * h)
    for x in range(0, w, 20):                      # stop-line trigger height
        cv2.line(img, (x, y_mid), (x + 10, y_mid), (200, 200, 200), 1)
    for d in dets:
        x, y, bw, bh = d["box"]
        c = COLORS.get(d["name"], (255, 255, 255))
        cv2.rectangle(img, (x, y), (x + bw, y + bh), c, 2)
        cv2.putText(img, f"{d['name']} {d['score']:.2f}", (x, max(y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    lines = [f"infer {ms:.0f} ms  {fps:.1f} fps"]
    if st["pending"]:
        lines.append("next: " + "  ".join(f"{a} {v} in {dt}s" for dt, a, v in st["pending"][:3]))
    if st["cooldown"]:
        lines.append("ignored: " + "  ".join(f"{n} {r}s" for n, r in st["cooldown"].items()))
    if st.get("arrows"):
        lines.append("arrow votes: " + " ".join(f"{k}:{v}" for k, v in st["arrows"].items()))
    if st["wait_out_of_view"]:
        lines.append("re-arm when out of view: " + " ".join(st["wait_out_of_view"]))
    if st["red_hold"] or st["red_now"]:
        lines.append("RED HOLD (waiting for green)" if st["red_hold"] else "red light seen")
    for i, text in enumerate(lines):
        y = 22 + 20 * i
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok else None


# ---- outputs --------------------------------------------------------------------------
class Horn:
    def __init__(self, pin):
        self.dev = None
        if pin is not None:
            try:
                from gpiozero import PWMOutputDevice
                self.dev = PWMOutputDevice(int(pin), frequency=2000)
            except Exception as exc:            # no buzzer is not a reason to stop the run
                print(f"[vision] horn on GPIO{pin} unavailable: {exc}", flush=True)

    def set(self, on):
        if self.dev is not None:
            if on:
                self.dev.value = 0.08
            else:
                self.dev.off()

    def close(self):
        if self.dev is not None:
            self.dev.off()
            self.dev.close()


class Sender:
    """UDP JSON to bench.pi_run. command_id makes a resend harmless."""

    def __init__(self, port, dry=False):
        self.addr, self.dry, self.n = ("127.0.0.1", port), dry, 0
        self.sock = None if dry else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, c):
        self.n += 1
        msg = dict(c, command_id=f"vision-{os.getpid()}-{self.n}")
        msg.setdefault("reason", "vision")
        if self.sock is not None:
            self.sock.sendto(json.dumps(msg).encode(), self.addr)
        return msg


def carry_out(cmds, sender, horn, log, t, verbose=True):
    for c in cmds:
        if "local" in c:
            horn.set(c["on"])
            rec = dict(t=round(t, 3), local=c)
        else:
            rec = dict(t=round(t, 3), sent=sender.send(c))
        log.write(json.dumps(rec) + "\n")
        if verbose:
            print("[vision]", json.dumps(rec), flush=True)


# ---- frame hand-over (pi_run side) ----------------------------------------------------
class VisionLink:
    """Owned by bench.pi_run. offer() costs one 0.9 MB copy, and only when the vision
    process has asked for a frame; otherwise it returns at once."""

    def __init__(self, cmd_port, out_dir, model_dir=DEFAULT_MODEL, fps=8.0, horn_pin=None, threads=2,
                 view_port=None):
        import multiprocessing as mp
        from multiprocessing import shared_memory
        ctx = mp.get_context("spawn")
        self.shm = shared_memory.SharedMemory(create=True, size=int(np.prod(FRAME_SHAPE)))
        self.view = np.ndarray(FRAME_SHAPE, np.uint8, buffer=self.shm.buf)
        self.want, self.ready, self.quit = ctx.Event(), ctx.Event(), ctx.Event()
        self.meta = ctx.Array("d", 2, lock=False)
        self.offered = 0
        self.proc = ctx.Process(target=worker_main, daemon=True, name="vision",
                                args=(self.shm.name, self.want, self.ready, self.quit, self.meta,
                                      cmd_port, str(out_dir), str(model_dir), float(fps),
                                      horn_pin, int(threads), view_port))
        self.proc.start()

    def offer(self, image, capture, index):
        if not self.want.is_set() or image.shape != FRAME_SHAPE:
            return False
        self.want.clear()
        self.view[:] = image
        self.meta[0], self.meta[1] = float(capture), float(index)
        self.ready.set()
        self.offered += 1
        return True

    def alive(self):
        return self.proc.is_alive()

    def close(self):
        self.quit.set()
        self.proc.join(timeout=3)
        if self.proc.is_alive():
            self.proc.terminate()
        del self.view
        self.shm.close()
        self.shm.unlink()


# ---- vision process -------------------------------------------------------------------
def worker_main(shm_name, want, ready, quit, meta, cmd_port, out_dir, model_dir, fps,
                horn_pin, threads, view_port=None):
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(name=shm_name)
    view = np.ndarray(FRAME_SHAPE, np.uint8, buffer=shm.buf)
    rules, horn, sender = R.Rules(FRAME_SHAPE[0]), Horn(horn_pin), Sender(cmd_port)
    log = open(Path(out_dir) / "vision.jsonl", "w", buffering=1)
    live = None
    if view_port:
        try:
            live = LiveView(int(view_port))
            print(f"[vision] live view: http://{own_address()}:{view_port}", flush=True)
        except OSError as exc:                      # port taken: run on without the view
            print(f"[vision] live view on port {view_port} unavailable: {exc}", flush=True)
    fps_now, t_prev = 0.0, None
    try:
        detect = make_detector(model_dir, threads)
        print(f"[vision] ready: model={model_dir} fps<={fps} threads={threads} "
              f"horn={'GPIO' + str(horn_pin) if horn_pin is not None else 'log only'}", flush=True)
        period, next_at, n = 1.0 / max(fps, 0.5), 0.0, 0
        while not quit.is_set():
            now = time.monotonic()
            carry_out(rules.tick(now), sender, horn, log, now)
            if now < next_at:
                time.sleep(min(0.02, next_at - now))
                continue
            want.set()
            if not ready.wait(0.1):
                continue
            ready.clear()
            frame = view.copy()
            capture, index = meta[0], int(meta[1])
            next_at = time.monotonic() + period
            tic = time.perf_counter()
            dets = detect(frame)
            ms = (time.perf_counter() - tic) * 1e3
            now = time.monotonic()
            cmds = rules.update(now, dets)
            log.write(json.dumps(dict(t=round(now, 3), capture=round(capture, 3), frame=index,
                                      infer_ms=round(ms, 1), dets=dets)) + "\n")
            carry_out(cmds, sender, horn, log, now)
            if t_prev is not None and now > t_prev:
                fps_now = 0.8 * fps_now + 0.2 / (now - t_prev)
            t_prev = now
            if live is not None and live.watching():
                jpg = annotate(frame, dets, rules.status(now), ms, fps_now)
                if jpg is not None:
                    live.push(jpg)
            n += 1
            if n % 40 == 0:
                print(f"[vision] {n} frames, infer {ms:.0f} ms, last {[d['name'] for d in dets]}", flush=True)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        log.write(json.dumps(dict(error=f"{type(exc).__name__}: {exc}")) + "\n")
    finally:
        now = time.monotonic()
        carry_out(rules.shutdown(), sender, horn, log, now)
        horn.close()
        if live is not None:
            live.close()
        log.close()
        del view
        shm.close()


# ---- replay ---------------------------------------------------------------------------
def replay(folder, fps, model_dir, threads):
    import cv2
    files = sorted(Path(folder).glob("*.jpg"))
    if not files:
        sys.exit(f"no .jpg in {folder}")
    detect, rules = make_detector(model_dir, threads), R.Rules(FRAME_SHAPE[0])
    sender, horn = Sender(0, dry=True), Horn(None)
    out = Path(folder) / "vision_replay.jsonl"
    counts, ms_all = {}, []
    with out.open("w") as log:
        for i, f in enumerate(files):
            t = i / fps
            frame = cv2.imread(str(f))
            tic = time.perf_counter()
            dets = detect(frame)
            ms_all.append((time.perf_counter() - tic) * 1e3)
            for d in dets:
                counts[d["name"]] = counts.get(d["name"], 0) + 1
            log.write(json.dumps(dict(t=round(t, 2), file=f.name, dets=dets)) + "\n")
            cmds = rules.update(t, dets)
            for c in cmds:
                print(f"{t:7.2f}s {f.name}  ->", json.dumps(c), flush=True)
            carry_out(cmds, sender, horn, log, t, verbose=False)
    print(f"\n{len(files)} frames, infer median {np.median(ms_all):.1f} ms; detections per class: {counts}")
    print("rule log:", *[f"  {t:7.2f}s {s}" for t, s in rules.log], sep="\n")
    print("written:", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", required=True, help="folder of saved upright frames (*.jpg)")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="frames per second of the saved images (pi_run --save-every 3 at 30 fps = 10)")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    replay(a.replay, a.fps, a.model, a.threads)


if __name__ == "__main__":
    main()
