"""Step 0: the PyTorch copy of A5b must give the same outputs as the original ONNX
(OpenCV DNN, as the bench and the Pi run it), and so must a re-exported ONNX.

    python -m cnn_finetune.convert_check [--frames DIR]

Inputs: bench renders (clean + field-log camera effects) at many poses, plus saved real
frames if --frames is given (upright jpgs, e.g. Pi pi_runs/<run>/).
"""
import argparse
import glob
import json
import os
import tempfile

import cv2
import numpy as np
import torch

from .model import A5B_ONNX, export_onnx, load_a5b, preprocess_bgr


def bench_frames(n=200, seed=0):
    from bench.world import Camera, Renderer, Track
    tr, cam = Track(), Camera()
    r = Renderer(tr, cam)
    rng = np.random.default_rng(seed)
    route = np.asarray(cam.route)
    out = []
    for _ in range(n):
        i = rng.integers(len(route))
        a, b = route[i], route[(i + 1) % len(route)]
        yaw = float(np.arctan2(b[0] - a[0], b[1] - a[1])) + rng.normal(0, 0.15)
        if rng.random() < 0.5:
            yaw += np.pi
        pose = np.array([a[0] + rng.normal(0, 2), a[1] + rng.normal(0, 2), yaw])
        out.append(r.render_raw(pose))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", help="folder with upright real camera jpgs")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    cv2.setNumThreads(1)
    imgs = bench_frames(a.n)
    if a.frames:
        files = sorted(glob.glob(os.path.join(os.path.expanduser(a.frames), "*.jpg")))[: a.n]
        imgs += [cv2.imread(f) for f in files]
    x = np.stack([preprocess_bgr(im) for im in imgs])
    ref = cv2.dnn.readNetFromONNX(np.frombuffer(A5B_ONNX.read_bytes(), np.uint8))
    y_ref = []
    for xi in x:
        ref.setInput(xi[None])
        y_ref.append(ref.forward()[0].copy())
    y_ref = np.array(y_ref)
    model = load_a5b()
    with torch.no_grad():
        y_t = model(torch.from_numpy(x)).numpy()
    res = dict(frames=len(x), output_range=[float(y_ref.min()), float(y_ref.max())],
               torch_vs_onnx_max_abs=float(np.abs(y_t - y_ref).max()))
    try:                                   # torch's ONNX export needs the onnx package
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "re.onnx")
            export_onnx(model, p)
            re = cv2.dnn.readNetFromONNX(p)
            y_re = []
            for xi in x:
                re.setInput(xi[None])
                y_re.append(re.forward()[0].copy())
        res["reexport_vs_onnx_max_abs"] = float(np.abs(np.array(y_re) - y_ref).max())
    except Exception as exc:
        res["reexport_vs_onnx_max_abs"] = f"skipped: {type(exc).__name__}: {exc}"
    res["ok"] = res["torch_vs_onnx_max_abs"] < 1e-4
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
