"""PilotNet lane policy, the same layers as the team's train_policy.LanePolicy (A5b), and
the A5b weights read from its ONNX file with OpenCV (no onnx package needed).

Input:  1x3x66x200 float, YUV of the bottom half of the upright raw camera frame,
        3x3 Gaussian blur, /255  (preprocess() below = bench TeamCNN._preprocess)
Output: [linear, angular] in [-1, 1]; angular + = left (team convention)
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
A5B_ONNX = ROOT / "bench" / "_vendor" / "tdt_models_2026-11-01" / "models" / "lane-v5b" / "policy.onnx"


class LanePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2), nn.ELU(),
            nn.Conv2d(24, 36, 5, 2), nn.ELU(),
            nn.Conv2d(36, 48, 5, 2), nn.ELU(),
            nn.Conv2d(48, 64, 3), nn.ELU(),
            nn.Conv2d(64, 64, 3), nn.ELU(),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1152, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 2), nn.Tanh(),
        )

    def forward(self, image):
        return self.net(image)


def preprocess_bgr(image_bgr):
    """Upright raw BGR camera frame (H x W x 3, uint8) -> 3x66x200 float32 (bench TeamCNN)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    half = rgb[rgb.shape[0] // 2:]
    yuv = cv2.cvtColor(half, cv2.COLOR_RGB2YUV)
    yuv = cv2.GaussianBlur(yuv, (3, 3), 0)
    yuv = cv2.resize(yuv, (200, 66))
    return np.ascontiguousarray(yuv.transpose(2, 0, 1), dtype=np.float32) / 255.0


def load_a5b(path=A5B_ONNX):
    """LanePolicy with the A5b weights (ONNX initialisers via OpenCV's importer)."""
    net = cv2.dnn.readNetFromONNX(np.frombuffer(Path(path).read_bytes(), np.uint8))
    blobs = []
    for name in net.getLayerNames():
        layer = net.getLayer(net.getLayerId(name))
        if layer.type in ("Convolution", "Gemm", "InnerProduct"):
            blobs.append([np.asarray(b) for b in layer.blobs])
    model = LanePolicy()
    targets = [m for m in model.net if isinstance(m, (nn.Conv2d, nn.Linear))]
    if len(targets) != len(blobs):
        raise ValueError(f"layer count mismatch: {len(targets)} vs {len(blobs)}")
    with torch.no_grad():
        for m, (w, b) in zip(targets, blobs):
            if tuple(m.weight.shape) != w.shape:
                raise ValueError(f"weight shape {w.shape} vs {tuple(m.weight.shape)}")
            m.weight.copy_(torch.from_numpy(w.astype(np.float32)))
            m.bias.copy_(torch.from_numpy(b.reshape(-1).astype(np.float32)))
    return model.eval()


def export_onnx(model, path):
    """Same interface as the team policy.onnx: input 1x3x66x200, output 1x2."""
    model = model.eval().cpu()
    torch.onnx.export(model, torch.zeros(1, 3, 66, 200), str(path), input_names=["image"],
                      output_names=["command"], opset_version=13, dynamo=False)
