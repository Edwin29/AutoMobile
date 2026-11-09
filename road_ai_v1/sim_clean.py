"""Matched closed-loop outer-track trials. Renderer supplies rectified BGR frames."""
import io
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "kit"))
sys.path.insert(0, str(HERE.parent / "map_localization/outer_driver"))
from train import RoadNet
from validate_sim import Plant, Driver, CENTER_AHEAD, DT


class CNNDriver(Driver):
    @torch.inference_mode()
    def boundary_pixels_bev(self, und):
        frame = cv2.resize(und, (320, 240), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()[None] / 255
        labels = self.net(inputs).argmax(1)[0].numpy().astype(np.uint8)
        labels = cv2.resize(labels, self.size, interpolation=cv2.INTER_NEAREST)
        ground = cv2.warpPerspective(labels, self.M, self.bev, flags=cv2.INTER_NEAREST, borderValue=255)
        adjacent = cv2.dilate((ground == 0).astype(np.uint8), np.ones((3, 3), np.uint8))
        return np.nonzero((ground == 1) & (adjacent > 0))


def run(direction, method, net):
    driver = CNNDriver(direction) if method == "CNN" else Driver(direction)
    driver.net = net
    plant = Plant(direction, pivot_behind_center_cm=CENTER_AHEAD)
    for tick in range(3500):
        frame = plant.capture_array()
        command = driver.step(frame, DT, plant.out, rectified=True)
        plant.set(*command)
        if driver.reason:
            break
    return dict(direction=direction, method=method, reason=driver.reason or "timeout",
                laps=plant.travel/(2*np.pi), seconds=(tick+1)*DT,
                minimum_clearance_cm=min(row[4] for row in plant.rows))


if __name__ == "__main__":
    cv2.setNumThreads(1)
    torch.set_num_threads(3)
    model = RoadNet().eval()
    with zipfile.ZipFile(sys.argv[1]) as archive:
        model.load_state_dict(torch.load(io.BytesIO(archive.read("best.pt")), weights_only=True, map_location="cpu"))
    results = []
    for direction in ("ccw", "cw"):
        for method in ("HSV", "CNN"):
            results.append(run(direction, method, model))
            print(results[-1], flush=True)
            (HERE/"closed_loop_clean.json").write_text(json.dumps(results, indent=2))
