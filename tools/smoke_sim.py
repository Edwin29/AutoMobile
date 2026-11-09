"""Exercise real simulation entry points for three frames, with no GPIO."""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'road_ai_v1'))
from sim_clean import CNNDriver, Driver, RoadNet, Plant, DT, CENTER_AHEAD


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    model = RoadNet().eval()
    model.load_state_dict(torch.load(ROOT / 'road_ai_v1/results/clean/best.pt',
                                    weights_only=True, map_location='cpu'))
    for cls in (Driver, CNNDriver):
        driver = cls('ccw')
        driver.net = model
        plant = Plant('ccw', pivot_behind_center_cm=CENTER_AHEAD)
        for _ in range(3):
            frame = plant.capture_array()
            command = driver.step(frame, DT, plant.out, rectified=True)
            assert np.isfinite(command).all() and not driver.reason, driver.reason
            plant.set(*command)
        print(cls.__name__, 'three closed-loop frames passed')


if __name__ == '__main__':
    main()
