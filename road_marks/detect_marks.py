"""Detect track markings and signs from an upright BGR photo. No GPIO or camera access."""
from pathlib import Path
import cv2
import ncnn
import numpy as np

ROOT = Path(__file__).resolve().parent
NAMES = ("cross", "dashline", "green", "horn", "line", "red", "right", "slow", "stop")
MARKS = ("cross", "dashline", "line")


class Detector:
    def __init__(self, folder=ROOT / "best_1_416_ncnn_model", size=416, threads=2):
        self.size = size
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = threads
        if self.net.load_param(str(folder / "model.ncnn.param")):
            raise OSError("Cannot read NCNN parameters")
        if self.net.load_model(str(folder / "model.ncnn.bin")):
            raise OSError("Cannot read NCNN weights")

    def detect(self, frame, confidence=0.25):
        h, w = frame.shape[:2]
        scale = min(self.size / w, self.size / h)
        rw, rh = round(w * scale), round(h * scale)
        x0, y0 = (self.size-rw)//2, (self.size-rh)//2
        square = np.full((self.size, self.size, 3), 114, np.uint8)
        square[y0:y0+rh, x0:x0+rw] = cv2.resize(frame, (rw, rh))
        rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255
        with self.net.create_extractor() as ex:
            if ex.input("in0", ncnn.Mat(tensor).clone()):
                raise RuntimeError("NCNN input failed")
            status, output = ex.extract("out0")
            if status:
                raise RuntimeError("NCNN inference failed")
            rows = np.array(output).T
        if rows.ndim != 2 or rows.shape[1] != 4+len(NAMES):
            raise ValueError(f"Unexpected detector output {rows.shape}")
        ids = rows[:, 4:].argmax(axis=1)
        scores = rows[np.arange(len(rows)), 4+ids]
        result = []
        for idx in np.unique(ids[scores >= confidence]):
            use = (ids == idx) & (scores >= confidence)
            boxes = rows[use, :4].copy()
            boxes[:, :2] -= boxes[:, 2:]/2
            conf = scores[use]
            for j in np.asarray(cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), confidence, .45)).flatten():
                x, y, bw, bh = boxes[j]
                box = np.clip([(x-x0)/scale,(y-y0)/scale,(x+bw-x0)/scale,(y+bh-y0)/scale],
                              [0,0,0,0], [w,h,w,h]).tolist()
                result.append(dict(kind=NAMES[idx], score=float(conf[j]), box=box))
        return sorted(result, key=lambda d: -d["score"])


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    img = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        parser.error("Cannot decode image")
    print(json.dumps(Detector().detect(img), indent=2))
