"""Replay a captured Pi session through the clean RoadNet checkpoint."""
import csv
import io
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

from build_data import Driver, HERE
sys.path.insert(0, str(HERE / "kit"))
from train import RoadNet, road_edge


def main(session, checkpoint, out):
    out.mkdir(exist_ok=True)
    masks = out / "labels"
    masks.mkdir(exist_ok=True)
    camera = Driver("ccw")
    roi = cv2.imread(str(HERE / "kit/data/evaluation_roi.png"), 0) > 0
    net = RoadNet().eval()
    torch.set_num_threads(4)
    with zipfile.ZipFile(checkpoint) as z:
        net.load_state_dict(torch.load(io.BytesIO(z.read("best.pt")), map_location="cpu", weights_only=True))
    with (session / "frames.csv").open() as f:
        rows = list(csv.DictReader(f))
    video = cv2.VideoWriter(str(out / "review.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 2, (640,240))
    if not video.isOpened():
        raise IOError("video writer")
    try:
        with (out / "frame_stats.csv").open("w", newline="") as f, torch.inference_mode():
            writer = csv.writer(f)
            writer.writerow(["file","sensor_ns","elapsed_s","edge_pixels_roi","road_fraction_roi"])
            for row in rows:
                name = row["file"]
                frame = cv2.imread(str(session / name))
                rect = cv2.resize(cv2.remap(frame,*camera.remap,cv2.INTER_LINEAR),(320,240),interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(rect,cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None]/255
                labels = net(tensor).argmax(1)[0].numpy().astype(np.uint8)
                dest = masks / name
                dest.parent.mkdir(exist_ok=True)
                if not cv2.imwrite(str(dest),labels):
                    raise IOError(dest)
                edge = road_edge(labels).astype(bool) & roi
                overlay = rect.copy()
                overlay[cv2.dilate(edge.astype(np.uint8),np.ones((2,2),np.uint8))>0] = (255,255,0)
                elapsed = (int(row["sensor_ns"])-int(rows[0]["sensor_ns"]))/1e9
                cv2.putText(overlay,f"{dest.stem}  {elapsed:.1f}s",(5,18),0,.5,(255,255,255),1,cv2.LINE_AA)
                video.write(np.hstack([rect,overlay]))
                writer.writerow([name,row["sensor_ns"],round(elapsed,3),int(edge.sum()),round(float((labels[roi]==1).mean()),4)])
    finally:
        video.release()


if __name__ == "__main__":
    main(*(Path(arg) for arg in sys.argv[1:]))
