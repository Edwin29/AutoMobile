"""Offline, score unused floor marks on the real recording up to the first latched stop."""
from pathlib import Path
import csv
import json
import sys
import cv2
import numpy as np
from controller import Driver, advance, axes, LENS_AHEAD

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from replay_pose import inputs, geometry, indices, RUN

def run():
    cv2.setNumThreads(1)
    _,fields,_ = geometry()
    with (RUN/"log.csv").open(encoding="utf8") as stream:
        log = list(csv.DictReader(stream))
    driver = Driver("ccw")
    last = -1
    for frame, image, cues, _ in inputs():
        for i in range(last+1,frame+1):
            row = log[i]
            prior = log[max(0,i-1)]
            dt = float(row["t"])-(float(prior["t"]) if i else 0.)
            wheels = (float(prior["left"]), float(prior["right"])) if i else (0.,0.)
            if i<frame:
                driver.pose=advance(driver.pose,wheels,dt)
        cmd=driver.step(image,dt,wheels,rectified=True)
        last=frame
        rotation=axes(driver.pose[2])
        camera=driver.pose[:2]+LENS_AHEAD*rotation[1]
        errors={}
        for kind in ("dashline","cross","line"):
            if kind in cues:
                points=cues[kind]@rotation + camera
                errors[kind]=fields[kind][indices(*points.T)].round(2).tolist()
        print(json.dumps(dict(frame=frame,time=float(row["t"]),residual=driver.residual,marks=errors,
                              proposed=cmd.round(3).tolist(),historical=[row["left"],row["right"]],stop=driver.reason)))
        if driver.reason:
            break

if __name__=="__main__":
    run()
