"""Replay saved frames with the actual logged wheel commands, never proposed commands."""
import csv
import json
from pathlib import Path
import cv2
from controller import Driver, advance

def run():
    cv2.setNumThreads(1)
    folder=Path(__file__).resolve().parents[2]/"junction_logic/visibility_review/run_160024"
    with (folder/"log.csv").open(encoding="utf8") as f:
        logs=list(csv.DictReader(f))
    driver=Driver("ccw")
    saved=0
    for pos,row in enumerate(logs):
        dt=float(row["t"])-(float(logs[pos-1]["t"]) if pos else 0)
        prior=logs[max(0,pos-1)]
        wheels=(float(prior["left"]),float(prior["right"])) if pos else (0.,0.)
        path=folder/("frame_%05d.jpg"%int(row["frame"]))
        if not path.exists():
            driver.pose=advance(driver.pose,wheels,dt)
            continue
        saved+=1
        driver.step(cv2.imread(str(path)),dt,wheels)
        if driver.reason:
            break
    print(json.dumps(dict(frames=saved,time_s=float(row["t"]),frame=row["frame"],
                          state=driver.reason or "end",residual_cm=driver.residual,margin_cm=driver.margin)))
if __name__=="__main__":
    run()
