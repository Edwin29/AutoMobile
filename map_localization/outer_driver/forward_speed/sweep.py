"""CCW field configuration. Change only translation of the simulated car."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from calibrated_lap import calibrate, OUT
from validate_sim import Plant,DT
from feedback_driver import Feedback,FixedGain
from controller import CENTER_AHEAD,axes


def run(scale):
    cv2.setNumThreads(1)
    first,second=json.loads((OUT/"cw_calibration.json").read_text())
    plant=Plant("ccw",pivot_behind_center_cm=CENTER_AHEAD,
                left_yaw_response=calibrate()["left_gain"],right_yaw_response=second["gain"],
                forward_scale=scale)
    driver=Feedback("ccw")
    driver.color_value_min=135
    driver.rim=np.load(HERE.parent/"photo_plus5mm_rim.npy")
    driver.yaw_tracker=FixedGain()
    driver.right_mix=driver.yaw_tracker.right=first["gain"]
    driver.record_edges=True
    rows=[]
    for _ in range(round(220/DT)):
        frame=plant.capture_array()
        command=driver.step(frame,DT,plant.out,rectified=True)
        plant.set(*command)
        basis=axes(plant.yaw)
        pivot=plant.pos-CENTER_AHEAD*basis[1]
        lateral,forward=(pivot-driver.pose[:2])@basis.T
        obs=driver.edge_observation
        ratio=float(np.mean(np.abs(obs[1])<2.5)) if obs is not None else float("nan")
        rows.append((plant.tick*DT,*pivot,math.degrees(plant.yaw),*driver.pose[:2],
                     math.degrees(driver.pose[2]),forward,lateral,ratio,driver.margin,
                     driver.distance/driver.arc[-1],*command))
        if driver.reason:break
    stem=HERE/f"ccw_{round(scale*100)}"
    with stem.with_suffix(".csv").open("w",newline="") as out:
        writer=csv.writer(out)
        writer.writerow("t,true_x,true_y,true_yaw,est_x,est_y,est_yaw,along_error_cm,lateral_error_cm,edge_inliers,forecast_cm,estimated_laps,left,right".split(","))
        writer.writerows(rows)
    a=np.array(rows,dtype=float)
    idx=np.argmax(abs(a[:,7]))
    # Gate near the latest real stop, outbound from corner two, referenced to true pivot.
    near=np.where((a[:,2]>74)&(a[:,1]>32)&(a[:,1]<43)&(a[:,3]<-65)&(a[:,3]>-110))[0]
    result=dict(scale=scale,end=driver.reason or "timeout",t=a[-1,0],
                true_laps=plant.travel/(2*np.pi),estimated_laps=a[-1,11],
                peak_abs_along_cm=abs(a[idx,7]),peak_signed_cm=a[idx,7],peak_t=a[idx,0],
                last_signed_cm=a[-1,7],gate_signed_cm=None,
                min_predicted_margin_cm=float(np.nanmin(a[:,10])))
    if len(near):
        gate=near[np.argmin(abs(a[near,1]-37.56))]
        result["gate_signed_cm"]=a[gate,7]
    stem.with_suffix(".json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("scale",type=float,choices=(.95,1.,1.05))
    run(parser.parse_args().scale)
