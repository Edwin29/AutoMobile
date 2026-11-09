"""Matched simulation-only output-gain adaptation. The controller is never passed the plant's yaw gain."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import cv2

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from calibrated_lap import OUT,CompensatedDriver,FixedGain,calibrate
from validate_sim import Plant,DT
from controller import WHEEL_SPEED,TRACK_CM,CENTER_AHEAD,axes


from feedback_driver import Feedback

def run(mode,case,stop_at,direction="ccw"):
    cv2.setNumThreads(1)
    cal=calibrate()
    strong=cal["left_gain"];weak=cal["test_observed"]/cal["test_uncorrected"]
    first,second=json.loads((OUT/"cw_calibration.json").read_text())
    plant=Plant(direction,pivot_behind_center_cm=CENTER_AHEAD,left_yaw_response=strong,right_yaw_response=second["gain"])
    d=Feedback(direction) if mode=="dynamic" else CompensatedDriver(direction,adaptive_yaw=True)
    d.right_mix=first["gain"]
    d.yaw_tracker=FixedGain();d.yaw_tracker.right=d.right_mix
    events=[]
    minimum=float("inf");crossed=0
    if case in ("near_reflection","blind"):
        from corner_visibility.experiment import textures
        surfaces=textures(plant.world)
    stem=f"{case}_{mode}_{direction}"
    with (HERE/f"{stem}.csv").open("w",newline="") as stream:
        w=csv.writer(stream)
        w.writerow(["t","x_body","y_body","yaw_deg","actual_margin_cm","L_next","R_next","left_output_gain_next","right_output_gain_next"])
        for tick in range(round(stop_at/DT)):
            t=(tick+1)*DT
            if (case in ("step","near_reflection","blind") and t>=12) or case=="weak":
                plant.left_yaw_response=weak
            if case=="paired_step" and t>=12:
                plant.left_yaw_response=weak
                plant.right_yaw_response=second["gain"]*weak/strong
            if case in ("near_reflection","blind"):
                active=265<=tick+1<295
                kind="near" if case=="near_reflection" else "both"
                plant.world.img=surfaces[kind if active else "normal"]
            frame=plant.capture_array()
            cmd=d.step(frame,DT,plant.out,rectified=True)
            plant.set(*cmd)
            margin=plant.rows[-1][4]
            minimum=min(minimum,margin);crossed+=int(margin<0)
            if getattr(d,"last_update",None):events.append(dict(t=t,**d.last_update))
            w.writerow([t,*plant.pos,math.degrees(plant.yaw),margin,*cmd,1/d.left_mix,1/d.right_mix])
            if d.reason:break
    result=dict(case=case,mode=mode,direction=direction,reason=d.reason or "test horizon",elapsed=t,laps=plant.travel/(2*np.pi),
                min_cm=minimum,crossing_frames=crossed,final_output_gain=1/d.left_mix,final_right_output_gain=1/d.right_mix,updates=events)
    (HERE/f"{stem}.json").write_text(json.dumps(result,indent=2))
    print(case,mode,direction,result["reason"],round(t,2),round(minimum,3),len(events),
          "L/R",round(1/d.left_mix,3),round(1/d.right_mix,3))
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("case",choices=["step","weak","unchanged","near_reflection","blind","paired_step"])
    ap.add_argument("mode",choices=["dynamic","fixed"])
    ap.add_argument("--seconds",type=float,default=220.)
    ap.add_argument("--direction",choices=["cw","ccw"],default="ccw")
    a=ap.parse_args();run(a.mode,a.case,a.seconds,a.direction)
