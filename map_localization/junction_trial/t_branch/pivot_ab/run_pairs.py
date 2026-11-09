"""Paired narrow T exit, driven only from the rendered camera and previous commands."""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

OUTPUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUTPUT.parent))
from pivot import run, DT
from yaw_gain import GainTracker
from controller import WHEEL_SPEED, TRACK_CM
from floor_motion import FloorMotion

CASES=("normal","delay","delay_slide","weak_clear","weak_blind")

class Comparison:
    def __init__(self,case,mode):
        self.case=case
        self.mode=mode
        self.prefix=OUTPUT/f"{case}_{mode}"
        self.start=None
        self.blind=False
        self.streak=0
        self.trace=[]
        self.events=[]
        self.shadow=GainTracker()
        self.motion=FloorMotion()
        self.flow_events=[]
        self.edge_events=[]

    def initialize(self,driver,prior,plant_gain):
        self.initial_gain=prior
        self.nominal=plant_gain
        self.shadow.right=prior
        driver.right_mix=prior

    def set_disturbance(self,plant,tick,rot):
        if self.start is None and plant.out[0]*plant.out[1]<0:
            self.start=tick
        elapsed=(tick-self.start)*DT if self.start is not None else -1
        plant.right_yaw_response=self.nominal
        plant.slide_vector=np.zeros(2)
        if elapsed>=0 and self.case.startswith("weak") and plant.out[0]*plant.out[1]<0:
            plant.right_yaw_response=self.nominal*.85
        if 0<=elapsed<2.5 and self.case.startswith("delay"):
            # Partial yaw from the other wheels. Not a measured four-wheel friction model.
            plant.right_yaw_response=self.nominal*.60
            if self.case=="delay_slide":
                # Stress assumption: during the delay slip per degree matches one photographed turn.
                plant.slide_vector=np.array([1.3,2.4])@rot*(90/52)
        self.blind=self.case=="weak_blind" and self.start is not None and 66<=tick-self.start<75

    def observe(self,driver,frame,applied,old_yaw,phase):
        self.shadow.event=None
        driver.last_update=None
        raw=math.degrees((applied[0]-applied[1])*WHEEL_SPEED/TRACK_CM*DT)
        delta=math.degrees(driver.pose[2]-old_yaw)
        self.mark_count=0
        if phase=="rotate" and raw>.12 and driver.matched and driver.residual<=.65 and delta>.25:
            self.streak+=1
        else:
            self.streak=0
        if self.streak<4:
            self.motion.reset()
            self.shadow.reset()
            driver.window.clear()
            return
        driver.update_output(old_yaw,DT,applied)
        marks=[m for m in driver.paint_detector.detect(frame,True) if m.kind=="cross"]
        self.mark_count=len(marks)
        self.shadow.update(marks,raw,DT,sum(applied)*WHEEL_SPEED*DT/2)
        if self.mode=="joint" and self.shadow.event:
            driver.yaw_tracker.right=self.shadow.right
        if self.mode=="joint_flow":
            ratio=self.motion.update(frame,driver.M,raw)
            if ratio is not None:
                old=driver.yaw_tracker.right
                new=float(np.clip(old+.25*(ratio-old),old-.05,old+.05))
                driver.yaw_tracker.right=new
                self.flow_events.append(dict(tick=len(self.trace)+1,ratio=ratio,before=old,after=new))
        if self.mode=="joint_edges" and driver.last_update:
            # Same accepted camera-corrected yaw interval as the established output learner.
            # This is a posterior measurement: prediction and observation are not independent.
            ratio=driver.last_update["ratio"]
            old=driver.yaw_tracker.right
            new=float(np.clip(old+.25*(ratio-old),old-.05,old+.05))
            driver.yaw_tracker.right=new
            self.edge_events.append(dict(tick=len(self.trace)+1,ratio=ratio,before=old,after=new))

    def record(self,tick,phase,driver,yaw,site_heading):
        if self.shadow.event or driver.last_update:
            self.events.append(dict(tick=tick,paint=self.shadow.event,output=driver.last_update))
        self.trace.append((tick,phase,self.blind,driver.yaw_tracker.right,driver.right_mix,
                           self.mark_count,math.degrees(yaw-site_heading)))

    def summary(self,rows):
        blind=[i for i,r in enumerate(self.trace) if r[2]]
        first_exit=next((r[0] for r in self.trace if r[1]=="exit"),len(self.trace))
        turning=[r[-1] for r in self.trace if r[0]<=first_exit and r[1] in ("rotate","wait","confirm","exit")]
        path=self.prefix.parent/(self.prefix.name+"_gains.csv")
        with path.open("w",encoding="utf8") as stream:
            stream.write("tick,phase,blind,yaw_gain,output_response,crossings_visible,true_heading_deg\n")
            for t,phase,b,yg,og,n,h in self.trace:
                stream.write(f"{t},{phase},{int(b)},{yg},{og},{n},{h}\n")
        return dict(case=self.case,mode=self.mode,first_spin=self.start,
                    fault_start=(self.start+66 if self.case=="weak_blind" and self.start else None),
                    fault_end=(self.start+75 if self.case=="weak_blind" and self.start else None),
                    paint_events=sum(e["paint"] is not None for e in self.events),
                    output_events=sum(e["output"] is not None for e in self.events),
                    eligible_frames_with_cross=sum(r[5]>0 for r in self.trace),
                    initial_gain=self.initial_gain,final_prediction=self.trace[-1][3],
                    final_output_multiplier=1/self.trace[-1][4],events=self.events,
                    flow_events=self.flow_events,edge_events=self.edge_events,
                    peak_pivot_overshoot_deg=max(0.,max(turning,default=0.)-90.),
                    at_last_blind_deg=float(rows[blind[-1],10]) if blind else None,
                    actual_blind_frames=len(blind))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("case",choices=CASES,nargs="+")
    parser.add_argument("--mode",choices=("output","joint","joint_flow","joint_edges"),nargs="+",default=("output","joint"))
    args=parser.parse_args()
    for case in args.case:
        for mode in args.mode:
            run("narrow","normal",recover=True,comparison=Comparison(case,mode))
