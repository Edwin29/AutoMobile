"""Closed-loop visibility faults at one reproducible CCW corner. No GPIO or live camera."""
import copy
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from calibrated_lap import CompensatedDriver, FixedGain, calibrate, OUT
from validate_sim import Plant, DT, env
from controller import CENTER_AHEAD, LENS_AHEAD, axes

ON=265  # 17.67 s, still turning, both boundary sides in the detector ROI (15 Hz)
LAST=420 # 28.00 s, beyond first corner
KINDS=("normal","inner","both","near","ghost")
DURATIONS=(1,5,15,30,60,90)
SHOW=30

def textures(world):
    clean=world.img.copy()
    n=clean.shape[0]
    grid=(np.arange(n)+.5)/env["PPC"]-env["EXT"]
    x,y=np.meshgrid(grid,-grid)
    island=env["BLOCKS"][3]  # adjacent to the outer corner approached from the photo-left start
    inner=env["rrect_sdf"](x,y,*island,env["BLOCK_R"])
    road=env["COL"]["road"]
    hidden=clean.copy(); hidden[inner<1.5]=road
    blind=clean.copy(); blind[world.sd<1.5]=road
    # Reflect a displaced coloured band on the dark road. Both genuine boundaries still exist.
    # Feather its outline; its saturated part passes the controller's actual HSV detector.
    alpha=.93*np.clip(1.-np.maximum(abs(inner-3.)-.6,0)/.5,0,1)
    alpha*=world.sd>0
    tint=np.array([98.,232.,120.])
    ghost=np.rint(clean+(tint-clean)*alpha[:,:,None]).astype(np.uint8)
    alpha_near=.93*np.clip(1.-np.maximum(abs(inner-1.65)-.30,0)/.40,0,1)*(world.sd>0)
    near=np.rint(clean+(tint-clean)*alpha_near[:,:,None]).astype(np.uint8)
    return dict(normal=clean,inner=hidden,both=blind,ghost=ghost,near=near)

def new_run():
    cal=calibrate()
    first,second=json.loads((OUT/"cw_calibration.json").read_text())
    plant=Plant("ccw",pivot_behind_center_cm=CENTER_AHEAD,left_yaw_response=cal["left_gain"],
                right_yaw_response=second["gain"])
    driver=CompensatedDriver("ccw",adaptive_yaw=True)
    driver.right_mix=first["gain"]
    driver.yaw_tracker=FixedGain()
    driver.yaw_tracker.right=driver.right_mix
    return plant,driver

def step(p,d):
    frame=p.capture_array()
    before=d.reason
    cmd=d.step(frame,DT,p.out,rectified=True)
    p.set(*cmd)
    actual=p.pos-CENTER_AHEAD*axes(p.yaw)[1]
    row=dict(tick=p.tick,x=p.pos[0],y=p.pos[1],yaw=p.yaw,
             px=d.pose[0],py=d.pose[1],pyaw=d.pose[2],
             error_cm=np.linalg.norm(d.pose[:2]-actual),
             error_deg=abs(math.degrees(math.remainder(d.pose[2]-p.yaw,2*math.pi))),
             clearance=p.rows[-1][4],left=cmd[0],right=cmd[1],reason=d.reason,
             matched=not before and d.reason!="unmatched road edges",residual=d.residual)
    return row,frame

def evidence(frame,d,p):
    # Exact same pixel mask/ROI as Driver.observe, then label observed edges with independent plant truth.
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    col=cv2.inRange(hsv,(20,100,130),(87,255,255))
    road=((hsv[:,:,2]<130)&(col==0)).astype(np.uint8)*255
    dark=cv2.warpPerspective(road,d.M,d.bev)>=128
    edge=(cv2.warpPerspective(col,d.M,d.bev)>=128)&(cv2.dilate(dark.astype(np.uint8),np.ones((3,3),np.uint8))!=0)
    y,x=np.nonzero(edge)
    pts=np.c_[x/5-40,50-y/5]
    pts=pts[(pts[:,1]>=6)&(pts[:,1]<=35)]
    xy=(pts+[0,LENS_AHEAD-CENTER_AHEAD])@axes(p.yaw)+p.pos
    inn=env["rrect_sdf"](*xy.T,*env["BLOCKS"][3],env["BLOCK_R"])
    out=-env["rrect_sdf"](*xy.T,*env["OUTER"],env["OUTER_R"])
    return {k:int(np.count_nonzero(v)) for k,v in dict(inner=abs(inn)<1,outer=abs(out)<1,
                                                        displaced=(inn>1.6)&(inn<4.5),all=np.ones(len(pts))).items()}

def simulate(selected=KINDS,durations=DURATIONS):
    cv2.setNumThreads(1)
    p0,d0=new_run()
    maps=textures(p0.world)
    pre=[]
    while p0.tick<ON-1:
        row,_=step(p0,d0); pre.append(row)
    assert not d0.reason
    path=HERE/"results.json"
    summary=json.loads(path.read_text()) if path.exists() else {}
    for kind in selected:
        for length in ((SHOW,) if kind=="normal" else durations):
            p=copy.deepcopy(p0,{id(p0.world):p0.world})
            d=copy.deepcopy(d0)
            rows=pre.copy()
            first_evidence=None
            clean_probe=None
            for tick in range(ON,LAST+1):
                active=tick<ON+length
                p.world.img=maps[kind if active else "normal"]
                row,frame=step(p,d)
                rows.append(row)
                if tick==ON:
                    first_evidence=evidence(frame,d,p)
                    if kind=="normal":assert min(first_evidence["inner"],first_evidence["outer"])>0
                    if kind=="inner":assert first_evidence["inner"]==0 and first_evidence["outer"]>0
                    if kind=="both":assert first_evidence["all"]==0
                    if kind=="ghost":assert first_evidence["displaced"]>20
                    if kind=="near":assert first_evidence["all"]>310
                if tick==ON+length and d.reason:
                    # Diagnostic probe only; never feeds back to the driver.
                    test=copy.copy(d); test.pose=d.pose.copy()
                    clean_probe=bool(test.observe(frame,True))
                if d.reason:
                    assert np.allclose(p.out,0)
                    if tick>=ON+length+5 and length not in (SHOW,90):break
            fault=[r for r in rows if ON<=r["tick"]<ON+length]
            post=[r for r in rows if r["tick"]>=ON+length]
            def accurate(r):
                return r["matched"] and not r["reason"] and r["error_cm"]<=.5 and r["error_deg"]<=1
            reacquired=next((r["tick"]-ON-length for j,r in enumerate(post)
                             if len(post[j:j+5])==5 and all(accurate(x) for x in post[j:j+5])),None)
            stop=next((r for r in rows if r["reason"]),None)
            result=dict(kind=kind,frames=length,stop_frame=stop["tick"] if stop else None,
                        reason=d.reason,corner_exit=bool(p.pos[0]>50 and p.pos[1]>-45 and not d.reason),
                        fault_max_xy_cm=max(r["error_cm"] for r in fault),fault_max_yaw_deg=max(r["error_deg"] for r in fault),
                        fault_frames_driving=sum(not r["reason"] for r in fault),
                        body_min_after_on_cm=min(r["clearance"] for r in rows if r["tick"]>=ON),
                        fault_end_xy_cm=fault[-1]["error_cm"],fault_end_deg=fault[-1]["error_deg"],
                        post_first_matched=post[0]["matched"],within_tolerance_all_fault=all(accurate(r) for r in fault),
                        post_settled_start_offset_frames=reacquired,clean_probe_after_latched_stop=clean_probe,
                        first_frame_edge_pixels=first_evidence)
            summary[f"{kind}_{length}"]=result
            if length==SHOW or (length==90 and kind in ("near","ghost")):
                filename=f"{kind}.csv" if length==SHOW else f"{kind}_90.csv"
                with (HERE/filename).open("w",newline="") as f:
                    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
            print(kind,length,result["stop_frame"],result["corner_exit"],round(result["fault_max_xy_cm"],2),
                  round(result["fault_max_yaw_deg"],2),reacquired,first_evidence,flush=True)
    path.write_text(json.dumps(summary,indent=2))

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("kinds",nargs="*")
    ap.add_argument("--frames",type=int,choices=DURATIONS)
    args=ap.parse_args()
    simulate(args.kinds or KINDS,(args.frames,) if args.frames else DURATIONS)

