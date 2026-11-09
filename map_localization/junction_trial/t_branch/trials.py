"""Matched T-junction trials. Ground truth is restricted to camera rendering, fault timing and scoring."""
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
from central_right import Feedback,FixedGain,Plant,OUT,DT,axes,calibrate,env,rim,CENTER_AHEAD

# Photo-left outer road -> horizontal interior; photo-top outer road -> vertical interior.
SITES={"wide":(np.array([0.,-87.25]),-np.pi/2),"narrow":(np.array([-86.25,0.]),0.)}
START=-49.
FRAME_ON=8
FRAME_OFF=158    # 10 s, same approach/entry interval for both choices

def path_for(turn,r):
    if turn=="straight":
        t=np.arange(START,67,.25)
        return np.c_[np.zeros(len(t)),t]
    t=np.arange(START,-r,.25)
    arc=np.linspace(0,np.pi/2,129)
    end=np.arange(r+.25,66.,.25)
    return np.vstack([np.c_[np.zeros(len(t)),t],np.c_[r*(1-np.cos(arc)),r*(np.sin(arc)-1)],np.c_[end,end*0]])

def clearance(x,y):
    sdf=env["rrect_sdf"]
    d=-sdf(x,y,*env["OUTER"],env["OUTER_R"])
    for b in env["BLOCKS"]:d=np.minimum(d,sdf(x,y,*b,env["BLOCK_R"]))
    return d

def route(site,turn):
    center,yaw=SITES[site]
    candidates=[]
    for radius in (range(8,39) if turn=="right" else [0]):
        path=path_for(turn,radius)@axes(yaw)+center
        fw=np.gradient(path,axis=0);fw/=np.linalg.norm(fw,axis=1)[:,None]
        right=np.c_[fw[:,1],-fw[:,0]]
        pos=path[:,None,:]+right[:,None,:]*rim[:,0,None]+fw[:,None,:]*(rim[:,1,None]+CENTER_AHEAD)
        candidates.append((float(clearance(*pos.transpose(2,0,1)).min()),radius,path))
    return max(candidates,key=lambda item:item[0])

def surface(world,site,condition):
    clean=world.img
    if condition=="normal":return clean
    center,yaw=SITES[site]
    rot=axes(yaw)
    axis=(np.arange(len(clean))+.5)/env["PPC"]-env["EXT"]
    x,y=np.meshgrid(axis,-axis)
    lat=(x-center[0])*rot[0,0]+(y-center[1])*rot[0,1]
    fwd=(x-center[0])*rot[1,0]+(y-center[1])*rot[1,1]
    modified=clean.copy()
    if condition=="hidden":
        # Erase the inside terrain to road colour. Outside edge and collision map stay intact.
        modified[(lat>7)&(fwd>-65)&(fwd<65)&(lat<75)&(world.sd<1.5)]=env["COL"]["road"]
    else:
        # Synthetic appearance of an earlier side road: dark reflected band and two bright borders.
        # It is visual only. The block under it remains forbidden to the physical footprint.
        layer=(lat>9)&(lat<49)&(world.sd<1.3)
        band=abs(fwd+31.)
        modified[layer&(band<5.5)]=env["COL"]["road"]
        modified[layer&(band>=5.5)&(band<6.5)]=(135,225,128)
    return modified

class ShortGapDriver(Feedback):
    """Try every fresh image against the map; bridge only a short gap from a trusted fix."""
    def __init__(self):
        super().__init__("cw")
        self.gap=0
        self.matched=False
        self.had_fix=False

    def observe(self,image,rectified):
        self.matched=super().observe(image,rectified)
        if self.matched:
            self.had_fix=True;self.gap=0
            return True
        self.gap+=1
        self.window.clear()  # do not learn turn gain from prediction alone
        # DT is fixed at 15 Hz in this test. No fresh fix -> stop after at most 3 frames.
        return self.had_fix and self.gap<=3 and self.margin is not None and self.margin>=1.

    def step(self,*args,**kw):
        cmd=super().step(*args,**kw)
        if not self.matched:self.window.clear()
        return cmd

def run(site,turn,condition):
    cv2.setNumThreads(1)
    center,yaw=SITES[site];rot=axes(yaw)
    geom_min,radius,path=route(site,turn)
    c=calibrate();first,second=json.loads((OUT/"cw_calibration.json").read_text())
    p=Plant("cw",pivot_behind_center_cm=CENTER_AHEAD,left_yaw_response=c["left_gain"],right_yaw_response=second["gain"])
    p.yaw=yaw;p.pos=path[0]+CENTER_AHEAD*rot[1]
    d=ShortGapDriver();d.right_mix=first["gain"];d.yaw_tracker=FixedGain();d.yaw_tracker.right=d.right_mix
    d.pose=np.r_[path[0],yaw];d.route=path;d.arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))];d.last_index=0
    clean=p.world.img;bad=surface(p.world,site,condition)
    rows=[];outcome="timeout";exit_name=None
    for i in range(620):
        tick=i+1
        p.world.img=bad if FRAME_ON<=tick<FRAME_OFF else clean
        frame=p.capture_array();cmd=d.step(frame,DT,p.out,rectified=True);p.set(*cmd)
        truth=p.pos-CENTER_AHEAD*axes(p.yaw)[1]
        loc=(p.pos-center)@rot.T
        yaw_err=abs(math.degrees(math.remainder(p.yaw-d.pose[2],2*np.pi)))
        rows.append([tick,p.tick*DT,*p.pos,p.yaw,p.rows[-1][4],*d.pose,np.linalg.norm(truth-d.pose[:2]),yaw_err,int(d.matched),*cmd])
        if d.reason:outcome=d.reason;break
        for label,k,angle in (("right",0,yaw+np.pi/2),("straight",1,yaw)):
            if loc[k]>=40:
                exit_name=label
                aligned=abs(loc[1-k])<2 and abs(math.degrees(math.remainder(p.yaw-angle,2*np.pi)))<5
                outcome="pass" if label==turn and aligned else "wrong exit or alignment"
                break
        if exit_name:break
    a=np.asarray(rows);post=a[a[:,0]>=FRAME_OFF]
    good=(post[:,9]<=.5)&(post[:,10]<=1)&(post[:,11]==1)
    settled=next((int(post[j,0]-FRAME_OFF) for j in range(len(post)-4) if good[j:j+5].all()),None)
    prefix=HERE/f"{site}_{turn}_{condition}"
    np.savetxt(prefix.with_suffix(".csv"),a,delimiter=",",header="frame,t,x,y,yaw,margin,est_x,est_y,est_yaw,pos_error,angle_error,matched,L,R",comments="")
    result=dict(site=site,command=turn,condition=condition,outcome=outcome,observed_exit=exit_name,elapsed_s=a[-1,1],min_cm=a[:,5].min(),crossed_frames=int((a[:,5]<0).sum()),
                max_error_cm=a[:,9].max(),max_yaw_deg=a[:,10].max(),settled_frames_after_restore=settled,
                missing_frames=int((a[:,11]==0).sum()),geometry_min_cm=geom_min,turn_radius_cm=radius)
    prefix.with_suffix(".json").write_text(json.dumps(result,indent=2))
    print(site,turn,condition,outcome,round(result["elapsed_s"],2),round(result["min_cm"],2),radius,flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("site",choices=SITES);ap.add_argument("condition",choices=("normal","hidden","fake"))
    args=ap.parse_args()
    for turn in ("straight","right"):run(args.site,turn,args.condition)

