"""Causal, offline along-route hypothesis replay. No motor or camera interfaces."""
import ast
import csv
import json
import math
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from controller import Driver, LENS_AHEAD, WHEEL_SPEED, axes

RUNS=HERE.parent/"field_runs"
LAYOUT=HERE.parents[2]/"junction_logic/visibility_review/source/sim_drive.py"
MARKS=np.array([m[:2] for m in next(ast.literal_eval(n.value) for n in
    ast.parse(LAYOUT.read_text(encoding="utf8")).body if isinstance(n,ast.Assign)
    and getattr(n.targets[0],"id",None)=="CROSSWALKS")])


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def crossing_centers(frame,driver):
    """Recognize a complete row of six parallel paint bars, anywhere on the floor."""
    und=cv2.remap(frame,*driver.remap,cv2.INTER_LINEAR)
    hsv=cv2.cvtColor(und,cv2.COLOR_BGR2HSV)
    white=((hsv[:,:,1]<65)&(hsv[:,:,2]>210)).astype(np.uint8)
    bars=[]
    for c in cv2.findContours(white,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
        if cv2.contourArea(c)<25:continue
        b=cv2.perspectiveTransform(c.astype(np.float32),driver.M)
        moment=cv2.moments(b)
        if moment["m00"]<1:continue
        xy=np.array([moment["m10"],moment["m01"]])/moment["m00"]
        center=(xy-[200,250])*[.2,-.2]
        width,length=np.ptp(b[:,0,:],axis=0)/5
        if 8<center[1]<70 and 1<width<5 and 7<length<23:
            bars.append(center)
    if len(bars)<6:return []
    bars=np.array(sorted(bars,key=lambda p:p[0]))
    found=[]
    for before in bars:
        band=bars[abs(bars[:,1]-before[1])<3]
        for i in range(len(band)-5):
            group=band[i:i+6]
            spacing=np.diff(group[:,0])
            if np.all((spacing>2.2)&(spacing<4.4)) and np.ptp(group[:,1])<3:
                center=group.mean(axis=0)
                if all(np.linalg.norm(center-prev)>1. for prev in found):
                    found.append(center)
    return found


def replay(folder,uncertainty_per_cm=.15):
    d=Driver("ccw")
    log=rows(folder.parent/(folder.name[:-6]+".csv"))
    index=rows(folder/"index.csv")
    assert len(index)==24
    # End-aligned ordering; the camera timestamp and post-step log timestamp differ.
    assert all(abs(float(a["t"])-float(b["t"]))<.20 for a,b in zip(log[-24:],index))
    lo,hi,center=-.5,.5,0.
    pending=deque(maxlen=3)
    accepted_mark=None
    result=[]
    for i,r in enumerate(log):
        if i:
            dt=float(r["t"])-float(log[i-1]["t"])
            cmd=(float(log[i-1]["applied_L"])+float(log[i-1]["applied_R"]))/2
            travel=abs(cmd)*WHEEL_SPEED*dt*float(r.get("prediction_forward_gain",1))
            # Hypothesized (not statistically calibrated) fractional distance uncertainty.
            lo-=uncertainty_per_cm*travel;hi+=uncertainty_per_cm*travel
        j=i-(len(log)-len(index))
        if j<0:continue
        sample=index[j]["file"]
        z=np.load(folder/(sample+".npz"))
        pose=z["prior"];points=z["points"]
        world=points@axes(pose[2])+pose[:2]
        assert np.allclose(d.sample(d.sdf,world),z["residual_cm"])
        side=points[:,0]<0
        d.pose=pose
        s=d.arc[d.nearest()]
        def on_route(offset):
            location=(s+offset)%d.arc[-1]
            return np.stack([np.interp(location,d.arc,d.route[:,axis]) for axis in (0,1)],axis=-1)
        # Correct along the road, retaining the logged lateral position and heading.
        offsets=np.unique(np.r_[np.arange(math.ceil(lo*4),math.floor(hi*4)+1)*.25,center])
        shifts=on_route(offsets)-on_route(0)
        aligned=world[None,:,:]+shifts[:,None,:]
        errors=abs(d.sample(d.sdf,aligned.reshape(-1,2))).reshape(len(offsets),-1)
        good=errors<2.5
        ratio=good.mean(axis=1)
        def rank(o):
            return int(np.argmin(abs(offsets-o)))
        raw_ratio=float(np.mean(abs(z["residual_cm"])<2.5))
        image=cv2.imread(str(folder/(sample+".png")))
        candidates=[]
        for mark in crossing_centers(image,d):
            seen=(mark+[0,LENS_AHEAD])@axes(pose[2])+pose[:2]
            for tag,known in enumerate(MARKS):
                dif=np.linalg.norm(seen+shifts-known,axis=1)
                k=int(np.argmin(dif))
                # Match identity and offset by paint alone; edge scoring follows.
                if dif[k]<3 and lo+.5<offsets[k]<hi-.5:
                    candidates.append((tag,k))
        proposed=None;edge_score=None;event=False
        before=(lo,hi)
        if len(candidates)==1:
            tag,k=candidates[0]
            proposed=float(offsets[k]);edge_score=float(ratio[k])
            sides_supported=all(m.sum()>=8 and good[k,m].mean()>=.8 for m in (side,~side))
            if sides_supported and edge_score>=raw_ratio+.12:
                pending.append((tag,proposed))
            else:pending.clear()
            if (accepted_mark is None and len(pending)==3 and len({v[0] for v in pending})==1
                    and np.ptp([v[1] for v in pending])<=1.0):
                center=float(np.median([v[1] for v in pending]))
                lo,hi=max(lo,center-1.),min(hi,center+1.)
                accepted_mark=tag;event=True
        else:pending.clear()
        # Until a verified update exists, preserve the original stop on this frame.
        k=rank(center)
        eligible=ratio[k]>=.55 and np.median(errors[k])<=1.8
        result.append(dict(frame=j,prior_range=before,range_after=[lo,hi],unique=len(candidates)==1,
            proposed=proposed,marker=int(candidates[0][0]) if len(candidates)==1 else None,
            checked_ratio=edge_score,confirmed=event,offset=center,baseline_ratio=raw_ratio,
            ratio=float(ratio[k]),points=len(points),matched=int(good[k].sum()),passes=bool(eligible)))
    return result


if __name__=="__main__":
    cv2.setNumThreads(1)
    output={p.name:replay(p) for p in sorted(RUNS.glob("ccw*_edges")) if p.is_dir()}
    (HERE/"interval_replay.json").write_text(json.dumps(output,indent=2),encoding="utf8")
    for key,data in output.items():
        print(key, "unique",sum(r["unique"] for r in data),"events",
              [(r["frame"],r["offset"]) for r in data if r["confirmed"]],"last",data[-1])
