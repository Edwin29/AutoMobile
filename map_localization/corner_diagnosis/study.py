"""Controlled first-corner tests. No device access. Coordinates are the original world X/Y."""
import csv
import json
import math
import sys
from pathlib import Path
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent/"c_drive_test"),str(HERE.parent/"ideal_drive")]
from simulate_c import Plant, lane, drive, DT, FWD, corner
import ideal_drive

# CCW reference, same geometric path used by the map controller, now in world X/Y.
route=ideal_drive.ref[::-1,::-1]
closed=np.vstack([route,route[0]])
arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(closed,axis=0),axis=1))]

def nearest(points):
    points=np.atleast_2d(points)
    return np.array([np.argmin(np.sum((route-p)**2,axis=1)) for p in points])

def ahead(idx,d):
    return np.array([np.interp((arc[idx]+d)%arc[-1],arc,closed[:,j]) for j in range(2)])

def curvature(q):
    return 2*q[0]/max(q@q,1e-6)

def run(mode,save=False):
    p=Plant("ccw")
    road=lane.DEFAULT_ROAD
    previous=0.
    rows=[]
    views=[]
    while p.tick*DT<45:
        img=p.capture_array()
        yaw=p.yaw
        rot=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
        cam=p.pos+FWD*rot[1]
        r,_=lane.analyze(img,p.world.M,p.world.size,None,"straight",road,18,0,2.5,0,0,None,0)
        if r is None:
            raise RuntimeError("Missing detection; this harness assumes valid baseline detections")
        road=r["road"]
        target=np.asarray(r["lookahead"])
        target_world=target@rot+cam
        idx=nearest(p.pos)[0]
        map_target=ahead(idx,10.)
        k_map=curvature((map_target-p.pos)@rot.T)
        k_c=float(r["curv_pp"])
        # Pure pursuit toward the identical world target from the body pivot.
        k_pivot=curvature(target+[0.,FWD])
        k={"C":k_c,"pivot_only":k_pivot,"track_only":k_c,"pivot_track":k_pivot,"map":k_map}[mode]
        track=13. if mode in ("track_only","pivot_track","map") else 18.
        filtered=.3*previous+.7*k
        out=drive.slew(p.out,drive.wheel_speeds(filtered,.3,track,10.),drive.ACCEL_PER_S*DT)
        previous=filtered
        p.set(*out)
        ci=nearest(target_world)[0]
        # unit normal points left (inward in CCW).
        tangent=route[(ci+1)%len(route)]-route[(ci-1)%len(route)]
        normal=np.array([-tangent[1],tangent[0]])/np.linalg.norm(tangent)
        target_dev=float((target_world-route[ci])@normal)
        raw_point=r["Ccm"][r["ia"]]@rot+cam
        ri=nearest(raw_point)[0]
        raw_t=route[(ri+1)%len(route)]-route[(ri-1)%len(route)]
        raw_dev=float((raw_point-route[ri])@np.array([-raw_t[1],raw_t[0]])/np.linalg.norm(raw_t))
        body_t=route[(idx+1)%len(route)]-route[(idx-1)%len(route)]
        body_dev=float((p.pos-route[idx])@np.array([-body_t[1],body_t[0]])/np.linalg.norm(body_t))
        actual_k=2*(out[0]-out[1])/(13.*sum(out)) if sum(out)>0 else 0.
        rows.append(dict(t=p.tick*DT,x=p.pos[0],y=p.pos[1],margin=p.rows[-1][4],inward_body=body_dev,
                         inward_raw=raw_dev,inward_target=target_dev,k_C=k_c,k_pivot=k_pivot,k_map=k_map,
                         k_applied=actual_k,left=out[0],right=out[1]))
        if save:
            views.append((img,cam,rot,r["Ccm"]@rot+cam,target_world,map_target,corner@rot+p.pos))
        if p.pos[1]>-45 and p.pos[0]>50:
            break
    else:
        raise RuntimeError("No exit")
    return rows,views,p.world

if __name__=="__main__":
    cv2.setNumThreads(1)
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("mode",choices=["C","pivot_only","track_only","pivot_track","map"])
    mode=ap.parse_args().mode
    rows,views,world=run(mode,mode=="C")
    with (HERE/(mode+".csv")).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    if mode=="C":
        old=np.genfromtxt(HERE.parent/"c_drive_test/ccw_base.csv",delimiter=",",names=True)
        err=max(abs(r["margin"]-old["body_clear_cm"][i]) for i,r in enumerate(rows))
        pos_err=max(np.hypot(r["x"]-old["x"][i],r["y"]-old["y"][i]) for i,r in enumerate(rows))
        assert max(err,pos_err)<.00001,(err,pos_err)
        from draw import render
        render(rows,views,world)
    print(mode,"steps",len(rows),"min",min(r["margin"] for r in rows),"max_inward_body",max(r["inward_body"] for r in rows),flush=True)
