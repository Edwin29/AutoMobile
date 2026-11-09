"""Generate rectified camera inputs and independent geometric masks from the team's map."""
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "map_localization/outer_driver"))
from validate_sim import env
from controller import Driver, axes, LENS_AHEAD

DATA = HERE / "kit/data"
W, H = 320, 240


def sites():
    # Group by physical site before sampling. All views/directions of a site share one split.
    for sx, sy, split in [(-1,-1,"train"),(1,-1,"train"),(-1,1,"val"),(1,1,"test")]:
        yield f"corner_{sx}_{sy}", split, "corner", (sx, sy)
    for side, split in [("north","train"),("south","train"),("west","val"),("east","test")]:
        vertical = side in ("west","east")
        center = {"north":(0,87.25),"south":(0,-87.25),"west":(-86.25,0),"east":(85.25,0)}[side]
        yaw = 0 if vertical else math.pi/2
        yield "T_"+side,split,"junction",(*center,yaw)
        yield "straight_"+side,split,"straight",(*(np.array(center)+38*axes(yaw)[1]),yaw)
    yield "central_cross","test","central",(0.,0.,0.)


def camera_pose(name,kind,params,rng):
    if kind=="corner":
        sx,sy=params
        angle=rng.uniform(.04,math.pi/2-.04)
        radius=rng.uniform(18,23)
        pivot=np.array([(64.75 if sx>0 else -66.75)+sx*radius*math.cos(angle),
                        sy*(67.75+radius*math.sin(angle))])
        yaw=math.atan2(-sx*math.sin(angle),sy*math.cos(angle))+rng.choice([0,math.pi])
    else:
        pivot=np.array(params[:2],dtype=float)
        yaw=params[2]+rng.choice([0,math.pi])
        if kind=="central":yaw=rng.choice([0,.5,1,1.5])*math.pi
        if kind in ("junction","central") and rng.random()<.45:
            # Pivot near the node while the car is part-way through a turn.
            yaw+=rng.uniform(-1.35,1.35)
            along=rng.uniform(-3,3)
        else:
            along=rng.uniform(-24,3) if kind!="straight" else rng.uniform(-8,8)
        pivot+=along*axes(yaw)[1]+rng.uniform(-2.4,2.4)*axes(yaw)[0]
    yaw+=math.radians(rng.uniform(-8,8))
    return pivot,yaw


def generate():
    cv2.setNumThreads(1)
    DATA.mkdir(parents=True,exist_ok=True)
    for name in ("images","masks","real"):
        (DATA/name).mkdir(exist_ok=True)
    world=env["World"]()
    camera=Driver("ccw")
    # Use the renderer's actual ray grid at the model resolution.
    lat=cv2.resize(world.lat,(W,H),interpolation=cv2.INTER_AREA)
    fwd=cv2.resize(world.fwd,(W,H),interpolation=cv2.INTER_AREA)
    sky=cv2.resize(world.sky.astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)>0
    roi=(~sky)&(fwd>=6)&(fwd<=35)
    cv2.imwrite(str(DATA/"evaluation_roi.png"),roi.astype(np.uint8)*255)
    rows=[]
    for site_index,(site,split,kind,params) in enumerate(sites()):
        for i in range(80 if split=="train" else 32):
            seed=34000+1000*site_index+i
            rng=np.random.default_rng(seed)
            for attempt in range(200):
                pivot,yaw=camera_pose(site,kind,params,rng)
                points=camera.rim@axes(yaw)+pivot
                if world.sd_at(points[:,0],points[:,1]).min()>=0:break
            else:raise RuntimeError(f"No on-road pose at {site}")
            cam=pivot+LENS_AHEAD*axes(yaw)[1]
            x=cam[0]+lat*math.cos(yaw)+fwd*math.sin(yaw)
            y=cam[1]-lat*math.sin(yaw)+fwd*math.cos(yaw)
            mx,my=world.to_px(x,y)
            im=cv2.remap(world.img,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=env["COL"]["floor"])
            im[sky]=env["COL"]["sky"]
            # The signed map distance, not RGB/HSV, supplies ground truth. Road paint stays drivable.
            mask=(cv2.remap(world.sd,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=-100)>0).astype(np.uint8)
            mask[sky]=0
            if rng.random()<.3:
                # Deliberately simple opaque sign surrogate. Only its visible silhouette gets class 2.
                ey,ex=np.nonzero((cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0)&roi)
                if len(ex):
                    k=rng.integers(len(ex));cx=int(ex[k]);bottom=int(ey[k])
                    length=int(rng.integers(18,55));width=int(rng.integers(6,15));top=bottom-length
                    cover=np.zeros_like(mask)
                    cv2.line(cover,(cx,top),(cx,bottom),1,int(rng.integers(2,5)))
                    if rng.random()<.5:cv2.circle(cover,(cx,top),width,1,-1)
                    else:cv2.rectangle(cover,(cx-width,top-width),(cx+width,top+width),1,-1)
                    im[cover>0]=rng.integers(20,205,3)
                    mask[cover>0]=2
            im=cv2.GaussianBlur(im,(3,3),.55)
            uid=f"{site}_{i:03}"
            cv2.imwrite(str(DATA/"images"/(uid+".jpg")),im,[cv2.IMWRITE_JPEG_QUALITY,94])
            cv2.imwrite(str(DATA/"masks"/(uid+".png")),mask)
            rows.append(dict(id=uid,site=site,split=split,kind=kind,seed=seed))
    with (DATA/"manifest.csv").open("w",newline="") as stream:
        w=csv.DictWriter(stream,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    # Real snapshots for diagnostic evaluation only; no map-pose pseudo-labels.
    field=ROOT/"map_localization/outer_driver/field_runs"
    for name,stem in {"seam":"ccw_edges_20261106_173559","glare":"ccw_edges_20261106_173933",
                      "curve":"ccw_v135_20261106_174607","curve_repeat":"ccw_forward127_20261106_184214"}.items():
        frame=cv2.imread(str(field/(stem+"_edges")/"23.png"))
        rect=cv2.remap(frame,camera.remap[0],camera.remap[1],cv2.INTER_LINEAR)
        cv2.imwrite(str(DATA/"real"/(name+".png")),cv2.resize(rect,(W,H),interpolation=cv2.INTER_AREA))
    # Exact, recorded V135 pipeline for real reference at the same ROI.
    for path in (DATA/"real").glob("*.png"):
        if path.stem not in ("seam","glare","curve","curve_repeat"):continue
        stem={"seam":"ccw_edges_20261106_173559","glare":"ccw_edges_20261106_173933",
              "curve":"ccw_v135_20261106_174607","curve_repeat":"ccw_forward127_20261106_184214"}[path.stem]
        raw=cv2.imread(str(field/(stem+"_edges")/"23.png"))
        rect=cv2.remap(raw,*camera.remap,cv2.INTER_LINEAR)
        hsv=cv2.cvtColor(rect,cv2.COLOR_BGR2HSV)
        c=cv2.inRange(hsv,(20,100,135),(87,255,255))
        r=np.uint8(hsv[:,:,2]<130)*255
        color=cv2.warpPerspective(c,camera.M,camera.bev)>=128
        dark=cv2.warpPerspective(r,camera.M,camera.bev)>=128
        yy,xx=np.nonzero(color&(cv2.dilate(dark.astype(np.uint8),np.ones((3,3),np.uint8))>0))
        keep=(50-yy/5>=6)&(50-yy/5<=35)
        uv=cv2.perspectiveTransform(np.float32(np.c_[xx[keep],yy[keep]])[:,None],np.linalg.inv(camera.M))[:,0]/2
        edges=np.zeros((H,W),np.uint8)
        for u,v in np.rint(uv).astype(int):
            if 0<=u<W and 0<=v<H:edges[v,u]=255
        cv2.imwrite(str(DATA/"real"/(path.stem+"_hsv.png")),edges)
    print({s:sum(r["split"]==s for r in rows) for s in ("train","val","test")})

if __name__=="__main__":generate()
