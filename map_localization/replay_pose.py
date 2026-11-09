"""Offline pose hypotheses on the Nov 5 outer-lap recording. No device access."""
import ast
import argparse
import csv
import json
import sys
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parent / "junction_logic/visibility_review"
sys.path[:0] = [str(REVIEW), str(HERE.parent/"road_marks")]
from map_pose_example import rounded_box
from replay import SRC, RUN, lane
from detect_marks import Detector


def geometry():
    keys = {"OUTER","OUTER_R","BLOCKS","BLOCK_R","CROSSWALKS","ROAD_MARKS","APPROACH","D_CAM","KV","YAW","LAG"}
    cfg = {}
    for node in ast.parse((SRC/"sim_drive.py").read_text(encoding="utf-8")).body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in keys:
                cfg[name]=({kw.arg:ast.literal_eval(kw.value) for kw in node.value.keywords}
                           if name=="YAW" else ast.literal_eval(node.value))
    axis=np.arange(-160,160.5,.5)
    x,y=np.meshgrid(axis,axis)
    dist=-rounded_box(x,y,cfg["OUTER"],cfg["OUTER_R"])
    for block in cfg["BLOCKS"]:
        dist=np.minimum(dist,rounded_box(x,y,block,cfg["BLOCK_R"]))
    fields={"edge":np.abs(dist), "cross":np.minimum.reduce(
        [np.hypot(x-a,y-b) for a,b,_ in cfg["CROSSWALKS"]])}
    for cls,kind in [("line","red"),("dashline","pink")]:
        ds=[]
        for color,(ax,ay),(bx,by) in cfg["ROAD_MARKS"]:
            if color!=kind: continue
            t=np.clip(((x-ax)*(bx-ax)+(y-ay)*(by-ay))/((bx-ax)**2+(by-ay)**2),0,1)
            ds.append(np.hypot(x-ax-t*(bx-ax),y-ay-t*(by-ay)))
        fields[cls]=np.minimum.reduce(ds)
    return cfg,fields,dist


def indices(x,y):
    return np.clip(np.rint((y+160)*2).astype(int),0,640),np.clip(np.rint((x+160)*2).astype(int),0,640)


def world(p,points):
    s=np.sin(p[:,2,None]); c=np.cos(p[:,2,None])
    return indices(p[:,0,None]+c*points[:,0]+s*points[:,1],p[:,1,None]-s*points[:,0]+c*points[:,1])


def inputs():
    c=np.load(SRC/"calib_checkerboard.npz"); g=dict(np.load(SRC/"ground.npz")); size=tuple(map(int,c["size"]))
    K=cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(c["K"],c["D"],size,np.eye(3),balance=0)
    maps=cv2.fisheye.initUndistortRectifyMap(c["K"],c["D"],np.eye(3),K,size,cv2.CV_16SC2)
    M,bev,_=lane.bev_transform(g,K,size[1])
    H=lane.ground_to_image(g,K)
    detector=Detector()
    for path in sorted(RUN.glob("frame_*.jpg")):
        raw=cv2.imread(str(path)); und=cv2.remap(raw,*maps,cv2.INTER_LINEAR)
        e,_=lane.edge_masks(und,M,bev)
        pts=lane.to_cm(np.argwhere(e>0)[:,::-1]); pts=pts[(pts[:,1]>=6)&(pts[:,1]<=35)]
        obs={"edge":pts[::max(1,int(np.ceil(len(pts)/100)))]} if len(pts)>=15 else {}
        for det in detector.detect(raw,confidence=.5):
            k=det["kind"]
            if k not in ("line","dashline","cross"): continue
            a,b,cx,d=det["box"]
            pix=np.array([[[(a+cx)/2,(b+d)/2]]],dtype=float)
            uv=cv2.fisheye.undistortPoints(pix,c["K"],c["D"],P=K).reshape(2)
            q=np.linalg.solve(H,[*uv,1])
            if q[2]<=0: continue
            pos=q[:2]/q[2]
            if 6<=pos[1]<=40:
                obs.setdefault(k,[]).append(pos)
        yield int(path.stem.split("_")[1]),und,{k:np.asarray(v) for k,v in obs.items()},H


def road_contours(dist):
    curves,_=cv2.findContours((dist>=0).astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    return [np.vstack((c[:,0],c[0,0]))/2-160 for c in curves]


def highlight_edges(image, contours, pose, H):
    s,c=np.sin(pose[2]),np.cos(pose[2])
    paths=[]
    for curve in contours:
        dx,dy=(curve-pose[:2]).T
        lat,fwd=c*dx-s*dy,s*dx+c*dy
        ids=np.flatnonzero((fwd>=6)&(fwd<=40))
        for part in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1):
            if len(part)<2: continue
            q=H@np.array([lat[part],fwd[part],np.ones(len(part))])
            paths.append(np.rint((q[:2]/q[2]).T).astype(np.int32))
    cv2.polylines(image,paths,False,(42,15,42),5,cv2.LINE_AA)
    cv2.polylines(image,paths,False,(236,35,255),3,cv2.LINE_AA)


def run(photo_start=False):
    cv2.setNumThreads(1)
    body=json.loads((HERE.parent/"track_guard/measurements.photo.json").read_text(encoding="utf-8"))["vehicle"]
    half_width,half_length=body["full_width_including_wheels"]/2,body["full_length"]/2
    board_fit=json.loads((HERE/"calibration_photos/board_fit.json").read_text(encoding="utf-8"))
    camera_ahead_of_body_center=half_length-board_fit["all"]["camera_behind_front_cm"]
    body_outline=np.array([[-half_width,-half_length],[half_width,-half_length],
                           [half_width,half_length],[-half_width,half_length]])
    cfg,fields,dist=geometry()
    contours=road_contours(dist)
    with (RUN/"log.csv").open(encoding="utf-8") as f: logs={int(r["frame"]):r for r in csv.DictReader(f)}
    samples=list(inputs())
    output=HERE/"photo_left_down" if photo_start else HERE
    output.mkdir(exist_ok=True)
    summary={}
    for label in ("normal","blind_510_550"):
        rng=np.random.default_rng(20261105); names=["ccw-S"] if photo_start else list(cfg["APPROACH"]); n=240
        cloud=[]
        for name in names:
            x,y,yaw=cfg["APPROACH"][name]
            angle=np.radians(yaw); before=rng.uniform(-5,35,n); off=rng.uniform(-6,6,n)
            if photo_start:
                # Photo right is map north (two red lines); photo down is map east.
                x,y,_=next(mark for mark in cfg["CROSSWALKS"] if mark[1]<-75)
                before=rng.uniform(-25,25,n)
            cloud.append(np.column_stack([x-before*np.sin(angle)+off*np.cos(angle),
                                          y-before*np.cos(angle)-off*np.sin(angle),
                                          angle+np.radians(rng.uniform(-20,20,n))]))
        cloud=np.asarray(cloud); bank_log=np.zeros(len(names)); history=[]; last=0
        fps=(len(samples)-1)/(float(logs[samples[-1][0]]["t"])-float(logs[samples[0][0]]["t"]))
        filename="replay.mp4" if label=="normal" else "replay_blind.mp4"
        video=cv2.VideoWriter(str(output/filename),cv2.VideoWriter_fourcc(*"mp4v"),fps,(800,360))
        if video is not None and not video.isOpened(): raise RuntimeError("video writer")
        for frame,und,features,H in samples:
            p=cloud.reshape(-1,3)
            for j in range(last+1,frame+1):
                dt=float(logs[j]["t"])-float(logs[j-1]["t"]); u=logs[max(0,j-cfg["LAG"])]
                l,r=float(u["left"]),float(u["right"]); v=cfg["KV"]*(l+r)/2
                k=cfg["YAW"]; diff=(l-r)/2
                omega=np.radians(k["kr"]*max(diff-k["dead"],0)+k["kl"]*min(diff+k["dead"],0)+k["c"])
                a=p[:,2].copy(); da=omega*dt+rng.normal(0,np.radians(6)*np.sqrt(dt),len(p))
                ds=v*dt+rng.normal(0,1.1*np.sqrt(dt),len(p)); lateral=rng.normal(0,.55*np.sqrt(dt),len(p))
                mid=a+da/2; b=a+da
                p[:,0]+=ds*np.sin(mid)+lateral*np.cos(mid)+cfg["D_CAM"]*(np.sin(b)-np.sin(a))
                p[:,1]+=ds*np.cos(mid)-lateral*np.sin(mid)+cfg["D_CAM"]*(np.cos(b)-np.cos(a))
                p[:,2]=b
            last=frame
            obs={} if label!="normal" and 510<=frame<=550 else features
            loss=np.zeros(len(p))
            for k,points in obs.items():
                ds=fields[k][world(p,points)]
                loss+=np.minimum(ds,6 if k=="edge" else 20).mean(axis=1)/(1.6 if k=="edge" else 3.0)
            loss=loss.reshape(len(names),n)
            w=np.exp(-loss); bank_log+=np.log(w.mean(axis=1)); w/=w.sum(axis=1,keepdims=True)
            best=int(bank_log.argmax()); pose=(w[best,:,None]*cloud[best]).sum(axis=0)
            spread=np.sqrt((w[best]*np.sum((cloud[best,:,:2]-pose[:2])**2,axis=1)).sum())
            history.append(dict(frame=frame,t=float(logs[frame]["t"]),start=names[best],x=pose[0],y=pose[1],
                                yaw_deg=np.degrees(pose[2])%360,within_branch_rms_cm=spread,observations=list(obs)))
            if video is not None:
                def camera_coords(X,Y):
                    dx=X-pose[0]; dy=Y-pose[1]; s,b=np.sin(pose[2]),np.cos(pose[2])
                    return b*dx-s*dy,s*dx+b*dy
                overlay=und.copy()
                highlight_edges(overlay,contours,pose,H)
                for kind,col in [('cross',(255,230,50)),('line',(0,0,255)),('dashline',(170,90,255))]:
                    ey,ex=np.nonzero(fields[kind]<1); la,fw=camera_coords(ex/2-160,ey/2-160)
                    mask=(fw>=6)&(fw<=40)
                    q=H@np.array([la[mask],fw[mask],np.ones(mask.sum())]); q=np.rint(q[:2]/q[2]).astype(int)
                    good=(q[0]>=0)&(q[0]<640)&(q[1]>=0)&(q[1]<480)
                    overlay[q[1,good],q[0,good]]=col
                canvas=np.zeros((360,800,3),np.uint8);canvas[:,:480]=cv2.resize(overlay,(480,360))
                mat=cv2.resize(np.where(dist[::-1]>=0,65,25).astype(np.uint8),(320,320))
                panel=cv2.cvtColor(mat,cv2.COLOR_GRAY2BGR)
                for group in range(len(names)):
                    xy=cloud[group,:,:2]; uv=np.rint(np.c_[160+xy[:,0],160-xy[:,1]]).astype(int).clip(0,319)
                    panel[uv[:,1],uv[:,0]]=(0,190,255) if group==best else (110,95,90)
                # Board-derived longitudinal mount; lateral offset assumed zero for this overlay.
                rect=body_outline-[0,camera_ahead_of_body_center]
                s,b=np.sin(pose[2]),np.cos(pose[2])
                xy=rect@np.array([[b,-s],[s,b]])+pose[:2]
                cv2.polylines(panel,[np.rint(np.c_[160+xy[:,0],160-xy[:,1]]).astype(np.int32)],True,(70,240,70),1)
                canvas[:320,480:]=cv2.rotate(panel,cv2.ROTATE_90_CLOCKWISE) if photo_start else panel
                cv2.putText(canvas,f'#{frame} {names[best]}   local RMS {spread:.1f} cm',(483,344),0,.54,(255,255,255),1)
                if not obs:
                    cv2.putText(canvas,'Prediction only - image withheld from filter',(6,26),0,.55,(20,220,255),2)
                video.write(canvas)
            if obs:
                draws=(rng.random((len(names),1))+np.arange(n))/n
                cloud=np.stack([row[np.searchsorted(weight.cumsum(),draw).clip(0,n-1)]
                                for row,weight,draw in zip(cloud,w,draws)])
        if video is not None: video.release()
        summary[label]=history
    (output/"replay_results.json").write_text(json.dumps(summary,indent=1),encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--photo-left-down",action="store_true",help="Start near the left outer crosswalk, toward the bottom-left corner in the user's photo.")
    run(parser.parse_args().photo_left_down)
