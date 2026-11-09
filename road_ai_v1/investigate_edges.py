"""Matched-prior attribution on real stop windows; no control changes."""
import csv
import io
import json
import sys
import zipfile
from pathlib import Path
import cv2
import numpy as np
import torch
from sim_clean import Driver,HERE,RoadNet
from controller import axes,LENS_AHEAD

def residual(driver,prior,bird):
    p=bird*np.array([.2,-.2])+[-40,50+LENS_AHEAD]
    return driver.sample(driver.sdf,p@axes(prior[2])+prior[:2])

def row_limits(edge,road):
    result=np.zeros_like(edge)
    for y in range(75,min(221,len(edge))):
        # Select the first observed road exit on each side of the vehicle centreline.
        if not road[y,200]: continue
        for seq in (np.arange(200,-1,-1),np.arange(200,edge.shape[1])):
            off=seq[~road[y,seq]]
            if len(off):
                near=seq[(abs(seq-off[0])<=1)&edge[y,seq]]
                if len(near): result[y,near[0]]=True
    return result

def main():
    cv2.setNumThreads(1);torch.set_num_threads(3)
    d=Driver("ccw"); net=RoadNet().eval(); kernel=np.ones((3,3),np.uint8)
    with zipfile.ZipFile(sys.argv[1]) as z:
        net.load_state_dict(torch.load(io.BytesIO(z.read("best.pt")),weights_only=True,map_location="cpu"))
    annotations=json.loads((HERE/"kit/data/real_points.json").read_text())
    out=HERE/"real_drive_review"
    records=[]; diagnostics=[]
    for path in sorted((HERE.parent/"map_localization/outer_driver/field_runs").glob("*_edges")):
        scene=next(v for k,v in [('173559','seam'),('173933','glare'),('174607','curve'),('184214','curve_repeat')] if k in path.name)
        d.color_value_min=135 if scene.startswith('curve') else 130
        for file in sorted(path.glob("*.npz")):
            prior=np.load(file)["prior"];raw=cv2.imread(str(file.with_suffix(".png")))
            rect=cv2.remap(raw,*d.remap,cv2.INTER_LINEAR)
            im=cv2.resize(rect,(320,240),interpolation=cv2.INTER_AREA)
            x=torch.from_numpy(cv2.cvtColor(im,cv2.COLOR_BGR2RGB).transpose(2,0,1).copy()).float()[None]/255
            with torch.inference_mode(): labels=net(x).argmax(1)[0].numpy().astype(np.uint8)
            full=cv2.resize(labels,d.size,interpolation=cv2.INTER_NEAREST)
            ground=cv2.warpPerspective(full,d.M,d.bev,flags=cv2.INTER_NEAREST,borderValue=255)
            cnn=(ground==1)&(cv2.dilate(np.uint8(ground==0),kernel)>0)
            cam_edge=(labels==1)&(cv2.dilate(np.uint8(labels==0),kernel)>0)
            img_first=cv2.warpPerspective(cv2.resize(cam_edge.astype(np.uint8),d.size,interpolation=cv2.INTER_NEAREST),d.M,d.bev,flags=cv2.INTER_NEAREST)>0
            hsv=np.zeros_like(cnn); hsv[Driver.boundary_pixels_bev(d,rect)]=True
            masks={'CNN':cnn,'HSV':hsv,'camera_edge_first':img_first,'first_exit':row_limits(cnn,ground==1),'colour_supported':cnn&(cv2.dilate(hsv.astype(np.uint8),np.ones((11,11),np.uint8))>0)}
            for name,mask in masks.items():
                d.pose=prior.copy();d.residual=None
                d.boundary_pixels_bev=lambda _,m=mask:np.nonzero(m)
                ok=d.observe(rect,True)
                records.append([scene,file.stem,name,int(ok),d.residual])
            if file.stem=='23':
                for index,curve in enumerate(annotations[scene]['true_boundary']):
                    line=np.zeros(labels.shape,np.uint8);cv2.polylines(line,[np.array(curve,np.int32)],False,1)
                    y,x=np.nonzero(line); bird=cv2.perspectiveTransform(np.float32(np.c_[x,y]*2)[:,None],d.M)[:,0]
                    bird=bird[(bird[:,1]>=75)&(bird[:,1]<=220)]
                    if len(bird):diagnostics.append([scene,'human segment '+str(index),len(bird),round(float(np.median(abs(residual(d,prior,bird)))),2)])
                y,x=np.nonzero(cnn);valid=(y>=75)&(y<=220)
                for name,mask in [('left',x<200),('right',x>=200),('near',y>=150),('far',y<150)]:
                    bird=np.c_[x[valid&mask],y[valid&mask]]
                    if len(bird):diagnostics.append([scene,name,len(bird),round(float(np.median(abs(residual(d,prior,bird)))),2)])
    with (out/'ablation.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['scene','frame','variant','accepted','median_abs_cm']);w.writerows(records)
    print(diagnostics)
    for scene in ('seam','glare','curve','curve_repeat'):
        print(scene,[(name,sum(r[3] for r in records if r[0]==scene and r[2]==name)) for name in masks])
    (out/'residual_regions.json').write_text(json.dumps(diagnostics,indent=2))

if __name__=='__main__':main()
