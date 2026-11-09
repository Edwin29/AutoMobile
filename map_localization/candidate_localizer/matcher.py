"""Local pose search with class-separated, orientation-aware image Chamfer costs.

All poses refer to the axle midpoint in cm and clockwise radians.
The matcher receives a calibrated floor homography, map features, and an image.
It never receives the simulated ground-truth pose.
"""
import math
import time

import cv2
import numpy as np

CLASSES=("boundary","cross","pink","red")


def extract(image):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsv)
    b,g,r=cv2.split(image.astype(np.int16))
    green=(h>=20)&(h<=87)&(s>=100)&(v>=130)
    road=v<130
    masks=[
        green,
        (v>150)&(s<65),
        (b>g+5)&(r>g+5)&(np.abs(r-b)<65),
        (r>g+50)&(r>b+50)&(r>110),
    ]
    near_road=cv2.dilate(road.astype(np.uint8),np.ones((5,5),np.uint8))>0
    result=[]
    for mask in masks:
        binary=mask.astype(np.uint8)*255
        edge=(cv2.Canny(binary,50,100)>0)&near_road
        gx=cv2.Sobel(binary,cv2.CV_32F,1,0,ksize=3)
        gy=cv2.Sobel(binary,cv2.CV_32F,0,1,ksize=3)
        tangent=np.stack((-gy,gx),axis=-1)
        tangent/=np.maximum(np.linalg.norm(tangent,axis=-1,keepdims=True),1e-6)
        result.append((edge,tangent))
    return result


def grid(prior,span_cm=6.,step_cm=2.,span_deg=20.,step_deg=5.):
    x=np.arange(-span_cm,span_cm+step_cm*.1,step_cm)
    a=np.radians(np.arange(-span_deg,span_deg+step_deg*.1,step_deg))
    return np.stack(np.meshgrid(x,x,a,indexing="ij"),axis=-1).reshape(-1,3)+prior


def distance_field(edge,tangent):
    if not edge.any():
        return np.full(edge.shape,12.,np.float32),np.zeros((*edge.shape,2),np.float32)
    dist,labels=cv2.distanceTransformWithLabels(
        (~edge).astype(np.uint8),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
    lut=np.zeros((int(labels.max())+1,2),np.float32)
    lut[labels[edge]]=tangent[edge]
    return dist,lut[labels]


class Matcher:
    def __init__(self,map_image,ppc,extent,ground_to_image,image_size=(320,240)):
        self.size=image_size
        self.H=ground_to_image.astype(np.float32)
        self.Hi=np.linalg.inv(self.H)
        self.features=[]
        for mask,tangent in extract(map_image):
            y,x=np.nonzero(mask)
            # Uniform spatial sampling by half-centimetre map cells.
            cells=np.c_[x//max(1,int(ppc/2)),y//max(1,int(ppc/2))]
            _,keep=np.unique(cells,axis=0,return_index=True)
            x,y=x[keep],y[keep]
            points=np.c_[(x+.5)/ppc-extent,extent-(y+.5)/ppc]
            directions=tangent[y,x]*[1,-1]
            self.features.append((points.astype(np.float32),directions.astype(np.float32)))

    def prepare(self,image,prior):
        image=cv2.resize(image,self.size,interpolation=cv2.INTER_AREA)
        features=extract(image)
        # A fixed camera ROI is shared by every candidate and every fault condition.
        self.roi=np.zeros(image.shape[:2],bool)
        self.roi[int(self.size[1]*.30):-4,4:-4]=True
        yy,xx=np.indices(image.shape[:2],dtype=np.float32)
        ground=np.stack((xx,yy,np.ones_like(xx)),axis=-1)@self.Hi.T
        forward=ground[:,:,1]/np.maximum(ground[:,:,2],1e-6)
        self.roi&=(ground[:,:,2]>0)&(forward>=6)&(forward<=60)
        self.observed=[]
        self.local=[]
        for (edge,tangent),(points,directions) in zip(features,self.features):
            edge&=self.roi
            field,nearest=distance_field(edge,tangent)
            self.observed.append((edge,field,nearest))
            keep=np.linalg.norm(points-prior[:2],axis=1)<100
            self.local.append((points[keep],directions[keep]))

    def project(self,poses,points,directions):
        n=len(poses)
        c=np.cos(poses[:,2]);s=np.sin(poses[:,2])
        transforms=np.zeros((n,3,3),np.float32)
        transforms[:,0,0]=c;transforms[:,0,1]=-s
        transforms[:,1,0]=s;transforms[:,1,1]=c
        transforms[:,0,2]=-c*poses[:,0]+s*poses[:,1]
        transforms[:,1,2]=-s*poses[:,0]-c*poses[:,1]-12.36
        transforms[:,2,2]=1
        homographies=self.H[None]@transforms
        p=np.c_[points,np.ones(len(points),np.float32)]
        q=p[None]@homographies.transpose(0,2,1)
        end=np.c_[points+directions,np.ones(len(points),np.float32)]
        end=end[None]@homographies.transpose(0,2,1)
        uv=q[...,:2]/np.maximum(q[...,2:3],1e-6)
        tangent=end[...,:2]/np.maximum(end[...,2:3],1e-6)-uv
        tangent/=np.maximum(np.linalg.norm(tangent,axis=-1,keepdims=True),1e-6)
        valid=(q[...,2]>0)&(uv[...,0]>=4)&(uv[...,0]<self.size[0]-4)&(
            uv[...,1]>=self.size[1]*.30)&(uv[...,1]<self.size[1]-4)
        floor=p[None]@transforms.transpose(0,2,1)
        valid&=(floor[...,1]>=6)&(floor[...,1]<=60)
        return uv,tangent,valid

    def forward(self,poses):
        scores=[];coverage=[];support=[]
        for start in range(0,len(poses),96):
            batch=poses[start:start+96]
            total=np.zeros(len(batch));weight=np.zeros(len(batch))
            hit_total=np.zeros(len(batch));point_total=np.zeros(len(batch))
            for (points,directions),(edge,field,nearest) in zip(self.local,self.observed):
                if not len(points):continue
                uv,tangent,valid=self.project(batch,points,directions)
                u=np.clip(uv[...,0],0,self.size[0]-1).astype(np.float32)
                v=np.clip(uv[...,1],0,self.size[1]-1).astype(np.float32)
                distances=cv2.remap(field,u,v,cv2.INTER_LINEAR)
                normals=nearest[np.rint(v).astype(int),np.rint(u).astype(int)]
                alignment=np.abs((normals*tangent).sum(axis=-1)).clip(0,1)
                loss=np.minimum(distances/12.,1)+.15*(1-alignment)
                count=valid.sum(axis=1)
                present=count>=8
                total+=present*(loss*valid).sum(axis=1)/np.maximum(count,1)
                weight+=present
                hit_total+=((distances<=3)&valid).sum(axis=1)
                point_total+=count
            score=total/np.maximum(weight,1)
            score[(weight==0)|(point_total<30)]=np.inf
            scores.extend(score);coverage.extend(hit_total/np.maximum(point_total,1))
            support.extend(point_total)
        return np.array(scores),np.array(coverage),np.array(support)

    def verify(self,pose,forward_score):
        # Reverse image-to-projected-map distance is computed only for finalists.
        loss=[];hits=[];counts=[]
        for (points,directions),(edge,_,_) in zip(self.local,self.observed):
            if edge.sum()<8:continue
            uv,_,valid=self.project(pose[None],points,directions)
            predicted=np.zeros(edge.shape,np.uint8)
            coords=np.rint(uv[0,valid[0]]).astype(int)
            if len(coords):
                predicted[coords[:,1],coords[:,0]]=1
                dist=cv2.distanceTransform(1-predicted,cv2.DIST_L2,5)
            else:
                dist=np.full(edge.shape,12.,np.float32)
            values=dist[edge]
            loss.append(np.minimum(values/12,1).mean())
            hits.append(int((values<=3).sum()));counts.append(len(values))
        reverse=float(np.mean(loss)) if loss else 1.
        return .5*forward_score+.5*reverse,sum(hits)/max(1,sum(counts))

    def search(self,image,prior):
        started=time.perf_counter()
        self.prepare(image,prior)
        if sum(edge.sum() for edge,_,_ in self.observed)<30:
            return dict(accepted=False,reason="insufficient features",pose=None,candidates=0,
                        seconds=time.perf_counter()-started),None
        coarse=grid(prior)
        coarse_score,_,_=self.forward(coarse)
        finite=np.flatnonzero(np.isfinite(coarse_score))
        if not len(finite):
            return dict(accepted=False,reason="insufficient features",pose=None,seconds=time.perf_counter()-started),None
        seeds=[]
        for idx in finite[np.argsort(coarse_score[finite])]:
            candidate=coarse[idx]
            if all(np.linalg.norm(candidate[:2]-seed[:2])>=2 or
                   abs(candidate[2]-seed[2])>=math.radians(5) for seed in seeds):
                seeds.append(candidate)
            if len(seeds)==4:break
        refined=np.concatenate([grid(s,2.,.5,5.,1.) for s in seeds])
        within=(np.max(np.abs(refined[:,:2]-prior[:2]),axis=1)<=6.001)&(
            np.abs(refined[:,2]-prior[2])<=math.radians(20.001))
        refined=np.unique(np.round(refined[within],6),axis=0)
        score,coverage,support=self.forward(refined)
        # Preserve separate local minima, plus the best fine candidates.
        order=np.argsort(score)
        finalists=list(order[:12])
        for idx in order:
            if all(np.linalg.norm(refined[idx,:2]-refined[j,:2])>1.5 or
                   abs(refined[idx,2]-refined[j,2])>math.radians(3) for j in finalists):
                finalists.append(int(idx))
            if len(finalists)>=28:break
        checked=[]
        for idx in finalists:
            final,reverse_coverage=self.verify(refined[idx],score[idx])
            checked.append((final,idx,reverse_coverage))
        checked.sort()
        best,idx,reverse_coverage=checked[0]
        distinct=[r[0] for r in checked[1:] if np.linalg.norm(refined[r[1],:2]-refined[idx,:2])>1.5
                  or abs(refined[r[1],2]-refined[idx,2])>math.radians(3)]
        gap=min(distinct)-best if distinct else 0.
        boundary=(np.any(np.abs(refined[idx,:2]-prior[:2])>=5.99) or
                  abs(refined[idx,2]-prior[2])>=math.radians(19.99))
        accepted=bool(best<.32 and coverage[idx]>.55 and reverse_coverage>.55 and gap>.015 and not boundary)
        reason="accepted" if accepted else "ambiguous or poor fit"
        if boundary:reason="search boundary"
        result=dict(accepted=accepted,reason=reason,pose=refined[idx].tolist(),score=float(best),
                    gap=float(gap),forward_coverage=float(coverage[idx]),reverse_coverage=float(reverse_coverage),
                    candidates=len(coarse)+len(refined),seconds=time.perf_counter()-started)
        diagnostic=dict(coarse=coarse,coarse_score=coarse_score,refined=refined,
                        refined_score=score,checked=np.array(checked))
        return result,diagnostic
