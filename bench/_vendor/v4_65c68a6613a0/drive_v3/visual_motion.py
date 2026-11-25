"""Sparse floor VO. Motion estimate is not an independent sensor from map vision."""
from dataclasses import dataclass
import numpy as np
import cv2

from .geometry import wrap


@dataclass
class MotionObservation:
    start_s:float
    end_s:float
    translation_body_cm:np.ndarray
    yaw_rad:float
    inliers:int
    rms_cm:float
    status:str


class FloorVO:
    def __init__(self,geometry,cfg):
        self.g=geometry;self.cfg=cfg;self.previous=None

    def update(self,debug,now):
        gray=cv2.cvtColor(debug["bev"],cv2.COLOR_BGR2GRAY)
        mask=(debug["valid"] & debug["road"] & ~debug["glare_suspect"]).astype(np.uint8)*255
        # Include paint features only on/adjacent to observed floor, not elevated objects.
        support=cv2.dilate((debug["road"] & debug["valid"]).astype(np.uint8),np.ones((7,7),np.uint8))>0
        mask[(debug["white"]|debug["pink"]|debug["red"]) & support & debug["valid"]]=255
        old=self.previous
        self.previous=(gray,mask,now)
        if old is None or not 0<now-old[2]<=self.cfg["vo_max_dt_s"]:
            return None
        a=cv2.goodFeaturesToTrack(old[0],100,.02,6,mask=old[1])
        if a is None or len(a)<self.cfg["vo_min_inliers"]:
            return None
        b,ok,_=cv2.calcOpticalFlowPyrLK(old[0],gray,a,None,winSize=(15,15),maxLevel=2)
        back,ok2,_=cv2.calcOpticalFlowPyrLK(gray,old[0],b,None,winSize=(15,15),maxLevel=2)
        valid=(ok.ravel()>0)&(ok2.ravel()>0)&(np.linalg.norm(back[:,0]-a[:,0],axis=1)<1.)
        uv=np.rint(b[:,0]).astype(int)
        inside=(uv[:,0]>=0)&(uv[:,0]<mask.shape[1])&(uv[:,1]>=0)&(uv[:,1]<mask.shape[0])
        valid &= inside
        indices=np.flatnonzero(valid)
        valid[indices] &= mask[uv[indices,1],uv[indices,0]]>0
        if valid.sum()<self.cfg["vo_min_inliers"]:
            return None
        prev=self.g.pixels_to_body(a[valid,0]);curr=self.g.pixels_to_body(b[valid,0])
        A,inliers=cv2.estimateAffinePartial2D(prev,curr,method=cv2.RANSAC,ransacReprojThreshold=.7)
        if A is None or inliers.sum()<self.cfg["vo_min_inliers"]:
            return None
        keep=inliers.ravel()>0
        # Repeated collinear dashes can fit well while translation is ambiguous.
        # Require two-dimensional support, not only a small RANSAC residual.
        if np.linalg.eigvalsh(np.cov(prev[keep].T))[0]<2.0:
            return None
        u,_,vt=np.linalg.svd(A[:,:2]);R=u@vt
        if not .95<np.linalg.norm(A[:,0])<1.05 or np.linalg.det(R)<0:
            return None
        t=np.median(curr[keep]-prev[keep]@R.T,axis=0)
        rms=float(np.sqrt(np.mean(np.sum((curr[keep]-(prev[keep]@R.T+t))**2,axis=1))))
        if rms>self.cfg["vo_max_rms_cm"]:
            return None
        # Body motion is inverse of motion of static floor in the camera/body frame.
        translation=-R.T@t
        yaw=float(wrap(np.arctan2(R[1,0],R[0,0])))
        return MotionObservation(old[2],now,translation,yaw,int(keep.sum()),rms,
                                 "stationary" if np.linalg.norm(translation)<.03 and abs(yaw)<.003 else "moving")
