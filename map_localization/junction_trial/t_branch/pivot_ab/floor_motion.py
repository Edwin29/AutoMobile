"""Independent ground-plane yaw from matched camera pixels; never receives simulated pose."""
import cv2
import numpy as np
import math

class FloorMotion:
    def __init__(self):
        self.reset()

    def reset(self):
        self.previous=None
        self.samples=[]

    def track(self,frame,H):
        current=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        old=self.previous
        self.previous=current
        if old is None:return None
        features=cv2.goodFeaturesToTrack(old,180,.01,4)
        if features is None or len(features)<6:return None
        new,ok,_=cv2.calcOpticalFlowPyrLK(old,current,features,None,winSize=(19,19),maxLevel=2)
        back,back_ok,_=cv2.calcOpticalFlowPyrLK(current,old,new,None,winSize=(19,19),maxLevel=2)
        valid=(ok.ravel()>0)&(back_ok.ravel()>0)&(np.linalg.norm((features-back)[:,0],axis=1)<.8)
        def ground(points):
            return (cv2.perspectiveTransform(points,H)[:,0]-[200,250])*[.2,-.2]
        p,q=ground(features),ground(new)
        for pts in (p,q):
            valid&=(pts[:,1]>6)&(pts[:,1]<35)&(np.abs(pts[:,0])<32)
        p,q=p[valid],q[valid]
        if len(p)<6:return None
        fit,mask=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=.45)
        if fit is None:return None
        keep=mask.ravel().astype(bool)
        if keep.sum()<6 or keep.mean()<.70 or np.max(np.ptp(p[keep],axis=0))<9:return None
        if abs(np.linalg.norm(fit[:,0])-1)>.04:return None
        return math.degrees(math.atan2(fit[1,0],fit[0,0]))

    def update(self,frame,H,raw):
        delta=self.track(frame,H)
        if delta is None:
            self.samples.clear()
            return None
        self.samples.append((raw,delta))
        a=np.array(self.samples)
        total,visual=a.sum(axis=0)
        if total<16:return None
        self.samples.clear()
        ratio=visual/total
        if len(a)<4 or not .4<=ratio<=1.35 or visual<8:return None
        if np.max(np.abs(np.cumsum(a[:,1])-ratio*np.cumsum(a[:,0])))>2:return None
        return float(ratio)
