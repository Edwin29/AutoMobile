"""Independent first-corner yaw estimate from real floor feature tracks and PWM log."""
import csv
import json
import math
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[1]/"junction_logic/visibility_review/run_160024"
from controller import WHEEL_SPEED, TRACK_CM

class Ground:
    def __init__(self):
        with np.load(HERE/"track.npz") as p:
            self.K,self.D,self.rect,self.M=(p[k] for k in ("K","D","rect_K","M"))
        v,u=np.indices((480,640))
        local=self.points(np.stack([u,v],axis=-1).reshape(-1,2)).reshape(480,640,2)
        x,y=local[:,:,0],local[:,:,1]
        self.roi=(abs(x)<32)&(y>7)&(y<40)&(v<425)

    def points(self,uv):
        uv=np.ascontiguousarray(uv,dtype=float).reshape(-1,1,2)
        rect=cv2.fisheye.undistortPoints(uv,self.K,self.D,P=self.rect)
        bev=cv2.perspectiveTransform(rect,self.M).reshape(-1,2)
        return np.column_stack([bev[:,0]/5-40,50-bev[:,1]/5])

    def read(self,i):
        img=cv2.imread(str(RUN/f"frame_{i:05d}.jpg"))
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        # Require nearby ink or green mat; neutral highlights alone cannot supply tracks.
        ink=((hsv[:,:,1]>75)&(hsv[:,:,2]>65)).astype(np.uint8)
        near=cv2.dilate(ink,np.ones((9,9),np.uint8))
        mask=(self.roi&(near!=0)&(hsv[:,:,2]<246)).astype(np.uint8)*255
        return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),mask

def rotation(a,b):
    x=a-a.mean(axis=0)
    y=b-b.mean(axis=0)
    return math.degrees(math.atan2(np.sum(x[:,0]*y[:,1]-x[:,1]*y[:,0]),np.sum(x*y)))

def fit_pair(g,first,second):
    prev,ma=first
    curr,mb=second
    p=cv2.goodFeaturesToTrack(prev,350,.004,5,mask=ma)
    if p is None or len(p)<6:
        return {},None
    opts=dict(winSize=(25,25),maxLevel=3)
    q,fwd,_=cv2.calcOpticalFlowPyrLK(prev,curr,p,None,**opts)
    back,rev,_=cv2.calcOpticalFlowPyrLK(curr,prev,q,None,**opts)
    p,q=p[:,0],q[:,0]
    u,v=np.rint(q).astype(int).T
    valid=(fwd.ravel()>0)&(rev.ravel()>0)&(np.linalg.norm(p-back[:,0],axis=1)<.8)
    valid&=(u>=0)&(u<640)&(v>=0)&(v<480)&(mb[v.clip(0,479),u.clip(0,639)]!=0)
    p,q=p[valid],q[valid]
    if len(p)<6:return {},None
    a,b=g.points(p),g.points(q)
    mat,inliers=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=.4)
    if mat is None:return {},None
    keep=inliers.ravel()>0
    a,b=a[keep],b[keep]
    deg=rotation(a,b)
    scale=np.hypot(*mat[:,0])
    # Widely separated points constrain angle even on a single stripe.
    span=np.sqrt(np.mean(np.sum((a-a.mean(axis=0))**2,axis=1)))
    rad=np.radians(deg)
    R=np.array([[np.cos(rad),-np.sin(rad)],[np.sin(rad),np.cos(rad)]])
    err=np.linalg.norm((a-a.mean(0))@R.T-(b-b.mean(0)),axis=1)
    valid=len(a)>=8 and keep.mean()>=.60 and span>=3 and .96<=scale<=1.04 and np.median(err)<.24
    return dict(deg=deg,accepted=bool(valid),n=len(a),fraction=keep.mean(),span=span,scale=scale,
                median_cm=np.median(err)),(p[keep],q[keep])

def main():
    cv2.setNumThreads(1)
    cv2.setRNGSeed(71)
    g=Ground()
    with (RUN/"log.csv").open(encoding="utf8") as stream: log=list(csv.DictReader(stream))
    ts=np.array([float(r["t"]) for r in log])
    diff=np.array([float(r["left"])-float(r["right"]) for r in log])
    yaw=np.r_[0,np.cumsum(np.degrees(WHEEL_SPEED/TRACK_CM*diff[:-1]*np.diff(ts)))]
    cache={i:g.read(i) for i in range(0,341,5)}
    rows=[]
    for step in (5,10):
        for j in range(step,341,step):
            i=j-step
            r,tracks=fit_pair(g,cache[i],cache[j])
            rows.append(dict(step=step,start=i,end=j,t0=ts[i],t1=ts[j],pwm_deg=yaw[j]-yaw[i],**r))
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (HERE/"corner_yaw_pairs.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    for step in (5,10):
        for lo,hi in [(0,50),(50,100),(100,150),(150,200),(200,250),(250,300),(300,340),(0,340)]:
            rr=[r for r in rows if r["step"]==step and lo<=r["start"] and r["end"]<=hi]
            yes=[r for r in rr if r.get("accepted")]
            print(step,lo,hi,len(yes),len(rr),*[round(sum(r[k] for r in yes),1) for k in ("deg","pwm_deg")],round(yaw[hi]-yaw[lo],1))
if __name__=="__main__":main()
