"""Estimate left/right command-to-yaw gains from continuous views of printed markings."""
from dataclasses import dataclass
import math
import cv2
import numpy as np

def wrap_line(a):
    return (a+90.)%180.-90.

@dataclass
class Mark:
    kind: str
    center: np.ndarray   # camera-relative right, forward in cm
    angle: float         # transverse paint direction in the ground plane
    count: int

class PaintDetector:
    def __init__(self,assets):
        # One interpolation from the native image to a metric floor grid, including peripheral pixels.
        v,u=np.indices((180,300),dtype=float)
        bev=np.stack([u+50,v+40,np.ones_like(u)],axis=-1)
        rect=bev@np.linalg.inv(assets["M"]).T
        rect=rect[:,:,:2]/rect[:,:,2,None]
        norm=np.dstack([rect,np.ones_like(u)])@np.linalg.inv(assets["rect_K"]).T
        raw=cv2.fisheye.distortPoints(np.ascontiguousarray(norm[:,:,:2]).reshape(-1,1,2),
                                      assets["K"],assets["D"]).reshape(rect.shape)
        self.maps=[raw.astype(np.float32),rect.astype(np.float32)]
        self.kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))

    def detect(self,image,rectified=False):
        bird=cv2.remap(image,self.maps[int(rectified)],None,cv2.INTER_LINEAR)
        b,g,r=cv2.split(bird.astype(np.int16))
        gray=cv2.cvtColor(bird,cv2.COLOR_BGR2GRAY)
        contrast=cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,self.kernel)
        pink=(contrast>6)&(b>g+2)&(r>g+1)&(abs(r-b)<55)&(gray>50)
        white=(gray>150)&(abs(b-g)<52)&(abs(g-r)<65)
        result=[]
        # Blurred paint merges into thin lines in BEV. Verify repeated bright dashes
        # along each candidate in the original image before accepting it.
        joined=cv2.dilate(pink.astype(np.uint8),np.ones((5,5),np.uint8))
        native_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        for contour in cv2.findContours(joined,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
            x,y,w,h=cv2.boundingRect(contour)
            if max(w,h)<32:continue
            cmask=np.zeros((h,w),np.uint8)
            cv2.drawContours(cmask,[contour-np.array([x,y],np.int32)],-1,1,-1)
            ys,xs=np.nonzero(pink[y:y+h,x:x+w] & cmask.astype(bool))
            pts=np.column_stack([xs+x,ys+y]).astype(float)
            if len(pts)<18:continue
            center=pts.mean(axis=0)
            _,_,vt=np.linalg.svd(pts-center,full_matrices=False)
            along=(pts-center)@vt[0]
            if not 34<=np.ptp(along)<=115 or np.std((pts-center)@vt[1])>1.4:continue
            distances=np.arange(along.min()+1,along.max()-1,.35)
            samples=(center+distances[:,None]*vt[0]).astype(np.float32).reshape(1,-1,2)
            coords=cv2.remap(self.maps[int(rectified)],samples,None,cv2.INTER_LINEAR)
            intensity=cv2.remap(native_gray,coords,None,cv2.INTER_LINEAR).astype(np.float32)
            signal=(cv2.GaussianBlur(intensity,(0,0),1.5)-cv2.GaussianBlur(intensity,(0,0),6)).ravel()
            peaks=np.flatnonzero((signal[1:-1]>signal[:-2])&(signal[1:-1]>=signal[2:])&(signal[1:-1]>1.5))+1
            gaps=np.diff(distances[peaks])
            if len(gaps)<3 or np.mean((gaps>1.7)&(gaps<16))<.75:continue
            p=(center-[150,210])*[.2,-.2]
            angle=math.degrees(math.atan2(-vt[0,1],vt[0,0]))
            result.append(Mark("dash",p,wrap_line(angle),len(peaks)))
        centers=[]; angles=[]
        contours=cv2.findContours(white.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        for contour in contours:
            area=cv2.contourArea(contour)
            (cx,cy),(w,h),angle=cv2.minAreaRect(contour)
            lo,hi=sorted((w,h))
            if 70<=area<=1000 and 4<=lo<=19 and 23<=hi<=80 and hi>2.7*lo and area>.6*w*h:
                centers.append([(cx-150)/5,(210-cy)/5])
                angles.append(wrap_line(-angle-(90 if h>w else 0)))
        pts=np.asarray(centers)
        if not 4<=len(pts)<=20:
            return result
        hypotheses=[]
        for i,p in enumerate(pts[:-1]):
            directions=pts[i+1:]-p
            for vec in directions[np.sum(directions**2,axis=1)>3.**2]:
                unit=vec/np.linalg.norm(vec)
                along=(pts-p)@unit
                line=np.degrees(np.arctan2(unit[1],unit[0]))
                aligned=abs(wrap_line(np.asarray(angles)-line-90))<15
                ids=np.flatnonzero((abs((pts-p)@[-unit[1],unit[0]])<.8)&aligned)
                ids=ids[np.argsort(along[ids])]
                for grp in np.split(ids,np.flatnonzero(np.diff(along[ids])>4.8)+1):
                    if len(grp)<4: continue
                    span=np.ptp(along[grp])
                    if 7<=span<=23 and np.min(np.diff(along[grp]))>1.5:
                        hypotheses.append((len(grp),span,grp))
        used=set()
        for _,_,grp in sorted(hypotheses,key=lambda h:(h[0],h[1]),reverse=True):
            if used.intersection(grp):continue
            used.update(grp)
            values=pts[grp]
            center=values.mean(axis=0)
            _,_,vt=np.linalg.svd(values-center,full_matrices=False)
            angle=math.degrees(math.atan2(vt[0,1],vt[0,0]))
            result.append(Mark("cross",center,wrap_line(angle),len(grp)))
        return result

class GainTracker:
    """Disjoint learning windows. Local tracker does not require a reliable absolute map pose."""
    def __init__(self):
        self.left=13./18.
        self.right=1.    # no right-turn calibration yet
        self.previous=None
        self.samples=[]
        self.event=None

    def gain(self,signed):
        return self.left if signed<0 else self.right

    def reset(self):
        self.previous=None
        self.samples=[]

    def update(self,marks,raw_deg,dt,forward_cm):
        self.event=None
        if dt<=0 or dt>.55 or not np.isfinite([dt,raw_deg,forward_cm]).all():
            self.reset()
            return
        match=None
        if self.previous is not None:
            prev=self.previous
            da=raw_deg*self.gain(raw_deg)
            rad=math.radians(da)
            R=np.array([[math.cos(rad),-math.sin(rad)],[math.sin(rad),math.cos(rad)]])
            # Rotate about axle midpoint, then express predicted feature in the camera frame.
            predicted=R@(prev.center+[0,12.36]-[0,forward_cm])-[0,12.36]
            valid=[m for m in marks if m.kind==prev.kind and np.linalg.norm(m.center-predicted)<3.5
                   and abs(wrap_line(m.angle-prev.angle-da))<4.]
            if len(valid)==1:match=valid[0]
        if match is None:
            self.reset()
            if len(marks)==1:
                self.previous=marks[0]
            return
        delta=wrap_line(match.angle-self.previous.angle)
        self.previous=match
        if abs(raw_deg)<.04 or (self.samples and raw_deg*self.samples[-1][0]<=0):
            self.samples=[]
            return
        self.samples.append((raw_deg,delta,dt))
        values=np.array(self.samples)
        raw,visual,elapsed=values.sum(axis=0)
        if abs(raw)<16 or len(values)<4:
            if elapsed>4:self.samples=[]
            return
        self.samples=[]
        ratio=visual/raw
        # Reject a jump that dominated the entire fit, or conflicting angle observations.
        residual=np.cumsum(values[:,1])-ratio*np.cumsum(values[:,0])
        if not .4<=ratio<=1.35 or abs(visual)<8 or np.max(abs(residual))>2.:
            return
        old=self.gain(raw)
        new=float(np.clip(old+.25*(ratio-old),old-.05,old+.05))
        if raw<0:self.left=new
        else:self.right=new
        self.event=dict(mark=match.kind,raw_deg=raw,visual_deg=visual,before=old,after=new,seconds=elapsed)



