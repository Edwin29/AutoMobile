"""Camera-to-wheel controller. World frame x east, y north, yaw positive clockwise."""
from pathlib import Path
import math
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
TRACK_CM = 13.
WHEEL_SPEED = 16.
LENS_AHEAD = 12.36
CENTER_AHEAD = 2.25

def axes(yaw):
    s,c=math.sin(yaw), math.cos(yaw)
    return np.array([[c,-s],[s,c]])

def advance(pose, wheels, dt, yaw_gain=1.):
    left,right=np.asarray(wheels)*WHEEL_SPEED
    a=yaw_gain*(left-right)*dt/TRACK_CM
    mid=pose[2]+a/2
    d=(left+right)*dt/2*np.sinc(a/(2*np.pi))
    return pose+np.array([d*math.sin(mid),d*math.cos(mid),a])

class Driver:
    color_value_min=130
    def __init__(self,direction,adaptive_yaw=False):
        if direction not in ("cw","ccw"):
            raise ValueError("direction must be cw or ccw")
        self.yaw_tracker=None
        self.paint_motion=np.zeros(3)  # raw yaw degrees, elapsed seconds, forward cm
        self.paint_sign=0
        with np.load(HERE/"track.npz") as data:
            if adaptive_yaw:
                from yaw_gain import GainTracker, PaintDetector
                self.paint_detector=PaintDetector(data)
                self.yaw_tracker=GainTracker()
            self.route=data["route"][::1 if direction=="cw" else -1].copy()
            self.sdf=data["sdf"]
            self.M=data["M"]
            self.bev=tuple(map(int,data["bev"]))
            self.size=tuple(map(int,data["size"]))
            self.remap=cv2.fisheye.initUndistortRectifyMap(data["K"],data["D"],np.eye(3),
                                      data["rect_K"],self.size,cv2.CV_16SC2)
        self.gy,self.gx=np.gradient(self.sdf,.25)
        self.route=np.vstack([self.route,self.route[:1]])
        self.arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(self.route,axis=0),axis=1))]
        yaw=-np.pi/2 if direction=="cw" else np.pi/2
        self.pose=np.r_[np.array([21.5,-87.25])-CENTER_AHEAD*axes(yaw)[1],yaw]
        self.last_index=self.nearest()
        self.distance=0.
        self.target_laps=1
        self.out=np.zeros(2)
        self.filtered=0.
        self.good=0
        self.reason=""
        front,rear=14.5,-10.
        corners=np.array([[-8.,rear],[8.,rear],[8.,front],[-8.,front]])
        self.rim=np.concatenate([np.linspace(a,b,math.ceil(np.linalg.norm(b-a)/.25),endpoint=False)
                                 for a,b in zip(corners,np.roll(corners,-1,axis=0))])
        self.residual=None
        self.margin=None

    def nearest(self):
        return int(np.argmin(np.sum((self.route[:-1]-self.pose[:2])**2,axis=1)))

    def wheel_targets(self,speed,curvature):
        return speed*(1+np.array([1.,-1.])*curvature*TRACK_CM/2)

    def sample(self,field,points):
        x,y=np.clip((points+130)*4,0,len(field)-1.001).T
        i,j=x.astype(int),y.astype(int)
        u,v=x-i,y-j
        return (1-v)*((1-u)*field[j,i]+u*field[j,i+1])+v*((1-u)*field[j+1,i]+u*field[j+1,i+1])

    def boundary_pixels_bev(self,und):
        hsv=cv2.cvtColor(und,cv2.COLOR_BGR2HSV)
        colored=cv2.inRange(hsv,(20,100,self.color_value_min),(87,255,255))
        road=((hsv[:,:,2]<130)&(colored==0)).astype(np.uint8)*255
        color_bev=cv2.warpPerspective(colored,self.M,self.bev)>=128
        road_bev=(cv2.warpPerspective(road,self.M,self.bev)>=128).astype(np.uint8)
        return np.nonzero(color_bev & (cv2.dilate(road_bev,np.ones((3,3),np.uint8))>0))

    def observe(self,image,rectified):
        if getattr(self,"record_edges",False):
            self.edge_observation=None
        if image is None or image.shape!=(self.size[1],self.size[0],3):
            return False
        und=image if rectified else cv2.remap(image,*self.remap,cv2.INTER_LINEAR)
        y,x=self.boundary_pixels_bev(und)
        p=np.column_stack([x/5.-40,50.-y/5.])
        p=p[(p[:,1]>=6)&(p[:,1]<=35)]
        if len(p)<20 or np.ptp(p[:,1])<5:
            return False
        p=p[::max(1,math.ceil(len(p)/140))].copy()
        p[:,1]+=LENS_AHEAD
        rot=axes(self.pose[2])
        xy=p@rot+self.pose[:2]
        residual=self.sample(self.sdf,xy)
        self.residual=float(np.median(np.abs(residual)))
        valid=np.abs(residual)<2.5
        if getattr(self,"record_edges",False):
            self.edge_observation=(p.copy(),residual.copy(),self.pose.copy())
        if valid.mean()<.55 or self.residual>1.8:
            return False
        # Linearize map distance at observed edges. Regularization preserves
        # dead-reckoned progress along a straight, where edges give little information.
        xy=xy[valid]
        gradient=np.column_stack([self.sample(f,xy) for f in (self.gx,self.gy)])
        dxy=p[valid,1,None]*rot[0]-p[valid,0,None]*rot[1]
        J=np.column_stack([gradient,(gradient*dxy).sum(axis=1)/25])
        correction=np.linalg.solve(J.T@J/len(J)+np.diag([.55,.55,.55]),
                                   -(J.T@residual[valid])/len(J))
        self.pose+=np.clip(correction,[-.45,-.45,-.40],[.45,.45,.40])*[1,1,1/25]
        return True

    def stop(self,reason):
        self.reason=reason
        self.out[:]=0
        return self.out.copy()

    def predict(self,applied,dt):
        raw_yaw=(applied[0]-applied[1])*WHEEL_SPEED*dt/TRACK_CM
        gain=1.
        if self.yaw_tracker is not None:
            gain=self.yaw_tracker.gain(raw_yaw)
            sign=np.sign(raw_yaw)
            if sign and self.paint_sign and sign!=self.paint_sign:
                self.yaw_tracker.reset()
            if sign:
                self.paint_sign=sign
            self.paint_motion += [math.degrees(raw_yaw),dt,sum(applied)*WHEEL_SPEED*dt/2]
        self.pose=advance(self.pose,applied,dt,gain)

    def step(self,frame,dt,applied,rectified=False):
        if self.reason:
            return self.out.copy()
        if self.yaw_tracker is not None:
            self.yaw_tracker.event=None
        if not 0<dt<=.35 or not np.isfinite(applied).all():
            return self.stop("stale timing")
        self.predict(applied,dt)
        self.residual=None
        if not self.observe(frame,rectified):
            return self.stop("unmatched road edges")
        idx=self.nearest()
        length=self.arc[-1]
        move=(self.arc[idx]-self.arc[self.last_index]+length/2)%length-length/2
        if abs(move)>4:
            return self.stop("route discontinuity")
        self.last_index=idx
        self.distance+=move
        if self.distance>=self.target_laps*length:
            return self.stop("lap complete")
        ahead=(self.arc[idx]+10)%length
        target=np.array([np.interp(ahead,self.arc,self.route[:,j]) for j in range(2)])
        q=(target-self.pose[:2])@axes(self.pose[2]).T
        if q[1]<=0:
            return self.stop("target behind")
        k=2*q[0]/max(q@q,1e-6)
        self.filtered=.3*self.filtered+.7*k
        v=.3/(1+10*abs(self.filtered))
        requested=self.wheel_targets(v,self.filtered)
        next_out=self.out+np.clip(requested-self.out,-.6*dt,.6*dt)
        margins=[]
        forecast_gains={1.}
        if self.yaw_tracker is not None:
            forecast_gains.add(self.yaw_tracker.gain(next_out[0]-next_out[1]))
        for h in (0.,.15,.30):
            for factor in forecast_gains:
                predicted=advance(self.pose,next_out,h,factor)
                margins.append(self.sample(self.sdf,self.rim@axes(predicted[2])+predicted[:2]).min())
        self.margin=float(min(margins))
        if self.margin<.25:
            return self.stop("footprint limit")
        self.good+=1
        if self.good>=3:
            self.out=next_out
        if self.yaw_tracker is not None:
            marks=self.paint_detector.detect(frame,rectified)
            self.yaw_tracker.update(marks,*self.paint_motion)
            self.paint_motion[:]=0.
            self.paint_sign=0
        return self.out.copy()
