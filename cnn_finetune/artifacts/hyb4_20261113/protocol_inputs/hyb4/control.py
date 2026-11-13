"""Body-frame path contract and signed control. No simulator state accepted."""
from dataclasses import dataclass
import numpy as np


def wrap(x):
    return (x+np.pi)%(2*np.pi)-np.pi


def advance(v,w,dt):
    a=w*dt
    return np.array([v*dt*np.sinc(a/(2*np.pi))*np.sin(a/2),
                     v*dt*np.sinc(a/(2*np.pi))*np.cos(a/2),a])


@dataclass(frozen=True)
class LocalPath:
    points_cm: np.ndarray
    speeds_cm_s: np.ndarray
    capture_s: float

    def __post_init__(self):
        p=np.array(self.points_cm,dtype=float,copy=True);s=np.array(self.speeds_cm_s,dtype=float,copy=True)
        if p.ndim!=2 or p.shape[1]!=2 or len(p)<5 or s.shape!=(len(p),):raise ValueError('Expected Nx2 path and N speeds')
        if not np.isfinite(p).all() or not np.isfinite(s).all() or not np.isfinite(self.capture_s):raise ValueError('Nonfinite path')
        if np.any(s<0) or np.any(s>12):raise ValueError('Forward body speed must be in [0,12]cm/s')
        if np.any(np.linalg.norm(np.diff(p,axis=0),axis=1)<1e-5):raise ValueError('Repeated path points')
        p.setflags(write=False);s.setflags(write=False)
        object.__setattr__(self,'points_cm',p);object.__setattr__(self,'speeds_cm_s',s)


class SignedDrive:
    """Known nominal model, signed velocities, ratio-preserving saturation.

    Desired duty is a time-average. Pulse density below floor is clock-based,
    so a slower inference loop does not silently change the nominal average.
    """
    def __init__(self,gain=16.8,track_cm=13.,turn_gain=.6,floor=.36,cap=1.):
        self.gain=gain;self.track=track_cm;self.turn_gain=turn_gain;self.floor=floor;self.cap=cap
        self.area=np.zeros(2);self.last_sign=np.zeros(2)

    def inverse(self,v,w):
        differential=w*self.track/(2*self.turn_gain)
        desired=np.array([v+differential,v-differential])/self.gain
        return desired/max(1.,float(np.max(abs(desired)))/self.cap)

    def intended_motion(self,desired):
        wheels=np.asarray(desired)*self.gain
        return float(wheels.mean()),float((wheels[0]-wheels[1])*self.turn_gain/self.track)

    def applied_motion(self,pwm):
        p=np.asarray(pwm,float);effective=np.where(abs(p)>=.30,p,0.)
        return self.intended_motion(effective)

    def allocate(self,desired,dt):
        if not np.isfinite(dt) or not 0<dt<=.25:raise ValueError('Invalid command interval')
        x=np.asarray(desired,float)
        if x.shape!=(2,) or not np.isfinite(x).all() or np.max(abs(x))>self.cap+1e-9:raise ValueError('Invalid signed command')
        out=x.copy()
        for j in range(2):
            sign=np.sign(x[j])
            if sign!=self.last_sign[j]:self.area[j]=0.
            if 0<abs(x[j])<self.floor:
                self.area[j]+=abs(x[j])*dt
                if self.area[j]>=.5*self.floor*dt:
                    out[j]=sign*self.floor;self.area[j]-=self.floor*dt
                else:out[j]=0.
            else:self.area[j]=0.
            self.last_sign[j]=sign
        return out

    def reset(self):
        self.area[:]=0;self.last_sign[:]=0


class AppliedMotion:
    """Causal nominal actuator response driven exclusively by acknowledged PWM."""
    def __init__(self,drive=None,delay=.05,tau=.10):
        self.drive=drive or SignedDrive();self.delay=delay;self.tau=tau
        self.t=0.;self.vw=np.zeros(2);self.commands=[(-1e9,np.zeros(2))]

    def at(self,t):
        if not np.isfinite(t):raise ValueError('Nonfinite clock')
        if t<self.t-1e-9:raise ValueError('Clock moved backward')
        while self.t<t-1e-10:
            dt=min(.005,t-self.t);pwm=self.commands[0][1]
            for stamp,value in self.commands:
                if stamp<=self.t-self.delay+1e-10:pwm=value
                else:break
            target=np.array(self.drive.applied_motion(pwm))
            self.vw+=(1-np.exp(-dt/self.tau))*(target-self.vw);self.t+=dt
        return self.vw.copy()

    def acknowledge(self,t,pwm):
        value=np.asarray(pwm,float)
        if value.shape!=(2,) or not np.isfinite(value).all() or max(abs(value))>1:raise ValueError('Invalid applied PWM')
        if not np.isfinite(t) or t<self.commands[-1][0]:raise ValueError('Unordered applied PWM')
        self.at(t);self.commands.append((float(t),value.copy()))
        if len(self.commands)>1000:self.commands=self.commands[-500:]


class PathTracker:
    """FF curvature + heading/lateral feedback in the packet's body frame.

    275ms preview is the declared midpoint of nominal 150..400ms effective
    response priors, not an actual vehicle parameter or an identified value.
    """
    def __init__(self,preview_s=.275):
        self.drive=SignedDrive();self.motion=AppliedMotion(self.drive)
        self.preview_s=preview_s;self.last_t=None

    def desired(self,path,now):
        age=now-path.capture_s
        if not np.isfinite(now) or age<0 or age>.25:raise ValueError('Unavailable or stale path')
        if self.last_t is not None and now<=self.last_t:raise ValueError('Non-increasing control clock')
        v,w=self.motion.at(now);pred=advance(v,w,age+self.preview_s)
        pts=path.points_cm;distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))]
        derivative=np.gradient(pts,distance,axis=0,edge_order=2)
        heading=np.unwrap(np.arctan2(derivative[:,0],derivative[:,1]))
        curvature=np.gradient(heading,distance,edge_order=2)
        near=int(np.argmin(np.linalg.norm(pts-pred[:2],axis=1)))
        # Numerical endpoint derivatives are not used for steering.
        near=int(np.clip(near,1,len(pts)-2));h=heading[near]
        lateral=float((pred[:2]-pts[near])@np.array([np.cos(h),-np.sin(h)]))
        target=float(path.speeds_cm_s[near]);head=float(wrap(h-pred[2]))
        omega=target*curvature[near]+1.3*head-.08*lateral
        self.last_t=float(now)
        return self.drive.inverse(target,omega),dict(target_v=target,target_w=float(omega),
            curvature=float(curvature[near]),lateral_cm=lateral,heading_rad=head,preview_s=self.preview_s,
            nominal_v=float(v),nominal_w=float(w),packet_age_s=float(age))

    def acknowledge(self,t,pwm):self.motion.acknowledge(t,pwm)
