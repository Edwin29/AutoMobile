"""Direct teacher: mean/0.66 and LEFT-positive curvature/0.095."""
import numpy as np
from bench.world import Track, to_world, advance
from bench.vehicle import Vehicle, PRESETS
from .teacher import Teacher

class DirectTeacher(Teacher):
    curvature_cap = 0.085
    def __init__(self,direction,lookahead=18,variant=1):
        super().__init__(direction,lookahead,variant=8)
        self.direct_variant=variant
        self.corner_bias = 0.0 if variant == 1 else 2.0
        self.track = Track()
        self.rim = Vehicle(PRESETS["field_estimate"],[0,0,0]).footprint(step=4)
    def step(self,pose):
        k,mean=self.target(pose)
        if self.direct_variant >= 3:
            horizon = 1.2 if self.direct_variant == 3 else 2.0
            candidates=np.unique(np.r_[k,np.linspace(-.085,.085,35)])
            predicted=np.tile(pose,(len(candidates),1)).astype(float)
            params=getattr(self,'sim_params',PRESETS['field_estimate'])
            actual_gain=np.where(candidates<0,params.turn_eff_left,params.turn_eff_right)
            gain=np.where(candidates<0,self.motor.gain_left,self.motor.gain_right)
            v=6.7
            w=v*candidates*actual_gain/gain
            margin=np.full(len(candidates),100.)
            for _ in range(int(horizon/.2)):
                angle=w*.2
                predicted[:,0]+=v*.2*np.sinc(angle/(2*np.pi))*np.sin(predicted[:,2]+angle/2)
                predicted[:,1]+=v*.2*np.sinc(angle/(2*np.pi))*np.cos(predicted[:,2]+angle/2)
                predicted[:,2]+=angle
                c=np.cos(predicted[:,2,None]);sn=np.sin(predicted[:,2,None])
                points=np.stack([self.rim[None,:,0]*c+self.rim[None,:,1]*sn+predicted[:,None,0],
                                 -self.rim[None,:,0]*sn+self.rim[None,:,1]*c+predicted[:,None,1]],axis=-1)
                margin=np.minimum(margin,self.track.clearance(points).min(axis=1))
            route_error=np.min(np.sum((self.route[None,:,:]-predicted[:,None,:2])**2,axis=2),axis=1)
            costs=400*np.maximum(0,1.0-margin)**2 + route_error + 800*(candidates-k)**2
            k=float(candidates[int(np.argmin(costs))])
        label=np.array([mean/0.66,-k/0.095])
        pwm,note=self.motor.mix(mean,k,0.66)
        self.last_pwm=np.array(pwm)
        return self.last_pwm,dict(k=k,requested_mean=mean,used_mean=mean,label=label.tolist(),allocator=note)
