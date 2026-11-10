"""Experimental rolling allocation. No hardware; limits are provisional.

The planner rolls out this SAME controller, rather than accepting a geometric
spline that the motor allocator subsequently refuses. Curvature clipping is
explicit and included in those rollouts, never a hidden promise of tracking.
"""
import numpy as np
from .control import Controller, ControlResult
from .geometry import path_geometry


def rolling_allocate(model, speed, omega, speed_cap):
    cfg=model.cfg
    lo,hi=cfg['pwm_min'],cfg['pwm_max']
    if speed_cap < lo*cfg['wheel_cm_s_per_duty']*model.forward-1e-9:
        return None
    requested=model.inverse(max(speed,1e-6),omega)
    # Parameterize a positive-wheel arc by differential / mean duty.
    ratio=float((requested[0]-requested[1])/max(requested.sum(),1e-9))
    limit=(hi-lo)/(hi+lo)
    ratio=float(np.clip(ratio,-limit,limit))
    factors=np.array([1+ratio,1-ratio])
    lower=lo/factors.min()
    upper=min(hi/factors.max(),speed_cap/(cfg['wheel_cm_s_per_duty']*model.forward))
    # A low speed cap may also reduce available curvature.
    if lower>upper:
        mean_cap=speed_cap/(cfg['wheel_cm_s_per_duty']*model.forward)
        ratio=float(np.clip(ratio,-max(0.,1-lo/mean_cap),max(0.,1-lo/mean_cap)))
        factors=np.array([1+ratio,1-ratio]);lower=lo/factors.min()
        upper=min(hi/factors.max(),mean_cap)
    mean=float(np.clip(speed/(cfg['wheel_cm_s_per_duty']*model.forward),lower,upper))
    pwm=mean*factors
    return pwm


class RollingController(Controller):
    def step(self,path,now,dt,clock_id,quality_ok):
        if not np.isfinite([now,dt]).all() or not 0<dt<=self.cfg['watchdog_s']:
            return self.stop('control_timing_invalid')
        if path.clock_id!=clock_id or now>path.valid_until_s or path.speed_limit_cm_s<=0:
            return self.stop(path.reason if path.speed_limit_cm_s<=0 else 'stale_path')
        p=np.asarray(path.points_body_cm,float)
        if len(p)<3 or not np.isfinite(p).all():return self.stop('invalid_path')
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        near=int(np.argmin(np.linalg.norm(p,axis=1)))
        distance=path.lookahead_cm or 12.
        target=min(len(p)-1,int(np.searchsorted(arc,arc[near]+distance)))
        q=p[target];curvature=2*q[0]/max(q@q,1.)
        preferred=min(path.speed_limit_cm_s,self.cfg.get('preferred_speed_cm_s',4.8))
        requested=rolling_allocate(self.model,preferred,preferred*curvature+self.trim,path.speed_limit_cm_s)
        if requested is None:return self.stop('speed_cap_below_rolling_floor')
        delivered=self.pwm+np.clip(requested-self.pwm,-self.cfg['slew_per_s']*dt,self.cfg['slew_per_s']*dt)
        self.pwm=delivered
        actual=delivered.copy()
        if np.any(actual<self.cfg['pwm_min']-1e-9):actual[:]=0.
        v,w=self.model.forward_model(actual);self.speed=v
        return ControlResult(actual,v,w,'rolling_limited' if abs(w-v*curvature)>1e-3 else 'tracking',self.trim)
