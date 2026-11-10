"""Body-feedback controller and a shared bounded PWM response model. No GPIO."""
from dataclasses import dataclass
import numpy as np

from .geometry import path_geometry,wrap


@dataclass
class ControlResult:
    pwm:np.ndarray
    v_cm_s:float
    omega_rad_s:float
    reason:str
    trim:float


class ResponseModel:
    def __init__(self,cfg,vehicle):
        self.cfg=cfg;self.track=vehicle["track_effective_cm"]
        self.forward=cfg["forward_gain"];self.left=cfg["left_yaw_gain"];self.right=cfg["right_yaw_gain"]
        self.version=0

    def forward_model(self,pwm):
        p=np.asarray(pwm,float)
        effective=np.where(np.abs(p)>=self.cfg["pwm_min"],p,0.)
        speed=effective*self.cfg["wheel_cm_s_per_duty"]
        raw=(speed[0]-speed[1])/self.track
        return float(speed.mean()*self.forward),float(raw*(self.left if raw<0 else self.right))

    def inverse(self,v,omega):
        gain=self.left if omega<0 else self.right
        base=v/self.forward
        differential=omega/max(gain,.1)*self.track/2
        return np.array([base+differential,base-differential])/self.cfg["wheel_cm_s_per_duty"]

    def adapt(self,pwm,motion,allowed):
        if not self.cfg["adaptive"] or not allowed or motion is None:return False
        dt=motion.end_s-motion.start_s
        if not .02<dt<.5 or motion.status!="moving":return False
        p=np.asarray(pwm)
        if (p<self.cfg["pwm_min"]).any() or (p>self.cfg["pwm_max"]-.02).any():return False
        nominal_v=p.mean()*self.cfg["wheel_cm_s_per_duty"]
        nominal_w=(p[0]-p[1])*self.cfg["wheel_cm_s_per_duty"]/self.track
        lower,upper=self.cfg["adapt_gain_bounds"];alpha=self.cfg["adapt_alpha"]
        if abs(nominal_w)<.03 and nominal_v>1.:
            ratio=motion.translation_body_cm[1]/dt/nominal_v
            if not lower<=ratio<=upper:return False
            self.forward=(1-alpha)*self.forward+alpha*ratio
        elif abs(nominal_w)>self.cfg["min_observable_turn_rad_s"]:
            ratio=motion.yaw_rad/dt/nominal_w
            if not lower<=ratio<=upper:return False
            if nominal_w<0:self.left=(1-alpha)*self.left+alpha*ratio
            else:self.right=(1-alpha)*self.right+alpha*ratio
        else:return False
        self.version+=1;return True


class Controller:
    def __init__(self,model,cfg,mode="feedback"):
        if mode not in ("feedback","pure_pursuit"):raise ValueError("Unknown controller")
        self.model=model;self.cfg=cfg;self.mode=mode
        self.trim=0.;self.speed=0.;self.pwm=np.zeros(2);self.last=None

    def stop(self,reason):
        self.speed=0.;self.pwm[:]=0
        return ControlResult(self.pwm.copy(),0.,0.,reason,self.trim)

    def step(self,path,now,dt,clock_id,quality_ok):
        if not np.isfinite([now,dt]).all() or not 0<dt<=self.cfg["watchdog_s"]:
            return self.stop("control_timing_invalid")
        if not np.isfinite([path.valid_until_s,path.speed_limit_cm_s]).all():return self.stop("invalid_path_limits")
        if path.clock_id!=clock_id or now>path.valid_until_s or path.speed_limit_cm_s<=0:
            return self.stop(path.reason if path.speed_limit_cm_s<=0 else "stale_path")
        points=np.asarray(path.points_body_cm,float)
        if len(points)<3 or not np.isfinite(points).all():return self.stop("invalid_path")
        s,heading,curvature=path_geometry(points)
        near=int(np.argmin(np.linalg.norm(points,axis=1)))
        ahead=min(len(points)-1,near+max(1,round(self.speed*self.cfg["preview_s"]/max(np.median(np.diff(s)),.1))))
        k=float(curvature[ahead])
        # Preferred cruising speed can be below the planner's checked upper bound.
        # The existing ratio-preserving deadband lift may use that reserve.
        preferred=min(path.speed_limit_cm_s,self.cfg.get('preferred_speed_cm_s',path.speed_limit_cm_s))
        target=preferred/(1+self.cfg["curvature_slowdown_cm"]*abs(k))
        self.speed+=float(np.clip(target-self.speed,-self.cfg["decel_cm_s2"]*dt,self.cfg["accel_cm_s2"]*dt))
        # Path in current body frame: path right of car -> positive/right correction.
        theta=float(wrap(heading[near]))
        right=np.array([np.cos(theta),-np.sin(theta)])
        e=float(-points[near]@right);psi=float(wrap(-theta))
        if path.reference_lateral_error_cm is not None:e=path.reference_lateral_error_cm
        if path.reference_heading_error_rad is not None:psi=path.reference_heading_error_rad
        if self.mode=="pure_pursuit":
            lookahead=10. if path.lookahead_cm is None else path.lookahead_cm
            target_index=int(np.searchsorted(s,min(s[-1],s[near]+lookahead)))
            q=points[min(target_index,len(points)-1)]
            omega=self.speed*2*q[0]/max(q@q,1.)+self.trim
        else:
            reference_k=k if path.reference_curvature_cm_inv is None else path.reference_curvature_cm_inv
            omega=self.speed*reference_k-self.cfg["lateral_gain"]*e-self.cfg["heading_gain"]*psi+self.trim
        omega=float(np.clip(omega,-self.cfg["max_omega_rad_s"],self.cfg["max_omega_rad_s"]))
        requested=self.model.inverse(self.speed,omega)
        limited=False
        if requested.min()<0:
            # No hidden counterrotation; request a different path/speed instead.
            return self.stop("reverse_required")
        if requested.max()>self.cfg["pwm_max"]:
            requested*=self.cfg["pwm_max"]/requested.max();limited=True
        nonzero=requested[requested>1e-6]
        if len(nonzero) and nonzero.min()<self.cfg["pwm_min"] and self.speed>=2.:
            scale=self.cfg["pwm_min"]/nonzero.min()
            scaled=requested*scale
            sv,sw=self.model.forward_model(scaled)
            # Preserve curvature by scaling both sides together, only within the
            # planner's cap. Curvature slowdown is a preference, not permission to stall.
            if sv>path.speed_limit_cm_s+1e-6 or abs(sw)>self.cfg["max_omega_rad_s"] or scaled.max()>self.cfg["pwm_max"]:
                return self.stop("deadband_path_infeasible")
            requested=scaled;limited=True
        delivered=self.pwm+np.clip(requested-self.pwm,-self.cfg["slew_per_s"]*dt,self.cfg["slew_per_s"]*dt)
        limited |= bool(np.max(np.abs(delivered-requested))>.005)
        self.pwm=np.clip(delivered,0,self.cfg["pwm_max"])
        if (self.cfg["trim_enabled"] and quality_ok and not limited and abs(k)<self.cfg["trim_straight_curvature"]
                and (self.pwm>=self.cfg["pwm_min"]).all() and abs(e)<4 and abs(psi)<.15):
            delta=float(np.clip(-self.cfg["trim_ki"]*e*dt,-self.cfg["trim_rate_rad_s2"]*dt,self.cfg["trim_rate_rad_s2"]*dt))
            self.trim=float(np.clip(self.trim+delta,-self.cfg["trim_max_rad_s"],self.cfg["trim_max_rad_s"]))
        actual=self.pwm.copy()
        # During startup, do not energize just the faster side while the other
        # remains below its modeled start threshold. Keep internal ramp state.
        if np.any((actual>1e-6)&(actual<self.cfg["pwm_min"]-1e-9)):
            actual[:]=0.;limited=True
        v,w=self.model.forward_model(actual)
        return ControlResult(actual,v,w,"limited" if limited else "tracking",self.trim)
