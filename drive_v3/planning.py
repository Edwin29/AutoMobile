"""Local spline candidates, bounded uncertainty footprint checks, and planning FSM."""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import make_interp_spline

from .geometry import to_body,to_world,path_geometry,footprint,sigma_poses,wrap
from .contracts import PathDecision
from .path_acceptance import evaluate_prefix, commitment_distance


class LocalPlanner:
    def __init__(self,track,route,cfg,vehicle):
        self.track=track;self.route=route;self.cfg=cfg;self.vehicle=vehicle
        self.rim=footprint(vehicle);self.offset=0.;self.switched=-1e9
        self.state="CRUISE";self.last_observed_distance=0.;self.last_diagnostics={}
        self.completed_route=False
        self.good_after_hold=0

    def set_route(self,route):
        self.route=route;self.state="APPROACH";self.completed_route=False

    def stop(self,now,clock,reason):
        self.state="REOBSERVE";self.last_diagnostics={"reason":reason}
        return PathDecision([],0.,now+self.cfg["path_valid_s"],clock,reason)

    def _margin(self,body_path,beliefs):
        _,heading,_=path_geometry(body_path)
        worst=float("inf")
        for belief in beliefs:
            for pose in sigma_poses(belief.mean,belief.P,self.cfg["sigma_multiplier"]):
                world=to_world(body_path,pose)
                angles=heading+pose[2]
                c,s=np.cos(angles),np.sin(angles)
                p=np.empty((len(world),len(self.rim),2))
                p[:,:,0]=world[:,None,0]+c[:,None]*self.rim[None,:,0]+s[:,None]*self.rim[None,:,1]
                p[:,:,1]=world[:,None,1]-s[:,None]*self.rim[None,:,0]+c[:,None]*self.rim[None,:,1]
                distance=self.track.clearance if self.cfg.get('distance_mode','grid_lower_bound')=='exact' else self.track.clearance_fast
                worst=min(worst,float(distance(p).min()))
        return worst

    def _observed_margin(self,path,observation):
        if observation is None:return float("inf")
        _,head,_=path_geometry(path)
        worst=float("inf")
        for f in observation.fragments:
            if f.side_hint=="unknown" or f.quality.get("length_cm",0)<6:continue
            p=np.asarray(f.points_cm);t=np.gradient(p,axis=0)
            t/=np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-9)
            # Check only sufficiently parallel local fragments; far cross-road fragments are not lane IDs.
            for q,a in zip(path,head):
                j=int(np.argmin(np.linalg.norm(p-q,axis=1)))
                direction=np.array([np.sin(a),np.cos(a)])
                if abs(direction@t[j])<.8 or np.linalg.norm(p[j]-q)>20:continue
                right=np.array([t[j,1],-t[j,0]])
                interior=right if f.side_hint=="left" else -right
                footprint_points=to_world(self.rim,[*q,a])
                # Require longitudinal overlap; do not extrapolate a finite fragment indefinitely.
                projected=(footprint_points-p[j])@t[j]
                valid=np.abs(projected)<2
                if valid.any():worst=min(worst,float(((footprint_points[valid]-p[j])@interior).min()))
        return worst

    def _visual_candidate(self,observation):
        if observation is None:return None
        sides={}
        for side in ("left","right"):
            choices=[f for f in observation.fragments if f.side_hint==side and f.quality.get("length_cm",0)>12]
            if not choices:return None
            p=np.asarray(max(choices,key=lambda f:f.quality["length_cm"]).points_cm)
            if np.ptp(p[:,1])<10:return None
            order=np.argsort(p[:,1]);sides[side]=p[order]
        a,b=sides["left"],sides["right"]
        start=max(a[0,1],b[0,1]);end=min(a[-1,1],b[-1,1])
        if end-start<10:return None
        y=np.linspace(start,end,20)
        left=np.interp(y,a[:,1],a[:,0]);right=np.interp(y,b[:,1],b[:,0]);width=right-left
        if np.any(width<17) or np.any(width>30) or np.ptp(width)>4:return None
        return np.c_[(left+right)/2,y]

    def plan(self,estimator,observation,now,clock,mission_stop=False,speed_cap=None,_horizon=None):
        if _horizon is None:self._attempt_trace=[]
        if mission_stop:return self.stop(now,clock,"mission_hold")
        beliefs=estimator.hypotheses;primary=beliefs[0]
        age=now-primary.last_observed_s
        if age<1e-6:self.last_observed_distance=estimator.motion_distance
        if age>self.cfg["prediction_only_s"] or estimator.motion_distance-self.last_observed_distance>self.cfg["prediction_only_cm"]:
            return self.stop(now,clock,"observation_budget_exhausted")
        if any(np.sqrt(h.P[2,2])>self.cfg["max_yaw_sigma_rad"] or np.sqrt(np.max(np.diag(h.P)[:2]))>self.cfg["max_xy_sigma_cm"] for h in beliefs):
            return self.stop(now,clock,"uncertainty_too_large")
        route=self.route;i=route.nearest(primary.mean[:2]);arc=route.s[i]
        remaining=route.s[-1]-arc
        if not route.closed and remaining<1.5 and abs(wrap(primary.mean[2]-route.heading[-1]))<self.cfg["settle_heading_rad"]:
            self.completed_route=True;return self.stop(now,clock,"route_complete")
        horizon=min(self.cfg["horizon_cm"],remaining) if not route.closed else self.cfg["horizon_cm"]
        if _horizon is not None:horizon=min(horizon,_horizon)
        if horizon<2:return self.stop(now,clock,"route_end_unsettled")
        world=route.sample(arc+np.arange(0,horizon+.01,self.cfg["spacing_cm"]))
        base=to_body(world,primary.mean)
        s,head,k=path_geometry(base)
        candidates=[]
        bounded=self.cfg.get('acceptance_mode','legacy_full')=='bounded_prefix'
        requested_speed=self.cfg['normal_speed_cm_s'] if age<.15 else self.cfg['degraded_speed_cm_s']
        if speed_cap is not None:requested_speed=min(requested_speed,speed_cap)
        commitment=commitment_distance(self.cfg,requested_speed,age)
        rejected={"curvature":0,"footprint":0,"observed_boundary":0}
        attempt={"horizon_cm":float(horizon),"candidates":[]}
        self._attempt_trace.append(attempt)
        current=self._margin(np.array([[0.,0.],[0.,.01],[0.,.02]]),beliefs)
        visual=self._visual_candidate(observation)
        for offset in self.cfg["offsets_cm"]:
            trace={"offset_cm":float(offset)};attempt["candidates"].append(trace)
            right=np.c_[np.cos(head),-np.sin(head)]
            p=base+offset*right
            # Quintic blend corrects the initial position/heading without an instantaneous jump.
            r=np.clip(s/min(self.cfg["join_cm"],horizon*.8),0,1)
            blend=1-(10*r**3-15*r**4+6*r**5)
            p-=blend[:,None]*p[0]
            # Initial derivative is aligned to body forward using a short smooth correction.
            tangent=p[1]-p[0]
            p[:,0]-=tangent[0]/max(s[1],.1)*s*blend
            try:
                spline=make_interp_spline(s,p,k=3)
                dense=spline(np.linspace(0,s[-1],max(4,int(s[-1]/.5)+1)))
                _,_,curv=path_geometry(dense)
            except ValueError:
                trace['rejected']='invalid_spline';continue
            trace['max_curvature_cm_inv']=float(np.max(np.abs(curv)))
            if np.max(np.abs(curv))>self.cfg["max_curvature_cm_inv"]:
                trace['rejected']='curvature'
                rejected["curvature"]+=1;continue
            if bounded:
                allowed,reason,acceptance=evaluate_prefix(self.track,self.rim,dense,beliefs,self.cfg,commitment)
                trace['acceptance']=acceptance
                if not allowed:
                    trace['rejected']=reason;rejected['footprint']+=1;continue
                margin=acceptance['short_minimum_cm']
            else:margin=self._margin(dense,beliefs)
            observed=self._observed_margin(dense,observation)
            trace.update(margin_cm=margin,observed_margin_cm=observed if np.isfinite(observed) else None)
            recovering=False
            if not bounded and margin<self.cfg["clearance_cm"]:
                # Only slight existing penetration, visible corridor, improving end,
                # and no further penetration across retained pose hypotheses.
                recovering=(visual is not None and -self.cfg["recovery_penetration_cm"]<=current<self.cfg["clearance_cm"]
                            and margin>=current-.05 and self._margin(dense[-3:],beliefs)>self.cfg["clearance_cm"])
                if not recovering:
                    trace['rejected']='footprint';rejected["footprint"]+=1;continue
            observed_limit=-self.cfg.get('allowed_penetration_cm',.5) if bounded else 0.
            if observed<observed_limit and not recovering:
                trace['rejected']='observed_boundary';rejected["observed_boundary"]+=1;continue
            trace['rejected']=None
            if bounded:recovering=acceptance['initial_worst_cm']<self.cfg['clearance_cm']
            score=-margin+.25*abs(offset)+2*float(np.mean(curv**2))
            if bounded:score+=2*max(0.,self.cfg['clearance_cm']-acceptance['mean_minimum_cm'])
            candidates.append((score,offset,dense,margin,"recovery" if recovering else "map"))
        if visual is not None and age>.05:
            # Partial visual corridor cannot authorize a junction turn or extrapolated unseen floor.
            if self.state=="CRUISE" and visual[0,1]<18:
                p=np.vstack(([0.,0.],visual))
                if np.all(np.diff(p[:,1])>0):
                    _,_,k=path_geometry(p)
                    if np.max(np.abs(k))<.04:
                        margin=self._margin(p,beliefs)
                        if margin>=self.cfg["clearance_cm"]:candidates.append((-margin+.5,0.,p,margin,"observed"))
        if not candidates:
            # Check shorter common motions before declaring the entire route blocked.
            smaller=next((v for v in (16.,10.,6.) if v<horizon-1),None)
            if smaller is not None:return self.plan(estimator,observation,now,clock,mission_stop,speed_cap,smaller)
            self.good_after_hold=0
            result=self.stop(now,clock,"no_feasible_candidate")
            self.last_diagnostics.update(rejected=rejected,current_margin_cm=current,horizon_cm=horizon,attempts=self._attempt_trace)
            return result
        candidates.sort(key=lambda c:c[0]);best=candidates[0]
        old=next((c for c in candidates if c[1]==self.offset),None)
        if old and best[1]!=self.offset and (now-self.switched<self.cfg["switch_hold_s"] or old[0]-best[0]<self.cfg["switch_improvement"]):best=old
        if best[1]!=self.offset:self.switched=now
        self.offset=best[1]
        if self.state=="REOBSERVE":
            self.good_after_hold+=1
            if self.good_after_hold<3:return self.stop(now,clock,"reacquisition_confirmation")
        if best[4]=="recovery":self.state="RECOVERY"
        elif route.junction:
            self.state="APPROACH" if arc<15 else ("TURN" if remaining>15 else "SETTLE")
        else:self.state="CRUISE"
        speed=self.cfg["normal_speed_cm_s"] if age<.15 else self.cfg["degraded_speed_cm_s"]
        if speed_cap is not None:speed=min(speed,speed_cap)
        if horizon<self.cfg["horizon_cm"] or self.state=="RECOVERY":speed=min(speed,self.cfg["degraded_speed_cm_s"])
        stop_distance=speed*self.cfg["stop_latency_s"]+speed**2/(2*self.cfg["brake_cm_s2"])
        if stop_distance>horizon:return self.stop(now,clock,"insufficient_stopping_horizon")
        self.last_diagnostics={"state":self.state,"margin_cm":best[3],"source":best[4],"candidates":len(candidates),"offset_cm":best[1],"horizon_cm":horizon,"rejected":rejected,"attempts":self._attempt_trace}
        reference=base[0]+best[1]*np.array([np.cos(head[0]),-np.sin(head[0])])
        lateral=float(-reference@np.array([np.cos(head[0]),-np.sin(head[0])]))
        return PathDecision(best[2].tolist(),speed,now+self.cfg["path_valid_s"],clock,"tracking",
                            lateral,float(wrap(-head[0])),float(path_geometry(base)[2][0]))
