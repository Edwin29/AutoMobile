"""v3 experiment: executable rollouts, soft road costs and bounded recovery.

Not v4; not an obstacle detector. Uses only estimated poses/observations, never
simulation truth. Representative pose samples are costs, not probability bounds.
"""
import copy
import numpy as np
from .planning import LocalPlanner
from .contracts import PathDecision
from .geometry import to_body, to_world, path_geometry, advance, sigma_poses
from .path_acceptance import margin_profile
from .recovery_limits import depth_rejection


class RecoveryPlanner(LocalPlanner):
    def __init__(self,track,route,cfg,vehicle,controller):
        super().__init__(track,route,cfg,vehicle)
        self.controller=controller;self.last_time=None;self.outside_travel=0.;self.last_motion_distance=None

    def plan(self,estimator,observation,now,clock,mission_stop=False,speed_cap=None):
        if mission_stop:return self.stop(now,clock,'mission_hold')
        primary=estimator.primary;age=now-primary.last_observed_s
        if self.cfg.get('allow_existing_deep_recovery',False):
            current_distances=self.track.clearance(to_world(self.rim,primary.mean))
            travelled=0. if self.last_motion_distance is None else max(0.,estimator.motion_distance-self.last_motion_distance)
            self.last_motion_distance=estimator.motion_distance
            self.outside_travel=self.outside_travel+travelled if current_distances.min()<0 else 0.
            if current_distances.max()<0:return self.stop(now,clock,'estimated_no_road_overlap')
            if self.outside_travel>self.cfg.get('max_recovery_travel_cm',49.):return self.stop(now,clock,'recovery_travel_budget')
        route=self.route;index=route.nearest(primary.mean[:2]);arc=route.s[index]
        base=to_body(route.sample(arc+np.arange(0,51.,1.)),primary.mean)
        # A visible straight corridor may align lateral/heading components when
        # global matching is unavailable. Longitudinal start/route remains a prior.
        source='map_aligned';visual=self._visual_candidate(observation)
        if age>self.cfg.get('coast_s',1.2):
            if visual is None:return self.stop(now,clock,'no_local_or_map_support')
            vb=np.polyfit(visual[:,1],visual[:,0],1)
            near=(base[:,1]>=visual[0,1])&(base[:,1]<=visual[-1,1])
            if near.sum()<3:return self.stop(now,clock,'local_alignment_unavailable')
            mb=np.polyfit(base[near,1],base[near,0],1)
            base[:,0]+=(vb[0]-mb[0])*base[:,1]+vb[1]-mb[1]
            source='local_aligned_map_shape'
        if np.sqrt(primary.P[2,2])>.5:return self.stop(now,clock,'heading_unresolved')
        speed=min(self.cfg['normal_speed_cm_s'],speed_cap if speed_cap is not None else float('inf'))
        dt=self.cfg.get('rollout_dt_s',.2);duration=self.cfg.get('rollout_s',4.)
        first_dt=dt if self.last_time is None else now-self.last_time
        self.last_time=now
        n=int(np.ceil(duration/dt)) if self.cfg.get('exact_rollout_horizon',False) else int(duration/dt)
        candidates=[];traces=[]
        _,heading,_=path_geometry(base);right=np.c_[np.cos(heading),-np.sin(heading)]
        for offset in self.cfg.get('rollout_offsets_cm',[-2.,0.,2.]):
            for lookahead in self.cfg.get('rollout_lookaheads_cm',[10.,16.]):
                reference=base+offset*right
                control=copy.deepcopy(self.controller);pose=np.zeros(3);poses=[pose.copy()]
                first=None
                for j in range(n):
                    p=to_body(reference,pose)
                    path=PathDecision(p.tolist(),speed,now+(j+1)*dt+.5,clock,'tracking',lookahead_cm=lookahead)
                    out=control.step(path,now+j*dt,first_dt if j==0 else dt,clock,False)
                    if first is None:first=out
                    hold=min(dt,duration-j*dt) if self.cfg.get('exact_rollout_horizon',False) else dt
                    pose=advance(pose,out.v_cm_s,out.omega_rad_s,hold);poses.append(pose.copy())
                poses=np.asarray(poses)
                nominal=margin_profile(self.track,self.rim,poses[:,:2],primary.mean,poses[:,2])
                # Stop/latency interval is tested separately from the longer rollout.
                stop_s=dt+self.cfg['stop_latency_s']+max(control.speed,first.v_cm_s)/self.cfg['brake_cm_s2']
                end=min(n,max(1,int(np.ceil(stop_s/dt))))
                profiles=np.array([margin_profile(self.track,self.rim,poses[:end+1,:2],p,poses[:end+1,2])
                    for p in sigma_poses(primary.mean,primary.P,1.)])
                # Relative degradation is costed per hypothesis, not min(mins).
                worsening=np.maximum(0.,profiles[:,0]-profiles.min(axis=1))
                distance=np.linalg.norm(np.diff(poses[:,:2],axis=0),axis=1).sum()
                reason=None
                reason=depth_rejection(nominal,self.cfg.get('hard_road_depth_cm',6.),self.cfg.get('allow_existing_deep_recovery',False))
                if nominal[0]<0 and nominal[-1]<nominal[0]+.1:reason='not_returning'
                if nominal[0]<-1 and nominal.min()<nominal[0]-1.5:reason='recovery_worsens'
                if distance<.1:reason='motor_unavailable'
                # Cross-road/tangent labels are not absolute obstacle identities.
                moving=np.r_[True,np.linalg.norm(np.diff(poses[:,:2],axis=0),axis=1)>1e-8]
                observed=self._observed_margin(poses[moving,:2],observation) if moving.sum()>2 else float('inf')
                intrusion=np.maximum(0.,-nominal)
                score=(4*intrusion.mean()+2*intrusion.max()+.35*worsening.mean()
                       +.2*max(0.,-profiles.min())+.12*abs(offset)+.04*abs(offset-self.offset)
                       -.025*distance+.1*max(0.,-observed))
                traces.append(dict(offset_cm=offset,lookahead_cm=lookahead,rejected=reason,score=float(score),
                    nominal_min_cm=float(nominal.min()),nominal_end_cm=float(nominal[-1]),
                    sampled_min_cm=float(profiles.min()),rollout_distance_cm=float(distance),first_pwm=first.pwm.tolist()))
                if reason is None:candidates.append((score,reference,offset,lookahead,nominal,poses))
        if not candidates:
            result=self.stop(now,clock,'no_returning_rollout');self.last_diagnostics['candidates']=traces;return result
        score,reference,offset,lookahead,nominal,poses=min(candidates,key=lambda x:x[0])
        self.offset=offset;self.state='RECOVERY' if nominal[0]<0 else 'CRUISE'
        self.last_diagnostics=dict(source=source,state=self.state,score=float(score),candidates=traces,
            recovery_predicted_travel_cm=self.outside_travel,
            nominal_min_cm=float(nominal.min()),nominal_start_cm=float(nominal[0]),
            selected_offset_cm=offset,lookahead_cm=lookahead,rollout_body=poses.tolist())
        return PathDecision(reference.tolist(),speed,now+self.cfg['path_valid_s'],clock,'tracking',lookahead_cm=lookahead)
