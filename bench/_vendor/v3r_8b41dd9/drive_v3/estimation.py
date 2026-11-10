"""Bounded hypothesis EKF with robust semantic map fitting and capture-time replay."""
from dataclasses import dataclass
import copy
import math
import numpy as np
from scipy.optimize import least_squares

from .geometry import advance, axes, to_world, wrap
from .contracts import EvidenceSource
from .boundary_evidence import sample_evidence, weighted_huber


@dataclass
class Belief:
    mean:np.ndarray
    P:np.ndarray
    score:float=0.
    associations:dict|None=None
    last_observed_s:float=-1e9


class Estimator:
    def __init__(self,track,pose,cfg):
        self.track=track;self.cfg=cfg
        self.hypotheses=[Belief(np.asarray(pose,float),np.diag(np.square(cfg["initial_sigma"])),associations={})]
        self.last_search=-1e9;self.last_diagnostics={};self.motion_distance=0.

    @property
    def primary(self):
        return self.hypotheses[0]

    def snapshot(self):
        return copy.deepcopy((self.hypotheses,self.last_search,self.last_diagnostics,self.motion_distance))

    def restore(self,snapshot):
        self.hypotheses,self.last_search,self.last_diagnostics,self.motion_distance=copy.deepcopy(snapshot)

    def predict(self,v,omega,dt):
        if not np.isfinite([v,omega,dt]).all() or not 0<=dt<=2.:
            raise ValueError("Invalid prediction interval")
        for h in self.hypotheses:
            yaw=h.mean[2]+omega*dt/2
            F=np.eye(3);F[:2,2]=v*dt*np.sinc(omega*dt/(2*np.pi))*np.array([np.cos(yaw),-np.sin(yaw)])
            if self.cfg.get('motion_scaled_noise',False):
                # Match old moving noise at 5cm/s; retain a small stationary
                # drift floor. PWM is not a wheel encoder or a stop measurement.
                activity=abs(v)/5.+abs(omega)/.5
                Q=np.diag(np.square(self.cfg["motion_sigma_per_s"]))*(activity*dt)
                Q+=np.diag(np.square(self.cfg.get('stationary_sigma_per_s',[.03,.03,.003])))*dt
            else:Q=np.diag(np.square(self.cfg["motion_sigma_per_s"]))*dt
            h.mean=advance(h.mean,v,omega,dt);h.P=F@h.P@F.T+Q
        self.motion_distance+=abs(v)*dt  # predicted travel only, never includes pose corrections

    def predict_visual(self,motion,dt):
        """Replace PWM prediction for this interval; never an extra absolute pose fix."""
        duration=motion.end_s-motion.start_s
        angle=motion.yaw_rad
        a=np.sinc(angle/np.pi);b=(1-np.cos(angle))/angle if abs(angle)>1e-7 else angle/2
        velocity=np.linalg.solve(np.array([[a,b],[-b,a]]),motion.translation_body_cm)/duration
        theta=angle*dt/duration
        a=np.sinc(theta/np.pi);b=(1-np.cos(theta))/theta if abs(theta)>1e-7 else theta/2
        delta=np.array([[a,b],[-b,a]])@velocity*dt
        fraction=dt/duration
        Q=np.diag([max(.1,motion.rms_cm)**2,max(.1,motion.rms_cm)**2,.01**2])*fraction
        for h in self.hypotheses:
            rotation=axes(h.mean[2]);world=delta@rotation
            F=np.eye(3);F[:2,2]=[world[1],-world[0]]
            G=np.eye(3);G[:2,:2]=rotation.T
            h.mean[:2]+=world;h.mean[2]=wrap(h.mean[2]+theta)
            h.P=F@h.P@F.T+G@Q@G.T
        self.motion_distance+=float(np.linalg.norm(velocity)*dt)

    def _data(self,observation):
        if self.cfg.get('length_weighted_boundaries',False):
            return sample_evidence(observation.fragments,self.cfg)
        points=[];weights=[];groups=[]
        for f in sorted(observation.fragments,key=lambda x:-x.quality.get("length_cm",0))[:self.cfg["max_fragments"]]:
            if f.source != EvidenceSource.OBSERVED:
                continue
            p=np.asarray(f.points_cm)
            if len(p)<3:
                continue
            indices=np.unique(np.linspace(0,len(p)-1,min(len(p),self.cfg["max_points_per_fragment"])).astype(int))
            points.extend(p[indices]);weights.extend([1/math.sqrt(len(indices))]*len(indices))
            groups.extend([f.id]*len(indices))
        return np.asarray(points).reshape(-1,2),np.asarray(weights),groups

    def _fit(self,observation,prior,seed,association=None):
        p,w,groups=self._data(observation)
        world=to_world(p,seed)
        # Bind each whole fragment to a boundary identity instead of per-pixel nearest edges.
        ds=self.track.distances(world)
        ids=np.zeros(len(p),int)
        usable=np.ones(len(p),bool)
        for key in sorted(set(groups)):
            mask=np.array([v==key for v in groups])
            ids[mask]=int(np.argmin(np.median(np.abs(ds[mask]),axis=0)))
            # Exclude only fragments that cannot reach ANY boundary anywhere in
            # this bounded optimization box (1-Lipschitz geometric lower bound).
            reach=np.sqrt(2)*self.cfg["fit_bound_cm"]+2*np.linalg.norm(p[mask],axis=1)*np.sin(self.cfg["fit_bound_yaw_rad"]/2)
            if np.median(np.min(np.abs(ds[mask]),axis=1)-reach)>self.cfg["inlier_cm"]:
                usable[mask]=False
        discarded=int((~usable).sum())
        p,w,ids=p[usable],w[usable],ids[usable]
        groups=[g for g,keep in zip(groups,usable) if keep]
        markers=[];assoc={}
        for mark in observation.landmarks:
            if mark.source!=EvidenceSource.OBSERVED or mark.kind.endswith("_patch") or mark.flags and "view_or_roi_clipped" in mark.flags:
                continue
            mw=to_world([mark.center_cm],seed)[0]
            kind="dashline" if mark.kind=="pink_line_candidate" else mark.kind
            candidates=[m for m in self.track.landmarks if m["kind"]==kind]
            if not candidates:
                continue
            candidates.sort(key=lambda m:np.linalg.norm(mw-m["xy"]))
            chosen=next((m for m in candidates if association and association.get(mark.id)==m["id"]),candidates[0])
            if np.linalg.norm(mw-chosen["xy"])>self.cfg["landmark_search_cm"]:
                continue
            normal=None
            if "endpoints" in chosen:
                tangent=np.diff(np.array(chosen["endpoints"]),axis=0)[0]
                normal=np.array([-tangent[1],tangent[0]])/np.linalg.norm(tangent)
            elif chosen["kind"]=="crosswalk":normal=np.array([0.,1.]) if chosen["approach"]=="ns" else np.array([1.,0.])
            # A partially seen transverse marking constrains distance normal to
            # the line, not the unobserved full stripe group's lateral center.
            markers.append((np.array(mark.center_cm),np.array(chosen["xy"]),chosen["sigma_cm"],normal))
            assoc[mark.id]=chosen["id"]
        sigma=self.cfg["measurement_sigma_cm"]
        if len(p)<4 and not markers:return None

        def residual(q):
            r=self.track.boundary_residual(to_world(p,q),ids)/sigma
            if not self.cfg.get('length_weighted_boundaries',False):r=r*w
            for center,xy,msigma,normal in markers:
                delta=(to_world([center],q)[0]-xy)/math.hypot(self.cfg["landmark_sigma_cm"],msigma)
                r=np.r_[r,delta if normal is None else delta@normal]
            return r

        bound=np.array([self.cfg["fit_bound_cm"],self.cfg["fit_bound_cm"],self.cfg["fit_bound_yaw_rad"]])
        # Keep yaw unwrapped around seed during optimization.
        robust_loss='huber'
        if self.cfg.get('length_weighted_boundaries',False):
            robust_loss=weighted_huber(np.r_[w,np.ones(len(residual(seed))-len(p))])
        fit=least_squares(residual,seed,bounds=(seed-bound,seed+bound),loss=robust_loss,
                          f_scale=self.cfg["huber_scale"],max_nfev=self.cfg["fit_iterations"],x_scale=[1,1,.05])
        raw=self.track.boundary_residual(to_world(p,fit.x),ids)
        inliers=np.abs(raw)<self.cfg["inlier_cm"]
        mass=w*w if self.cfg.get('length_weighted_boundaries',False) else np.ones(len(raw))
        rms=float(np.sqrt(np.average(np.minimum(raw**2,25.),weights=mass))) if len(raw) else 0.
        fraction=float(np.average(inliers,weights=mass)) if len(inliers) else 1.
        landmark_residual=residual(fit.x)[len(p):]
        marker_error=float(np.sqrt(np.mean(landmark_residual**2))) if len(landmark_residual) else 0.
        if rms>self.cfg["postfit_rms_cm"] or fraction<self.cfg["min_inlier_fraction"] or marker_error>3.:
            return None
        # Measurement information, including weak/null directions. Never invert into fake certainty.
        J=fit.jac; info=J.T@J
        eig,V=np.linalg.eigh(info)
        keep=eig>max(float(eig.max())*1e-5,1e-6)
        if not keep.any():
            return None
        H=V[:,keep].T
        R=np.diag(1/np.maximum(eig[keep],1e-8))
        difference=fit.x-prior.mean;difference[2]=wrap(difference[2])
        innovation=H@difference
        S=H@prior.P@H.T+R
        nis=float(innovation@np.linalg.solve(S,innovation))
        K=prior.P@H.T@np.linalg.inv(S)
        mean=prior.mean+K@innovation
        A=np.eye(3)-K@H
        covariance=A@prior.P@A.T+K@R@K.T
        if nis>self.cfg["innovation_limit"]:
            # Alternative recovery needs independent support across spatial pieces, not one edge.
            if len(p)<3:return None
            spread=np.linalg.eigvalsh(np.cov(p.T))
            if len(set(groups))<2 or spread[0]<2. or rms>self.cfg["postfit_rms_cm"]*.6:
                return None
            mean=prior.mean+V[:,keep]@(V[:,keep].T@difference)
            covariance=np.linalg.inv(np.linalg.inv(prior.P)+info)+np.diag([.5,.5,.002])
        mean[2]=wrap(mean[2])
        # Evidence-normalized residual + bounded preference for continuity.
        score=rms+.5*marker_error+min(nis,50)*.005
        return Belief(mean,(covariance+covariance.T)/2,score,assoc,observation.stamp.capture_s), {
            "rms_cm":rms,"inlier_fraction":fraction,"nis":nis,"landmark_residual_sigma":marker_error,
            "unreachable_points":discarded,
            "boundary_evidence_mass":float((w*w).sum()),
            "boundary_weighting":"observed_length" if self.cfg.get('length_weighted_boundaries',False) else "equal_fragment",
            "unweighted_rms_cm":float(np.sqrt(np.mean(np.minimum(raw**2,25.)))) if len(raw) else 0.,
            "information_eigenvalues":eig.tolist(),"boundary_ids":sorted(set(map(int,ids))),
            "relocalized":nis>self.cfg["innovation_limit"]}

    def observe(self,observation):
        now=observation.stamp.capture_s
        if now is None:
            self.last_diagnostics={"status":"missing_capture_time"};return False
        candidates=[]
        for prior in self.hypotheses:
            fitted=self._fit(observation,prior,prior.mean.copy())
            if fitted is not None:
                candidates.append(fitted)
        needs_search=not candidates and now-self.primary.last_observed_s>=self.cfg["search_after_s"]
        if needs_search and now-self.last_search>=self.cfg["search_period_s"]:
            self.last_search=now
            prior=self.primary
            for lateral in (-self.cfg["search_lateral_cm"],0,self.cfg["search_lateral_cm"]):
                for forward in (-self.cfg["search_longitudinal_cm"],0,self.cfg["search_longitudinal_cm"]):
                    if lateral==forward==0:continue
                    seed=prior.mean.copy();seed[:2]+=np.array([lateral,forward])@axes(seed[2])
                    fitted=self._fit(observation,prior,seed)
                    if fitted is not None:candidates.append(fitted)
        # Only branch an actually ambiguous landmark, rather than multiplying all assignments.
        if candidates:
            prior=self.primary
            for mark in observation.landmarks:
                if mark.kind!="blue_circle_candidate":continue
                position=to_world([mark.center_cm],prior.mean)[0]
                options=sorted([m for m in self.track.landmarks if m["kind"]==mark.kind],
                               key=lambda m:np.linalg.norm(position-m["xy"]))[:2]
                if len(options)==2 and np.linalg.norm(position-options[1]["xy"])<self.cfg["landmark_search_cm"]:
                    alternate=self._fit(observation,prior,prior.mean.copy(),{mark.id:options[1]["id"]})
                    if alternate is not None:candidates.append(alternate)
                break
        if not candidates:
            self.last_diagnostics={"status":"prediction_only","hypotheses":len(self.hypotheses)}
            return False
        candidates.sort(key=lambda x:x[0].score)
        selected=[]
        for h,diag in candidates:
            if h.score>candidates[0][0].score+self.cfg["ambiguity_score_gap"]:continue
            if any(np.linalg.norm(h.mean[:2]-a.mean[:2])<self.cfg["merge_cm"] and abs(wrap(h.mean[2]-a.mean[2]))<self.cfg["merge_rad"] for a in selected):continue
            selected.append(h)
            if len(selected)>=self.cfg["max_hypotheses"]:break
        self.hypotheses=selected
        self.last_diagnostics={"status":"observed",**candidates[0][1],"hypotheses":len(selected)}
        return True


class TimedEstimator:
    """Bounded event replay for out-of-sequence observations; no double fusion.

    Commands are body velocities predicted from the *applied* PWM at each change.
    Snapshots after events allow pruning without inventing a fresh initial pose.
    """
    def __init__(self,estimator,start_s,clock_id):
        self.estimator=estimator;self.base=estimator.snapshot();self.base_time=start_s
        self.base_command=(0.,0.);self.now=start_s;self.clock_id=clock_id
        self.events=[];self.seen=set();self.last_observation_key=None
        self.cache=[]
        self.motions=[]
        self.command_sequence=0

    def _propagate(self,command,start,end):
        cuts=sorted({start,end}|{v for m in self.motions for v in (m.start_s,m.end_s) if start<v<end})
        for a,b in zip(cuts,cuts[1:]):
            match=next((m for m in self.motions if m.start_s<=a+1e-9 and m.end_s>=b-1e-9),None)
            if match is not None:self.estimator.predict_visual(match,b-a)
            else:
                while b-a>1.:self.estimator.predict(*command,1.);a+=1.
                self.estimator.predict(*command,b-a)

    def _replay(self,now):
        self.estimator.restore(self.base);t=self.base_time;command=self.base_command;snapshots=[]
        prefix=0
        while prefix<min(len(self.events),len(self.cache)) and self.events[prefix] is self.cache[prefix][0]:
            snapshots.append(self.cache[prefix][1]);prefix+=1
        if prefix:
            self.estimator.restore(snapshots[-1][0]);command=snapshots[-1][1];t=self.events[prefix-1][0]
        for event in self.events[prefix:]:
            at,kind,key,data=event
            self._propagate(command,t,at);t=at
            if kind==0:command=data
            else:self.estimator.observe(data)
            snapshots.append((self.estimator.snapshot(),command))
        # Large idle intervals are subdivided, never silently clipped.
        self._propagate(command,t,now)
        self.cache=list(zip(self.events,snapshots))
        cutoff=now-self.estimator.cfg["max_history_s"]
        n=sum(e[0]<cutoff for e in self.events)
        if n:
            self.base=snapshots[n-1][0];self.base_command=snapshots[n-1][1];self.base_time=self.events[n-1][0]
            self.events=self.events[n:]
            self.cache=self.cache[n:]
            self.seen={(self.clock_id,e[2]) for e in self.events if e[1]==1}
            self.motions=[m for m in self.motions if m.end_s>self.base_time]

    def step(self,now,observation=None,motion=None):
        if now<self.now:raise ValueError("Runtime time went backwards")
        self.now=now
        if motion is not None:
            if (self.base_time<=motion.start_s<motion.end_s<=now and motion.rms_cm<=self.estimator.cfg["vo_max_rms_cm"]
                    and motion.inliers>=self.estimator.cfg["vo_min_inliers"]
                    and not any(abs(m.start_s-motion.start_s)<1e-9 for m in self.motions)):
                self.motions.append(motion)
                self.cache=[item for item in self.cache if item[0][0]<=motion.start_s]
        if observation is not None:
            s=observation.stamp;key=(s.clock_id,s.sequence)
            if s.clock_id!=self.clock_id:raise ValueError("Mixed timestamp clocks")
            if key not in self.seen:
                if s.capture_s is None or not self.base_time<=s.capture_s<=now or now-s.capture_s>self.estimator.cfg["max_observation_age_s"]:
                    self._replay(now);return False
                # A copied latest-frame buffer is still the same exposure even
                # if its consumer accidentally assigns a new sequence number.
                if not any(e[1]==1 and e[0]==s.capture_s for e in self.events):
                    self.events.append((s.capture_s,1,s.sequence,observation));self.seen.add(key)
                    self.events.sort(key=lambda e:(e[0],e[1],e[2]))
        self._replay(now)
        return self.estimator.last_diagnostics.get("status")=="observed"

    def command(self,now,v,omega):
        if now!=self.now:raise ValueError("Command must use current runtime timestamp")
        self.command_sequence+=1
        self.events.append((now,0,self.command_sequence,(float(v),float(omega))))
        self.events.sort(key=lambda e:(e[0],e[1],e[2]))
