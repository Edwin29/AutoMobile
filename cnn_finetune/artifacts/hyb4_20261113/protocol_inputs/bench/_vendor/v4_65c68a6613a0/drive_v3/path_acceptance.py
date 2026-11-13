"""Bounded-prefix road clearance policy; not an obstacle collision guarantee."""
import numpy as np
from .geometry import path_geometry, sigma_poses, to_world


def commitment_distance(cfg, speed, observation_age):
    # Upper speed and brake/latency are provisional until physically measured.
    bound=max(0.,speed)*cfg.get('speed_bound_factor',1.5)
    time=cfg.get('replan_period_s',.2)+cfg['stop_latency_s']+max(0.,observation_age)
    return bound*time+bound*bound/(2*cfg['brake_cm_s2'])+cfg.get('commitment_extra_cm',.5)


def margin_profile(track, rim, path, pose, headings=None):
    if headings is None:headings=path_geometry(path)[1]
    world=to_world(path,pose);angle=headings+pose[2]
    c,s=np.cos(angle),np.sin(angle)
    p=np.empty((len(world),len(rim),2))
    p[:,:,0]=world[:,None,0]+c[:,None]*rim[None,:,0]+s[:,None]*rim[None,:,1]
    p[:,:,1]=world[:,None,1]-s[:,None]*rim[None,:,0]+c[:,None]*rim[None,:,1]
    return track.clearance(p).min(axis=1)


def judge_profiles(short_profiles, mean_profiles, cfg, short_means=None):
    """Each candidate is compared to ITS OWN initial clearance, never global min.

    Epsilon applies only above target; an already deficient candidate cannot
    ratchet downward by epsilon at every replanning step.
    """
    short=np.asarray(short_profiles);means=np.asarray(mean_profiles)
    target=cfg['clearance_cm'];eps=cfg.get('clearance_tolerance_cm',.05)
    depth=cfg.get('allowed_penetration_cm',.5);tol=1e-7
    initial=short[:,0];minimum=short.min(axis=1)
    recovery_tolerance=cfg.get('recovery_motion_tolerance_cm',0.)
    floor=np.where(initial>=target,target-eps,initial-recovery_tolerance)
    details=dict(short_minimum_cm=float(minimum.min()),mean_minimum_cm=float(means.min()),
        initial_worst_cm=float(initial.min()),worst_change_cm=float((minimum-initial).min()))
    support_depth=cfg.get('uncertain_penetration_cm',depth)
    if np.any(minimum < -support_depth-tol):return False,'penetration_depth',details
    if short_means is not None and np.min(short_means)<-depth-tol:
        return False,'nominal_penetration_depth',details
    if np.any(minimum < floor-tol):return False,'candidate_worsening',details
    if recovery_tolerance>0 and np.any(short[:,-1]<np.minimum(initial,target)-tol):
        return False,'candidate_end_worsening',details
    for profile in (short[:1] if short_means is None else short_means):
        required=min(profile[0],target)
        if profile[0]<0:required=min(target,profile[0]+cfg.get('recovery_progress_cm',.05))
        if profile[-1]<required-tol:return False,'nominal_short_no_recovery',details
    # The nominal path may accept slight existing intrusion, but must recover.
    for profile in means:
        if profile.min() < -depth-tol:return False,'nominal_long_penetration',details
        required=min(profile[0],target)
        if profile[0]<0:required=min(target,profile[0]+cfg.get('recovery_progress_cm',.05))
        if profile[-1]<required-tol:return False,'nominal_no_recovery',details
    return True,'bounded_prefix',details


def evaluate_prefix(track,rim,path,beliefs,cfg,commitment):
    arc,head,_=path_geometry(path)
    # Actual current footprint has body heading zero, irrespective of numerical
    # spline derivative at its first sample. Future samples keep their headings.
    head=head.copy();head[0]=0.
    if arc[-1]+1e-6<commitment:return False,'insufficient_commitment_horizon',{}
    end=min(len(arc)-1,int(np.searchsorted(arc,commitment)))
    short=[];means=[];short_means=[]
    for belief in beliefs:
        poses=sigma_poses(belief.mean,belief.P,cfg['sigma_multiplier'])
        for j,pose in enumerate(poses):
            if j==0:
                profile=margin_profile(track,rim,path,pose,head)
                means.append(profile);short.append(profile[:end+1]);short_means.append(profile[:end+1])
            else:short.append(margin_profile(track,rim,path[:end+1],pose,head[:end+1]))
    ok,reason,details=judge_profiles(short,means,cfg,short_means)
    details.update(commitment_cm=float(commitment),checked_prefix_cm=float(arc[end]),
        sampled_poses=len(short),long_mean_hypotheses=len(means))
    return ok,reason,details
