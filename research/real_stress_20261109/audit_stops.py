"""Grade simulated failed observations at known truth; never used to command the car."""
import json
import numpy as np
from experiment import OUT, driver_for, Plant, axes, CENTER_AHEAD, LENS_AHEAD, save, candidate_grid

records=[]
for path in sorted(OUT.glob('*_stop.npz')):
    z=np.load(path)
    direction=path.name.split('_')[0]
    d=driver_for(direction)
    p=Plant(direction,pivot_behind_center_cm=CENTER_AHEAD)
    truth=z['true_pose']; prior=z['prior']; points=z['points']
    clean=p.world.render((*(truth[:2]+LENS_AHEAD*axes(truth[2])[1]),truth[2]))
    d.pose=prior.copy()
    accepted=d.observe(clean,True)
    observation=d.edge_observation
    r=observation[1] if observation is not None else np.array([])
    q=d.sample(d.sdf,points@axes(truth[2])+truth[:2])
    rec=dict(name=path.stem.removesuffix('_stop'),
             corrupt_at_truth_pass=bool(np.mean(abs(q)<2.5)>=.55 and np.median(abs(q))<=1.8),
             corrupt_at_truth_inliers=float(np.mean(abs(q)<2.5)),
             corrupt_at_truth_median=float(np.median(abs(q))),
             clean_at_same_prior_pass=bool(accepted),
             clean_at_same_prior_inliers=float(np.mean(abs(r)<2.5)) if len(r) else None,
             clean_at_same_prior_median=float(np.median(abs(r))) if len(r) else None,
             prior_position_error_cm=float(np.linalg.norm(prior[:2]-truth[:2])))
    offsets, scores, passed, best, constrained=candidate_grid(d,points,prior)
    best_pose=prior.copy()
    best_pose[:2]+=offsets[best,:2]@axes(prior[2])
    best_pose[2]+=np.deg2rad(offsets[best,2])
    true_offset=(truth[:2]-prior[:2])@axes(prior[2]).T
    rec.update(best_candidate_position_error_cm=float(np.linalg.norm(best_pose[:2]-truth[:2])),
               best_candidate_yaw_error_deg=float(np.rad2deg(best_pose[2]-truth[2])),
               best_candidate_pass=bool(passed[best]),
               true_lateral_longitudinal_offset=true_offset.tolist())
    records.append(rec)
save(OUT/'stop_audit.json',records)
for row in records:print(json.dumps(row),flush=True)
