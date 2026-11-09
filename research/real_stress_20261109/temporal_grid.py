"""Independent searches of stored frames; does NOT replace/reintegrate the logged trajectory."""
import csv
import json
import numpy as np
from experiment import CORE, OUT, driver_for, candidate_grid, axes, save

d = driver_for('ccw')
records = []
for folder in sorted((CORE/'field_runs').glob('*_edges')):
    if not (folder/'23.npz').exists(): continue
    rows = []
    for f in sorted(folder.glob('[0-9][0-9].npz')):
        z = np.load(f)
        p, prior = z['points'], z['prior']
        off, scores, passed, best, constrained = candidate_grid(d, p, prior)
        r = d.sample(d.sdf, p@axes(prior[2])+prior[:2])
        small = (np.linalg.norm(off[:, :2], axis=1) <= 1) & (abs(off[:,2]) <= 1)
        row = dict(frame=int(f.stem), prior_inliers=float(np.mean(abs(r)<2.5)),
                   prior_median=float(np.median(abs(r))), pass_count=int(passed.sum()),
                   small_pass_count=int((small&passed).sum()),
                   best_lat=off[best,0], best_lon=off[best,1], best_yaw=off[best,2],
                   best_loss=scores[best,0], best_inliers=scores[best,2], best_pass=bool(passed[best]))
        rows.append(row)
    with (OUT/(folder.name+'_temporal.csv')).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    result = dict(run=folder.name, frames=len(rows),
                  any_candidate_pass=sum(r['pass_count']>0 for r in rows),
                  small_candidate_pass=sum(r['small_pass_count']>0 for r in rows),
                  lowest_loss_pass=sum(r['best_pass'] for r in rows),
                  best_longitudinal_range=[min(r['best_lon'] for r in rows),max(r['best_lon'] for r in rows)],
                  last=rows[-1])
    records.append(result)
    print(json.dumps(result), flush=True)
save(OUT/'temporal_grid.json', records)
