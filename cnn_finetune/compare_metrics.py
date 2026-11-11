"""Paired outcomes, clearance, post-corner heading recovery and wheel-floor ratios."""
import json
from pathlib import Path
import numpy as np
from bench.world import Camera,wrap

def extra(folder,direction):
    r=np.asarray(Camera().route);r=r if direction=='cw' else r[::-1]
    tang=np.roll(r,-2,axis=0)-np.roll(r,2,axis=0);heading=np.arctan2(tang[:,0],tang[:,1])
    arc=np.linalg.norm(np.roll(r,-1,axis=0)-r,axis=1)
    k=wrap(np.roll(heading,-3)-np.roll(heading,3))/np.maximum(sum(np.roll(arc,j) for j in range(-3,3)),1e-6)
    events=[];pending=None;previous=False;floor_count=corner_count=0
    for line in (Path(folder)/'steps.jsonl').read_text().splitlines():
        s=json.loads(line);pose=np.array(s['truth']);i=int(np.argmin(np.sum((r-pose[:2])**2,axis=1)));corner=bool(abs(k[i])>.02)
        if previous and not corner:pending=s['t']
        if pending is not None and abs(wrap(pose[2]-heading[i]))<=np.radians(5):events.append(s['t']-pending);pending=None
        if corner:corner_count+=1;floor_count+=min(s['pwm'])<.30-1e-9
        previous=corner
    return dict(post_corner_recovery_s=events,unrecovered_at_end=pending is not None,corner_frames=corner_count,inner_below_floor_fraction=floor_count/max(corner_count,1))

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--matrix',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    rows=json.loads(Path(a.matrix).read_text())
    for r in rows:r['extra']=extra(r['path'],r['direction'])
    Path(a.out).write_text(json.dumps(rows,indent=2),encoding='utf-8')
