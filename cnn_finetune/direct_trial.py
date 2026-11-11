"""Fast controller-only trials: same physics, timing and scorer, no camera rendering.
These are diagnostic tests, not the realcam acceptance matrix.
"""
import json
from pathlib import Path
import numpy as np
from bench.world import Track,Camera
from bench.vehicle import Vehicle,PRESETS
from bench.score import Scorer
from cnn_finetune.direct_teacher import DirectTeacher as Teacher


def trial(vehicle,direction,variant):
    track=Track(); camera=Camera(); params=PRESETS[vehicle]
    rng=np.random.default_rng(41)
    pose=track.start_pose(direction)+[rng.uniform(-1,1),rng.uniform(-2,2),np.radians(rng.uniform(-3,3))]
    car=Vehicle(params,pose); scorer=Scorer(track,car.footprint(),camera.route,direction,pose)
    teacher=Teacher(direction,lookahead=12 if variant==4 else 25 if variant==5 else 18,variant=variant)
    teacher.sim_params=params
    t=0.; capture=1/30; pwm=np.zeros(2)
    def advance(target):
        nonlocal t
        while t<target-1e-10 and scorer.end is None:
            dt=min(.02,target-t);car.step(t,dt);t+=dt;scorer.sample(t,dt,car.pose,pwm)
    while capture<200 and scorer.end is None:
        advance(capture); observed=car.pose.copy();done=capture+.036;advance(done)
        if scorer.end:break
        pwm,info=teacher.step(observed);car.command(done,pwm);capture=max(capture+1/30,done)
    return dict(vehicle=vehicle,direction=direction,variant=variant,**scorer.summary(scorer.end or 'time_limit'))

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--variant',type=int,required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    rows=[]
    for v in ['field_estimate','weak_left_turn','sluggish','unequal_sides']:
        for d in ['cw','ccw']:
            r=trial(v,d,a.variant);rows.append(r);print(v,d,r['verified_lap'],r['min_margin_cm'],flush=True)
    Path(a.out).write_text(json.dumps(rows,indent=2),encoding='utf-8')
