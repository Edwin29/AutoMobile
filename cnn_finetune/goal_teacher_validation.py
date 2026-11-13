"""Freeze the mechanism-selected teacher, then validate on new development seeds."""
import json
import time
import numpy as np

from bench.world import to_world
from .goal_development import OUT,REPORT,KEYS,PRESETS,sha
from .goal_teacher_probe import FastTeacher,BASE
from .goal_exit_reference_probe import centered_reference
from .junction_stage02 import dump

VALIDATION_SEEDS=list(range(5201,5209))
SELECTED=dict(end_cm=35.,lateral_gain=.08,heading_gain=1.3,speed_scale=1.)


def legal_start(scene):
    """Condition solely on initial legality, before either controller runs.

    Keep yaw and longitudinal offset; project a sampled illegal axle lateral
    offset to the incoming centerline. Record every projection for both arms.
    """
    p=scene.car.pose.copy();before=float(scene.track.clearance(to_world(scene.rim,p)).min())
    if before<0:
        normal=np.array([scene.di[1],-scene.di[0]])
        p[:2]-=normal*float((p[:2]-scene.c)@normal)
        scene.car.pose=p;scene.initial_pose=p.copy();scene.history=[(0.,p.copy())]
    after=float(scene.track.clearance(to_world(scene.rim,p)).min())
    assert after>=0
    return dict(projected=before<0,original_margin_cm=before,initial_margin_cm=after,pose=p.tolist())


def main():
    dst=OUT/'teacher_validation';dst.mkdir(exist_ok=False)
    dump(REPORT/'teacher_selected_before_validation.json',dict(selected=SELECTED,selected_candidate=5,
        reason='24/24 normal and mild clean on selection; lower median angular acceleration and no time regression',
        seeds=VALIDATION_SEEDS,conditions=['normal','mild','harsh'],presets=PRESETS,keys=KEYS,
        initial_rule='Keep original draw if legal; otherwise project only initial axle lateral offset to road center before either arm; report all projections',
        harsh='Existing unmodified harsh envelope; feasibility diagnostic, not acceptance envelope certification',
        source_sha256=sha(__file__),reference_probe_source_sha256=sha(__file__.replace('goal_teacher_validation','goal_exit_reference_probe'))))
    rows=[];start=time.perf_counter()
    for seed in VALIDATION_SEEDS:
        for preset in PRESETS:
            for key in KEYS:
                for condition in ['normal','mild','harsh']:
                    base=FastTeacher(key,condition,seed,preset);initial=legal_start(base)
                    for name in ['original','centered']:
                        env=base.clone()
                        if name=='centered':
                            env.points,env.h,env.s,env.k,margin=centered_reference(env,SELECTED['end_cm'])
                            assert margin>=0
                            env.controls={k:SELECTED[k] for k in BASE}
                        env.finish();r=dict(env.summary(),arm=name,initial_sampling=initial)
                        folder=dst/(name+'_'+'_'.join(key)+f'_{condition}_{seed}_{preset}');folder.mkdir()
                        np.savez_compressed(folder/'trace.npz',physics=np.asarray(env.physics))
                        dump(folder/'result.json',r);rows.append(dict(r,artifact=str(folder)))
        dump(REPORT/'teacher_validation_progress.json',rows)
        print('teacher independent validation',seed,len(rows),round(time.perf_counter()-start,1),'s',flush=True)
    summary={arm:{c:dict(n=len(rs),stable=sum(r['stable_success'] for r in rs),clean=sum(r['clean_stable'] for r in rs),
        projected_initial=sum(r['initial_sampling']['projected'] for r in rs),median_s=float(np.median([r['simulated_s'] for r in rs])),
        median_alpha=float(np.median([r['angular_accel_rms_rad_s2'] for r in rs])))
        for c in ['normal','mild','harsh'] for rs in [[r for r in rows if r['arm']==arm and r['condition']==c]]} for arm in ['original','centered']}
    dump(REPORT/'teacher_validation.json',dict(summary=summary,runs=rows));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
