"""Test a specific mechanism: delayed centering leaves little disturbance margin.

Taper the existing outgoing lateral offset to zero sooner; retain the corner
entry, source route, speed and teacher family. This is development-only.
"""
import itertools
import json
import time
import numpy as np

from .goal_development import OUT,REPORT,KEYS,PRESETS,SEEDS,sha
from .goal_teacher_probe import FastTeacher,BASE
from .junction_stage02 import dump
from .junction_feasibility import geometry


def centered_reference(scene, end_cm):
    p=scene.points.copy();normal=np.array([scene.do[1],-scene.do[0]])
    forward=(p-scene.c)@scene.do;lateral=(p-scene.c)@normal
    u=np.clip((forward-15)/(end_cm-15),0,1);taper=6*u**5-15*u**4+10*u**3
    p-=taper[:,None]*lateral[:,None]*normal
    margin,h=geometry(p,scene.rim,scene.track)
    s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    curvature=np.gradient(np.unwrap(h),s)
    return p,h,s,curvature,float(margin.min())


def main():
    candidates=[dict(end_cm=None,lateral_gain=.16)]+[dict(end_cm=e,lateral_gain=l) for e,l in itertools.product([25.,30.,35.],[.08,.16])]
    dst=OUT/'exit_reference_probe';dst.mkdir(exist_ok=False)
    dump(REPORT/'exit_reference_protocol.json',dict(hypothesis='Outgoing teacher offset reduces body clearance before the second push; earlier centering may preserve margin.',
        candidates=candidates,source_sha256=sha(__file__),seeds=SEEDS,presets=PRESETS,keys=KEYS,conditions=['normal','mild'],
        rule='All predeclared candidates reported; not a final test. Reject negative static clearance. No change to disturbance envelope.',
        gain_probe_best_mild_clean='18/24; none of 12 original gain/speed candidates improved it'))
    rows=[];geometries={};started=time.perf_counter()
    for seed in SEEDS:
        for preset in PRESETS:
            for key in KEYS:
                for condition in ['normal','mild']:
                    base=FastTeacher(key,condition,seed,preset)
                    for i,config in enumerate(candidates):
                        env=base.clone();env.controls=dict(BASE,lateral_gain=config['lateral_gain'])
                        if config['end_cm'] is not None:
                            env.points,env.h,env.s,env.k,margin=centered_reference(env,config['end_cm'])
                            geometries[str((i,key))]=margin
                            if margin<0:
                                rows.append(dict(candidate=i,key=key,condition=condition,seed=seed,preset=preset,static_rejected=True));continue
                        env.finish();r=dict(env.summary(),candidate=i,config=config,static_rejected=False)
                        folder=dst/(f'{i}_'+'_'.join(key)+f'_{condition}_{seed}_{preset}');folder.mkdir()
                        np.savez_compressed(folder/'trace.npz',physics=np.asarray(env.physics))
                        dump(folder/'result.json',r);rows.append(dict(r,artifact=str(folder)))
            print('exit centering probe',seed,preset,len(rows),round(time.perf_counter()-started,1),'s',flush=True)
            dump(REPORT/'exit_reference_progress.json',rows)
    summary=[]
    for i,config in enumerate(candidates):
        entry=dict(candidate=i,config=config)
        for condition in ['normal','mild']:
            rs=[r for r in rows if r['candidate']==i and r['condition']==condition]
            valid=[r for r in rs if not r['static_rejected']]
            entry[condition]=dict(n=len(rs),static_rejected=len(rs)-len(valid),clean=sum(r['clean_stable'] for r in valid),
                stable=sum(r['stable_success'] for r in valid),median_s=float(np.median([r['simulated_s'] for r in valid])) if valid else None,
                median_alpha=float(np.median([r['angular_accel_rms_rad_s2'] for r in valid])) if valid else None)
        summary.append(entry)
    dump(REPORT/'exit_reference_probe.json',dict(summary=summary,geometry_min_margin_cm=geometries,runs=rows));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
