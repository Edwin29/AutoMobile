"""Execution parity, signed action and metric boundary checks for stages 0--1."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from cnn_finetune.junction_stage02 import OUT, REPORT, ROOT, dump, metrics


def launch(spec,dst):
    if (dst/'result.json').exists():return dst
    with (OUT/(dst.name+'.log')).open('w',encoding='utf-8') as f:
        subprocess.run([sys.executable,'-m','cnn_finetune.junction_stage02','bench','--spec',json.dumps(spec),'--out',str(dst)],
                       cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    return dst


def parity():
    specs=[dict(junction='south',came_from='N',exit='right',seed=1301,mixer='v4'),
           dict(junction='north',came_from='S',exit='left',seed=1303,mixer='direct'),
           dict(junction='center',came_from='E',exit='right',seed=1307,mixer='v4',variation='mild',effects=['realcam_light'])]
    tasks=[]
    for i,s in enumerate(specs):
        for collect in [False,True]:tasks.append((dict(s,collect=collect),OUT/f'parity_{i}_{int(collect)}'))
    OUT.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda item:launch(*item),tasks))
    result=[]
    for i,s in enumerate(specs):
        a,b=[OUT/f'parity_{i}_{j}' for j in [0,1]]
        aa=np.load(a/'physics.npz');bb=np.load(b/'physics.npz')
        frames=[json.loads(line) for line in (a/'steps.jsonl').read_text().splitlines()]
        frames2=[json.loads(line) for line in (b/'steps.jsonl').read_text().splitlines()]
        records=json.loads((b/'collection.json').read_text())
        wall_keys={'ms','ms_cnn','ms_hyb2','ms_cnn_path','ms_map'}
        def close(a,b):
            if isinstance(a,dict):return a.keys()==b.keys() and all(close(a[k],b[k]) for k in a if k not in wall_keys)
            if isinstance(a,list):return len(a)==len(b) and all(close(x,y) for x,y in zip(a,b))
            if isinstance(a,(float,int)) and not isinstance(a,bool):return np.isclose(a,b,atol=1e-6,rtol=1e-7,equal_nan=True)
            return a==b
        same_frames=close(frames,frames2)
        # Each additional record corresponds to exactly one original decision.
        checks=dict(spec=s,exact_physics=np.array_equal(aa['samples'],bb['samples']),
                    exact_clearance=np.array_equal(aa['margins'],bb['margins']), steps_within_tolerance=same_frames,
                    exact_commands=all(x['pwm']==y['pwm'] for x,y in zip(frames,frames2)),
                    numeric_tolerance=dict(atol=1e-6,rtol=1e-7),
                    max_map_pose_difference=float(max(np.max(np.abs(np.asarray(x['est'])-np.asarray(y['est']))) for x,y in zip(frames,frames2) if x.get('est') is not None)),
                    frames=len(frames),collection_frames=len(records),excluded_wall_clock_fields=sorted(wall_keys),
                    latency_36ms=all(abs(r['apply_s']-r['capture_s']-.036)<1e-9 for r in records),
                    proposals_equal_applied=all(r['proposed_pwm']==r['applied_pwm'] for r in records))
        assert all(checks[k] for k in ['exact_physics','exact_clearance','steps_within_tolerance','exact_commands','latency_36ms','proposals_equal_applied'])
        assert len(frames)==len(records)
        result.append(checks)
    dump(REPORT/'parity.json',result);print('PARITY PASS',flush=True)


def unit_checks():
    from bench.vehicle import Vehicle, PRESETS
    from bench.controllers import HybridJunction
    from bench.run import timing_profile
    from bench.world import Track
    from cnn_finetune.junction_feasibility import SignedAllocator
    ctrl=HybridJunction('ccw',Track().start_pose('ccw'),timing_profile('cnn_pi'))
    results=[]
    for label,pwm in [('forward',[.4,.4]),('inner_stop',[.5,0]),('reverse',[-.4,-.4]),
                       ('pivot_right',[.4,-.4]),('pivot_left',[-.4,.4]),('low_turn',[.35,.30]),('stop',[0,0])]:
        car=Vehicle(PRESETS['field_estimate'],[0,0,0]);car.command(0,pwm)
        for i in range(200):car.step(i*.02,.02)
        ctrl.on_applied(10+len(results),pwm)
        mv,mw=ctrl.driver.body_cmds[-1][1:]
        assert np.sign(mv)==np.sign(car.v) and np.sign(mw)==np.sign(car.omega),(label,mv,mw,car.v,car.omega)
        assert np.allclose(ctrl.driver.pwm,pwm)
        results.append(dict(action=label,pwm=pwm,actual_v=car.v,actual_omega=car.omega,map_v=mv,map_omega=mw))
    # Delay and stop/resume: physical speed changes after command+actuation delay.
    car=Vehicle(PRESETS['field_estimate'],[0,0,0]);car.command(0,[.4,.4])
    car.step(0,.02);assert car.v==0
    for i in range(1,100):car.step(i*.02,.02)
    car.command(2,[0,0])
    for i in range(100,200):car.step(i*.02,.02)
    stopped=car.v;assert stopped<1e-6
    car.command(4,[.4,.4])
    for i in range(200,300):car.step(i*.02,.02)
    assert car.v>6
    # Signed dither mean including near-zero/reverse; one accumulator step per command.
    alloc=SignedAllocator([.3,.3]);outs=np.array([alloc.apply([.1,-.15]) for _ in range(600)])
    assert np.allclose(outs.mean(axis=0),[.1,-.15],atol=.00051) and alloc.calls==600
    # Metric boundaries: heading alone does not recover; nonrecovery kept; STOP masking.
    t=np.arange(.02,4.02,.02);poses=np.zeros((len(t),3));poses[:,1]=45
    v=np.zeros(len(t));w=v.copy();margin=np.ones(len(t));event=[dict(t0=0,duration=1)]
    good=metrics(t,poses,v,w,margin,[0,0],np.array([0,1]),event)
    poses[:,0]=2;bad=metrics(t,poses,v,w,margin,[0,0],np.array([0,1]),event)
    assert good['stable_success'] and not bad['stable_success'] and bad['unrecovered']==1 and bad['recovery'][0]['recovery_s'] is None
    poses[:,0]=0;masked=metrics(t,poses,v,w,margin,[0,0],np.array([0,1]),mandatory=np.ones(len(t),bool))
    assert masked['unnecessary_stops']==0 and good['unnecessary_stops']==1
    interrupted=metrics(t,poses,v,w,margin,[0,0],np.array([0,1]),[dict(t0=3.9,duration=.7)])
    assert interrupted['unrecovered']==1 and interrupted['recovery'][0]['interrupted_during_event']
    # Cache body hash is not checked by legacy loader: explicitly check both ourselves.
    import pickle,hashlib
    from bench.junction import PLAN_CACHE
    cache=pickle.loads(PLAN_CACHE.read_bytes())
    rim=Vehicle(PRESETS['field_estimate'],np.zeros(3)).footprint()
    assert cache['meta']['rim_sha256']==hashlib.sha256(np.ascontiguousarray(rim).tobytes()).hexdigest()
    dump(REPORT/'action_metric_checks.json',dict(passed=True,actions=results,delay_stop_resume=True,
        signed_dither_mean=outs.mean(axis=0),dither_calls=alloc.calls,metric_nonrecovery=True,mandatory_stop_mask=True,cache_body_hash=True))
    print('ACTION / METRICS PASS',flush=True)


if __name__=='__main__':
    unit_checks();parity()
