"""Predeclared gain/speed diagnostic of the existing offline path teacher.

No neural changes. Rendering can be skipped because this teacher never reads
images. The fast execution path is checked against the rendered teacher.
"""
import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np
from torch import nn

from bench.world import advance, wrap
from .goal_development import LocalScene, OUT, REPORT, KEYS, PRESETS, SEEDS, sha
from .junction_stage02 import dump

BASE = dict(heading_gain=1.3,lateral_gain=.16,speed_scale=1.)
CANDIDATES=[BASE]+[dict(heading_gain=h,lateral_gain=l,speed_scale=s)
    for h,l,s in itertools.product([1.3,2.0],[.08,.16,.24],[.8,1.]) if (h,l,s)!=(1.3,.16,1.)]


class FastTeacher(LocalScene):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,encoder=nn.Identity(),**kwargs)
        self.controls=dict(BASE)

    def teacher(self,observed,v,w):
        predicted=advance(observed,v,w,.036+self.p.actuation_delay_s+self.p.time_constant_s)
        near=self.idx+int(np.argmin(np.linalg.norm(self.points[self.idx:min(len(self.points),self.idx+100)]-predicted[:2],axis=1)))
        self.idx=max(self.idx-2,near);k=float(self.k[self.idx]);h=self.h[self.idx]
        lat=float((predicted[:2]-self.points[self.idx])@np.array([np.cos(h),-np.sin(h)]));herr=wrap(h-predicted[2])
        target=float(self.speed[self.idx])*self.controls['speed_scale']
        if self.teacher_mode=='planned':self.planned_v=min(target,self.planned_v+.036);v=self.planned_v
        else:v=target
        omega=v*k+self.controls['heading_gain']*herr-self.controls['lateral_gain']*lat
        eff=self.p.turn_eff_left if omega<0 else self.p.turn_eff_right;diff=omega*self.p.track_cm/(2*eff)
        desired=np.array([(v+diff)/self.p.cm_s_per_duty_left,(v-diff)/self.p.cm_s_per_duty_right])
        return desired/max(1,float(max(abs(desired))))

    def step(self,policy=None,intervention=None):
        assert policy is None and intervention is None
        self.advance_to(self.capture)
        if self.outcome!='time_limit':return False
        label=self.teacher(self.car.pose.copy(),self.car.v,self.car.omega)
        done=self.capture+.036;self.advance_to(done)
        if self.outcome!='time_limit':return False
        applied=self.alloc.apply(label);self.car.command(done,applied)
        self.records.append(dict(capture=self.capture,done=done,action=label.copy(),applied=applied.copy()))
        self.previous=applied.copy();self.capture=max(self.capture+1/30,done)
        return True


def check():
    from .junction_learning import Encoder,setup
    setup();encoder=Encoder().cuda().eval()
    a=LocalScene(('east','W','right'),'mild',4101,'sluggish',encoder)
    b=FastTeacher(('east','W','right'),'mild',4101,'sluggish')
    a.finish(until=18);b.finish(until=18)
    np.testing.assert_array_equal(a.physics,b.physics)
    np.testing.assert_array_equal([r['applied'] for r in a.records],[r['applied'] for r in b.records])
    dump(REPORT/'teacher_fast_parity.json',dict(physics_exact=True,commands_exact=True,seconds=a.t,frames=len(a.records),source_sha256=sha(__file__)))
    print('fast teacher exact parity',a.t,len(a.records),flush=True)


def run():
    check()
    dst=OUT/'teacher_probe';dst.mkdir(exist_ok=False)
    dump(REPORT/'teacher_probe_protocol.json',dict(candidates=CANDIDATES,seeds=SEEDS,presets=PRESETS,keys=KEYS,
         conditions=['normal','mild'],selection='diagnostic only, no replacement; report safety, speed and smoothness tradeoffs',
         source_sha256=sha(__file__),note='Same rendered-physics teacher and reference; only gains and nominal speed vary. Full cross, no case exclusions.'))
    rows=[];start=time.perf_counter()
    for seed in SEEDS:
        for preset in PRESETS:
            for key in KEYS:
                for condition in ['normal','mild']:
                    base=FastTeacher(key,condition,seed,preset)
                    for i,config in enumerate(CANDIDATES):
                        env=base.clone();env.controls=config;env.finish()
                        r=dict(env.summary(),candidate=i,controls=config)
                        folder=dst/(f'{i}_'+ '_'.join(key)+f'_{condition}_{seed}_{preset}');folder.mkdir()
                        np.savez_compressed(folder/'trace.npz',physics=np.asarray(env.physics),commands=np.array([[q['capture'],q['done'],*q['action'],*q['applied']] for q in env.records]))
                        dump(folder/'result.json',r);rows.append(dict(r,artifact=str(folder)))
            dump(REPORT/'teacher_probe_progress.json',rows)
            print('teacher gain probe',seed,preset,len(rows),'runs',round(time.perf_counter()-start,1),'s',flush=True)
    summary=[]
    for i,config in enumerate(CANDIDATES):
        selected=[r for r in rows if r['candidate']==i];entry=dict(candidate=i,controls=config)
        for condition in ['normal','mild']:
            rs=[r for r in selected if r['condition']==condition]
            entry[condition]=dict(n=len(rs),clean=sum(r['clean_stable'] for r in rs),stable=sum(r['stable_success'] for r in rs),
                median_s=float(np.median([r['simulated_s'] for r in rs])),median_alpha=float(np.median([r['angular_accel_rms_rad_s2'] for r in rs])))
        summary.append(entry)
    dump(REPORT/'teacher_probe.json',dict(summary=summary,runs=rows));print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['check','run']);a=p.parse_args()
    check() if a.task=='check' else run()
