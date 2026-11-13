"""All 28 mission movements: offline reference/teacher coverage audit.

Held references remain held. This diagnostic does not silently adopt them or
give the actor junction identity. Straight paths use existing stage-02 files.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np

from bench.world import Track
from bench.vehicle import Vehicle,PRESETS as VEHICLES,VARIATIONS
from bench.junction import movement,heading_of
from .goal_teacher_probe import FastTeacher,BASE
from .goal_teacher_validation import SELECTED,legal_start
from .goal_exit_reference_probe import centered_reference
from .junction_learning import REGISTRY,CONDITIONS as LEVELS
from .junction_feasibility import SignedAllocator,events_for
from .goal_development import ROOT,KEYS as PILOT,PRESETS,sha
from .junction_stage02 import dump,OUT as STAGE_OUT,REPORT as STAGE_REPORT

OUT=Path('D:/autodrive_cnn/runs/goal_teacher_coverage_20261113')
REPORT=ROOT/'cnn_finetune/results/goal_teacher_coverage_20261113'
INVENTORY=ROOT/'cnn_finetune/results/goal_development_20261113/coverage_inventory.json'
KEYS=[tuple(r['key']) for r in json.loads(INVENTORY.read_text())['required_movements']]
SPEED={tuple(r['key']):r['speed'] for r in json.loads((STAGE_REPORT/'selection.json').read_text())}
SEEDS=[6201,6202]


class CoverageTeacher(FastTeacher):
    def __init__(self,key,condition,seed,preset,variant):
        # Explicit fast-physics construction avoids the neural collector's
        # accepted-reference assertion without changing its registry or code.
        self.key=tuple(key);self.condition=condition;self.seed=seed;self.preset=preset
        self.level,self.profile=LEVELS[condition];self.p=VEHICLES[preset]
        self.c,self.di,self.do,self.arm=movement(*key);self.track=Track()
        if key[2]=='straight':
            z=np.load(STAGE_OUT/'paths'/'_'.join(key)/'path.npz')
            self.points=z['points'];self.h=z['heading']
            self.s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(self.points,axis=0),axis=1))]
            self.k=np.gradient(np.unwrap(self.h),self.s);self.speed=np.full(len(self.s),SPEED[self.key])
            self.teacher_mode='legacy';self.reference_status='straight_reference_from_stage02'
        else:
            spec=REGISTRY[self.key];z=np.load(spec['reference'])
            self.points=z['points'];self.h=z['heading'];self.s=z['s'];self.k=z['curvature'];self.speed=z['nominal_speed_cm_s']
            self.teacher_mode=spec['mode'];self.reference_status=spec['status']
        self.idx=0;self.planned_v=0.;rng=np.random.default_rng(seed)
        initial=np.r_[self.points[0]+rng.uniform(-1,1)*np.array([self.di[1],-self.di[0]])+rng.uniform(-2,2)*self.di,
                      heading_of(self.di)+np.radians(rng.uniform(-3,3))]
        self.car=Vehicle(self.p,initial,replace(VARIATIONS[self.level],seed=seed));self.rim=self.car.footprint()
        self.alloc=SignedAllocator([.36,.36]);self.events=events_for(seed,self.level)
        self.t=0.;self.capture=1/30;self.previous=np.zeros(2);self.history=[(0.,initial.copy())]
        self.physics=[];self.records=[];self.hidden=None;self.since=None;self.outcome='time_limit';self.exposure=0.
        self.initial_pose=initial.copy();self.controls=dict(BASE);self.initial_sampling=legal_start(self)
        self.variant=variant;self.static_candidate_margin=None
        if variant=='centered':
            self.points,self.h,self.s,self.k,self.static_candidate_margin=centered_reference(self,SELECTED['end_cm'])
            self.controls={k:SELECTED[k] for k in BASE}
    def summary(self):
        return dict(super().summary(),reference_status=self.reference_status,variant=self.variant,
                    initial_sampling=self.initial_sampling,static_candidate_margin_cm=self.static_candidate_margin)


def check():
    for key in PILOT:
        for variant in ['original','centered']:
            a=FastTeacher(key,'mild',6201,'sluggish');legal_start(a)
            if variant=='centered':
                a.points,a.h,a.s,a.k,_=centered_reference(a,SELECTED['end_cm']);a.controls={k:SELECTED[k] for k in BASE}
            b=CoverageTeacher(key,'mild',6201,'sluggish',variant)
            a.finish(until=8);b.finish(until=8)
            np.testing.assert_array_equal(a.physics,b.physics)
            np.testing.assert_array_equal([r['applied'] for r in a.records],[r['applied'] for r in b.records])
    print('all eight pilot-variant physics/command parity checks passed',flush=True)


def run():
    OUT.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    protocol=dict(source_sha256=sha(__file__),keys=KEYS,seeds=SEEDS,presets=PRESETS,conditions=['normal','mild'],
        variants=['original','centered'],inventory_sha256=sha(INVENTORY),
        initial_rule='same legal-start projection as previous teacher validation, record all projections',
        scope='Offline teacher coverage: 28 movements including eight straights and six held turns. No actor or final exam.',
        reference_rule='Held stays held. Negative static centered geometry is counted as rejection, not dropped from denominator.',
        illumination='No camera teacher, mild_light not duplicated; no visual robustness claim',
        selection='no model or reference selection; report every required movement')
    dump(REPORT/'protocol.json',protocol)
    rows=[];start=time.perf_counter()
    for key in KEYS:
        for seed in SEEDS:
            for preset in PRESETS:
                for condition in ['normal','mild']:
                    starts=[]
                    for variant in ['original','centered']:
                        env=CoverageTeacher(key,condition,seed,preset,variant);starts.append(env.car.pose.copy())
                        dst=OUT/('_'.join(key)+f'_{seed}_{preset}_{condition}_{variant}');dst.mkdir()
                        if env.static_candidate_margin is not None and env.static_candidate_margin<0:
                            r=dict(key=key,seed=seed,preset=preset,condition=condition,variant=variant,static_rejected=True,
                                   static_candidate_margin_cm=env.static_candidate_margin,clean_stable=False,stable_success=False,
                                   reference_status=env.reference_status,initial_sampling=env.initial_sampling)
                        else:
                            env.finish();r=dict(env.summary(),static_rejected=False)
                            np.savez_compressed(dst/'physics.npz',samples=np.asarray(env.physics))
                        r['artifact']=str(dst);dump(dst/'result.json',r);rows.append(r)
                    np.testing.assert_array_equal(*starts)
        dump(REPORT/'progress.json',rows)
        print('coverage',key,len(rows),'runs',round(time.perf_counter()-start,1),'s',flush=True)
    summary=[]
    for key in KEYS:
        item=dict(key=key)
        for variant in ['original','centered']:
            item[variant]={}
            for condition in ['normal','mild']:
                a=[r for r in rows if tuple(r['key'])==key and r['variant']==variant and r['condition']==condition]
                item[variant][condition]=dict(n=len(a),clean=sum(r['clean_stable'] for r in a),stable=sum(r['stable_success'] for r in a),
                    static_rejected=sum(r['static_rejected'] for r in a),projected_initial=sum(r['initial_sampling']['projected'] for r in a))
        summary.append(item)
    dump(REPORT/'coverage.json',dict(summary=summary,runs=rows,final_exam_used=False));print('coverage audit complete',len(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['check','run']);a=p.parse_args();globals()[a.task]()
