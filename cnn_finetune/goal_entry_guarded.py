"""Abstain from model-predicted unsafe recovery commands.

This removes the risky fallback of the previous teacher. Finite scenarios and
nominal model uncertainty remain; it is not a formally verified safety shield.
"""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .goal_entry_reactive import ReactivePhase,REPORT as PARENT_REPORT
from .goal_entry_recovery_search import actions,capture_root,short_branch
from .goal_entry_robust import decide,forecast_duration,DURATIONS,REPORT as ROOT_REPORT
from .goal_phase_disturbance import PhaseTeacher,registry,TURNS,PHASES
from .goal_boundary_teacher import ForecastInput
from .goal_boundary_teacher_v2 import observed_residual
from .goal_teacher_probe import FastTeacher
from .goal_teacher_coverage import ROOT,PRESETS
from .goal_continuous_clearance import continuous_clearance
from .goal_development import sha
from .junction_stage02 import dump

OUT=Path('D:/autodrive_cnn/runs/goal_entry_guarded_20261113')
REPORT=ROOT/'cnn_finetune/results/goal_entry_guarded_20261113'


class GuardedPhase(ReactivePhase):
    def __init__(self,key,seed,preset,phase):super().__init__(key,seed,preset,phase,'guarded')

    def teacher(self,observed,v,w):
        if self.program is None:
            residual=observed_residual(self)
            if np.linalg.norm(residual[:2])>.25 or abs(residual[2])>.005:
                snapshot=ForecastInput.capture(self);programs=actions()
                baseline=min(forecast_duration(snapshot,programs[:1],residual,d)['min_margin'][0] for d in DURATIONS)
                rejected=False
                if baseline>=.3:selected=0;margin=float(baseline)
                else:
                    pred=decide(snapshot,programs,residual);selected=pred['index'];margin=float(pred['worst_margins'][selected])
                    if selected!=0 and not (margin>=.01 and pred['moving_all_scenarios'][selected]):
                        rejected=True;selected=0
                self.decisions.append(dict(t=self.t,index=selected,residual=residual,predicted_proposal_margin=margin,
                                           baseline_predicted_margin=float(baseline),unsafe_proposal_rejected=rejected))
                if selected!=0:self.program=programs[selected].copy();self.program_frame=0
        base=FastTeacher.teacher(self,observed,v,w)
        if self.program is None:return base
        s,d=self.program[:2] if self.program_frame<7 else self.program[2:]
        self.program_frame+=1
        if self.program_frame==14:self.program=None
        if s==1. and d==0.:return base
        mean=float(base.mean())*s;delta=float((base[0]-base[1])/2)+d;self.modified_frames+=1
        return np.clip([mean+delta,mean-delta],-1,1)


def freeze():
    parent=json.loads((PARENT_REPORT/'protocol.json').read_text())
    roots=json.loads((ROOT_REPORT/'results.json').read_text())['rows'];derived=[];checked=[]
    for r in roots:
        d=r['decision'];allowed=d['worst_margins'][d['index']]>=.01 and d['moving_all_scenarios'][d['index']]
        derived.append(dict(clean=r['result']['continuous_clean'] if allowed else r['source']['continuous_clean'],
            clean_moving=r['result']['clean_moving'] if allowed else r['source']['continuous_clean'],baseline_clean=r['source']['continuous_clean']))
    examples=[next(r for r in roots if r['decision']['index']!=0 and r['decision']['worst_margins'][r['decision']['index']]<0),
              next(r for r in roots if r['decision']['index']!=0 and r['decision']['worst_margins'][r['decision']['index']]>=.01 and r['decision']['moving_all_scenarios'][r['decision']['index']])]
    for r in examples:
        root,_=capture_root(dict(source=r['source'],root_mode='post_observation'))
        def convert():
            env=GuardedPhase.__new__(GuardedPhase);env.__dict__.update(copy.deepcopy(root.__dict__));del env.__dict__['teacher'];env.initialize('guarded');return env
        a=convert();b=convert();b.events=[]
        ua=a.teacher(a.car.pose.copy(),a.car.v,a.car.omega);ub=b.teacher(b.car.pose.copy(),b.car.v,b.car.omega)
        np.testing.assert_array_equal(ua,ub)
        d=r['decision'];allowed=d['worst_margins'][d['index']]>=.01 and d['moving_all_scenarios'][d['index']]
        assert a.decisions[0]['index']==(d['index'] if allowed else 0)
        checked.append(dict(root=r['index'],accepted=bool(allowed),future_event_mutation_invariant=True))
    OUT.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    dump(REPORT/'protocol.json',dict(source_sha256=sha(__file__),parent_protocol_sha256=sha(PARENT_REPORT/'protocol.json'),
        parent_source_sha256=sha(ROOT/'cnn_finetune/goal_entry_reactive.py'),dependencies=parent['dependencies'],module_hashes=parent['module_hashes'],
        keys=TURNS,seeds=parent['seeds'],presets=PRESETS,phases=PHASES,
        change='If selected nonzero proposal is predicted unsafe in any duration or not moving in all durations, keep baseline command; never execute risk-minimizing unsafe fallback',
        development_root_derivation=dict(n=240,clean=sum(r['clean'] for r in derived),clean_moving=sum(r['clean_moving'] for r in derived),
            gains=sum(r['clean'] and not r['baseline_clean'] for r in derived),losses=sum(r['baseline_clean'] and not r['clean'] for r in derived)),
        provenance='Rule derived exclusively from6501-6504 root results. Prospective6601/6602 reactive results were still running and not inspected when this rule was frozen.',
        checks=checked,final_exam_used=False,new_training=False,scope='offline teacher only, same causal trigger and14-command commitment; model safety remains approximate'))
    shutil.copy2(__file__,OUT/'source_snapshot.py');print('guarded candidate frozen',flush=True)


def run_one(item):
    key,seed,preset,phase=item;env=GuardedPhase(key,seed,preset,phase).finish();trace=np.asarray(env.physics)
    if phase=='none':
        baseline=PhaseTeacher(key,seed,preset,'reserve',phase,registry()).finish()
        np.testing.assert_array_equal(trace,baseline.physics);assert len(env.decisions)==0
    margin=continuous_clearance(trace[:,1:4],env.p,env.track)
    dst=OUT/'rollouts'/('_'.join(key)+f'_{seed}_{preset}_{phase}');dst.mkdir(parents=True,exist_ok=False)
    r=dict(env.summary(),profile='guarded',artifact=str(dst),continuous_min_clearance_cm=float(margin.min()))
    r['continuous_clean']=bool(r['stable_success'] and r['initial_valid'] and margin.min()>=-1e-9)
    r['clean_moving']=bool(r['continuous_clean'] and r['translation_pause_s']==0 and trace[:,4].min()>=-.01)
    np.savez_compressed(dst/'physics.npz',physics=trace);dump(dst/'result.json',r);dump(dst/'decisions.json',env.decisions)
    return r


def run():
    p=json.loads((REPORT/'protocol.json').read_text());assert sha(__file__)==p['source_sha256']
    for f,h in p['dependencies'].items():assert sha(f)==h,f
    for m,h in p['module_hashes'].items():assert sha(ROOT/'cnn_finetune'/(m+'.py'))==h,m
    assert sha(ROOT/'cnn_finetune/goal_entry_reactive.py')==p['parent_source_sha256']
    assert sha(PARENT_REPORT/'protocol.json')==p['parent_protocol_sha256']
    cases=list(itertools.product(TURNS,p['seeds'],PRESETS,PHASES));rows=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=3) as pool:
        for i,r in enumerate(pool.map(run_one,cases),1):
            rows.append(r)
            if i%24==0:
                dump(REPORT/'progress.json',rows);print('guarded teacher',i,'/',len(cases),'elapsed',round(time.perf_counter()-started,1),flush=True)
    summary={}
    for phase in PHASES:
        a=[r for r in rows if r['phase']==phase]
        summary[phase]=dict(n=len(a),clean=sum(r['continuous_clean'] for r in a),clean_moving=sum(r['clean_moving'] for r in a),
            exposed=sum(r['disturbance_exposed'] for r in a),pauses=sum(r['translation_pause_s']>0 for r in a),
            outcomes={v:sum(r['outcome']==v for r in a) for v in sorted(set(r['outcome'] for r in a))})
    dump(REPORT/'results.json',dict(summary=summary,rows=rows,none_physics_identical=True,final_exam_used=False));print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['freeze','run']);a=p.parse_args();globals()[a.task]()
