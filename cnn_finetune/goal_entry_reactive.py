"""Offline teacher with causal observed-motion triggering and committed actions.

The environment may place phase disturbances. This controller does not read
their phase, schedule, duration, onset flag, or future random state.
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
from .goal_phase_disturbance import PhaseTeacher,registry,TURNS,PHASES
from .goal_teacher_coverage import ROOT,PRESETS
from .goal_teacher_probe import FastTeacher
from .goal_entry_recovery_search import actions,capture_root,short_branch
from .goal_entry_forecast import forecast,choose,REPORT as FORECAST_REPORT
from .goal_entry_robust import decide,forecast_duration,DURATIONS,REPORT as ROBUST_REPORT
from .goal_boundary_teacher import ForecastInput
from .goal_boundary_teacher_v2 import observed_residual
from .goal_continuous_clearance import continuous_clearance
from .goal_development import sha
from .junction_stage02 import dump

OUT=Path('D:/autodrive_cnn/runs/goal_entry_reactive_20261113')
REPORT=ROOT/'cnn_finetune/results/goal_entry_reactive_20261113'


class ReactivePhase(PhaseTeacher):
    def __init__(self,key,seed,preset,phase,profile):
        super().__init__(key,seed,preset,'reserve',phase,registry());self.initialize(profile)

    def initialize(self,profile):
        self.profile_name=profile;self.program=None;self.program_frame=0;self.decisions=[];self.modified_frames=0

    def teacher(self,observed,v,w):
        if self.program is None:
            residual=observed_residual(self)
            if np.linalg.norm(residual[:2])>.25 or abs(residual[2])>.005:
                snapshot=ForecastInput.capture(self);programs=actions()
                if self.profile_name=='hold035':
                    prediction=forecast(snapshot,programs,residual,'hold035');selected=choose(prediction,programs)
                    margin=float(prediction['min_margin'][selected])
                else:
                    # Exactly equivalent fast path for the unchanged action.
                    baseline=min(forecast_duration(snapshot,programs[:1],residual,d)['min_margin'][0] for d in DURATIONS)
                    if baseline>=.3:selected=0;margin=float(baseline)
                    else:
                        prediction=decide(snapshot,programs,residual);selected=prediction['index'];margin=float(prediction['worst_margins'][selected])
                self.decisions.append(dict(t=self.t,index=selected,residual=residual,predicted_margin=margin))
                if selected!=0:self.program=programs[selected].copy();self.program_frame=0
        base=FastTeacher.teacher(self,observed,v,w)
        if self.program is None:return base
        s,d=self.program[:2] if self.program_frame<7 else self.program[2:]
        self.program_frame+=1
        if self.program_frame==14:self.program=None
        if s==1. and d==0.:return base
        mean=float(base.mean())*s;delta=float((base[0]-base[1])/2)+d
        self.modified_frames+=1
        return np.clip([mean+delta,mean-delta],-1,1)

    def summary(self):
        return dict(super().summary(),reactive_profile=self.profile_name,decisions=len(self.decisions),
                    nonzero_decisions=sum(r['index']!=0 for r in self.decisions),modified_frames=self.modified_frames)


def checks():
    checks=[]
    for profile,path in [('hold035',FORECAST_REPORT/'hold035.json'),('robust',ROBUST_REPORT/'results.json')]:
        rows=json.loads(path.read_text())['rows']
        selected=[r for r in rows if r['decision']['index']!=0][:2]+[next(r for r in rows if r['decision']['index']==0)]
        for r in selected:
            root,_=capture_root(dict(source=r['source'],root_mode='post_observation'))
            def convert():
                env=ReactivePhase.__new__(ReactivePhase);env.__dict__.update(copy.deepcopy(root.__dict__))
                del env.__dict__['teacher'];env.initialize(profile);return env
            a=convert();b=convert();b.events=[dict(kind='push',t0=a.t+.1,duration=.9,lateral_cm=999,yaw_deg=99)]
            ua=a.teacher(a.car.pose.copy(),a.car.v,a.car.omega);ub=b.teacher(b.car.pose.copy(),b.car.v,b.car.omega)
            np.testing.assert_array_equal(ua,ub);assert a.decisions[0]['index']==r['decision']['index']
            if r['decision']['index']!=0:
                env=convert()
                for _ in range(14):env.step()
                branch=short_branch(root,r['decision']['action'])
                np.testing.assert_array_equal(env.physics,branch.physics[:len(env.physics)])
            checks.append(dict(profile=profile,index=r['index'],decision_index=r['decision']['index'],future_event_mutation_invariant=True,
                committed_prefix_exact=r['decision']['index']!=0))
    return checks


def freeze():
    checked=checks();parent=json.loads((ROBUST_REPORT/'protocol.json').read_text());OUT.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    modules=['goal_entry_forecast','goal_entry_robust','goal_entry_recovery_search','goal_phase_disturbance','goal_boundary_teacher_v2']
    dump(REPORT/'protocol.json',dict(source_sha256=sha(__file__),dependencies=parent['dependencies'],
        module_hashes={m:sha(ROOT/'cnn_finetune'/(m+'.py')) for m in modules},
        keys=TURNS,seeds=[6601,6602],presets=PRESETS,phases=PHASES,profiles=['baseline','hold035','robust'],
        trigger='observed translation residual magnitude>.25cm/s or yaw residual magnitude>.005rad/s; no event metadata',
        execution='nonzero action program committed14 commands; after completion observe again; zero proposal does not block reassessment',
        scope='prospective two-seed local offline teacher comparison; no images, no independent final examination',
        timing='offline planning walltime not added to modeled36ms; this expert is not a deployable hardware controller',
        none='mild stochastic vehicle dynamics with no explicit shove/power event; not fixed normal condition',
        checks=checked,no_automatic_adoption=True,final_exam_used=False,new_training=False))
    shutil.copy2(__file__,OUT/'source_snapshot.py');print('six causal-trigger checks passed',flush=True)


def run_one(item):
    key,seed,preset,phase=item;rows=[];reference=None
    for profile in ['baseline','hold035','robust']:
        env=PhaseTeacher(key,seed,preset,'reserve',phase,registry()) if profile=='baseline' else ReactivePhase(key,seed,preset,phase,profile)
        env.finish();trace=np.asarray(env.physics);margin=continuous_clearance(trace[:,1:4],env.p,env.track)
        if phase=='none':
            if reference is None:reference=trace
            else:np.testing.assert_array_equal(trace,reference);assert len(env.decisions)==0
        dst=OUT/profile/('_'.join(key)+f'_{seed}_{preset}_{phase}');dst.mkdir(parents=True,exist_ok=False)
        r=dict(env.summary(),profile=profile,artifact=str(dst),continuous_min_clearance_cm=float(margin.min()))
        r['continuous_clean']=bool(r['stable_success'] and r['initial_valid'] and margin.min()>=-1e-9)
        r['clean_moving']=bool(r['continuous_clean'] and r['translation_pause_s']==0 and trace[:,4].min()>=-.01)
        np.savez_compressed(dst/'physics.npz',physics=trace);dump(dst/'result.json',r)
        if profile!='baseline':dump(dst/'decisions.json',env.decisions)
        rows.append(r)
    return rows


def run():
    p=json.loads((REPORT/'protocol.json').read_text());assert sha(__file__)==p['source_sha256']
    for f,h in p['dependencies'].items():assert sha(f)==h,f
    for m,h in p['module_hashes'].items():assert sha(ROOT/'cnn_finetune'/(m+'.py'))==h,m
    cases=list(itertools.product(TURNS,p['seeds'],PRESETS,PHASES));rows=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=3) as pool:
        for i,batch in enumerate(pool.map(run_one,cases),1):
            rows.extend(batch)
            if i%24==0:
                dump(REPORT/'progress.json',rows);print('reactive teacher',i,'/',len(cases),'rollouts',len(rows),'elapsed',round(time.perf_counter()-started,1),flush=True)
    summary={}
    for profile in ['baseline','hold035','robust']:
        summary[profile]={}
        for phase in PHASES:
            a=[r for r in rows if r['profile']==profile and r['phase']==phase]
            summary[profile][phase]=dict(n=len(a),clean=sum(r['continuous_clean'] for r in a),clean_moving=sum(r['clean_moving'] for r in a),
                exposed=sum(r['disturbance_exposed'] for r in a),pauses=sum(r['translation_pause_s']>0 for r in a),
                outcomes={v:sum(r['outcome']==v for r in a) for v in sorted(set(r['outcome'] for r in a))})
    dump(REPORT/'results.json',dict(summary=summary,rows=rows,none_physics_identical=True,final_exam_used=False));print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['freeze','run']);a=p.parse_args();globals()[a.task]()
