"""Distinguish clearance-path gains from changed clock-event exposure.

Truth pose is used by the environment to place a disturbance, never by a
student. Both arms remain offline privileged teachers. No final exam.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time
import numpy as np
from bench.world import wrap,Track
from bench.junction import heading_of
from bench.vehicle import PRESETS as VEHICLES
from .goal_clearance_paths import ReserveTeacher,REPORT as SEARCH_REPORT
from .goal_clearance_validation import REPORT as VALIDATION_REPORT
from .goal_teacher_coverage import CoverageTeacher,KEYS,PRESETS,ROOT
from .goal_continuous_clearance import continuous_clearance
from .junction_feasibility import events_for
from .goal_development import sha
from .junction_stage02 import dump

OUT=Path('D:/autodrive_cnn/runs/goal_phase_disturbance_20261113')
REPORT=ROOT/'cnn_finetune/results/goal_phase_disturbance_20261113'
TURNS=[k for k in KEYS if k[2]!='straight']
PHASES=['none','entry','middle','exit']


class PhaseTeacher(ReserveTeacher):
    def __init__(self,key,seed,preset,arm,phase,registry):
        if arm=='baseline':CoverageTeacher.__init__(self,key,'mild',seed,preset,'centered')
        else:ReserveTeacher.__init__(self,key,'mild',seed,preset,'same_mode',registry)
        self.arm_name=arm;self.phase=phase;self.events=[];self.trigger=None
        self.event_template=events_for(seed,'mild')[:2]

    def ready(self):
        rel=self.car.pose[:2]-self.c
        if self.phase=='entry':return float(rel@self.di)>=-15
        if self.phase=='middle':
            total=wrap(heading_of(self.do)-heading_of(self.di))
            turned=wrap(self.car.pose[2]-heading_of(self.di))*np.sign(total)
            return turned>=np.pi/4
        if self.phase=='exit':return float(rel@self.do)>=15
        return False

    def advance_to(self,target):
        while self.t<target-1e-10 and self.outcome=='time_limit':
            if self.trigger is None and self.ready():
                self.trigger=dict(time=self.t,pose=self.car.pose.copy())
                self.events=[dict(e,t0=self.t) for e in self.event_template]
            super().advance_to(min(target,self.t+.02))

    def summary(self):
        r=CoverageTeacher.summary(self)
        return dict(r,arm=self.arm_name,phase=self.phase,trigger=self.trigger,
                    disturbance_exposed=self.exposure>0,events=self.events)


def registry():return {tuple(r['key']):r for r in json.loads((SEARCH_REPORT/'geometry.json').read_text())}


def checks():
    reg=registry();n=0;max_physics_error=0.;max_command_error=0.
    for arm in ['baseline','reserve']:
        for preset in PRESETS:
            for phase in PHASES:
                key=('east','W','left')
                a=PhaseTeacher(key,6501,preset,arm,phase,reg).finish()
                b=CoverageTeacher(key,'mild',6501,preset,'centered') if arm=='baseline' else ReserveTeacher(key,'mild',6501,preset,'same_mode',reg)
                b.events=copy.deepcopy(a.events);b.finish()
                # Substep target subtraction introduces roundoff at ~1e-12.
                np.testing.assert_allclose(a.physics,b.physics,rtol=0,atol=1e-8)
                aa=np.array([r['applied'] for r in a.records]);bb=np.array([r['applied'] for r in b.records])
                np.testing.assert_allclose(aa,bb,rtol=0,atol=1e-10)
                max_physics_error=max(max_physics_error,float(np.max(abs(np.asarray(a.physics)-b.physics))))
                max_command_error=max(max_command_error,float(np.max(abs(aa-bb))))
                assert (a.trigger is None)==(phase=='none')
                n+=1
    return dict(physics_and_command_replay_checks=n,phase_triggering_changes_only_event_time=True,
                physics_atol=1e-8,command_atol=1e-10,rtol=0,max_physics_error=max_physics_error,max_command_error=max_command_error)


def freeze():
    checked=checks();p=json.loads((VALIDATION_REPORT/'protocol.json').read_text())
    for f,h in p['hashes'].items():assert sha(f)==h,f
    OUT.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    dump(REPORT/'protocol.json',dict(source_sha256=sha(__file__),dependencies=p['hashes'],
        keys=TURNS,seeds=list(range(6501,6505)),presets=PRESETS,phases=PHASES,
        arms=['baseline','reserve'],candidate='same_mode frozen globally before these experiments',
        environment='mild stochastic vehicle parameters throughout; one compound shove + power event or none',
        event_distribution='first amplitude/duration pair from unchanged events_for(seed,mild); only start time changed',
        triggers=dict(entry='inbound axle progress >= -15cm',middle='signed turn angle >=45deg',exit='outgoing axle progress >=15cm'),
        trigger_sampling='each unchanged physics step, max20ms; current pose only, no future state',
        missing_exposure='record and retain; never discard failure or untriggered case',
        scope='diagnostic teacher comparison, not full mission or independent final exam; no image/lighting claim',
        changed_acceptance=False,changed_actor_inputs=False,final_exam_used=False,checks=checked))
    shutil.copy2(__file__,OUT/'source_snapshot.py');print(json.dumps(checked),flush=True)


def run(arm):
    p=json.loads((REPORT/'protocol.json').read_text());assert sha(__file__)==p['source_sha256']
    for f,h in p['dependencies'].items():assert sha(f)==h,f
    reg=registry();track=Track();rows=[];started=time.perf_counter()
    for key in TURNS:
        for seed in p['seeds']:
            for preset in PRESETS:
                for phase in PHASES:
                    dst=OUT/'rollouts'/arm/('_'.join(key)+f'_{seed}_{preset}_{phase}');dst.mkdir(parents=True,exist_ok=False)
                    env=PhaseTeacher(key,seed,preset,arm,phase,reg).finish();trace=np.asarray(env.physics)
                    margin=continuous_clearance(trace[:,1:4],VEHICLES[preset],track)
                    r=dict(env.summary(),artifact=str(dst),continuous_min_clearance_cm=float(margin.min()))
                    r['continuous_clean']=bool(r['clean_stable'] and margin.min()>=-1e-9)
                    dump(dst/'result.json',r);np.savez_compressed(dst/'physics.npz',samples=trace);rows.append(r)
        dump(REPORT/(arm+'_progress.json'),rows)
        print('phase',arm,key,len(rows),'clean',sum(r['clean_stable'] for r in rows),'elapsed',round(time.perf_counter()-started,1),flush=True)
    dump(REPORT/(arm+'.json'),rows)


def summarize():
    rows={a:json.loads((REPORT/(a+'.json')).read_text()) for a in ['baseline','reserve']}
    assert all(len(v)==960 for v in rows.values())
    def ident(r):return tuple(r['key']),r['seed'],r['preset'],r['phase']
    maps={a:{ident(r):r for r in rr} for a,rr in rows.items()}
    assert maps['baseline'].keys()==maps['reserve'].keys()
    for k,a in maps['baseline'].items():
        b=maps['reserve'][k];assert a['initial_sampling']==b['initial_sampling']
        # If both trigger, event magnitudes and durations must be identical.
        if a['trigger'] and b['trigger']:
            assert [{k:v for k,v in e.items() if k!='t0'} for e in a['events']]==[{k:v for k,v in e.items() if k!='t0'} for e in b['events']]
    def cell(rr):
        return dict(n=len(rr),clean=sum(r['clean_stable'] for r in rr),stable=sum(r['stable_success'] for r in rr),
            continuous_clean=sum(r['continuous_clean'] for r in rr),exposed=sum(r['disturbance_exposed'] for r in rr),
            pauses=sum(r['translation_pause_s']>0 for r in rr),
            median_time_s=float(np.median([r['simulated_s'] for r in rr])),
            outcomes={v:sum(r['outcome']==v for r in rr) for v in sorted(set(r['outcome'] for r in rr))})
    summary={a:{phase:cell([r for r in rr if r['phase']==phase]) for phase in PHASES} for a,rr in rows.items()}
    paired={}
    for phase in PHASES:
        kk=[k for k in maps['baseline'] if k[-1]==phase]
        paired[phase]=dict(gains=sum(maps['reserve'][k]['clean_stable'] and not maps['baseline'][k]['clean_stable'] for k in kk),
                          losses=sum(maps['baseline'][k]['clean_stable'] and not maps['reserve'][k]['clean_stable'] for k in kk))
    movements=[dict(key=key,**{a:{phase:cell([r for r in rr if tuple(r['key'])==key and r['phase']==phase]) for phase in PHASES} for a,rr in rows.items()}) for key in TURNS]
    dump(REPORT/'summary.json',dict(summary=summary,paired=paired,movements=movements,
        limitations='Four shared seeds, three vehicle presets; each phase tested separately with one compound event. Not certification or full-mission student evidence.'))
    print(json.dumps(dict(summary=summary,paired=paired)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['freeze','run','summarize']);p.add_argument('--arm',choices=['baseline','reserve']);a=p.parse_args()
    run(a.arm) if a.task=='run' else globals()[a.task]()
