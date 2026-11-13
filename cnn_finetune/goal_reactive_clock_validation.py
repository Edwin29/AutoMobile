"""Frozen hold035 teacher: all28 movements with repeated clock disturbances.

Prospective development validation. No images, student training or final exam.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from .goal_entry_reactive import ReactivePhase
from .goal_phase_disturbance import PhaseTeacher,registry
from .goal_clearance_paths import ReserveTeacher,REPORT as PATH_REPORT
from .goal_teacher_coverage import KEYS,PRESETS,ROOT,STAGE_OUT,STAGE_REPORT
from .goal_development import LocalScene,sha
from .junction_learning import REGISTRY,TEACHER_REPORT
from .junction_feasibility import events_for
from .goal_continuous_clearance import continuous_clearance
from .junction_stage02 import dump

OUT=Path('D:/autodrive_cnn/runs/goal_reactive_clock_validation_20261113')
REPORT=ROOT/'cnn_finetune/results/goal_reactive_clock_validation_20261113'


class ReactiveClockTeacher(ReactivePhase):
    advance_to=LocalScene.advance_to

    def __init__(self,key,condition,seed,preset,reg=None):
        ReserveTeacher.__init__(self,key,condition,seed,preset,'same_mode',registry() if reg is None else reg)
        self.initialize('hold035')
        assert not hasattr(self,'phase') and not hasattr(self,'trigger')

    def summary(self):
        return dict(ReserveTeacher.summary(self),reactive_profile=self.profile_name,decisions=len(self.decisions),
            nonzero_decisions=sum(r['index']!=0 for r in self.decisions),modified_frames=self.modified_frames)


def checks():
    # Check the new constructor against the frozen phase implementation while
    # temporarily giving the adapter the same phase environment for this test.
    rows=[]
    for key in [('center','N','right'),('east','W','left'),('north','E','straight')]:
        for preset in PRESETS:
            a=ReactivePhase(key,6501,preset,'entry','hold035').finish()
            b=ReactiveClockTeacher(key,'mild',6501,preset)
            b.phase='entry';b.trigger=None;b.events=[];b.event_template=events_for(6501,'mild')[:2]
            b.advance_to=PhaseTeacher.advance_to.__get__(b);b.finish()
            np.testing.assert_array_equal(a.physics,b.physics)
            np.testing.assert_array_equal([r['applied'] for r in a.records],[r['applied'] for r in b.records])
            rows.append(dict(key=key,preset=preset,phase_adapter_physics_exact=True,commands_exact=True))
    return rows


def freeze():
    checked=checks();OUT.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    files={Path(__file__).resolve(),ROOT/'bench/world/track.json',TEACHER_REPORT/'teacher_registry.json',STAGE_REPORT/'selection.json',PATH_REPORT/'geometry.json'}
    for name,module in list(sys.modules.items()):
        f=getattr(module,'__file__',None)
        if f and f.endswith('.py') and (name.startswith('cnn_finetune.') or name.startswith('bench.')):files.add(Path(f).resolve())
    for r in json.loads((PATH_REPORT/'geometry.json').read_text()):files.add(Path(r['reference']))
    for r in REGISTRY.values():files.add(Path(r['reference']))
    for key in KEYS:
        if key[2]=='straight':files.add(STAGE_OUT/'paths'/'_'.join(key)/'path.npz')
    dump(REPORT/'protocol.json',dict(source_sha256=sha(__file__),hashes={str(f):sha(f) for f in sorted(files)},
        keys=KEYS,seeds=list(range(6701,6709)),presets=PRESETS,conditions=['normal','mild'],arms=['baseline','hold035'],
        candidate='exact frozen hold035 teacher method, same clearance references and speed modes; no phase/onset fields in clock adapter',
        selection_evidence='goal_entry_reactive_20261113: prospective phase validation selected hold035 before6701-6708 outcomes',
        disturbances='unchanged events_for: two compound shove/power events at7-8s and15-16s; mild continuous vehicle variations',
        primary='stable55cm exit with continuous full-body clearance>=0; retain sparse legacy metrics separately',
        checks=checked,expected_rollouts=2688,
        limits='local privileged teacher, eight shared seeds; no vision/strong-lighting/student/full-mission evidence',
        no_automatic_adoption=True,final_exam_used=False,new_training=False))
    shutil.copy2(__file__,OUT/'source_snapshot.py');print('frozen; nine adapter parity cases passed',flush=True)


def run_one(item):
    key,seed,preset,condition=item;rows=[];initial=None;baseline_trace=None;reg=registry()
    for arm in ['baseline','hold035']:
        env=ReserveTeacher(key,condition,seed,preset,'same_mode',reg) if arm=='baseline' else ReactiveClockTeacher(key,condition,seed,preset,reg)
        if initial is None:initial=env.initial_pose.copy()
        else:np.testing.assert_array_equal(initial,env.initial_pose)
        env.finish();trace=np.asarray(env.physics)
        if condition=='normal':
            if baseline_trace is None:baseline_trace=trace
            else:np.testing.assert_array_equal(trace,baseline_trace);assert len(env.decisions)==0
        margin=continuous_clearance(trace[:,1:4],env.p,env.track)
        dst=OUT/arm/('_'.join(key)+f'_{seed}_{preset}_{condition}');dst.mkdir(parents=True,exist_ok=False)
        r=dict(env.summary(),arm=arm,artifact=str(dst),continuous_min_clearance_cm=float(margin.min()))
        r['continuous_clean']=bool(r['stable_success'] and r['initial_valid'] and margin.min()>=-1e-9)
        r['clean_forward']=bool(r['continuous_clean'] and r['translation_pause_s']==0 and trace[:,4].min()>=-.01)
        np.savez_compressed(dst/'physics.npz',physics=trace);dump(dst/'result.json',r);dump(dst/'events.json',env.events)
        if arm=='hold035':dump(dst/'decisions.json',env.decisions)
        rows.append(r)
    return rows


def run():
    p=json.loads((REPORT/'protocol.json').read_text());assert sha(__file__)==p['source_sha256']
    for f,h in p['hashes'].items():assert sha(f)==h,f
    cases=list(itertools.product(KEYS,p['seeds'],PRESETS,p['conditions']));rows=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=3) as pool:
        for i,batch in enumerate(pool.map(run_one,cases),1):
            rows.extend(batch)
            if i%48==0:
                dump(REPORT/'progress.json',rows);print('clock validation',i,'/',len(cases),'rollouts',len(rows),'elapsed',round(time.perf_counter()-started,1),flush=True)
    summary={}
    for arm in p['arms']:
        summary[arm]={}
        for condition in p['conditions']:
            a=[r for r in rows if r['arm']==arm and r['condition']==condition]
            summary[arm][condition]=dict(n=len(a),clean=sum(r['continuous_clean'] for r in a),stable=sum(r['stable_success'] for r in a),
                pauses=sum(r['translation_pause_s']>0 for r in a),projected_starts=sum(r['initial_sampling']['projected'] for r in a),
                median_time_s=float(np.median([r['simulated_s'] for r in a])),
                outcomes={v:sum(r['outcome']==v for r in a) for v in sorted(set(r['outcome'] for r in a))})
    dump(REPORT/'results.json',dict(summary=summary,rows=rows,normal_physics_identical=True,final_exam_used=False));print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['freeze','run']);a=p.parse_args();globals()[a.task]()
