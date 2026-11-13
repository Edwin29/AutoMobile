"""All28 visual demonstrations from a prospectively checked recovery teacher.

Every failed episode is retained. Only complete continuous-footprint-clean
demonstrations are eligible for the initial supervised baseline. No DAgger
takeover labels or outcome-loss claims are made by this collection.
"""
import argparse
import itertools
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from .goal_visual_recovery import VisualRecoveryScene,DIRECTION,REPORT as BRIDGE_REPORT
from .goal_reactive_clock_validation import REPORT as VALIDATION_REPORT
from .goal_teacher_coverage import ROOT,KEYS,PRESETS
from .junction_learning import Encoder,CHECKPOINT,setup
from .goal_continuous_clearance import continuous_clearance
from .goal_development import sha
from .junction_stage02 import dump

OUT=Path('D:/autodrive_cnn/runs/goal_all_movement_data_20261113')
REPORT=ROOT/'cnn_finetune/results/goal_all_movement_data_20261113'
SEEDS=[6801,6802]
CONDITIONS=['normal','mild','mild_light']


def freeze():
    validation=json.loads((VALIDATION_REPORT/'results.json').read_text());s=validation['summary']
    assert validation['normal_physics_identical']
    assert s['hold035']['normal']['clean']>=s['baseline']['normal']['clean']
    assert s['hold035']['mild']['clean']>s['baseline']['mild']['clean']
    for condition in ['normal','mild']:
        for outcome in ['wrong_exit','severe_departure']:
            assert s['hold035'][condition]['outcomes'].get(outcome,0)<=s['baseline'][condition]['outcomes'].get(outcome,0)
    bridge=json.loads((BRIDGE_REPORT/'bridge_checks.json').read_text())
    assert bridge['source_sha256']==sha(ROOT/'cnn_finetune/goal_visual_recovery.py')
    assert bridge['encoder_sha256']==sha(CHECKPOINT)
    assert all(r['physics_exact'] and r['commands_exact'] for r in bridge['rows'])
    files={Path(__file__).resolve(),CHECKPOINT,VALIDATION_REPORT/'results.json',BRIDGE_REPORT/'bridge_checks.json',ROOT/'bench/world/camera_track.npz'}
    backgrounds=sorted((ROOT/'bench/world/realcam').glob('bg_*.jpg'))
    assert len(backgrounds)>=3
    files.update(backgrounds)
    inherited=json.loads((VALIDATION_REPORT/'protocol.json').read_text())['hashes']
    for f,h in inherited.items():assert sha(f)==h,f
    files.update(Path(f) for f in inherited)
    for name,module in list(sys.modules.items()):
        f=getattr(module,'__file__',None)
        if f and f.endswith('.py') and (name.startswith('cnn_finetune.') or name.startswith('bench.')):files.add(Path(f).resolve())
    OUT.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    dump(REPORT/'protocol.json',dict(source_sha256=sha(__file__),hashes={str(f):sha(f) for f in sorted(files)},
        keys=KEYS,seeds=SEEDS,presets=PRESETS,conditions=CONDITIONS,expected_episodes=504,
        background_files=[str(f) for f in backgrounds],
        teacher='frozen hold035 repeated-event adapter with clearance references; no automatic driving-model replacement',
        eligibility='stable55cm exit, legal initial pose, continuous full-body clearance>=0 for entire episode; retain every failure with eligible=false',
        labels='actual expert commands at capture; only demonstrations, no unexecuted committed expert program queried on student trajectories',
        input_contract='image history + previous sentPWM + explicit straight[0,0]/left[1,0]/right[0,1] instruction',
        images='existing lower-half YUV blur3 66x200 uint8; frozen A5b encoder features cached, image frames retained for later CNN experiments',
        conditions_note='mild and mild_light share physical seeds/events, differ in camera profile; paired physical identity checked after collection',
        split='6801/6802 training only. Subsequent visual development seeds6901/6902; final900000-999999 untouched',
        quality_gate='Improvement over reference teacher is a data-source research gate, not the full goal acceptance criteria',
        intended_next='three independent recurrent-head initializations on identical data/order, then closed-loop assessment; outcome-loss ablation only with a separately validated label protocol',
        final_exam_used=False,new_training=False))
    shutil.copy2(__file__,OUT/'source_snapshot.py');print('visual collection frozen504episodes',flush=True)


def collect(seed):
    setup();p=json.loads((REPORT/'protocol.json').read_text());assert seed in SEEDS
    for f,h in p['hashes'].items():assert sha(f)==h,f
    assert [str(f) for f in sorted((ROOT/'bench/world/realcam').glob('bg_*.jpg'))]==p['background_files']
    encoder=Encoder().cuda().eval();rows=[];started=time.perf_counter()
    for key,preset,condition in itertools.product(KEYS,PRESETS,CONDITIONS):
        name='_'.join(key)+f'_{seed}_{preset}_{condition}';base=OUT/'data'/name
        candidates=[base]+sorted(base.parent.glob(name+'_attempt_*')) if base.parent.exists() else [base]
        completed=[d for d in candidates if (d/'complete.json').exists()]
        if completed:
            r=json.loads((completed[-1]/'complete.json').read_text());assert sha(Path(r['artifact'])/'sequence.npz')==r['sequence_sha256']
            assert r['source_sha256']==p['source_sha256'] and r['encoder_sha256']==sha(CHECKPOINT)
            assert sha(Path(r['artifact'])/'physics.npz')==r['physics_sha256']
        else:
            dst=base;attempt=1
            while dst.exists():dst=base.with_name(name+f'_attempt_{attempt}');attempt+=1
            env=VisualRecoveryScene(key,condition,seed,preset,encoder).finish()
            trace=np.asarray(env.physics);m=continuous_clearance(trace[:,1:4],env.p,env.track)
            r=env.save(dst,save_images=True)
            with np.load(dst/'sequence.npz') as z:
                assert z['f'].shape==(len(env.records),1152) and z['aux'].shape==(len(env.records),4)
                np.testing.assert_array_equal(z['label'],z['action'])
                np.testing.assert_allclose(z['aux'][1:,:2],z['applied'][:-1],atol=3e-8,rtol=0)
                np.testing.assert_array_equal(z['aux'][:,-2:],np.tile(DIRECTION[key[2]],(len(env.records),1)))
                assert z['label_valid'].all() and z['supervise'].all()
            r.update(continuous_min_clearance_cm=float(m.min()),eligible=bool(r['stable_success'] and r['initial_valid'] and m.min()>=-1e-9),
                sequence_sha256=sha(dst/'sequence.npz'),physics_sha256=sha(dst/'physics.npz'),
                source_sha256=p['source_sha256'],encoder_sha256=sha(CHECKPOINT))
            dump(dst/'complete.json',r)
        rows.append(r)
        if len(rows)%9==0:
            dump(REPORT/f'seed_{seed}_progress.json',rows)
            print('visual data',seed,len(rows),'/252','eligible',sum(r['eligible'] for r in rows),'elapsed',round(time.perf_counter()-started,1),flush=True)
    dump(REPORT/f'seed_{seed}.json',rows);print('visual seed complete',seed,len(rows),flush=True)


def summarize():
    rows=[]
    for seed in SEEDS:
        r=json.loads((REPORT/f'seed_{seed}.json').read_text());assert len(r)==252;rows+=r
    assert len({(tuple(r['key']),r['seed'],r['preset'],r['condition']) for r in rows})==504
    by_key={(tuple(r['key']),r['seed'],r['preset'],r['condition']):r for r in rows}
    paired=0
    for key,seed,preset in itertools.product(KEYS,SEEDS,PRESETS):
        a=by_key[key,seed,preset,'mild'];b=by_key[key,seed,preset,'mild_light']
        np.testing.assert_array_equal(np.load(Path(a['artifact'])/'physics.npz')['samples'],np.load(Path(b['artifact'])/'physics.npz')['samples']);paired+=1
    coverage=[]
    for key in KEYS:
        coverage.append(dict(key=key,conditions={c:dict(n=sum(tuple(r['key'])==key and r['condition']==c for r in rows),
            eligible=sum(tuple(r['key'])==key and r['condition']==c and r['eligible'] for r in rows)) for c in CONDITIONS}))
    dump(REPORT/'data.json',dict(rows=rows,coverage=coverage,n=len(rows),eligible=sum(r['eligible'] for r in rows),
        physical_lighting_pairs_exact=paired,final_exam_used=False))
    print('visual data summarized',len(rows),'eligible',sum(r['eligible'] for r in rows),'paired',paired,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['freeze','collect','summarize']);p.add_argument('--seed',type=int);a=p.parse_args()
    collect(a.seed) if a.task=='collect' else globals()[a.task]()
