"""Balanced initial-state demonstrations with independently stressed lighting.

Normal physics is identical for typical/harsh-light camera pairs. These new
training seeds were selected from pose bins BEFORE teacher outcomes. Existing
compound-disturbance demonstrations remain separate and are not relabeled.
"""
import argparse
import itertools
import json
from pathlib import Path
import time
import numpy as np
import torch
from bench.world import Camera,Renderer
from bench.realcam import RealCam
from .goal_visual_recovery import VisualRecoveryScene,DIRECTION
from .goal_reactive_clock_validation import ReactiveClockTeacher
from .goal_initial_coverage_preflight import REPORT as PREFLIGHT_REPORT
from .goal_all_movement_data import REPORT as OLD_REPORT
from .goal_teacher_coverage import ROOT,KEYS,PRESETS
from .junction_learning import Encoder,CHECKPOINT,setup
from .goal_continuous_clearance import continuous_clearance
from .goal_development import sha
from .junction_stage02 import dump

OUT=Path('D:/autodrive_cnn/runs/goal_balanced_visual_data_20261113')
REPORT=ROOT/'cnn_finetune/results/goal_balanced_visual_data_20261113'
CONDITIONS=['normal','normal_light']


class BalancedScene(VisualRecoveryScene):
    def __init__(self,key,condition,seed,preset,encoder):
        if condition not in CONDITIONS:raise ValueError(condition)
        ReactiveClockTeacher.__init__(self,key,'normal',seed,preset)
        self.condition=condition;self.profile='typical' if condition=='normal' else 'harsh_light'
        camera=Camera();renderer=Renderer(self.track,camera)
        self.camera=RealCam(renderer,camera,self.profile,seed);self.encoder=encoder;self.actor_evaluation=False


def freeze():
    setup();preflight=json.loads((PREFLIGHT_REPORT/'results.json').read_text());s=preflight['summary']
    assert s['n']==672 and s['clean']==672 and s['projected']==0 and not s['missing_strata']
    old=json.loads((OLD_REPORT/'protocol.json').read_text())
    for f,h in old['hashes'].items():assert sha(f)==h,f
    encoder=Encoder().cuda().eval();checks=[]
    selected=json.loads((PREFLIGHT_REPORT/'protocol.json').read_text())['selected'];seed=next(iter(selected.values()))['seed']
    for key in [('center','N','right'),('north','E','left'),('west','N','straight')]:
        a=VisualRecoveryScene(key,'normal',seed,'field_estimate',encoder).finish()
        for condition in CONDITIONS:
            b=BalancedScene(key,condition,seed,'field_estimate',encoder).finish()
            np.testing.assert_array_equal(a.physics,b.physics)
            np.testing.assert_array_equal([r['applied'] for r in a.records],[r['applied'] for r in b.records])
            if condition=='normal':np.testing.assert_array_equal([r['f'] for r in a.records],[r['f'] for r in b.records])
            else:assert any(not np.array_equal(x['f'],y['f']) for x,y in zip(a.records,b.records))
            checks.append(dict(key=key,condition=condition,physics_exact=True,commands_exact=True,
                feature_relation='exact' if condition=='normal' else 'changed_by_light'))
    OUT.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    dump(REPORT/'protocol.json',dict(source_sha256=sha(__file__),parent_protocol_sha256=sha(OLD_REPORT/'protocol.json'),
        preflight_sha256=sha(PREFLIGHT_REPORT/'results.json'),selected=selected,keys=KEYS,presets=PRESETS,conditions=CONDITIONS,
        expected=1344,groups='selected seed insertion order alternating between worker0/1;672episodes per worker',
        checks=checks,eligibility='stable exit,initial valid,all physics samples continuous full-body clearance>=0,zero translation pause',
        scope='Training data only: normal and strong-light-only, complementing old mild/mild_light data; not replacing compound-disturbance evaluation',
        input_contract='unchanged image history,previous sentPWM,direction; original model architecture and reference files unchanged',
        no_final_exam=True,no_dev6901_or6902_in_training=True))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print('balanced visual data frozen1344; six bridge checks passed',flush=True)


def collect(group):
    setup();p=json.loads((REPORT/'protocol.json').read_text());assert group in [0,1] and sha(__file__)==p['source_sha256']
    assert sha(OLD_REPORT/'protocol.json')==p['parent_protocol_sha256'] and sha(PREFLIGHT_REPORT/'results.json')==p['preflight_sha256']
    old=json.loads((OLD_REPORT/'protocol.json').read_text())
    for f,h in old['hashes'].items():assert sha(f)==h,f
    assert [str(f) for f in sorted((ROOT/'bench/world/realcam').glob('bg_*.jpg'))]==old['background_files']
    teacher={(tuple(r['key']),r['preset'],r['seed']):r for r in json.loads((PREFLIGHT_REPORT/'results.json').read_text())['rows']}
    encoder=Encoder().cuda().eval();rows=[];started=time.perf_counter()
    choices=list(p['selected'].items())[group::2]
    for (stratum,spec),key,preset,condition in itertools.product(choices,KEYS,PRESETS,CONDITIONS):
        seed=spec['seed'];name='_'.join(key)+f'_{seed}_{preset}_{condition}';base=OUT/'data'/name
        candidates=[base]+sorted(base.parent.glob(name+'_attempt_*')) if base.parent.exists() else [base]
        completed=[d for d in candidates if (d/'complete.json').exists()]
        if completed:
            r=json.loads((completed[-1]/'complete.json').read_text());assert r['source_sha256']==p['source_sha256']
            assert sha(Path(r['artifact'])/'sequence.npz')==r['sequence_sha256']
            assert sha(Path(r['artifact'])/'physics.npz')==r['physics_sha256']
        else:
            dst=base;attempt=1
            while dst.exists():dst=base.with_name(name+f'_attempt_{attempt}');attempt+=1
            env=BalancedScene(key,condition,seed,preset,encoder).finish();trace=np.asarray(env.physics)
            expected=np.load(Path(teacher[key,preset,seed]['artifact'])/'physics.npz')['physics'];np.testing.assert_array_equal(trace,expected)
            margin=continuous_clearance(trace[:,1:4],env.p,env.track);r=env.save(dst,save_images=True)
            with np.load(dst/'sequence.npz') as z:
                np.testing.assert_array_equal(z['label'],z['action'])
                np.testing.assert_allclose(z['aux'][1:,:2],z['applied'][:-1],atol=3e-8,rtol=0)
                np.testing.assert_array_equal(z['aux'][:,-2:],np.tile(DIRECTION[key[2]],(len(env.records),1)))
                assert z['label_valid'].all() and z['supervise'].all()
            r.update(stratum=stratum,seed_draw=spec,continuous_min_clearance_cm=float(margin.min()),
                eligible=bool(r['stable_success'] and r['initial_valid'] and margin.min()>=-1e-9 and r['translation_pause_s']==0),
                preflight_physics_exact=True,source_sha256=p['source_sha256'],encoder_sha256=sha(CHECKPOINT),
                sequence_sha256=sha(dst/'sequence.npz'),physics_sha256=sha(dst/'physics.npz'))
            dump(dst/'complete.json',r)
        rows.append(r)
        if len(rows)%24==0:
            dump(REPORT/f'group_{group}_progress.json',rows)
            print('balanced visual',group,len(rows),'/672','eligible',sum(r['eligible'] for r in rows),'elapsed',round(time.perf_counter()-started,1),flush=True)
    dump(REPORT/f'group_{group}.json',rows);print('balanced visual group complete',group,len(rows),flush=True)


def summarize():
    rows=[]
    for group in [0,1]:
        a=json.loads((REPORT/f'group_{group}.json').read_text());assert len(a)==672;rows+=a
    assert len({(tuple(r['key']),r['seed'],r['preset'],r['condition']) for r in rows})==1344
    lookup={(tuple(r['key']),r['seed'],r['preset'],r['condition']):r for r in rows};paired=0
    for r in rows:
        if r['condition']!='normal':continue
        b=lookup[tuple(r['key']),r['seed'],r['preset'],'normal_light']
        np.testing.assert_array_equal(np.load(Path(r['artifact'])/'physics.npz')['samples'],np.load(Path(b['artifact'])/'physics.npz')['samples']);paired+=1
    summary=dict(n=len(rows),eligible=sum(r['eligible'] for r in rows),physical_lighting_pairs_exact=paired,
        no_dev6901_or6902_in_training=not any(r['seed'] in [6901,6902] for r in rows),final_exam_used=False)
    dump(REPORT/'data.json',dict(summary=summary,rows=rows));print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['freeze','collect','summarize']);p.add_argument('--group',type=int);a=p.parse_args()
    collect(a.group) if a.task=='collect' else globals()[a.task]()
