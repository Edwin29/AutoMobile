"""Matched full-episode dataset aggregation after diagnosing early entry drift.

Both arms share a corrected teacher, replay pool, initialization, optimizer and
update budget. Only the second data source differs: more teacher versus full
student-observation queries. The acting student never receives oracle state.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree

from bench.junction import movement
from bench.world import Track,wrap
from bench.vehicle import PRESETS as VEHICLES,Vehicle
from .goal_development import LocalScene,KEYS,PRESETS,CONDITIONS,ROOT,sha,OUT as PREVIOUS_OUT
from .goal_teacher_probe import FastTeacher,BASE
from .goal_exit_reference_probe import centered_reference
from .goal_teacher_validation import legal_start,SELECTED
from .junction_learning import REGISTRY,Encoder,model_at,setup
from .junction_stage02 import dump

OUT=Path('D:/autodrive_cnn/runs/goal_full_dagger_20261113')
REPORT=ROOT/'cnn_finetune/results/goal_full_dagger_20261113'
TRAIN_SEEDS=[6101,6102]
INITIAL=PREVIOUS_OUT/'convergence_probe/best.pt'


class Labeler:
    teacher=FastTeacher.teacher
    def __init__(self,key,preset):
        self.key=tuple(key);self.p=VEHICLES[preset];self.track=Track();self.rim=Vehicle(self.p,np.zeros(3)).footprint()
        self.c,self.di,self.do,self.arm=movement(*key)
        row=REGISTRY[self.key];z=np.load(row['reference'])
        self.points=z['points'];self.h=z['heading'];self.s=z['s'];self.k=z['curvature'];self.speed=z['nominal_speed_cm_s']
        self.teacher_mode=row['mode'];self.idx=0;self.planned_v=0.;self.controls={k:SELECTED[k] for k in BASE}
        self.points,self.h,self.s,self.k,margin=centered_reference(self,SELECTED['end_cm']);assert margin>=0


class QueryScene(LocalScene):
    teacher=FastTeacher.teacher
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.points,self.h,self.s,self.k,margin=centered_reference(self,SELECTED['end_cm']);assert margin>=0
        self.controls={k:SELECTED[k] for k in BASE}
        self.initial_sampling=legal_start(self)
    def summary(self):return dict(super().summary(),initial_sampling=self.initial_sampling)


def supervision_mask(key,preset,physics,captures):
    labeler=Labeler(key,preset)
    ids=np.searchsorted(physics[:,0],captures)
    ids=np.clip(ids,0,len(physics)-1)
    assert np.max(abs(physics[ids,0]-captures))<1e-8
    distance,near=cKDTree(labeler.points).query(physics[ids,1:3])
    angle=abs(np.degrees(wrap(physics[ids,3]-labeler.h[near])))
    return ((physics[ids,6]>=-3)&(distance<=12)&(angle<=60)).astype(np.float32)


def freeze():
    REPORT.mkdir(parents=True,exist_ok=False);OUT.mkdir(parents=True,exist_ok=False)
    protocol=dict(hypothesis='Entry drift starts at 3-6s, before old branch roots; full student-visited trajectories improve feedback recovery',
        train_seeds=TRAIN_SEEDS,vehicles=PRESETS,keys=KEYS,conditions=CONDITIONS,
        teacher=SELECTED,teacher_limit='Modified teacher is not certified: new-seed mild clean 81/96 and harsh not solved',
        initial_checkpoint=str(INITIAL),initial_sha256=sha(INITIAL),encoder_frozen=True,architecture_changed=False,
        sources=dict(common='clean teacher episodes from new collection plus 60 old clean episodes relabeled with same corrected teacher',
                     teacher_only='second batch drawn from same teacher pool',full_dagger='second batch from full acting-student episodes'),
        actor_during_collection='fixed initial checkpoint; no teacher intervention',
        query_mask='margin at capture >=-3cm, distance to reference <=12cm, heading difference <=60deg; additionally existing done-margin mask',
        supervision='These are oracle queries, not claims that every query leads to a clean recovery. Failure originals retained.',
        training=dict(updates=1200,lr=1e-4,min_lr=1e-5,seed=17,batch_episodes=8,
                      source_mix='0.5 common teacher + 0.5 second source, each four episodes one per movement',
                      objective='equal-episode turn-weighted SmoothL1 beta .05; no preference term',
                      selection='use fixed final update 1200, no checkpoint selection by observed driving'),
        evaluation='same frozen local development matrix 4101/4102 x three vehicles x four movements x three conditions; NOT final exam',
        source_sha256=sha(__file__),final_exam_used=False)
    dump(REPORT/'protocol.json',protocol)
    snap=OUT/'source_snapshot';snap.mkdir()
    for name in ['goal_full_dagger.py','goal_development.py','junction_learning.py','goal_teacher_probe.py','goal_exit_reference_probe.py','goal_teacher_validation.py']:
        shutil.copy2(ROOT/'cnn_finetune'/name,snap/name)


def collect(arm):
    setup();protocol=json.loads((REPORT/'protocol.json').read_text());assert sha(__file__)==protocol['source_sha256'];assert sha(INITIAL)==protocol['initial_sha256']
    encoder=Encoder().cuda().eval();model=None if arm=='teacher' else model_at(INITIAL);rows=[]
    for seed in TRAIN_SEEDS:
        for preset in PRESETS:
            for key in KEYS:
                for condition in CONDITIONS:
                    dst=OUT/'data'/arm/('_'.join(key)+f'_{condition}_{seed}_{preset}')
                    if (dst/'complete.json').exists():r=json.loads((dst/'complete.json').read_text())
                    else:
                        env=QueryScene(key,condition,seed,preset,encoder).finish(model)
                        r=env.save(dst,save_images=True)
                        with np.load(dst/'sequence.npz') as z:arrays={k:z[k] for k in z.files}
                        arrays['supervise']=supervision_mask(key,preset,np.asarray(env.physics),arrays['capture'])
                        np.savez_compressed(dst/'sequence.npz',**arrays)
                        r['supervised_frames']=int(arrays['supervise'].sum());r['source']=arm
                        dump(dst/'complete.json',r)
                    rows.append(r)
            dump(REPORT/(arm+'_progress.json'),rows)
            print('full episode collection',arm,seed,preset,len(rows),'clean',sum(r['clean_stable'] for r in rows),flush=True)
    dump(REPORT/(arm+'_data.json'),rows)


def relabel_replay():
    old=ROOT/'cnn_finetune/results/junction_learning_20261113/teacher_data.json'
    source=[r for r in json.loads(old.read_text()) if r['clean_stable']];rows=[]
    for r in source:
        origin=Path(r['artifact']);dst=OUT/'data/replay'/origin.name
        if (dst/'complete.json').exists():rows.append(json.loads((dst/'complete.json').read_text()));continue
        dst.mkdir(parents=True,exist_ok=False)
        with np.load(origin/'sequence.npz') as z:arrays={k:z[k] for k in z.files}
        physics=np.load(origin/'physics.npz')['samples'];ids=np.searchsorted(physics[:,0],arrays['capture']);ids=np.clip(ids,0,len(physics)-1)
        assert np.max(abs(physics[ids,0]-arrays['capture']))<1e-8
        labeler=Labeler(r['key'],r['preset']);labels=[]
        for i in ids:labels.append(labeler.teacher(physics[i,1:4],physics[i,4],physics[i,5]))
        arrays['label']=np.asarray(labels,np.float32)
        arrays['supervise']=supervision_mask(r['key'],r['preset'],physics,arrays['capture'])
        np.savez_compressed(dst/'sequence.npz',**arrays)
        row=dict(r,artifact=str(dst),original_artifact=str(origin),source='replay',labels_recomputed=True,
                 supervised_frames=int(arrays['supervise'].sum()))
        dump(dst/'complete.json',row);rows.append(row)
    dump(REPORT/'replay_data.json',rows);print('relabelled replay',len(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['freeze','collect','replay']);p.add_argument('--arm',choices=['teacher','student']);a=p.parse_args()
    if a.task=='freeze':freeze()
    elif a.task=='collect':collect(a.arm)
    else:relabel_replay()
