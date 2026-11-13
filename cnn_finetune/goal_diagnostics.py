"""Matched teacher-observation error diagnosis after the frozen local baseline.

This is teacher-forced replay, not a closed-loop success claim. Average within
episodes first. Camera-profile comparisons use identical teacher physics.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from bench.junction import movement
from .goal_development import OUT, REPORT, MODELS, ROOT, analyze
from .junction_learning import model_at, setup
from .junction_stage02 import dump


def open_loop():
    setup();rows=json.loads((REPORT/'teacher.json').read_text());result=[]
    models={a:model_at(p) for a,p in MODELS.items()}
    for r in rows:
        path=Path(r['artifact'])
        with np.load(path/'sequence.npz') as z:
            f=torch.from_numpy(z['f'][None]).cuda();aux=torch.from_numpy(z['aux'][None]).cuda()
            labels=z['label'];times=z['capture'];margin=z['margin']
        tr=np.load(path/'physics.npz')['samples']
        xy=np.c_[np.interp(times,tr[:,0],tr[:,1]),np.interp(times,tr[:,0],tr[:,2])]
        c,di,do,_=movement(*r['key']);rel=xy-c
        phases=np.where(rel@di < -15,'approach',np.where(rel@do > 15,'exit','corner'))
        for name,model in models.items():
            with torch.no_grad():pred,_=model(f,aux)
            y=pred[0].cpu().numpy();wheel=abs(y-labels).mean(-1)
            steer=abs((y[:,0]-y[:,1])-(labels[:,0]-labels[:,1]))
            throttle=abs(y.mean(-1)-labels.mean(-1))
            record=dict(arm=name,key=r['key'],condition=r['condition'],seed=r['seed'],preset=r['preset'],
                        teacher_clean=r['clean_stable'],wheel_mae=float(wheel.mean()),diff_mae=float(steer.mean()),
                        mean_mae=float(throttle.mean()),phases={})
            for phase in ['approach','corner','exit']:
                mask=(phases==phase)&(margin>=0)
                record['phases'][phase]=dict(frames=int(mask.sum()),wheel_mae=float(wheel[mask].mean()) if mask.any() else None,
                    diff_mae=float(steer[mask].mean()) if mask.any() else None)
            result.append(record)
    pairs=[]
    lookup={(tuple(r['key']),r['seed'],r['preset'],r['condition']):r for r in rows}
    for r in rows:
        if r['condition']!='mild':continue
        other=lookup[(tuple(r['key']),r['seed'],r['preset'],'mild_light')]
        a=np.load(Path(r['artifact'])/'physics.npz')['samples'];b=np.load(Path(other['artifact'])/'physics.npz')['samples']
        np.testing.assert_array_equal(a,b)
        for arm in MODELS:
            match=lambda c:next(x for x in result if x['arm']==arm and x['key']==r['key'] and x['seed']==r['seed'] and x['preset']==r['preset'] and x['condition']==c)
            low,high=match('mild'),match('mild_light')
            pairs.append(dict(arm=arm,key=r['key'],seed=r['seed'],preset=r['preset'],wheel_mae_change=high['wheel_mae']-low['wheel_mae'],
                              diff_mae_change=high['diff_mae']-low['diff_mae']))
    aggregate={}
    for arm in MODELS:
        aggregate[arm]={}
        for condition in ['normal','mild','mild_light']:
            rs=[r for r in result if r['arm']==arm and r['condition']==condition]
            aggregate[arm][condition]={k:float(np.mean([r[k] for r in rs])) for k in ['wheel_mae','diff_mae','mean_mae']}
            aggregate[arm][condition]['phase_wheel_mae']={ph:float(np.mean([r['phases'][ph]['wheel_mae'] for r in rs if r['phases'][ph]['wheel_mae'] is not None])) for ph in ['approach','corner','exit']}
    dump(REPORT/'teacher_replay.json',dict(aggregate=aggregate,episodes=result,paired_camera_changes=pairs,
         caveat='Teacher observations, teacher previous commands, recurrent full prefix. Not student closed-loop distribution or causal isolation of just light amplitude.',
         matched_camera_profile_physics=True))
    print('OPEN_LOOP',json.dumps(aggregate),flush=True)


def complete_baseline():
    # The teacher and BC workers were started separately. Keep at most two
    # image-rendering processes running; outcome starts when teacher finishes.
    while not (REPORT/'teacher.json').exists():time.sleep(5)
    if not (REPORT/'outcome.json').exists():
        subprocess.run([sys.executable,'-m','cnn_finetune.goal_development','evaluate','--arm','outcome'],cwd=ROOT,check=True)
    while not (REPORT/'bc.json').exists():time.sleep(5)
    analyze();open_loop()
    dump(REPORT/'diagnostics_complete.json',dict(completed=True,new_training=False,final_exam_used=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['complete','open_loop']);a=p.parse_args()
    complete_baseline() if a.task=='complete' else open_loop()
