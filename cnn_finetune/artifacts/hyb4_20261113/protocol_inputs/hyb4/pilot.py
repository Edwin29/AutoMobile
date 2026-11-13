"""Prerequisite gate: exact local paths and the new signed controller.

Truth is intentionally used by the named oracle packet producer. It is not a
CNN result. No new learning or controller sweep on these development seeds.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from .control import LocalPath,PathTracker
from cnn_finetune.goal_clearance_paths import ReserveTeacher,REPORT as REFERENCES
from cnn_finetune.goal_continuous_clearance import continuous_clearance
from cnn_finetune.junction_learning import setup
from cnn_finetune.junction_stage02 import dump

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'cnn_finetune/results/hyb4_prerequisite_20261113'
OUT=Path('D:/autodrive_cnn/runs/hyb4_prerequisite_20261113')
KEYS=[('north','E','left'),('north','W','right'),('center','N','right'),('east','W','right')]
SEEDS=[25001,25002]
PRESETS=['field_estimate','sluggish','unequal_sides']


def sha(file):return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def oracle_packet(points,s,speeds,pose,capture):
    """Offline labels only: fixed 0.5cm spacing, -2..21.5cm along reference.

    Endpoint tangent extension avoids repeated endpoint labels. Frame x=right,
    y=forward, clockwise positive yaw; no vehicle response parameter used.
    """
    near=int(np.argmin(np.linalg.norm(points-pose[:2],axis=1)))
    grid=s[near]+np.arange(48)*.5-2.
    world=np.c_[np.interp(grid,s,points[:,0]),np.interp(grid,s,points[:,1])]
    for mask,i,j in [(grid<s[0],0,1),(grid>s[-1],-1,-2)]:
        tangent=(points[i]-points[j])/(s[i]-s[j])
        world[mask]=points[i]+(grid[mask]-s[i])[:,None]*tangent
    delta=world-pose[:2];h=pose[2]
    body=np.c_[delta@np.array([np.cos(h),-np.sin(h)]),delta@np.array([np.sin(h),np.cos(h)])]
    return LocalPath(body,np.interp(grid,s,speeds),capture)


def freeze():
    REPORT.mkdir(exist_ok=False);OUT.mkdir(parents=True,exist_ok=False)
    files={Path(__file__),ROOT/'hyb4/control.py',ROOT/'hyb4/test_control.py',REFERENCES/'geometry.json'}
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and f.endswith('.py') and Path(f).is_file() and Path(f).resolve().is_relative_to(ROOT):files.add(Path(f).resolve())
    for row in json.loads((REFERENCES/'geometry.json').read_text()):
        if tuple(row['key']) in KEYS:files.add(Path(row['reference']))
    dump(REPORT/'protocol.json',dict(keys=KEYS,seeds=SEEDS,presets=PRESETS,condition='normal',n=24,
        hashes={str(f):sha(f) for f in sorted(files)},scope='Oracle capture-frame local paths; prerequisite only, no trained CNN or image-based control claim',
        gate='>=22/24 continuous-clean,24/24 stable,0wrong/severe,0 unnecessary pauses; otherwise hold training and image expansion',
        model='fixed nominal gain16.8 track13 efficiency.6 delay.05 tau.10;275ms preview; no true response inputs',
        packet='48 points at 0.5cm spacing,nearest reference arc-2..+21.5cm; true capture pose transforms labels;36ms latency',
        allocation='signed floor.36,integrated-duty pulse density,dt.036s; no legacy pivot/split',
        final_exam_used=False,new_training=False))
    print('protocol frozen',flush=True)


def run_one(key,seed,preset,registry):
    env=ReserveTeacher(key,'normal',seed,preset,'same_mode',registry);controller=PathTracker();records=[]
    while env.capture<40 and env.outcome=='time_limit':
        env.advance_to(env.capture)
        if env.outcome!='time_limit':break
        capture=env.capture;done=capture+.036
        packet=oracle_packet(env.points,env.s,env.speed,env.car.pose.copy(),capture)
        env.advance_to(done)
        if env.outcome!='time_limit':break
        desired,info=controller.desired(packet,done)
        applied=controller.drive.allocate(desired,.036)
        env.car.command(done,applied);controller.acknowledge(done,applied)
        records.append(dict(capture=capture,done=done,desired=desired,applied=applied,**info))
        env.records.append(dict(capture=capture,done=done,action=desired,applied=applied))
        env.previous=applied.copy();env.capture=done
    physics=np.asarray(env.physics)
    clearance=continuous_clearance(np.vstack([env.initial_pose,physics[:,1:4]]),env.p,env.track)
    result=env.summary();result.update(continuous_min_clearance_cm=float(clearance.min()),
        continuous_clean=bool(result['stable_success'] and result['initial_valid'] and clearance.min()>=-1e-9),
        signed_inner_frames=sum(min(r['applied'])<0<max(r['applied']) for r in records),
        oracle_path=True,controller_true_preset=False)
    dst=OUT/('_'.join(key)+f'_{seed}_{preset}');dst.mkdir(exist_ok=False)
    np.savez_compressed(dst/'physics.npz',samples=physics,initial=env.initial_pose,continuous_clearance=clearance)
    dump(dst/'records.json',records);dump(dst/'events.json',env.events)
    result['artifact']=str(dst);dump(dst/'complete.json',result)
    return result


def run():
    setup();protocol=json.loads((REPORT/'protocol.json').read_text())
    for file,value in protocol['hashes'].items():assert sha(file)==value,file
    registry={tuple(r['key']):r for r in json.loads((REFERENCES/'geometry.json').read_text())}
    rows=[];start=time.perf_counter()
    for seed in SEEDS:
        for key in KEYS:
            for preset in PRESETS:
                dst=OUT/('_'.join(key)+f'_{seed}_{preset}')/'complete.json'
                r=json.loads(dst.read_text()) if dst.exists() else run_one(key,seed,preset,registry)
                rows.append(r);dump(REPORT/'progress.json',rows)
                print(len(rows),'/24',key,preset,r['outcome'],'clean',r['continuous_clean'],
                      'clearance',round(r['continuous_min_clearance_cm'],3),'seconds',round(time.perf_counter()-start,1),flush=True)
    result=dict(n=len(rows),clean=sum(r['continuous_clean'] for r in rows),stable=sum(r['stable_success'] for r in rows),
        wrong=sum(r['outcome']=='wrong_exit' for r in rows),severe=sum(r['outcome']=='severe_departure' for r in rows),
        pauses=sum(r['translation_pause_s']>0 for r in rows),new_training=False,final_exam_used=False)
    result['gate_pass']=result['clean']>=22 and result['stable']==24 and result['wrong']==result['severe']==result['pauses']==0
    dump(REPORT/'runs.json',rows);dump(REPORT/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['freeze','run']);a=p.parse_args();globals()[a.task]()
