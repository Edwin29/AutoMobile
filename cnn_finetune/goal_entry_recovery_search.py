"""Constructive counterfactuals for observed phase-entry failures.

Future realized disturbances score branches OFFLINE. This is not an online
policy, a certified optimizer, or a source of already-approved student labels.
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
from scipy.stats import qmc
from bench.world import Track
from bench.vehicle import PRESETS
from .goal_phase_disturbance import PhaseTeacher,registry,REPORT as PHASE_REPORT
from .goal_teacher_probe import FastTeacher
from .goal_continuous_clearance import continuous_clearance
from .goal_development import ROOT,sha
from .junction_stage02 import dump

OUT=Path('D:/autodrive_cnn/runs/goal_entry_recovery_search_20261113')
REPORT=ROOT/'cnn_finetune/results/goal_entry_recovery_search_20261113'


def fork(scene):
    branch=copy.copy(scene)
    for field in ['car','alloc','previous','events','trigger','event_template']:
        setattr(branch,field,copy.deepcopy(getattr(scene,field)))
    branch.physics=list(scene.physics);branch.records=list(scene.records);branch.history=list(scene.history)
    branch.teacher=FastTeacher.teacher.__get__(branch)
    return branch


def actions():
    grid=[[1.,0.,1.,0.]]
    grid += [[s,d,s,d] for s,d in itertools.product([0.,.5,1.,1.5],[-.6,-.4,-.2,0.,.2,.4,.6]) if (s,d)!=(1.,0.)]
    x=qmc.Sobol(4,scramble=True,seed=73921).random_base2(6)
    x=x*np.array([2.,1.2,2.,1.2])+np.array([0.,-.6,0.,-.6])
    return np.r_[np.asarray(grid),x]


def freeze():
    audit=json.loads((PHASE_REPORT/'entry_response_audit.json').read_text())
    rows=json.loads((PHASE_REPORT/'reserve.json').read_text())
    selected=[r for r in rows if r['phase']=='entry' and not r['continuous_clean']]
    before={(tuple(r['key']),r['seed'],r['preset']) for r in audit['rows'] if r['arm']=='reserve' and r['incursion_before_possible_response']}
    cases=[]
    for r in selected:
        cases.append(dict(source=r,root_mode='post_observation'))
        if (tuple(r['key']),r['seed'],r['preset']) in before:cases.append(dict(source=r,root_mode='advance_0p5'))
    assert len(cases)==99
    OUT.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    deps=json.loads((PHASE_REPORT/'protocol.json').read_text())
    dump(REPORT/'protocol.json',dict(source_sha256=sha(__file__),dependencies=deps['dependencies'],
        phase_source_sha256=deps['source_sha256'],cases=cases,actions=actions(),
        roots='all93 reserve entry failures: first truth observation after shove; additionally six latency failures start .5s before original onset (offline diagnostic only)',
        branch='two7-command segments (.252s each) of teacher mean-PWM scale and differential offset, clipped to[-1,1], then unchanged teacher',
        scoring='60 commands (2.16s) exact stochastic simulator including realized future; continuous footprint margin',
        continuation='top3 clearance and top3 strictly moving short branches plus unchanged baseline to55cm gate; retain all branches',
        interpretation='A clean full branch proves only one constructive alternative; failed finite search does not prove physical impossibility. Future selection is not a causal policy.',
        actor_changed=False,new_training=False,final_exam_used=False))
    shutil.copy2(__file__,OUT/'source_snapshot.py')


def capture_root(case):
    r=case['source'];env=PhaseTeacher(tuple(r['key']),r['seed'],r['preset'],'reserve','entry',registry())
    roots=[];base=FastTeacher.teacher.__get__(env)
    threshold=r['trigger']['time']-.5
    def observed(pose,v,w):
        ready=(env.trigger is not None and env.t>env.trigger['time']+1e-10) if case['root_mode']=='post_observation' else env.t>=threshold
        if ready and not roots:roots.append(fork(env))
        return base(pose,v,w)
    env.teacher=observed;env.finish()
    stored=np.load(Path(r['artifact'])/'physics.npz')['samples'];np.testing.assert_array_equal(env.physics,stored)
    assert len(roots)==1
    root=roots[0]
    basebranch=fork(root).finish();np.testing.assert_array_equal(basebranch.physics,stored)
    return root,stored


def short_branch(root,action):
    branch=fork(root);teacher=FastTeacher.teacher.__get__(branch);count=[0]
    def intervene(pose,v,w):
        u=teacher(pose,v,w);i=count[0];count[0]+=1
        if i>=14:return u
        s,d=action[:2] if i<7 else action[2:]
        if s==1. and d==0.:return u
        mean=float(u.mean())*s;diff=float((u[0]-u[1])/2)+d
        return np.clip([mean+diff,mean-diff],-1,1)
    branch.teacher=intervene
    for _ in range(60):
        if not branch.step():break
    branch.teacher=teacher
    return branch


def search_one(item):
    case_index,case=item;root,stored=capture_root(case);r=case['source'];track=Track();p=PRESETS[r['preset']]
    start=len(root.physics)-1;destination=OUT/'cases'/f'{case_index:03d}';destination.mkdir(parents=True,exist_ok=False)
    prefix=continuous_clearance(np.asarray(root.physics)[:,1:4],p,track).min()
    rows=[]
    for i,action in enumerate(actions()):
        branch=short_branch(root,action);trace=np.asarray(branch.physics)[start:]
        margin=continuous_clearance(trace[:,1:4],p,track)
        row=dict(index=i,action=action,min_margin_cm=float(min(prefix,margin.min())),
                 short_clean=bool(min(prefix,margin.min())>=-1e-9),strictly_moving=bool(trace[:,4].min()>=.5),
                 forward_velocity_min=float(trace[:,4].min()),outcome=branch.outcome,short_end_s=branch.t)
        rows.append(row)
        if i==0:np.testing.assert_array_equal(branch.physics,stored[:len(branch.physics)])
    ranked=sorted(rows,key=lambda r:r['min_margin_cm'],reverse=True)
    moving=[r for r in ranked if r['strictly_moving']]
    ids=sorted({0}|{r['index'] for r in ranked[:3]+moving[:3]})
    full=[]
    for i in ids:
        branch=short_branch(root,actions()[i]);branch.finish()
        trace=np.asarray(branch.physics);margin=continuous_clearance(trace[:,1:4],p,track)
        result=branch.summary();result.update(index=i,action=actions()[i],continuous_min_clearance_cm=float(margin.min()),
            continuous_clean=bool(result['stable_success'] and result['initial_valid'] and margin.min()>=-1e-9),
            final_event_exposed=bool(branch.exposure>0))
        result['clean_moving']=bool(result['continuous_clean'] and result['translation_pause_s']==0 and trace[:,4].min()>=-.01)
        # Early reverse is not accepted as smooth forward cornering; full trace
        # includes the initial standstill, hence use a reverse threshold only.
        np.savez_compressed(destination/f'branch_{i}.npz',physics=trace,commands=np.asarray([x['applied'] for x in branch.records]))
        full.append(result)
        if i==0:np.testing.assert_array_equal(trace,stored)
    result=dict(case_index=case_index,case=case,root_capture_s=root.capture,root_pose=root.car.pose.copy(),
        root_margin_cm=float(prefix),short_branches=rows,full_branches=full,
        any_clean=any(r['continuous_clean'] and r['final_event_exposed'] for r in full),
        any_clean_moving=any(r['clean_moving'] and r['final_event_exposed'] for r in full),
        checks=dict(original_replay_exact=True,root_continuation_exact=True,unchanged_short_and_full_exact=True))
    dump(destination/'result.json',result);return result


def run():
    protocol=json.loads((REPORT/'protocol.json').read_text());assert sha(__file__)==protocol['source_sha256']
    for f,h in protocol['dependencies'].items():assert sha(f)==h,f
    assert sha(ROOT/'cnn_finetune/goal_phase_disturbance.py')==protocol['phase_source_sha256']
    rows=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=3) as pool:
        for r in pool.map(search_one,enumerate(protocol['cases'])):
            rows.append(r);dump(REPORT/'progress.json',rows)
            print('entry alternatives',len(rows),'/',len(protocol['cases']),r['case']['root_mode'],'clean',r['any_clean'],'moving',r['any_clean_moving'],'elapsed',round(time.perf_counter()-started,1),flush=True)
    summary={}
    for mode in ['post_observation','advance_0p5']:
        a=[r for r in rows if r['case']['root_mode']==mode]
        summary[mode]=dict(cases=len(a),short_branches=sum(len(r['short_branches']) for r in a),
            full_branches=sum(len(r['full_branches']) for r in a),clean_found=sum(r['any_clean'] for r in a),
            clean_moving_found=sum(r['any_clean_moving'] for r in a))
    dump(REPORT/'results.json',dict(summary=summary,rows=rows,final_exam_used=False));print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['freeze','run']);a=p.parse_args();globals()[a.task]()
