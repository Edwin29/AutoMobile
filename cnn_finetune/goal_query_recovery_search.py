"""Constructive offline action search on pre-next-push query failures.

Future rollouts select diagnostic counterfactuals, never actions of an evaluated
student. A failed finite search does not establish physical impossibility.
"""
import itertools
import json
from pathlib import Path
import numpy as np
from .goal_full_dagger import OUT as DATA,REPORT as DATA_REPORT,sha
from .goal_nominal_constraint import OUT,REPORT
from .goal_teacher_probe import FastTeacher,BASE
from .goal_teacher_validation import SELECTED,legal_start
from .goal_exit_reference_probe import centered_reference
from .junction_stage02 import dump


def main():
    cases=[r for r in json.loads((DATA_REPORT/'query_failure_timing.json').read_text())['rows'] if r['before_next_push']]
    sources=json.loads((DATA_REPORT/'student_data.json').read_text())
    actions=[dict(speed_scale=s,diff_offset=d) for s,d in itertools.product([0.,.5,1.,1.2],[-.25,-.12,0.,.12,.25])]
    destination=OUT/'query_recovery';destination.mkdir(exist_ok=False)
    dump(REPORT/'query_recovery_protocol.json',dict(source_sha256=sha(__file__),cases=cases,actions=actions,
        intervention_frames=10,intervention_s=.36,continuation='same corrected teacher',
        selection='development roots onroad at takeover, teacher incurs before next push or no push exists',
        use='offline constructive feasibility only, future used to score branches; not online controller',
        limits='finite action grid, correlated roots, no impossibility inference'))
    results=[]
    for case_index,r in enumerate(cases):
        source=sources[r['episode']]
        with np.load(Path(source['artifact'])/'sequence.npz') as z:recorded_actions=z['action']
        original=np.load(Path(source['artifact'])/'physics.npz')['samples']
        env=FastTeacher(tuple(r['key']),r['condition'],r['seed'],r['preset']);legal_start(env)
        env.points,env.h,env.s,env.k,_=centered_reference(env,SELECTED['end_cm'])
        env.controls={k:SELECTED[k] for k in BASE};oracle=FastTeacher.teacher.__get__(env,FastTeacher);index=[0]
        def recorded(observed,v,w):
            oracle(observed,v,w);i=index[0];index[0]+=1;return recorded_actions[i]
        env.teacher=recorded
        root=min([3.,5.,8.,12.,16.],key=lambda x:abs(x-r['root_s']))
        env.finish(until=root);assert env.t==r['root_s']
        np.testing.assert_array_equal(np.asarray(env.physics),original[:len(env.physics)])
        rows=[]
        for action_index,action in enumerate(actions):
            branch=env.clone();teacher=FastTeacher.teacher.__get__(branch,FastTeacher);start=len(branch.physics)-1
            def intervention(observed,v,w,teacher=teacher,action=action):
                u=teacher(observed,v,w)
                if action['speed_scale']==1. and action['diff_offset']==0.:return u
                mean=float(u.mean())*action['speed_scale'];diff=float((u[0]-u[1])/2)+action['diff_offset']
                return np.clip([mean+diff,mean-diff],-1,1)
            branch.teacher=intervention
            for _ in range(10):
                if not branch.step():break
            branch.teacher=teacher;branch.finish();trace=np.asarray(branch.physics)[start:]
            bad=np.flatnonzero(trace[:,6]<0);first=None if not len(bad) else float(trace[bad[0],0])
            next_push=r['next_push_s'];before=trace if next_push is None else trace[trace[:,0]<next_push]
            row=dict(action=action,index=action_index,outcome=branch.outcome,min_margin_cm=float(trace[:,6].min()),
                first_incursion_s=first,clean_to_next_push=bool(before[:,6].min()>=0),
                clean_continuation=bool(branch.outcome=='stable_exit' and trace[:,6].min()>=0))
            np.savez_compressed(destination/f'{case_index}_{action_index}.npz',physics=trace)
            if action['speed_scale']==1. and action['diff_offset']==0.:
                expected=np.load(DATA/'query_audit'/f"{r['episode']}_{root}.npz")['physics']
                np.testing.assert_array_equal(trace,expected)
            rows.append(row)
        best=max(rows,key=lambda a:(a['clean_continuation'],a['clean_to_next_push'],a['min_margin_cm']))
        results.append(dict(case=r,best=best,branches=rows,
            incursion_before_new_camera_action_possible=r['seconds_to_incursion']<.036-1e-9))
        dump(REPORT/'query_recovery_progress.json',results)
        print('query recovery',case_index+1,'/',len(cases),'clean',best['clean_continuation'],
              'until next push',best['clean_to_next_push'],flush=True)
    summary=dict(cases=len(cases),branches=len(cases)*len(actions),
        clean_branch_found=sum(r['best']['clean_continuation'] for r in results),
        clean_until_next_push_found=sum(r['best']['clean_to_next_push'] for r in results),
        incursion_before_first_new_command=sum(r['incursion_before_new_camera_action_possible'] for r in results))
    dump(REPORT/'query_recovery.json',dict(summary=summary,results=results));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
