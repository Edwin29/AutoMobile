"""Constructive recovery search on failed teacher states after a push starts.

Counterfactual lookahead is OFFLINE ONLY. Finding a clean branch demonstrates
one feasible recovery; failing this limited grid does not prove impossibility.
"""
import itertools
import json
import numpy as np

from .goal_development import OUT,REPORT,sha
from .goal_teacher_probe import FastTeacher,BASE
from .goal_exit_reference_probe import centered_reference
from .goal_teacher_validation import SELECTED,legal_start
from .junction_stage02 import dump


def main():
    old=json.loads((REPORT/'teacher_validation.json').read_text())
    failed=[r for r in old['runs'] if r['arm']=='centered' and r['condition']=='mild' and not r['clean_stable']]
    dst=OUT/'recovery_feasibility';dst.mkdir(exist_ok=False)
    actions=[dict(speed_scale=s,diff_offset=d) for s,d in itertools.product([0.,.5,1.],[-.25,-.12,0.,.12,.25])]
    actions.append(dict(speed_scale=1.,diff_offset=0.))
    # Last entry duplicates unchanged continuation deliberately: baseline clone check.
    dump(REPORT/'recovery_feasibility_protocol.json',dict(source_sha256=sha(__file__),cases=len(failed),actions=actions,
         root_rule='72ms after the push immediately preceding the first boundary overlap; failure-selected DEVELOPMENT cases',
         intervention_steps=10,intervention_s=.36,continuation='same centered teacher to local endpoint, max 40s',
         known_future_used='offline branch scoring and diagnostic case selection only',
         interpretation='Constructive lower bound on recoverable cases; no online performance claim, no impossibility claim for unsuccessful grid'))
    results=[]
    for case_index,r in enumerate(failed):
        trace=np.load(PathLike(r['artifact'])/'trace.npz')['physics'];first=float(trace[np.flatnonzero(trace[:,6]<0)[0],0])
        base=FastTeacher(tuple(r['key']),'mild',r['seed'],r['preset']);legal_start(base)
        base.points,base.h,base.s,base.k,_=centered_reference(base,SELECTED['end_cm'])
        base.controls={k:SELECTED[k] for k in BASE}
        pushes=[e for e in base.events if e['kind']=='push' and e['t0']<=first]
        if not pushes:
            results.append(dict(case=r,skipped='departure before any push'));continue
        event=max(pushes,key=lambda e:e['t0']);root_time=event['t0']+.072;base.finish(until=root_time)
        candidates=[]
        for index,action in enumerate(actions):
            env=base.clone();teacher=env.teacher
            def intervene(observed,v,w,teacher=teacher,action=action):
                u=teacher(observed,v,w);mean=u.mean()*action['speed_scale'];diff=(u[0]-u[1])/2+action['diff_offset']
                return np.clip([mean+diff,mean-diff],-1,1)
            env.teacher=intervene
            for _ in range(10):
                if not env.step():break
            env.teacher=teacher;env.finish();row=dict(env.summary(),action=action,index=index)
            candidates.append(row)
            np.savez_compressed(dst/f'{case_index}_{index}.npz',physics=np.asarray(env.physics))
        a=candidates[actions.index(dict(speed_scale=1.,diff_offset=0.))];b=candidates[-1]
        assert a['min_margin_cm']==b['min_margin_cm'] and a['simulated_s']==b['simulated_s']
        best=max(candidates,key=lambda x:(x['clean_stable'],x['stable_success'],x['min_margin_cm']))
        result=dict(key=r['key'],seed=r['seed'],preset=r['preset'],root_s=base.t,root_margin_cm=base.physics[-1][-1],
             baseline_margin_cm=a['min_margin_cm'],best=best,candidates=candidates)
        results.append(result)
        print('recovery feasibility',case_index+1,'/',len(failed),'clean',best['clean_stable'],'margin',round(best['min_margin_cm'],3),flush=True)
    dump(REPORT/'recovery_feasibility.json',dict(results=results,cases=len(failed),clean_branch_found=sum(r.get('best',{}).get('clean_stable',False) for r in results)))


def PathLike(p):
    from pathlib import Path
    return Path(p)


if __name__=='__main__':main()
