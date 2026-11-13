"""Audit whether oracle queries from student states actually recover cleanly.

Replay recorded student actions through the exact physics (no image rendering),
fork at predeclared times, and let the corrected teacher finish. This audits
supervision quality; never feeds future information into the student policy.
"""
import json
import numpy as np
from pathlib import Path
from .goal_full_dagger import OUT,REPORT,sha
from .goal_teacher_probe import FastTeacher,BASE
from .goal_teacher_validation import SELECTED,legal_start
from .goal_exit_reference_probe import centered_reference
from .junction_stage02 import dump


def main():
    rows=json.loads((REPORT/'student_data.json').read_text());roots=[3.,5.,8.,12.,16.]
    dst=OUT/'query_audit';dst.mkdir(exist_ok=False)
    dump(REPORT/'query_audit_protocol.json',dict(roots_s=roots,episodes=len(rows),source_sha256=sha(__file__),
         scope='Offline teacher continuation from acting-student states; same realized future disturbances. Not a student success score.',
         parity_required='full physics replay from saved proposed actions must match source episode'))
    results=[]
    for episode,r in enumerate(rows):
        with np.load(Path(r['artifact'])/'sequence.npz') as z:actions=z['action']
        original=np.load(Path(r['artifact'])/'physics.npz')['samples']
        env=FastTeacher(tuple(r['key']),r['condition'],r['seed'],r['preset']);legal_start(env)
        env.points,env.h,env.s,env.k,_=centered_reference(env,SELECTED['end_cm']);env.controls={k:SELECTED[k] for k in BASE}
        oracle=FastTeacher.teacher.__get__(env,FastTeacher);index=[0]
        def recorded(observed,v,w):
            oracle(observed,v,w)  # retain the query controller's causal path index
            i=index[0];index[0]+=1
            return actions[i] if i<len(actions) else np.zeros(2)
        env.teacher=recorded
        for target in roots:
            env.finish(until=target)
            if env.outcome!='time_limit':continue
            branch=env.clone();branch.teacher=FastTeacher.teacher.__get__(branch,FastTeacher)
            start=len(branch.physics);root_margin=float(branch.physics[-1][6]);branch.finish()
            trace=np.asarray(branch.physics);future=trace[start-1:]
            row=dict(episode=episode,key=r['key'],condition=r['condition'],seed=r['seed'],preset=r['preset'],root_s=env.t,
                root_margin_cm=root_margin,onroad_root=root_margin>=0,teacher_outcome=branch.outcome,
                future_min_margin_cm=float(future[:,6].min()),
                clean_teacher_continuation=branch.outcome=='stable_exit' and future[:,6].min()>=0,
                stable_teacher_continuation=branch.outcome=='stable_exit')
            np.savez_compressed(dst/f'{episode}_{target}.npz',physics=future);results.append(row)
        env.finish();np.testing.assert_array_equal(np.asarray(env.physics),original)
        if (episode+1)%12==0:
            dump(REPORT/'query_audit_progress.json',results);print('oracle query audit',episode+1,'episodes',len(results),'roots',flush=True)
    onroad=[r for r in results if r['onroad_root']]
    summary=dict(roots=len(results),onroad_roots=len(onroad),onroad_clean_continuations=sum(r['clean_teacher_continuation'] for r in onroad),
        stable_continuations=sum(r['stable_teacher_continuation'] for r in results),exact_physics_replays=len(rows),
        by_root={str(t):dict(n=len(rs),onroad=len(valid),clean=sum(r['clean_teacher_continuation'] for r in valid))
                 for t in roots for rs in [[r for r in results if abs(r['root_s']-t)<.05]] for valid in [[r for r in rs if r['onroad_root']]]})
    dump(REPORT/'query_audit.json',dict(summary=summary,results=results));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
