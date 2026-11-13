"""Prospective normal-condition validation for all required movements."""
import json
import time
import numpy as np
from .goal_boundary_commit import OUT,REPORT,CommittedTeacher
from .goal_teacher_coverage import CoverageTeacher,KEYS
from .goal_development import ROOT,sha
from .junction_stage02 import dump


def main():
    p=json.loads((REPORT/'all_normal_protocol.json').read_text())
    assert sha(ROOT/'cnn_finetune/goal_boundary_commit.py')==p['source_sha256']
    rows=[];started=time.perf_counter()
    for key in KEYS:
        for seed in p['seeds']:
            for preset in p['presets']:
                pair={}
                for arm,cls in [('baseline',CoverageTeacher),('committed',CommittedTeacher)]:
                    dst=OUT/'all_normal'/arm/('_'.join(key)+f'_{seed}_{preset}');dst.mkdir(parents=True,exist_ok=False)
                    env=cls(key,'normal',seed,preset,'centered').finish();r=dict(env.summary(),arm=arm,artifact=str(dst))
                    np.savez_compressed(dst/'physics.npz',samples=np.asarray(env.physics));dump(dst/'result.json',r)
                    if arm=='committed':dump(dst/'guard.json',env.guard_log)
                    rows.append(r);pair[arm]=env.initial_pose
                np.testing.assert_array_equal(pair['baseline'],pair['committed'])
        dump(REPORT/'all_normal_progress.json',rows)
        print('normal validation',key,'episodes',len(rows),'elapsed',round(time.perf_counter()-started,1),flush=True)
    summary={}
    for arm in ['baseline','committed']:
        a=[r for r in rows if r['arm']==arm]
        summary[arm]=dict(n=len(a),clean=sum(r['clean_stable'] for r in a),stable=sum(r['stable_success'] for r in a),
            pauses=sum(r['translation_pause_s']>0 for r in a),median_time_s=float(np.median([r['simulated_s'] for r in a])))
    dump(REPORT/'all_normal.json',dict(summary=summary,rows=rows,source_sha256=sha(__file__)));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
