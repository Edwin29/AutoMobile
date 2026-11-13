"""Matched offline teacher audit for the already declared6901 nominal screen."""
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import numpy as np
from .goal_reactive_clock_validation import ReactiveClockTeacher
from .goal_teacher_coverage import KEYS,PRESETS,ROOT
from .goal_continuous_clearance import continuous_clearance
from .goal_all_movement_data import REPORT as DATA_REPORT
from .goal_all_movement_eval import REPORT as EVAL_REPORT
from .goal_development import sha
from .junction_stage02 import dump

OUT=Path('D:/autodrive_cnn/runs/goal_visual_nominal_teacher_audit_20261113')
REPORT=ROOT/'cnn_finetune/results/goal_visual_nominal_teacher_audit_20261113'


def run_one(case):
    key,preset=case;env=ReactiveClockTeacher(key,'normal',6901,preset).finish()
    trace=np.asarray(env.physics);m=continuous_clearance(trace[:,1:4],env.p,env.track)
    dst=OUT/('_'.join(key)+'_'+preset);dst.mkdir()
    r=dict(env.summary(),artifact=str(dst),continuous_min_clearance_cm=float(m.min()),
        continuous_clean=bool(env.outcome=='stable_exit' and env.summary()['initial_valid'] and m.min()>=-1e-9))
    np.savez_compressed(dst/'physics.npz',physics=trace);dump(dst/'result.json',r);return r


def main():
    inherited=json.loads((DATA_REPORT/'protocol.json').read_text())
    for f,h in inherited['hashes'].items():assert sha(f)==h,f
    OUT.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    dump(REPORT/'protocol.json',dict(source_sha256=sha(__file__),data_protocol_sha256=sha(DATA_REPORT/'protocol.json'),
        visual_protocol_sha256=sha(EVAL_REPORT/'protocol.json'),keys=KEYS,presets=PRESETS,seed=6901,condition='normal',
        scope='Matched offline privileged teacher at same initial samples as visual nominal screen; not a universal upper bound or a learned model',
        execution='frozen teacher, no changes or parameter selection; no images needed because the teacher does not consume them',final_exam_used=False))
    with ProcessPoolExecutor(max_workers=3) as pool:rows=list(pool.map(run_one,itertools.product(KEYS,PRESETS)))
    reference=json.loads((EVAL_REPORT/'reference_nominal.json').read_text());by={(tuple(r['key']),r['preset']):r for r in reference}
    for r in rows:
        b=by[tuple(r['key']),r['preset']];np.testing.assert_array_equal(r['initial_sampling']['pose'],b['initial_sampling']['pose'])
    summary=dict(n=len(rows),teacher_clean=sum(r['continuous_clean'] for r in rows),reference_clean=sum(r['continuous_clean'] for r in reference),
        teacher_clean_reference_failed=sum(r['continuous_clean'] and not by[tuple(r['key']),r['preset']]['continuous_clean'] for r in rows),
        initial_poses_exact=True,final_exam_used=False)
    dump(REPORT/'results.json',dict(summary=summary,rows=rows));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
