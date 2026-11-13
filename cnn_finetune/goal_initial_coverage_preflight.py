"""Pre-render audit of stratified training starts, selected without outcomes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import numpy as np
from .goal_reactive_clock_validation import ReactiveClockTeacher
from .goal_teacher_coverage import ROOT,KEYS,PRESETS
from .goal_all_movement_data import REPORT as DATA_REPORT
from .goal_continuous_clearance import continuous_clearance
from .goal_development import sha
from .junction_stage02 import dump

OUT=Path('D:/autodrive_cnn/runs/goal_initial_coverage_preflight_20261113')
REPORT=ROOT/'cnn_finetune/results/goal_initial_coverage_preflight_20261113'
BINS={
    'negative_edge':((-1.,-.85),(-.5,.5)),
    'positive_edge':((.85,1.),(-.5,.5)),
    'negative_yaw':((-.2,.2),(-3.,-2.)),
    'positive_yaw':((-.2,.2),(2.,3.)),
    'negative_lateral_negative_yaw':((-.7,-.3),(-1.5,-.5)),
    'negative_lateral_positive_yaw':((-.7,-.3),(.5,1.5)),
    'positive_lateral_negative_yaw':((.3,.7),(-1.5,-.5)),
    'positive_lateral_positive_yaw':((.3,.7),(.5,1.5)),
}


def select():
    selected={}
    for seed in range(10001,11001):
        rng=np.random.default_rng(seed);lateral=rng.uniform(-1,1);along=rng.uniform(-2,2);yaw=rng.uniform(-3,3)
        for name,(lat_bounds,yaw_bounds) in BINS.items():
            if name not in selected and lat_bounds[0]<=lateral<=lat_bounds[1] and yaw_bounds[0]<=yaw<=yaw_bounds[1]:
                selected[name]=dict(seed=seed,lateral_draw_cm=lateral,along_draw_cm=along,yaw_draw_deg=yaw)
        if len(selected)==len(BINS):break
    assert len(selected)==8 and len({r['seed'] for r in selected.values()})==8
    return selected


def freeze():
    parent=json.loads((DATA_REPORT/'protocol.json').read_text())
    for f,h in parent['hashes'].items():assert sha(f)==h,f
    chosen=select();OUT.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    dump(REPORT/'protocol.json',dict(source_sha256=sha(__file__),parent_protocol_sha256=sha(DATA_REPORT/'protocol.json'),
        bins=BINS,selected=chosen,keys=KEYS,presets=PRESETS,condition='normal',expected=672,
        selection='First seed10001..11000 satisfying each initial-pose bin, before any teacher outcomes. All28moves x3cars retained; same legal-start projection as frozen earlier code, report every projection.',
        purpose='Check whether expanded training starts cover both lateral signs and yaw signs before expensive visual rendering; not changing evaluation starts or goal criteria',
        gate='Research source gate only: each movement needs at least one clean example with lateral<-.3, lateral>.3, yaw<-.5, yaw>.5. Report uncovered edge bins even when this coarse gate passes.',
        scope='Privileged teacher normal-physics preflight; no visual model, illumination, compound disturbance or final acceptance evidence',final_exam_used=False))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(chosen),flush=True)


def run_one(case):
    key,preset,name,spec=case;env=ReactiveClockTeacher(key,'normal',spec['seed'],preset).finish()
    trace=np.asarray(env.physics);m=continuous_clearance(trace[:,1:4],env.p,env.track)
    dst=OUT/('_'.join(key)+'_'+preset+'_'+name);dst.mkdir()
    r=dict(env.summary(),stratum=name,seed_draw=spec,artifact=str(dst),continuous_min_clearance_cm=float(m.min()),
        continuous_clean=bool(env.outcome=='stable_exit' and env.summary()['initial_valid'] and m.min()>=-1e-9))
    np.savez_compressed(dst/'physics.npz',physics=trace);dump(dst/'result.json',r);return r


def run():
    p=json.loads((REPORT/'protocol.json').read_text());assert sha(__file__)==p['source_sha256']
    assert sha(DATA_REPORT/'protocol.json')==p['parent_protocol_sha256']
    for f,h in json.loads((DATA_REPORT/'protocol.json').read_text())['hashes'].items():assert sha(f)==h,f
    cases=[(key,preset,name,spec) for key,preset,(name,spec) in itertools.product(KEYS,PRESETS,p['selected'].items())]
    rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for r in pool.map(run_one,cases):
            rows.append(r)
            if len(rows)%48==0:dump(REPORT/'progress.json',rows);print('initial coverage',len(rows),'/672',flush=True)
    coverage=[]
    for key in KEYS:
        a=[r for r in rows if tuple(r['key'])==key]
        clean=[r for r in a if r['continuous_clean'] and not r['initial_sampling']['projected']]
        signs={name:any(fn(r['seed_draw']) for r in clean) for name,fn in {
            'negative_lateral':lambda r:r['lateral_draw_cm']<-.3,'positive_lateral':lambda r:r['lateral_draw_cm']>.3,
            'negative_yaw':lambda r:r['yaw_draw_deg']<-.5,'positive_yaw':lambda r:r['yaw_draw_deg']>.5}.items()}
        coverage.append(dict(key=key,signs=signs,coarse_gate=all(signs.values()),
            strata={name:dict(n=sum(r['stratum']==name for r in a),clean=sum(r['stratum']==name and r['continuous_clean'] for r in a)) for name in BINS}))
    summary=dict(n=len(rows),clean=sum(r['continuous_clean'] for r in rows),projected=sum(r['initial_sampling']['projected'] for r in rows),
        coarse_source_gate=all(r['coarse_gate'] for r in coverage),missing_strata=[dict(key=r['key'],stratum=k) for r in coverage for k,v in r['strata'].items() if v['clean']==0])
    dump(REPORT/'results.json',dict(summary=summary,coverage=coverage,rows=rows,final_exam_used=False));print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['freeze','run']);a=p.parse_args();globals()[a.task]()
