"""Prospective all-movement paired test of a frozen clearance-path teacher.

Development evidence, never an independent final student examination.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from bench.vehicle import PRESETS as VEHICLES
from bench.world import Track
from .goal_clearance_paths import ReserveTeacher,REPORT as SEARCH_REPORT
from .goal_teacher_coverage import CoverageTeacher,KEYS,PRESETS,ROOT
from .goal_continuous_clearance import continuous_clearance
from .goal_development import sha
from .junction_stage02 import dump

OUT=Path('D:/autodrive_cnn/runs/goal_clearance_validation_20261113')
REPORT=ROOT/'cnn_finetune/results/goal_clearance_validation_20261113'


def freeze():
    OUT.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    files={Path(__file__).resolve()}
    for name,module in list(sys.modules.items()):
        f=getattr(module,'__file__',None)
        if f and (name.startswith('cnn_finetune.') or name.startswith('bench.')) and f.endswith('.py'):
            files.add(Path(f).resolve())
    registry=json.loads((SEARCH_REPORT/'geometry.json').read_text())
    files.add(SEARCH_REPORT/'geometry.json')
    for r in registry:files.add(Path(r['reference']))
    for key in KEYS:
        env=CoverageTeacher(key,'normal',6401,PRESETS[0],'centered')
        # Baseline arrays are saved, hashed and checked again in every case.
        dst=OUT/('baseline_reference_'+'_'.join(key)+'.npz')
        np.savez_compressed(dst,points=env.points,h=env.h,k=env.k,s=env.s,speed=env.speed)
        files.add(dst)
    dump(REPORT/'protocol.json',dict(seeds=list(range(6401,6409)),keys=KEYS,presets=PRESETS,
        conditions=['normal','mild'],arms=['baseline','reserve'],candidate='same_mode fixed globally; no movement-wise selection',
        selection_evidence=str(SEARCH_REPORT/'same_mode.json'),
        hashes={str(f):sha(f) for f in sorted(files)},
        initial_rule='paired identical sampled legal start; preserve every projection and every failure',
        scope='offline teacher only; mild is physical disturbances, no illumination evidence',
        analysis='all cells plus turn-only, paired clean changes, times, angular acceleration, pauses, continuous footprint sign',
        no_automatic_adoption=True,final_exam_used=False,actor_changed=False))
    shutil.copy2(__file__,OUT/'source_snapshot.py')


def run(arm):
    p=json.loads((REPORT/'protocol.json').read_text())
    for f,h in p['hashes'].items():assert sha(f)==h,f
    registry={tuple(r['key']):r for r in json.loads((SEARCH_REPORT/'geometry.json').read_text())}
    rows=[];start=time.perf_counter();track=Track()
    for key in KEYS:
        frozen=np.load(OUT/('baseline_reference_'+'_'.join(key)+'.npz'))
        for seed in p['seeds']:
            for preset in PRESETS:
                for condition in p['conditions']:
                    baseline=CoverageTeacher(key,condition,seed,preset,'centered')
                    for field in ['points','h','k','s','speed']:
                        np.testing.assert_array_equal(getattr(baseline,field),frozen[field])
                    env=baseline if arm=='baseline' else ReserveTeacher(key,condition,seed,preset,'same_mode',registry)
                    np.testing.assert_array_equal(env.initial_pose,baseline.initial_pose)
                    env.finish();trace=np.asarray(env.physics)
                    margin=continuous_clearance(trace[:,1:4],VEHICLES[preset],track)
                    dst=OUT/'rollouts'/arm/('_'.join(key)+f'_{seed}_{preset}_{condition}')
                    dst.mkdir(parents=True,exist_ok=False)
                    np.savez_compressed(dst/'physics.npz',samples=trace)
                    r=dict(env.summary(),arm=arm,artifact=str(dst),continuous_min_clearance_cm=float(margin.min()),
                        continuous_clean=bool(env.summary()['clean_stable'] and margin.min()>=-1e-9))
                    dump(dst/'result.json',r);rows.append(r)
        dump(REPORT/(arm+'_progress.json'),rows)
        print(arm,key,len(rows),'clean',sum(r['clean_stable'] for r in rows),'elapsed',round(time.perf_counter()-start,1),flush=True)
    dump(REPORT/(arm+'.json'),rows)


def summarize():
    runs={a:json.loads((REPORT/(a+'.json')).read_text()) for a in ['baseline','reserve']}
    assert all(len(r)==1344 for r in runs.values())
    def identity(r):return (tuple(r['key']),r['seed'],r['preset'],r['condition'])
    maps={a:{identity(r):r for r in rows} for a,rows in runs.items()}
    assert maps['baseline'].keys()==maps['reserve'].keys()
    for k,a in maps['baseline'].items():
        b=maps['reserve'][k]
        assert a['initial_sampling']==b['initial_sampling']
        if k[0][2]=='straight':
            np.testing.assert_array_equal(np.load(Path(a['artifact'])/'physics.npz')['samples'],np.load(Path(b['artifact'])/'physics.npz')['samples'])
    def cell(rows):
        return dict(n=len(rows),clean=sum(r['clean_stable'] for r in rows),continuous_clean=sum(r['continuous_clean'] for r in rows),
            stable=sum(r['stable_success'] for r in rows),pauses=sum(r['translation_pause_s']>0 for r in rows),
            projected_starts=sum(r['initial_sampling']['projected'] for r in rows),
            median_time_s=float(np.median([r['simulated_s'] for r in rows])),
            median_angular_accel=float(np.median([r['angular_accel_rms_rad_s2'] for r in rows])),
            outcomes={v:sum(r['outcome']==v for r in rows) for v in sorted(set(r['outcome'] for r in rows))})
    summary={a:{c:{group:cell([r for r in rows if r['condition']==c and (group=='all' or r['key'][2]!='straight')])
        for group in ['all','turns']} for c in ['normal','mild']} for a,rows in runs.items()}
    movements=[dict(key=key,**{a:{c:cell([r for r in rows if tuple(r['key'])==key and r['condition']==c])
        for c in ['normal','mild']} for a,rows in runs.items()}) for key in KEYS]
    paired={}
    for c in ['normal','mild']:
        keys=[k for k in maps['baseline'] if k[-1]==c]
        both=[k for k in keys if all(maps[a][k]['clean_stable'] for a in maps) and k[0][2]!='straight']
        paired[c]=dict(gains=sum(maps['reserve'][k]['clean_stable'] and not maps['baseline'][k]['clean_stable'] for k in keys),
            losses=sum(maps['baseline'][k]['clean_stable'] and not maps['reserve'][k]['clean_stable'] for k in keys),
            both_clean_turn_n=len(both),
            median_both_clean_turn_time_change_s=float(np.median([maps['reserve'][k]['simulated_s']-maps['baseline'][k]['simulated_s'] for k in both])),
            median_both_clean_turn_angular_accel_change=float(np.median([maps['reserve'][k]['angular_accel_rms_rad_s2']-maps['baseline'][k]['angular_accel_rms_rad_s2'] for k in both])))
    dump(REPORT/'summary.json',dict(summary=summary,paired=paired,movements=movements,
        checks=dict(straight_physics_identical=True,paired_initial_conditions_identical=True),
        limitations='Eight shared random seeds, not 1344 independent seeds. Local teacher, no vision/lighting, no student, no full mission, no final exam. Continuous footprint evaluated at physics timestamps.'))
    print(json.dumps(dict(summary=summary,paired=paired)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['freeze','run','summarize']);p.add_argument('--arm',choices=['baseline','reserve']);a=p.parse_args()
    run(a.arm) if a.task=='run' else globals()[a.task]()
