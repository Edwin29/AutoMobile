"""Reserve-constrained smooth turn references and matched speed comparison.

Geometry objective uses no disturbance outcomes. Required clearance is 90% of
the previously achieved clearance-first path, with a search discretization
buffer. This is a new offline reference candidate, never an automatic adoption.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.optimize import differential_evolution
from bench.vehicle import PRESETS as VEHICLES
from bench.world import Track
from .goal_teacher_coverage import CoverageTeacher,KEYS,PRESETS,SPEED,STAGE_OUT,ROOT
from .junction_cornering import path_for,shape_metrics,speed_profile,OUT as CORNER_OUT
from .goal_continuous_clearance import continuous_clearance
from .goal_development import sha
from .junction_stage02 import dump

OUT=Path('D:/autodrive_cnn/runs/goal_clearance_paths_20261113')
REPORT=ROOT/'cnn_finetune/results/goal_clearance_paths_20261113'
TURNS=[key for key in KEYS if key[2]!='straight']
BOUNDS=np.array([[6,30],[6,30],[1,13],[1,13],[-2,2],[-2,2]],float)


def freeze():
    OUT.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    dump(REPORT/'protocol.json',dict(source_sha256=sha(__file__),keys=KEYS,
        hypothesis='A reference that reserves more body clearance can improve perturbation tolerance without reverting to a pivot controller',
        geometry=dict(seed=20261113,target='0.9 times achieved continuous clearance of old clearance-first path',
            search_buffer_cm=.02,segments='three quintic Bezier segments with zero endpoint curvature',
            smooth_objective='12 max curvature +60 curvature derivative energy +.6 curvature variation +.002 length',
            maxiter=90,population=42,postcheck='continuous footprint at dense path samples must meet target; otherwise held'),
        speed_arms=['same_mode','planned'],speed_mode='same_mode retains current per-movement legacy/planned mode; planned applies spatial braking/acceleration to every turn',
        controller='same corrected teacher gains heading1.3 lateral.08; no boundary guard, to isolate reference/speed change',
        evaluation=dict(seeds=[6201,6202],presets=PRESETS,conditions=['normal','mild'],movements=28),
        straight='same original straight reference/controller in both arms; executed and retained',
        final_exam_used=False,selection='report all movements, all candidates and failures; no automatic replacement'))
    shutil.copy2(__file__,OUT/'source_snapshot.py')


def search_one(key):
    key=tuple(key);name='_'.join(key);track=Track();p=VEHICLES['field_estimate']
    old=np.load(STAGE_OUT/'paths'/name/'path.npz')
    old_margin=float(continuous_clearance(np.c_[old['points'],old['heading']],p,track).min())
    target=.9*old_margin;search_target=target+.02
    rng=np.random.default_rng(20261113);population=rng.uniform(BOUNDS[:,0],BOUNDS[:,1],(42,6))
    seeds=[json.loads((STAGE_OUT/'paths'/name/'path.json').read_text())['parameters']]
    seeds += [json.loads((CORNER_OUT/'paths'/name/(label+'.json')).read_text())['parameters'] for label in ['wide','reserve']]
    for i,x in enumerate(seeds):population[i]=np.clip(x,BOUNDS[:,0],BOUNDS[:,1])
    def objective(x):
        points,h,k,s=path_for(key,x,False)
        margin=float(continuous_clearance(np.c_[points,h],p,track).min())
        violation=max(0,search_target-margin)
        shape=shape_metrics(k,s)
        smooth=12*shape['max_abs_curvature_cm_inv']+60*shape['curvature_derivative_energy']+.6*shape['curvature_variation']+.002*s[-1]
        return 1000+10000*violation+smooth if violation>0 else smooth
    result=differential_evolution(objective,BOUNDS,init=population,seed=20261113,maxiter=90,polish=False,tol=.0005)
    points,h,k,s=path_for(key,result.x,True)
    minimum=float(continuous_clearance(np.c_[points,h],p,track).min())
    grid=np.linspace(0,s[-1],int(np.ceil(s[-1]/.25))+1)
    uniform=np.c_[np.interp(grid,s,points[:,0]),np.interp(grid,s,points[:,1])]
    headings=np.interp(grid,s,np.unwrap(h));curvature=np.interp(grid,s,k)
    grid_margin=float(continuous_clearance(np.c_[uniform,headings],p,track).min())
    approved=min(minimum,grid_margin)>=target
    dst=OUT/'paths'/name;dst.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(dst/'reference.npz',points=uniform,heading=headings,curvature=curvature,s=grid)
    row=dict(key=key,old_clearance_cm=old_margin,target_clearance_cm=target,dense_clearance_cm=minimum,
        grid_clearance_cm=grid_margin,geometry_accepted=approved,parameters=result.x,evaluations=result.nfev,
        reference=str(dst/'reference.npz'),**shape_metrics(curvature,grid))
    dump(dst/'geometry.json',row);return row


def search():
    p=json.loads((REPORT/'protocol.json').read_text());assert sha(__file__)==p['source_sha256']
    rows=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=3) as pool:
        for row in pool.map(search_one,TURNS):
            rows.append(row);dump(REPORT/'geometry_progress.json',rows)
            print('reserve geometry',row['key'],'clearance',round(row['dense_clearance_cm'],3),'target',round(row['target_clearance_cm'],3),
                  'radius',round(row['min_radius_cm'],2),'accepted',row['geometry_accepted'],'elapsed',round(time.perf_counter()-started,1),flush=True)
    dump(REPORT/'geometry.json',rows)


class ReserveTeacher(CoverageTeacher):
    def __init__(self,key,condition,seed,preset,arm,registry):
        super().__init__(key,condition,seed,preset,'centered');self.speed_arm=arm
        if key[2]=='straight':return
        r=registry[tuple(key)];assert r['geometry_accepted']
        z=np.load(r['reference']);np.testing.assert_allclose(z['points'][0],self.points[0],atol=1e-12,rtol=0)
        self.points=z['points'];self.h=z['heading'];self.k=z['curvature'];self.s=z['s']
        if arm=='planned':self.teacher_mode='planned'
        self.speed=speed_profile(self.k,self.s,SPEED[tuple(key)]) if self.teacher_mode=='planned' else SPEED[tuple(key)]/(1+5*abs(self.k))
        self.static_candidate_margin=r['dense_clearance_cm']
    def summary(self):return dict(super().summary(),speed_arm=self.speed_arm)


def evaluate(arm):
    p=json.loads((REPORT/'protocol.json').read_text());assert sha(__file__)==p['source_sha256']
    registry={tuple(r['key']):r for r in json.loads((REPORT/'geometry.json').read_text())};rows=[];started=time.perf_counter()
    baseline=json.loads((ROOT/'cnn_finetune/results/goal_teacher_coverage_20261113/coverage.json').read_text())['runs']
    for key in KEYS:
        for seed in p['evaluation']['seeds']:
            for preset in PRESETS:
                for condition in p['evaluation']['conditions']:
                    old=next(r for r in baseline if tuple(r['key'])==key and r['seed']==seed and r['preset']==preset and r['condition']==condition and r['variant']=='centered')
                    if key[2]!='straight' and not registry[key]['geometry_accepted']:
                        rows.append(dict(key=key,seed=seed,preset=preset,condition=condition,speed_arm=arm,static_rejected=True,
                                         clean_stable=False,stable_success=False,baseline=old));continue
                    dst=OUT/'rollouts'/arm/('_'.join(key)+f'_{seed}_{preset}_{condition}');dst.mkdir(parents=True,exist_ok=False)
                    env=ReserveTeacher(key,condition,seed,preset,arm,registry).finish();r=dict(env.summary(),static_rejected=False,artifact=str(dst))
                    np.testing.assert_allclose(env.initial_pose,old['initial_sampling']['pose'],atol=1e-12,rtol=0)
                    np.savez_compressed(dst/'physics.npz',samples=np.asarray(env.physics));dump(dst/'result.json',r)
                    rows.append(dict(r,baseline=old))
        dump(REPORT/(arm+'_progress.json'),rows)
        print('reserve evaluation',arm,key,len(rows),'clean',sum(r['clean_stable'] for r in rows),'elapsed',round(time.perf_counter()-started,1),flush=True)
    dump(REPORT/(arm+'.json'),rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['freeze','search','evaluate']);p.add_argument('--arm',choices=['same_mode','planned']);a=p.parse_args()
    evaluate(a.arm) if a.task=='evaluate' else globals()[a.task]()
