"""Test whether retaining a safe plan prevents receding-horizon deferral."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .goal_boundary_teacher import ForecastInput,ACTIONS,INTERVENTION
from .goal_boundary_teacher_v2 import ProjectionTeacher,replay_plans,observed_residual,PILOT
from .goal_teacher_probe import FastTeacher
from .goal_teacher_coverage import ROOT
from .goal_development import sha
from .junction_stage02 import dump

OUT=Path('D:/autodrive_cnn/runs/goal_boundary_commit_20261113')
REPORT=ROOT/'cnn_finetune/results/goal_boundary_commit_20261113'


class CommittedTeacher(ProjectionTeacher):
    def teacher(self,observed,v,w):
        snapshot=ForecastInput.capture(self);base=FastTeacher.teacher(self,observed,v,w)
        residual=observed_residual(self)
        first=snapshot.forecast(ACTIONS[:1]);first=replay_plans(snapshot,first['proposed'],residual)
        chosen=0;forecast=first;cached=False;safe_level=.05
        if first['min_margin'][0]<.05:
            if self.plan is not None and len(self.plan)>6:
                remaining=self.plan[1:,None]
                continuation=replay_plans(snapshot,remaining,residual)
                if continuation['min_margin'][0]>=0:
                    forecast=continuation;cached=True;safe_level=0.
            if not cached:
                plans=snapshot.forecast(ACTIONS)['proposed'];forecast=replay_plans(snapshot,plans,residual)
                cost=((plans[:INTERVENTION]-plans[:INTERVENTION,:1])**2).mean(axis=(0,2))
                safe=forecast['min_margin']>=.05
                if not safe.any():safe=forecast['min_margin']>=0;safe_level=0.
                if safe.any():chosen=int(np.argmin(np.where(safe,cost,np.inf)))
                else:
                    risk=np.maximum(0,.05-forecast['margin'])**2
                    chosen=int(np.argmin(cost+1000*risk.mean(axis=0)));safe_level=None
        self.plan=forecast['proposed'][:,chosen].copy()
        self.guard_log.append(dict(t=self.t,choice=chosen,cached_plan=cached,plan_length=len(self.plan),
            changed=bool(np.max(abs(self.plan[0]-base))>1e-12),safe_level=safe_level,
            baseline_predicted_margin=float(first['min_margin'][0]),chosen_predicted_margin=float(forecast['min_margin'][chosen]),
            observed_residual=residual))
        return self.plan[0].copy()


def freeze():
    OUT.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    dump(REPORT/'protocol.json',dict(source_sha256=sha(__file__),parent_sha256=sha(ROOT/'cnn_finetune/goal_boundary_teacher_v2.py'),
        hypothesis='When nominal forecast is unsafe, prioritize the still-safe remaining previous plan instead of perpetually postponing correction',
        change='retained plan has priority; no zero append; replan when remaining plan has <=6 frames or is unsafe',
        unchanged='same causal observed-push predictor, action grid, horizon, projection objective, physics, scoring, cases',
        keys=PILOT,seed=6202,presets=['field_estimate','sluggish'],conditions=['normal','mild'],
        interpretation='No formal recursive-feasibility guarantee; finite forecast and nominal plant remain approximations',final_exam_used=False))
    shutil.copy2(__file__,OUT/'source_snapshot.py')


def run():
    protocol=json.loads((REPORT/'protocol.json').read_text());assert sha(__file__)==protocol['source_sha256']
    prior=json.loads((ROOT/'cnn_finetune/results/goal_boundary_teacher_v2_20261113/observed_push.json').read_text())
    rows=[];start=time.perf_counter()
    for key in protocol['keys']:
        for preset in protocol['presets']:
            for condition in protocol['conditions']:
                dst=OUT/('_'.join(key)+f'_{preset}_{condition}');dst.mkdir(exist_ok=False)
                env=CommittedTeacher(tuple(key),condition,protocol['seed'],preset,'centered',observed_push=True).finish()
                result=env.summary();baseline=next(r for r in prior if r['key']==key and r['preset']==preset and r['condition']==condition)
                np.savez_compressed(dst/'physics.npz',samples=np.asarray(env.physics));dump(dst/'guard.json',env.guard_log);dump(dst/'result.json',result)
                rows.append(dict(result,artifact=str(dst),previous_projection=baseline));dump(REPORT/'progress.json',rows)
                print('committed',key,preset,condition,result['outcome'],'clean',result['clean_stable'],'margin',round(result['min_margin_cm'],3),
                      'time',round(result['simulated_s'],2),'elapsed',round(time.perf_counter()-start,1),flush=True)
    dump(REPORT/'pilot.json',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['freeze','run']);a=p.parse_args();globals()[a.task]()
