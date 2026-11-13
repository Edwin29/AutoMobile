"""Minimal-action projection with shifted feasible-plan fallback.

Compare nominal prediction against decaying extrapolation of already observed
body motion residual. Both are offline teachers, not actor inputs.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from bench.world import advance,wrap
from .goal_boundary_teacher import ForecastInput,ACTIONS,HORIZON,INTERVENTION,advance_batch
from .goal_teacher_coverage import CoverageTeacher,PRESETS,ROOT
from .goal_teacher_probe import FastTeacher
from .goal_development import sha
from .junction_stage02 import dump

OUT=Path('D:/autodrive_cnn/runs/goal_boundary_teacher_v2_20261113')
REPORT=ROOT/'cnn_finetune/results/goal_boundary_teacher_v2_20261113'
PILOT=[('north','S','left'),('east','W','left'),('center','E','right'),('south','N','right')]


def replay_plans(snapshot,plans,residual=None):
    """Re-evaluate explicit PWM plans, using past/current information only."""
    p=snapshot.p;n=plans.shape[1];pose=np.tile(snapshot.pose,(n,1));speed=np.tile(snapshot.wheel_speed,(n,1))
    error=np.tile(snapshot.error,(n,1));floor=snapshot.floor[None];t=snapshot.t
    commands=[(stamp,np.tile(u,(n,1))) for stamp,u in snapshot.commands]
    start=np.array([p.start_duty_left,p.start_duty_right]);gain=np.array([p.cm_s_per_duty_left,p.cm_s_per_duty_right])
    poses=[];times=[];outputs=[];residual=np.zeros(3) if residual is None else residual
    for u in plans:
        target_t=t+.036
        while t<target_t-1e-10:
            dt=min(.02,target_t-t);due=t-p.actuation_delay_s
            active=next((cmd for stamp,cmd in reversed(commands) if stamp<=due),commands[0][1])
            mag=abs(active);duty=np.sign(active)*np.where(mag>=start-1e-9,np.minimum(mag,p.max_duty),0.)
            desired=duty*gain
            speed=speed+(desired-speed)*(1-np.exp(-dt/p.time_constant_s)) if p.time_constant_s>0 else desired.copy()
            v=speed.mean(axis=1);raw=(speed[:,0]-speed[:,1])/p.track_cm
            w=raw*np.where(raw<0,p.turn_eff_left,p.turn_eff_right)+p.drift_rad_s*v/5
            pose=advance_batch(pose,v,w,dt)
            # Integral of a fixed observed world-motion residual with tau=.2s.
            integral=.2*(np.exp(-(t-snapshot.t)/.2)-np.exp(-(t+dt-snapshot.t)/.2))
            pose[:,:2]+=residual[:2]*integral;pose[:,2]=wrap(pose[:,2]+residual[2]*integral)
            t+=dt;poses.append(pose.copy());times.append(t)
        small=(abs(u)>0)&(abs(u)<floor);error=np.where(small,error+abs(u)/floor,0.);pulse=error>=.5
        output=np.where(small,np.where(pulse,np.sign(u)*floor,0.),u)
        error=np.where(small&pulse,error-1,error);commands.append((t,output.copy()));outputs.append(output)
    trace=np.asarray(poses);yaw=trace[:,:,2,None];co=np.cos(yaw);si=np.sin(yaw);rim=snapshot.rim
    points=np.stack([trace[:,:,0,None]+rim[None,None,:,0]*co+rim[None,None,:,1]*si,
                     trace[:,:,1,None]-rim[None,None,:,0]*si+rim[None,None,:,1]*co],axis=-1)
    margin=snapshot.track.clearance(points).min(axis=-1)
    return dict(poses=trace,times=np.asarray(times),margin=margin,min_margin=margin.min(axis=0),proposed=plans,applied=np.asarray(outputs))


def observed_residual(scene):
    if len(scene.physics)<2:return np.zeros(3)
    before,now=np.asarray(scene.physics[-2:]);dt=now[0]-before[0]
    predicted=advance(before[1:4],now[4],now[5],dt)
    return np.r_[(now[1:3]-predicted[:2])/dt,wrap(now[3]-predicted[2])/dt]


class ProjectionTeacher(CoverageTeacher):
    def __init__(self,*args,observed_push=False,**kwargs):
        super().__init__(*args,**kwargs);self.observed_push=observed_push;self.plan=None;self.guard_log=[]
    def teacher(self,observed,v,w):
        snapshot=ForecastInput.capture(self);base=FastTeacher.teacher(self,observed,v,w)
        residual=observed_residual(self) if self.observed_push else np.zeros(3)
        first=snapshot.forecast(ACTIONS[:1]);first=replay_plans(snapshot,first['proposed'],residual)
        chosen=0;forecast=first;cached_index=None;safe_level=.05
        if first['min_margin'][0]<.05:
            predicted=snapshot.forecast(ACTIONS);plans=predicted['proposed']
            if self.plan is not None:
                shifted=np.concatenate([self.plan[1:],np.zeros((1,2))])
                cached_index=plans.shape[1];plans=np.concatenate([plans,shifted[:,None]],axis=1)
            forecast=replay_plans(snapshot,plans,residual)
            # Prefer the least change from the nominal command sequence; do not
            # reward stationary alignment when a safe moving command exists.
            cost=((plans[:INTERVENTION]-plans[:INTERVENTION,:1])**2).mean(axis=(0,2))
            safe=forecast['min_margin']>=.05
            if not safe.any():safe=forecast['min_margin']>=0;safe_level=0.
            if safe.any():chosen=int(np.argmin(np.where(safe,cost,np.inf)))
            else:
                risk=np.maximum(0,.05-forecast['margin'])**2
                chosen=int(np.argmin(cost+1000*risk.mean(axis=0)));safe_level=None
        np.testing.assert_allclose(forecast['proposed'][0,0],base,atol=1e-12,rtol=0)
        self.plan=forecast['proposed'][:,chosen].copy()
        self.guard_log.append(dict(t=self.t,choice=chosen,cached_plan=chosen==cached_index,
            changed=bool(np.max(abs(self.plan[0]-base))>1e-12),safe_level=safe_level,
            baseline_predicted_margin=float(first['min_margin'][0]),chosen_predicted_margin=float(forecast['min_margin'][chosen]),
            observed_residual=residual))
        return self.plan[0].copy()
    def summary(self):
        return dict(super().summary(),observed_push=self.observed_push,guard_calls=len(self.guard_log),
            guard_interventions=sum(r['changed'] for r in self.guard_log),cached_plan_choices=sum(r['cached_plan'] for r in self.guard_log))


def check():
    for preset in PRESETS:
        env=CoverageTeacher(('north','S','left'),'normal',6202,preset,'centered');env.finish(until=2);env.advance_to(env.capture)
        snapshot=ForecastInput.capture(env);a=snapshot.forecast(ACTIONS);b=replay_plans(snapshot,a['proposed'])
        np.testing.assert_array_equal(a['poses'],b['poses']);np.testing.assert_array_equal(a['applied'],b['applied'])
        np.testing.assert_allclose(observed_residual(env),0,atol=1e-10,rtol=0)
        before=observed_residual(env);env.events=[dict(kind='push',t0=env.t+.01,duration=.1,lateral_cm=999,yaw_deg=90)]
        np.testing.assert_array_equal(before,observed_residual(env))
    # Recover an injected, already observed motion residual without reading events.
    env=CoverageTeacher(('east','W','left'),'normal',6202,'field_estimate','centered');env.finish(until=2)
    dt=env.physics[-1][0]-env.physics[-2][0];env.physics[-1][1]+=.02;env.physics[-1][3]+=.001
    np.testing.assert_allclose(observed_residual(env),[.02/dt,0,.001/dt],atol=1e-10,rtol=0)
    print('plan replay parity and causal motion residual checks passed',flush=True)


def freeze():
    OUT.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    dump(REPORT/'protocol.json',dict(source_sha256=sha(__file__),predictor_sha256=sha(ROOT/'cnn_finetune/goal_boundary_teacher.py'),
        hypotheses=['minimal-change projection avoids unnecessary stationary alignment','shifted prior plan preserves available continuation',
                    'already observed shove extrapolation improves response without future event access'],
        arms=['plan','observed_push'],keys=PILOT,seed=6202,presets=['field_estimate','sluggish'],conditions=['normal','mild'],
        horizon_s=.72,trigger_margin_cm=.05,safe_levels_cm=[.05,0],fallback_risk_weight=1000.,
        residual='last observed pose increment minus wheel-based increment; exponential world residual decay tau .2s',
        purpose='offline teacher, development-only; no global teacher replacement, no final exam',
        selection='fixed 16 cases per arm; no case exclusions or checkpoint tuning'))
    shutil.copy2(__file__,OUT/'source_snapshot.py')


def run(arm):
    protocol=json.loads((REPORT/'protocol.json').read_text());assert sha(__file__)==protocol['source_sha256']
    assert sha(ROOT/'cnn_finetune/goal_boundary_teacher.py')==protocol['predictor_sha256']
    old=json.loads((ROOT/'cnn_finetune/results/goal_teacher_coverage_20261113/coverage.json').read_text())['runs'];rows=[];started=time.perf_counter()
    for key in protocol['keys']:
        for preset in protocol['presets']:
            for condition in protocol['conditions']:
                dst=OUT/arm/('_'.join(key)+f'_{preset}_{condition}');dst.mkdir(parents=True,exist_ok=False)
                env=ProjectionTeacher(tuple(key),condition,protocol['seed'],preset,'centered',observed_push=arm=='observed_push').finish()
                result=env.summary();baseline=next(r for r in old if r['key']==key and r['preset']==preset and r['condition']==condition and r['seed']==protocol['seed'] and r['variant']=='centered')
                np.savez_compressed(dst/'physics.npz',samples=np.asarray(env.physics));dump(dst/'guard.json',env.guard_log);dump(dst/'result.json',result)
                rows.append(dict(result,artifact=str(dst),baseline=baseline));dump(REPORT/(arm+'_progress.json'),rows)
                print(arm,key,preset,condition,result['outcome'],'clean',result['clean_stable'],'margin',round(result['min_margin_cm'],3),
                      'elapsed',round(time.perf_counter()-started,1),flush=True)
    dump(REPORT/(arm+'.json'),rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['check','freeze','run']);p.add_argument('--arm',choices=['plan','observed_push']);a=p.parse_args()
    run(a.arm) if a.task=='run' else globals()[a.task]()
