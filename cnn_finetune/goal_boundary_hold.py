"""Hold the observed shove estimate through the actuator response interval.

No future event access. Holding is a conservative hypothesis, not knowledge of
when a shove ends. Retain every false alarm and resulting stop in evaluation.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from bench.world import wrap
from .goal_boundary_teacher import ForecastInput,ACTIONS,INTERVENTION,advance_batch
from .goal_boundary_teacher_v2 import observed_residual,replay_plans,PILOT
from .goal_boundary_commit import CommittedTeacher
from .goal_motion_prediction_audit import short_forecast
from .goal_teacher_probe import FastTeacher
from .goal_teacher_coverage import CoverageTeacher,PRESETS,ROOT
from .goal_development import sha
from .junction_stage02 import dump

OUT=Path('D:/autodrive_cnn/runs/goal_boundary_hold_20261113')
REPORT=ROOT/'cnn_finetune/results/goal_boundary_hold_20261113'


def replay_hold(snapshot,plans,residual):
    p=snapshot.p;n=plans.shape[1];pose=np.tile(snapshot.pose,(n,1));speed=np.tile(snapshot.wheel_speed,(n,1))
    error=np.tile(snapshot.error,(n,1));floor=snapshot.floor[None];t=snapshot.t
    commands=[(stamp,np.tile(u,(n,1))) for stamp,u in snapshot.commands]
    start=np.array([p.start_duty_left,p.start_duty_right]);gain=np.array([p.cm_s_per_duty_left,p.cm_s_per_duty_right])
    poses=[];times=[];outputs=[];hold=.036+p.actuation_delay_s
    def integral_to(elapsed):return min(elapsed,hold)+.2*(1-np.exp(-max(0,elapsed-hold)/.2))
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
            pose=advance_batch(pose,v,w,dt);integral=integral_to(t+dt-snapshot.t)-integral_to(t-snapshot.t)
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


class HeldMotionTeacher(CommittedTeacher):
    def teacher(self,observed,v,w):
        snapshot=ForecastInput.capture(self);base=FastTeacher.teacher(self,observed,v,w);residual=observed_residual(self)
        first=replay_hold(snapshot,snapshot.forecast(ACTIONS[:1])['proposed'],residual)
        chosen=0;forecast=first;cached=False;safe_level=.05
        if first['min_margin'][0]<.05:
            if self.plan is not None and len(self.plan)>6:
                continuation=replay_hold(snapshot,self.plan[1:,None],residual)
                if continuation['min_margin'][0]>=0:forecast=continuation;cached=True;safe_level=0.
            if not cached:
                plans=snapshot.forecast(ACTIONS)['proposed'];forecast=replay_hold(snapshot,plans,residual)
                cost=((plans[:INTERVENTION]-plans[:INTERVENTION,:1])**2).mean(axis=(0,2));safe=forecast['min_margin']>=.05
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


def check():
    for preset in PRESETS:
        env=CoverageTeacher(('north','S','left'),'normal',6202,preset,'centered');env.finish(until=2);env.advance_to(env.capture)
        s=ForecastInput.capture(env);plans=s.forecast(ACTIONS)['proposed']
        np.testing.assert_array_equal(replay_hold(s,plans,np.zeros(3))['poses'],replay_plans(s,plans,np.zeros(3))['poses'])
        residual=np.array([3.,1.,.1]);a=replay_hold(s,plans[:1,:1],residual)['poses'][-1,0]
        np.testing.assert_allclose(a,short_forecast(s,residual,True),atol=1e-10,rtol=0)
    a=CommittedTeacher(('north','S','left'),'normal',6202,'field_estimate','centered').finish(until=2)
    b=HeldMotionTeacher(('north','S','left'),'normal',6202,'field_estimate','centered').finish(until=2)
    np.testing.assert_array_equal(a.physics,b.physics)
    print('zero-residual exact parity, constant-motion integration, normal policy prefix checks passed',flush=True)


def freeze():
    OUT.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    dump(REPORT/'protocol.json',dict(source_sha256=sha(__file__),hypothesis='Observed shove must not immediately decay during the command response delay',
        change='hold current world-motion residual for .036+nominal actuator delay, then decay with tau .2s',
        unchanged='committed plan priority, nominal plant, objective, actions, reference, timing, scoring',
        keys=PILOT,seed=6202,presets=['field_estimate','sluggish'],conditions=['normal','mild'],
        future_information=False,final_exam_used=False,selection='fixed single candidate, all16 paired development cases'))
    shutil.copy2(__file__,OUT/'source_snapshot.py')


def run():
    protocol=json.loads((REPORT/'protocol.json').read_text());assert sha(__file__)==protocol['source_sha256']
    prior=json.loads((ROOT/'cnn_finetune/results/goal_boundary_commit_20261113/pilot.json').read_text());rows=[];started=time.perf_counter()
    for key in protocol['keys']:
        for preset in protocol['presets']:
            for condition in protocol['conditions']:
                dst=OUT/('_'.join(key)+f'_{preset}_{condition}');dst.mkdir(exist_ok=False)
                env=HeldMotionTeacher(tuple(key),condition,protocol['seed'],preset,'centered').finish();r=env.summary()
                baseline=next(x for x in prior if x['key']==key and x['preset']==preset and x['condition']==condition)
                np.savez_compressed(dst/'physics.npz',samples=np.asarray(env.physics));dump(dst/'guard.json',env.guard_log);dump(dst/'result.json',r)
                rows.append(dict(r,artifact=str(dst),previous_commit=baseline));dump(REPORT/'progress.json',rows)
                print('held motion',key,preset,condition,r['outcome'],'clean',r['clean_stable'],'margin',round(r['min_margin_cm'],3),
                      'elapsed',round(time.perf_counter()-started,1),flush=True)
    dump(REPORT/'pilot.json',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['check','freeze','run']);a=p.parse_args();globals()[a.task]()
