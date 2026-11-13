"""Causal nominal-plant footprint forecast for the OFFLINE teacher only.

ForecastInput contains current state, pending commands and nominal parameters.
It has no disturbance schedule, future random draws, camera, or actor access.
"""
import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import shutil
import time
import numpy as np

from bench.world import wrap
from .goal_teacher_coverage import CoverageTeacher,KEYS,PRESETS,ROOT
from .goal_teacher_probe import FastTeacher
from .junction_stage02 import dump
from .goal_development import sha

OUT=Path('D:/autodrive_cnn/runs/goal_boundary_teacher_20261113')
REPORT=ROOT/'cnn_finetune/results/goal_boundary_teacher_20261113'
ACTIONS=np.asarray([(1.,0.)]+[(s,d) for s,d in itertools.product([0.,.5,1.,1.2],[-.2,-.1,0.,.1,.2]) if (s,d)!=(1.,0.)])
HORIZON=20  # .72s, with exact 20ms/16ms integration and 36ms commands
INTERVENTION=4


def advance_batch(pose,v,w,dt):
    a=w*dt;distance=v*dt*np.sinc(a/(2*np.pi));out=pose.copy()
    out[:,0]+=distance*np.sin(pose[:,2]+a/2);out[:,1]+=distance*np.cos(pose[:,2]+a/2)
    out[:,2]=wrap(pose[:,2]+a);return out


@dataclass
class ForecastInput:
    p: object
    pose: np.ndarray
    wheel_speed: np.ndarray
    v: float
    w: float
    commands: list
    error: np.ndarray
    floor: np.ndarray
    t: float
    idx: int
    planned_v: float
    points: np.ndarray
    h: np.ndarray
    s: np.ndarray
    k: np.ndarray
    speed: np.ndarray
    mode: str
    controls: dict
    track: object
    rim: np.ndarray

    @classmethod
    def capture(cls,scene):
        return cls(scene.p,scene.car.pose.copy(),scene.car.speed.copy(),scene.car.v,scene.car.omega,
                   [(t,u.copy()) for t,u in scene.car.commands],scene.alloc.error.copy(),scene.alloc.floor.copy(),
                   scene.t,scene.idx,scene.planned_v,scene.points,scene.h,scene.s,scene.k,scene.speed,
                   scene.teacher_mode,dict(scene.controls),scene.track,scene.rim)

    def forecast(self,actions,horizon=HORIZON):
        p=self.p
        if p.pivot_slide_cm_per_rad!=0:raise ValueError('Pivot slide forecast not implemented; do not silently ignore it')
        n=len(actions);pose=np.tile(self.pose,(n,1));speed=np.tile(self.wheel_speed,(n,1))
        v=np.full(n,self.v);w=np.full(n,self.w);idx=np.full(n,self.idx,dtype=int);planned=np.full(n,self.planned_v)
        error=np.tile(self.error,(n,1));floor=self.floor[None]
        commands=[(t,np.tile(u,(n,1))) for t,u in self.commands]
        poses=[];times=[];applied=[];proposed=[];t=self.t
        start=np.array([p.start_duty_left,p.start_duty_right]);gain=np.array([p.cm_s_per_duty_left,p.cm_s_per_duty_right])
        for frame in range(horizon):
            predicted=advance_batch(pose,v,w,.036+p.actuation_delay_s+p.time_constant_s)
            candidates=np.minimum(idx[:,None]+np.arange(100)[None],len(self.points)-1)
            near=np.linalg.norm(self.points[candidates]-predicted[:,None,:2],axis=-1).argmin(axis=1)
            idx=candidates[np.arange(n),near]
            heading=self.h[idx];lateral=np.sum((predicted[:,:2]-self.points[idx])*np.c_[np.cos(heading),-np.sin(heading)],axis=1)
            target=self.speed[idx]*self.controls['speed_scale']
            planned=np.minimum(target,planned+.036) if self.mode=='planned' else target
            omega=planned*self.k[idx]+self.controls['heading_gain']*wrap(heading-predicted[:,2])-self.controls['lateral_gain']*lateral
            eff=np.where(omega<0,p.turn_eff_left,p.turn_eff_right);diff=omega*p.track_cm/(2*eff)
            u=np.c_[(planned+diff)/p.cm_s_per_duty_left,(planned-diff)/p.cm_s_per_duty_right]
            u/=np.maximum(1,np.max(abs(u),axis=1))[:,None]
            if frame<INTERVENTION:
                mean=u.mean(axis=1)*actions[:,0];delta=(u[:,0]-u[:,1])/2+actions[:,1]
                modified=np.clip(np.c_[mean+delta,mean-delta],-1,1)
                # Preserve exact baseline arithmetic.
                unchanged=(actions[:,0]==1)&(actions[:,1]==0);modified[unchanged]=u[unchanged];u=modified
            proposed.append(u.copy())
            target_t=t+.036
            while t<target_t-1e-10:
                dt=min(.02,target_t-t);due=t-p.actuation_delay_s
                active=next((cmd for stamp,cmd in reversed(commands) if stamp<=due),commands[0][1])
                mag=abs(active);duty=np.sign(active)*np.where(mag>=start-1e-9,np.minimum(mag,p.max_duty),0.)
                desired=duty*gain
                speed=speed+(desired-speed)*(1-np.exp(-dt/p.time_constant_s)) if p.time_constant_s>0 else desired.copy()
                v=speed.mean(axis=1);raw=(speed[:,0]-speed[:,1])/p.track_cm
                w=raw*np.where(raw<0,p.turn_eff_left,p.turn_eff_right)+p.drift_rad_s*v/5
                pose=advance_batch(pose,v,w,dt);t+=dt;poses.append(pose.copy());times.append(t)
            small=(abs(u)>0)&(abs(u)<floor)
            error=np.where(small,error+abs(u)/floor,0.)
            pulse=error>=.5;output=np.where(small,np.where(pulse,np.sign(u)*floor,0.),u)
            error=np.where(small&pulse,error-1,error);commands.append((t,output.copy()));applied.append(output)
        trace=np.asarray(poses);yaw=trace[:,:,2,None];co=np.cos(yaw);si=np.sin(yaw)
        points=np.stack([trace[:,:,0,None]+self.rim[None,None,:,0]*co+self.rim[None,None,:,1]*si,
                         trace[:,:,1,None]-self.rim[None,None,:,0]*si+self.rim[None,None,:,1]*co],axis=-1)
        margin=self.track.clearance(points).min(axis=-1)
        # Terminal alignment uses a local, forward reference window only.
        candidates=np.minimum(idx[:,None]+np.arange(100)[None],len(self.points)-1)
        near=np.linalg.norm(self.points[candidates]-pose[:,None,:2],axis=-1).argmin(axis=1);end=candidates[np.arange(n),near]
        alignment=np.sum((pose[:,:2]-self.points[end])**2,axis=1)+16*wrap(pose[:,2]-self.h[end])**2
        return dict(poses=trace,times=np.asarray(times),margin=margin,min_margin=margin.min(axis=0),
                    proposed=np.asarray(proposed),applied=np.asarray(applied),alignment=alignment,progress=self.s[end]-self.s[self.idx])


class BoundaryTeacher(CoverageTeacher):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.guard_log=[]
    def teacher(self,observed,v,w):
        snapshot=ForecastInput.capture(self)
        base=FastTeacher.teacher(self,observed,v,w)
        first=snapshot.forecast(ACTIONS[:1]);chosen=0;forecast=first
        if first['min_margin'][0]<.3:
            forecast=snapshot.forecast(ACTIONS)
            cost=forecast['alignment']+.05*(forecast['progress']-forecast['progress'][0])**2
            cost+=.2*((forecast['proposed'][0]-base)**2).sum(axis=1)
            safe=forecast['min_margin']>=0
            if safe.any():chosen=int(np.argmin(np.where(safe,cost,np.inf)))
            else:
                risk=np.maximum(0,.15-forecast['margin'])**2
                chosen=int(np.argmin(cost+100*risk.mean(axis=0)))
        np.testing.assert_allclose(forecast['proposed'][0,0],base,atol=1e-12,rtol=0)
        self.guard_log.append(dict(t=self.t,choice=chosen,baseline_predicted_margin=float(first['min_margin'][0]),
                                  chosen_predicted_margin=float(forecast['min_margin'][chosen])))
        return forecast['proposed'][0,chosen].copy()
    def summary(self):
        return dict(super().summary(),guard_calls=len(self.guard_log),guard_interventions=sum(r['choice']!=0 for r in self.guard_log))


def check():
    errors=[]
    for preset in PRESETS:
        for key in [('east','W','left'),('center','N','right')]:
            env=CoverageTeacher(key,'normal',6201,preset,'centered');env.finish(until=5)
            env.advance_to(env.capture);snapshot=ForecastInput.capture(env);prediction=snapshot.forecast(ACTIONS[:1])
            start=len(env.physics);cmd_start=len(env.records)
            for _ in range(HORIZON):env.step()
            actual=np.asarray(env.physics)[start:]
            error=float(np.max(abs(actual[:,1:4]-prediction['poses'][:,0])));errors.append(error)
            np.testing.assert_allclose(actual[:,0],prediction['times'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(actual[:,1:4],prediction['poses'][:,0],atol=1e-10,rtol=0)
            np.testing.assert_allclose([r['applied'] for r in env.records[cmd_start:]],prediction['applied'][:,0],atol=1e-12,rtol=0)
            before=snapshot.forecast(ACTIONS)
            env.events=[dict(kind='push',t0=env.t+.01,duration=.1,lateral_cm=999,yaw_deg=90)]
            after=snapshot.forecast(ACTIONS)
            np.testing.assert_array_equal(before['poses'],after['poses'])
    print('six nominal-plant forecast checks passed; max pose error',max(errors),flush=True)


def freeze():
    OUT.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    dump(REPORT/'protocol.json',dict(source_sha256=sha(__file__),horizon_s=HORIZON*.036,intervention_s=INTERVENTION*.036,
        actions=ACTIONS,trigger_margin_cm=.3,safe_candidate_margin_cm=0.,fallback='terminal alignment +100 mean hinge footprint risk',
        forecast='nominal plant, current actual pose/wheel speed/pending PWM/allocator state; no future disturbances or random values',
        purpose='offline teacher diagnostic only, not deployable actor or final exam',
        pilot_keys=[['north','S','left'],['east','W','left'],['center','E','right'],['south','N','right']],
        pilot_seed=6202,pilot_presets=['field_estimate','sluggish'],conditions=['normal','mild'],
        fixed_variant='centered',selection='single predefined candidate; retain every failure and compare with unchanged baseline'))
    shutil.copy2(__file__,OUT/'source_snapshot.py')


def run():
    protocol=json.loads((REPORT/'protocol.json').read_text());assert sha(__file__)==protocol['source_sha256'];rows=[];start=time.perf_counter()
    old=json.loads((ROOT/'cnn_finetune/results/goal_teacher_coverage_20261113/coverage.json').read_text())['runs']
    for key in protocol['pilot_keys']:
        for preset in protocol['pilot_presets']:
            for condition in protocol['conditions']:
                name='_'.join(key)+f'_{preset}_{condition}';dst=OUT/name;dst.mkdir(exist_ok=False)
                env=BoundaryTeacher(tuple(key),condition,protocol['pilot_seed'],preset,'centered').finish()
                result=env.summary();baseline=next(r for r in old if r['key']==key and r['preset']==preset and r['condition']==condition and r['seed']==protocol['pilot_seed'] and r['variant']=='centered')
                np.savez_compressed(dst/'physics.npz',samples=np.asarray(env.physics));dump(dst/'guard.json',env.guard_log)
                dump(dst/'result.json',result);rows.append(dict(result,artifact=str(dst),baseline=baseline))
                dump(REPORT/'progress.json',rows)
                print('boundary teacher',name,'clean',result['clean_stable'],'margin',round(result['min_margin_cm'],3),
                      'guards',result['guard_interventions'],'wall',round(time.perf_counter()-start,1),flush=True)
    dump(REPORT/'pilot.json',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['check','freeze','run']);a=p.parse_args();globals()[a.task]()
