"""Causal-input feedback forecast for two-stage offline recovery decisions.

The predictor takes only ForecastInput and an already observed motion residual.
Unlike old generate-then-replay forecasts, future nominal teacher commands
react to each predicted disturbed pose. No actual event schedule is read.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import time
import numpy as np
from bench.world import wrap
from .goal_boundary_teacher import ForecastInput,advance_batch
from .goal_boundary_teacher_v2 import observed_residual
from .goal_entry_recovery_search import actions,capture_root,short_branch
from .goal_phase_disturbance import REPORT as PHASE_REPORT,registry
from .goal_clearance_paths import ReserveTeacher
from .goal_continuous_clearance import continuous_clearance
from .goal_development import ROOT,sha
from .goal_teacher_coverage import PRESETS
from .junction_stage02 import dump

OUT=Path('D:/autodrive_cnn/runs/goal_entry_forecast_20261113')
REPORT=ROOT/'cnn_finetune/results/goal_entry_forecast_20261113'


def forecast(s,programs,residual,profile,horizon=60):
    p=s.p
    if p.pivot_slide_cm_per_rad!=0:raise ValueError('unsupported pivot slide')
    n=len(programs);pose=np.tile(s.pose,(n,1));speed=np.tile(s.wheel_speed,(n,1))
    v=np.full(n,s.v);w=np.full(n,s.w);idx=np.full(n,s.idx,dtype=int);planned=np.full(n,s.planned_v)
    error=np.tile(s.error,(n,1));floor=s.floor[None];t=s.t
    commands=[(t,np.tile(u,(n,1))) for t,u in s.commands]
    start=np.array([p.start_duty_left,p.start_duty_right]);gain=np.array([p.cm_s_per_duty_left,p.cm_s_per_duty_right])
    lateral=residual[:2]@np.array([np.cos(s.pose[2]),-np.sin(s.pose[2])])
    longitudinal=residual[:2]@np.array([np.sin(s.pose[2]),np.cos(s.pose[2])])
    def integral(elapsed):
        if profile=='constant':return elapsed
        if profile=='hold035':return min(elapsed,.35)
        if profile=='decay':
            hold=.036+p.actuation_delay_s
            return min(elapsed,hold)+.2*(1-np.exp(-max(0,elapsed-hold)/.2))
        raise ValueError(profile)
    poses=[];times=[];proposed=[];outputs=[];velocities=[]
    for frame in range(horizon):
        predicted=advance_batch(pose,v,w,.036+p.actuation_delay_s+p.time_constant_s)
        candidates=np.minimum(idx[:,None]+np.arange(100)[None],len(s.points)-1)
        near=np.linalg.norm(s.points[candidates]-predicted[:,None,:2],axis=-1).argmin(axis=1)
        idx=candidates[np.arange(n),near];heading=s.h[idx]
        lat=np.sum((predicted[:,:2]-s.points[idx])*np.c_[np.cos(heading),-np.sin(heading)],axis=1)
        target=s.speed[idx]*s.controls['speed_scale'];planned=np.minimum(target,planned+.036) if s.mode=='planned' else target
        omega=planned*s.k[idx]+s.controls['heading_gain']*wrap(heading-predicted[:,2])-s.controls['lateral_gain']*lat
        eff=np.where(omega<0,p.turn_eff_left,p.turn_eff_right);diff=omega*p.track_cm/(2*eff)
        u=np.c_[(planned+diff)/p.cm_s_per_duty_left,(planned-diff)/p.cm_s_per_duty_right]
        u/=np.maximum(1,np.max(abs(u),axis=1))[:,None]
        if frame<14:
            a=programs[:,:2] if frame<7 else programs[:,2:]
            mean=u.mean(axis=1)*a[:,0];delta=(u[:,0]-u[:,1])/2+a[:,1]
            modified=np.clip(np.c_[mean+delta,mean-delta],-1,1)
            unchanged=(a[:,0]==1)&(a[:,1]==0);modified[unchanged]=u[unchanged];u=modified
        proposed.append(u.copy());target_t=t+.036
        while t<target_t-1e-10:
            dt=min(.02,target_t-t);due=t-p.actuation_delay_s
            active=next((cmd for stamp,cmd in reversed(commands) if stamp<=due),commands[0][1])
            mag=abs(active);duty=np.sign(active)*np.where(mag>=start-1e-9,np.minimum(mag,p.max_duty),0.)
            desired=duty*gain;speed=speed+(desired-speed)*(1-np.exp(-dt/p.time_constant_s)) if p.time_constant_s>0 else desired.copy()
            v=speed.mean(axis=1);raw=(speed[:,0]-speed[:,1])/p.track_cm
            w=raw*np.where(raw<0,p.turn_eff_left,p.turn_eff_right)+p.drift_rad_s*v/5
            pose=advance_batch(pose,v,w,dt);amount=integral(t+dt-s.t)-integral(t-s.t)
            co=np.cos(pose[:,2]);si=np.sin(pose[:,2])
            pose[:,0]+=(lateral*co+longitudinal*si)*amount
            pose[:,1]+=(-lateral*si+longitudinal*co)*amount
            pose[:,2]=wrap(pose[:,2]+residual[2]*amount)
            t+=dt;poses.append(pose.copy());times.append(t);velocities.append(v.copy())
        small=(abs(u)>0)&(abs(u)<floor);error=np.where(small,error+abs(u)/floor,0.);pulse=error>=.5
        out=np.where(small,np.where(pulse,np.sign(u)*floor,0.),u)
        error=np.where(small&pulse,error-1,error);commands.append((t,out.copy()));outputs.append(out)
    trace=np.asarray(poses)
    margin=continuous_clearance(trace.reshape(-1,3),p,s.track).reshape(len(trace),n)
    return dict(poses=trace,times=np.asarray(times),margin=margin,min_margin=margin.min(0),
        proposed=np.asarray(proposed),applied=np.asarray(outputs),velocities=np.asarray(velocities))


def choose(prediction,programs):
    if prediction['min_margin'][0]>=.01:return 0
    moving=prediction['velocities'].min(0)>=.5
    safe=(prediction['min_margin']>=.01)&moving
    cost=.02*((programs[:,[0,2]]-1)**2).sum(1)+(programs[:,[1,3]]**2).sum(1)
    if safe.any():return int(np.argmin(np.where(safe,cost,np.inf)))
    risk=np.maximum(0,.05-prediction['margin'])**2
    return int(np.argmin(risk.mean(0)+.001*cost))


def check():
    rows=[];reg=registry();aa=actions()[[0,17,40]]
    for key in [('east','W','left'),('center','N','right')]:
        for preset in PRESETS:
            for pushed in [False,True]:
                env=ReserveTeacher(key,'normal',6501,preset,'same_mode',reg).finish(until=5);env.advance_to(env.capture)
                env.trigger=None;env.event_template=[];s=ForecastInput.capture(env)
                rate=2. if pushed else 0.;yaw=.04 if pushed else 0.
                residual=np.array([rate*np.cos(s.pose[2]),-rate*np.sin(s.pose[2]),yaw])
                pred=forecast(s,aa,residual,'constant');start=len(env.physics)-1
                if pushed:env.events=[dict(kind='push',t0=env.t,duration=10.,lateral_cm=rate*10,yaw_deg=float(np.degrees(yaw)*10))]
                max_error=0.
                for i,a in enumerate(aa):
                    branch=short_branch(env,a);trace=np.asarray(branch.physics)[start+1:]
                    np.testing.assert_allclose(trace[:,1:4],pred['poses'][:,i],atol=1e-8,rtol=0)
                    np.testing.assert_allclose([r['applied'] for r in branch.records[len(env.records):]],pred['applied'][:,i],atol=1e-10,rtol=0)
                    max_error=max(max_error,float(np.max(abs(trace[:,1:4]-pred['poses'][:,i]))))
                rows.append(dict(key=key,preset=preset,constant_push=pushed,programs=3,max_pose_error=max_error))
    return rows


def freeze():
    checked=check();OUT.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    p=json.loads((PHASE_REPORT/'protocol.json').read_text())
    dump(REPORT/'protocol.json',dict(source_sha256=sha(__file__),dependencies=p['dependencies'],
        phase_source_sha256=p['source_sha256'],search_source_sha256=sha(ROOT/'cnn_finetune/goal_entry_recovery_search.py'),
        source_rows_sha256=sha(PHASE_REPORT/'reserve.json'),profiles=['decay','hold035'],actions=actions(),
        cases='all240 reserve phase-entry cases, including147 previously clean; preserve denominator',
        forecast='nominal plant plus currently measured body-frame residual; future teacher reacts to predicted pose; no future events or random values',
        root='first observed motion after actual shove, selected offline; not a deployable event detector or complete causal policy',
        decision='baseline if predicted margin>=.01; else safe moving smallest command modification; otherwise mean squared hinge risk+.001 action cost',
        temporal_scope='one decision, two7-command adjustments then unchanged teacher through55cm exit',
        checks=checked,final_exam_used=False,new_training=False))
    shutil.copy2(__file__,OUT/'source_snapshot.py');print('36 forecast/plant parity cases passed',flush=True)


def run_one(item):
    index,r,profile=item;root,_=capture_root(dict(source=r,root_mode='post_observation'))
    residual=observed_residual(root);pred=forecast(ForecastInput.capture(root),actions(),residual,profile)
    selected=choose(pred,actions());dst=OUT/profile/f'{index:03d}';dst.mkdir(parents=True,exist_ok=False)
    # Write prediction/decision before realized continuation is evaluated.
    decision=dict(index=selected,action=actions()[selected],residual=residual,min_margins=pred['min_margin'],root_s=root.t)
    dump(dst/'decision.json',decision)
    branch=short_branch(root,actions()[selected]);branch.finish();trace=np.asarray(branch.physics)
    margin=continuous_clearance(trace[:,1:4],branch.p,branch.track);result=branch.summary()
    result.update(continuous_min_clearance_cm=float(margin.min()),continuous_clean=bool(result['stable_success'] and result['initial_valid'] and margin.min()>=-1e-9))
    result['clean_moving']=bool(result['continuous_clean'] and result['translation_pause_s']==0 and trace[:,4].min()>=-.01)
    np.savez_compressed(dst/'physics.npz',physics=trace);row=dict(source=r,result=result,decision=decision,profile=profile,index=index,artifact=str(dst))
    dump(dst/'result.json',row);return row


def run(profile):
    p=json.loads((REPORT/'protocol.json').read_text());assert sha(__file__)==p['source_sha256']
    for f,h in p['dependencies'].items():assert sha(f)==h,f
    assert sha(ROOT/'cnn_finetune/goal_phase_disturbance.py')==p['phase_source_sha256']
    assert sha(ROOT/'cnn_finetune/goal_entry_recovery_search.py')==p['search_source_sha256']
    assert sha(PHASE_REPORT/'reserve.json')==p['source_rows_sha256']
    cases=[r for r in json.loads((PHASE_REPORT/'reserve.json').read_text()) if r['phase']=='entry'];assert len(cases)==240
    rows=[];started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=3) as pool:
        for row in pool.map(run_one,[(i,r,profile) for i,r in enumerate(cases)]):
            rows.append(row)
            if len(rows)%20==0:
                dump(REPORT/(profile+'_progress.json'),rows)
                print('root prediction',profile,len(rows),'clean',sum(r['result']['continuous_clean'] for r in rows),'elapsed',round(time.perf_counter()-started,1),flush=True)
    summary=dict(n=len(rows),baseline_clean=sum(r['source']['continuous_clean'] for r in rows),clean=sum(r['result']['continuous_clean'] for r in rows),
        clean_moving=sum(r['result']['clean_moving'] for r in rows),
        gains=sum(r['result']['continuous_clean'] and not r['source']['continuous_clean'] for r in rows),
        losses=sum(r['source']['continuous_clean'] and not r['result']['continuous_clean'] for r in rows),
        interventions=sum(r['decision']['index']!=0 for r in rows))
    dump(REPORT/(profile+'.json'),dict(summary=summary,rows=rows));print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['freeze','run']);p.add_argument('--profile',choices=['decay','hold035']);a=p.parse_args()
    run(a.profile) if a.task=='run' else globals()[a.task]()
