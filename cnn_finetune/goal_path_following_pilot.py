"""Small known-path/visual-state feasibility experiment; no neural training.

Three arms share a controller and nominal motor model. Only the observation
source changes. Truth is confined to environment, scoring and named controls.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
from bench.world import Camera, Renderer, advance, wrap
from bench.realcam import RealCam, pose_from_history
from bench.junction import heading_of
from .goal_clearance_paths import ReserveTeacher, REPORT as PATH_REPORT
from .goal_continuous_clearance import continuous_clearance
from .goal_mission_map import DelayedImageMap
from .goal_development import ROOT, sha
from .junction_learning import setup
from .junction_stage02 import dump

OUT=Path('D:/autodrive_cnn/runs/goal_path_following_pilot_20261113')
REPORT=ROOT/'cnn_finetune/results/goal_path_following_pilot_20261113'
KEYS=[('north','E','left'),('north','W','right'),('center','N','right'),('east','W','right')]
SEEDS=[22001,22002]
PRESETS=['field_estimate','sluggish','unequal_sides']
ARMS=['truth_now','truth_delayed','vision']
CONDITIONS=['normal','mild','mild_light']


class NominalMotion:
    def __init__(self):
        self.t=0.;self.v=0.;self.w=0.;self.commands=[(-1e9,np.zeros(2))]
    def applied(self,t,pwm):self.commands.append((t,np.asarray(pwm).copy()))
    def at(self,t):
        while self.t<t-1e-10:
            dt=min(.01,t-self.t);pwm=self.commands[0][1]
            for stamp,command in self.commands:
                if stamp<=self.t-.05:pwm=command
            wheel=np.sign(pwm)*np.where(abs(pwm)>=.30,abs(pwm),0)*16.8
            a=1-np.exp(-dt/.10);self.v+=a*(wheel.mean()-self.v)
            self.w+=a*((wheel[0]-wheel[1])*.6/13-self.w);self.t+=dt
        return self.v,self.w


class PathController:
    """Fixed geometry feedback. No environment object or actual preset accepted."""
    def __init__(self,points,heading,curvature,speed,mode):
        self.points=points.copy();self.h=heading.copy();self.k=curvature.copy();self.speed=speed.copy()
        self.mode=mode;self.idx=0;self.planned_v=0.;self.motion=NominalMotion()
    def step(self,pose,t):
        v,w=self.motion.at(t);pred=advance(np.asarray(pose),v,w,.15)
        near=self.idx+int(np.argmin(np.linalg.norm(self.points[self.idx:min(len(self.points),self.idx+100)]-pred[:2],axis=1)))
        self.idx=max(self.idx-2,near);h=self.h[self.idx]
        lat=float((pred[:2]-self.points[self.idx])@np.array([np.cos(h),-np.sin(h)]))
        head=float(wrap(h-pred[2]));target=float(self.speed[self.idx])
        if self.mode=='planned':self.planned_v=min(target,self.planned_v+.036);target=self.planned_v
        omega=target*self.k[self.idx]+1.3*head-.08*lat
        diff=omega*13/(2*.6);desired=np.array([target+diff,target-diff])/16.8
        desired/=max(1,float(max(abs(desired))))
        return desired,dict(index=self.idx,lateral_cm=lat,heading_rad=head,nominal_v=v,nominal_w=w)
    def applied(self,t,pwm):self.motion.applied(t,pwm)


class DelayedTruth:
    """Diagnostic perfect capture-pose measurements, same 70ms/3frame timing.

    Replay nominal signed commands exactly like the image-map prediction;
    no true motor parameters, velocity, or future disturbances are accepted.
    """
    def __init__(self,hint):
        self.pose=hint.copy();self.stamp=0.;self.pending=[];self.commands=[(-1e9,0.,0.)];self.n=0
    def observe(self,pose,capture,now):
        if self.n%3==0:self.pending.append((capture+.070,capture,pose.copy()))
        self.n+=1
        while self.pending and self.pending[0][0]<=now+1e-10:
            _,self.stamp,self.pose=self.pending.pop(0)
        result=self.pose.copy();cuts=[self.stamp]+[x[0] for x in self.commands if self.stamp<x[0]<now]+[now]
        for a,b in zip(cuts,cuts[1:]):
            eligible=[x for x in self.commands if x[0]<=a];_,v,w=eligible[-1]
            result=advance(result,v,w,b-a)
        return result
    def applied(self,t,pwm):
        wheel=np.sign(pwm)*np.where(abs(pwm)>=.28,abs(pwm),0)*16.
        raw=(wheel[0]-wheel[1])/13;self.commands.append((t,float(wheel.mean()),float(raw*(.63 if raw<0 else .62))))


def controller_checks():
    # Symmetric signed steering and causal 70ms measurement publication.
    pts=np.c_[np.zeros(401),np.linspace(-45,55,401)]
    make=lambda:PathController(pts,np.zeros(401),np.zeros(401),np.full(401,5.),'legacy')
    straight,_=make().step(np.array([0.,-45.,0.]),.069)
    right,_=make().step(np.array([1.,-45.,0.]),.069)
    left,_=make().step(np.array([-1.,-45.,0.]),.069)
    assert straight[0]==straight[1] and right[0]<right[1] and left[0]>left[1]
    d=DelayedTruth(np.zeros(3));measurement=np.array([1.,2.,.1])
    np.testing.assert_array_equal(d.observe(measurement,.033,.069),np.zeros(3))
    np.testing.assert_allclose(d.observe(measurement,.069,.105),measurement,atol=1e-12,rtol=0)
    return dict(steering_signs=True,no_early_truth_publication=True)


def freeze():
    assert not (REPORT/'protocol.json').exists();checked=controller_checks()
    OUT.mkdir(exist_ok=True);REPORT.mkdir(exist_ok=True)
    files={Path(__file__).resolve(),PATH_REPORT/'geometry.json'}
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and file.endswith('.py') and Path(file).is_file() and Path(file).resolve().is_relative_to(ROOT):files.add(Path(file).resolve())
    vendor=ROOT/'bench/_vendor/v4_65c68a6613a0'
    files.update(p for p in vendor.rglob('*') if p.is_file() and p.suffix in ['.py','.json','.npz'])
    for row in json.loads((PATH_REPORT/'geometry.json').read_text()):
        if tuple(row['key']) in KEYS:files.add(Path(row['reference']))
    dump(REPORT/'protocol.json',dict(source_sha256=sha(__file__),hashes={str(f):sha(f) for f in sorted(files)},
        keys=KEYS,seeds=SEEDS,presets=PRESETS,arms=ARMS,conditions=CONDITIONS,checks=checked,
        purpose='Known smooth paths and existing image-map observer feasibility; no CNN learning or full mission claim',
        controller='same nominal16.8cm/s/duty,track13cm,turneff.6,lag.10s,delay.05s; FF curvature +1.3heading-.08lateral; .15s nominal prediction',
        observer='pinned v4 map on raw RealCam,70ms delayed snapshots every3frames; nominal unperturbed path-start hint; known path/map allowed in this diagnostic',
        timing='30Hz maximum,36ms command latency,.02s physics; current truth arm ideal; delayed truth shares70ms/3frame observation schedule',
        nominal_gate='Both truth controls and vision: >=22/24 continuous-clean;24/24 stable;0 wrong/severe;0 unnecessary pauses. Otherwise no compound expansion and no large training.',
        compound_gate='If nominal passes: each mild and mild_light >=21/24clean,0wrong/severe; vision within2 successes of delayed truth. Development feasibility only, not final acceptance.',
        diagnostics='If nominal fails, finish all24 per arm then analyze observation error before first body incursion; no gain sweeps on evaluation seeds.',
        input_boundary='vision controller accepts estimated pose and fixed path only; true pose used solely in diagnostic arms, rendering and scoring; no actual preset supplied to controller',
        final_exam_used=False,new_training=False))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print('frozen',flush=True)


def run_one(key,seed,preset,condition,arm):
    registry={tuple(r['key']):r for r in json.loads((PATH_REPORT/'geometry.json').read_text())}
    env=ReserveTeacher(key,condition,seed,preset,'same_mode',registry)
    control=PathController(env.points,env.h,env.k,env.speed,env.teacher_mode)
    hint=np.r_[env.points[0],heading_of(env.di)]
    observer=DelayedImageMap('cw',hint) if arm=='vision' else DelayedTruth(hint) if arm=='truth_delayed' else None
    camera=None
    if arm=='vision':
        observer.set_route(env.points);cam=Camera();camera=RealCam(Renderer(env.track,cam),cam,env.profile,seed)
    records=[];images=[];initial=env.initial_pose.copy()
    while env.capture<40 and env.outcome=='time_limit':
        env.advance_to(env.capture)
        if env.outcome!='time_limit':break
        capture=env.capture;done=capture+.036;capture_pose=env.car.pose.copy()
        image=None if camera is None else camera.frame(capture,pose_from_history(env.history),raw=True)
        env.advance_to(done)
        if env.outcome!='time_limit':break
        extra={}
        if arm=='vision':
            estimate=observer.observe(image,capture,done);pose=estimate['pose']
            d=observer.visible
            extra=dict(sxy=estimate['sxy'],map_status=d.map.status,map_capture=d.map.t,
                latest_observation=d.map.last_obs_t,matched=d.map.status in ['observed','relocalized'])
            if len(records)%28==0:images.append((len(records),image.copy()))
        elif arm=='truth_delayed':pose=observer.observe(capture_pose,capture,done)
        else:pose=env.car.pose.copy()
        desired,info=control.step(pose,done)
        applied=env.alloc.apply(desired);env.car.command(done,applied);control.applied(done,applied)
        if arm=='vision':observer.on_applied(done,applied)
        elif arm=='truth_delayed':observer.applied(done,applied)
        records.append(dict(capture=capture,done=done,estimate=pose.copy(),truth=env.car.pose.copy(),
            desired=desired,applied=applied,**info,**extra))
        # Environment records used by summary only; no supervised labels.
        env.records.append(dict(capture=capture,done=done,action=desired,applied=applied))
        env.previous=applied.copy();env.capture=max(capture+1/30,done)
    trace=np.asarray(env.physics);m=continuous_clearance(np.vstack([initial,trace[:,1:4]]),env.p,env.track)
    r=env.summary();r.update(arm=arm,continuous_min_clearance_cm=float(m.min()),
        continuous_clean=bool(r['stable_success'] and r['initial_valid'] and m.min()>=-1e-9),
        hint=hint.tolist(),controller_true_preset=False)
    estimated=np.array([x['estimate'] for x in records]);truth=np.array([x['truth'] for x in records])
    xy=np.linalg.norm(estimated[:,:2]-truth[:,:2],axis=1);yaw=abs(np.degrees(wrap(estimated[:,2]-truth[:,2])))
    first=np.where(m<0)[0];first_t=None if not len(first) else float(0 if first[0]==0 else trace[first[0]-1,0])
    before=np.array([x['done'] for x in records])<= (first_t if first_t is not None else env.t)
    r.update(position_error_p95_cm=float(np.percentile(xy,95)),heading_error_p95_deg=float(np.percentile(yaw,95)),
        first_incursion_s=first_t,pre_incursion_position_max_cm=float(xy[before].max()) if before.any() else None,
        pre_incursion_heading_max_deg=float(yaw[before].max()) if before.any() else None)
    if arm=='vision':r['matched_frame_fraction']=float(np.mean([x['matched'] for x in records]))
    dst=OUT/arm/('_'.join(key)+f'_{seed}_{preset}_{condition}');dst.mkdir(parents=True,exist_ok=False)
    dump(dst/'records.json',records);dump(dst/'events.json',env.events)
    np.savez_compressed(dst/'physics.npz',samples=trace,initial=initial,continuous_clearance=m)
    if images:np.savez_compressed(dst/'frames.npz',indices=np.array([x[0] for x in images]),images=np.array([x[1] for x in images]))
    r['artifact']=str(dst);dump(dst/'complete.json',r);return r


def run(arm,condition):
    setup();p=json.loads((REPORT/'protocol.json').read_text());assert sha(__file__)==p['source_sha256']
    for file,h in p['hashes'].items():assert sha(file)==h,file
    if condition!='normal':assert json.loads((REPORT/'normal_summary.json').read_text())['expand_compound']
    started=time.perf_counter();rows=[]
    for seed in SEEDS:
        for key in KEYS:
            for preset in PRESETS:
                dst=OUT/arm/('_'.join(key)+f'_{seed}_{preset}_{condition}')
                if (dst/'complete.json').exists():r=json.loads((dst/'complete.json').read_text())
                else:r=run_one(key,seed,preset,condition,arm)
                rows.append(r);dump(REPORT/f'{arm}_{condition}_progress.json',rows)
                print(arm,condition,len(rows),'/24',key,preset,r['outcome'],'clean',r['continuous_clean'],
                    'pose95',round(r['position_error_p95_cm'],2),'seconds',round(time.perf_counter()-started,1),flush=True)
    dump(REPORT/f'{arm}_{condition}.json',rows)


def summarize(condition):
    data={a:json.loads((REPORT/f'{a}_{condition}.json').read_text()) for a in ARMS};result={}
    for arm,rows in data.items():
        assert len(rows)==24
        result[arm]=dict(n=24,clean=sum(r['continuous_clean'] for r in rows),stable=sum(r['stable_success'] for r in rows),
            wrong=sum(r['outcome']=='wrong_exit' for r in rows),severe=sum(r['outcome']=='severe_departure' for r in rows),
            pauses=sum(r['translation_pause_s']>0 for r in rows),position_error_p95_median=float(np.median([r['position_error_p95_cm'] for r in rows])),
            median_time=float(np.median([r['simulated_s'] for r in rows])))
    for index in range(24):
        ref=data[ARMS[0]][index]
        for arm in ARMS[1:]:
            row=data[arm][index]
            assert all(row[k]==ref[k] for k in ['key','seed','preset','condition','initial_sampling'])
            assert json.loads((Path(row['artifact'])/'events.json').read_text())==json.loads((Path(ref['artifact'])/'events.json').read_text())
    gate=all(x['clean']>=22 and x['stable']==24 and x['wrong']==0 and x['severe']==0 and x['pauses']==0 for x in result.values())
    dump(REPORT/f'{condition}_summary.json',dict(summary=result,expand_compound=gate if condition=='normal' else None,
        paired_initials_events_exact=True,final_exam_used=False,new_training=False))
    print(json.dumps(result), 'expand',gate,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['freeze','run','summarize']);p.add_argument('--arm',choices=ARMS);p.add_argument('--condition',choices=CONDITIONS,default='normal');a=p.parse_args()
    if a.task=='freeze':freeze()
    elif a.task=='run':run(a.arm,a.condition)
    else:summarize(a.condition)
