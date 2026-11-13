"""Cornering refinement: bounded-clearance geometry, spatial speed planning, paired physics.

Privileged offline teacher only. No image policy, hardware, or model training.
All previous experiment files are read-only inputs.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from bench.world import Track, to_world, wrap, advance
from bench.vehicle import Vehicle, PRESETS, VARIATIONS
from bench.junction import movement, heading_of, ARM
from cnn_finetune.junction_feasibility import SignedAllocator, events_for
from cnn_finetune.junction_stage02 import OUT as OLD, REPORT as OLD_REPORT, ROOT, DEV, dump, metrics

OUT=Path('D:/autodrive_cnn/runs/junction_cornering_20261113')
REPORT=ROOT/'cnn_finetune/results/junction_cornering_20261113'
OLD_SELECTION=json.loads((OLD_REPORT/'selection.json').read_text())
KEYS=[tuple(r['key']) for r in OLD_SELECTION if r['key'][2]!='straight']
PRESETS_USED=['field_estimate','sluggish','unequal_sides','field_pivot_slip']


def old_speed(key):return next(r['speed'] for r in OLD_SELECTION if r['key']==list(key))


def curve(ctrl,n):
    """Bezier position, tangent and curvature using analytic derivatives."""
    ctrl=np.asarray(ctrl);u=np.linspace(0,1,n)[:,None]
    def evaluate(p):
        order=len(p)-1
        return sum(math.comb(order,i)*u**i*(1-u)**(order-i)*q for i,q in enumerate(p))
    order=len(ctrl)-1;p=evaluate(ctrl);d=evaluate(np.diff(ctrl,axis=0)*order)
    dd=evaluate(np.diff(ctrl,n=2,axis=0)*order*(order-1))
    h=np.arctan2(d[:,0],d[:,1]);k=(d[:,1]*dd[:,0]-d[:,0]*dd[:,1])/np.maximum(np.linalg.norm(d,axis=1)**3,1e-9)
    return p,h,k


def path_for(key,x,dense=False):
    c,di,do,_=movement(*key);a,b,h1,h2,oi,oo=x
    ni=np.array([di[1],-di[0]]);no=np.array([do[1],-do[0]])
    start=c-45*di;end=c+55*do;p0=c-a*di+oi*ni;p5=c+b*do+oo*no
    # Collinear first/last three controls make all segment joins curvature-continuous.
    pre=np.array([start,start+(45-a)/5*di,start+2*(45-a)/5*di,p0-2*(45-a)/5*di,p0-(45-a)/5*di,p0])
    core=np.array([p0,p0+h1*di,p0+2*h1*di,p5-2*h2*do,p5-h2*do,p5])
    post=np.array([p5,p5+(55-b)/5*do,p5+2*(55-b)/5*do,end-2*(55-b)/5*do,end-(55-b)/5*do,end])
    segments=[curve(z,501 if dense else 61) for z in [pre,core,post]]
    points=np.vstack([z[0][:-1] for z in segments]+[segments[-1][0][-1:]])
    h=np.concatenate([z[1][:-1] for z in segments]+[segments[-1][1][-1:]])
    k=np.concatenate([z[2][:-1] for z in segments]+[segments[-1][2][-1:]])
    s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    return points,h,k,s


def clearance(points,h,rim,track):
    co=np.cos(h)[:,None];si=np.sin(h)[:,None]
    world=np.stack([points[:,0,None]+rim[None,:,0]*co+rim[None,:,1]*si,
                    points[:,1,None]-rim[None,:,0]*si+rim[None,:,1]*co],axis=-1)
    return track.clearance(world).min(axis=1)


def shape_metrics(k,s):
    dk=np.gradient(k,s)
    return dict(max_abs_curvature_cm_inv=float(max(abs(k))),min_radius_cm=float(1/max(abs(k))),
                curvature_variation=float(np.sum(abs(np.diff(k)))),
                curvature_derivative_energy=float(np.trapz(dk*dk,s)))


def search():
    if (REPORT/'geometry.json').exists():raise FileExistsError('Completed geometry is immutable')
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    dump(REPORT/'protocol.json',dict(scope='24 turning movements; center left experimental, center right mission allowed',
         development_seeds=DEV,geometry_seed=314159,final_test_used=False,
         clearance_targets_cm=[.5,1.0],selection='field/sluggish x fixed/mild x seed1303, then freeze; no final test',
         speed_plan=dict(max_yaw_rate_rad_s=.45,lateral_acceleration_cm_s2=1.5,accel_cm_s2=1.0,decel_cm_s2=1.0),
         previous_protocol_sha256=hashlib.sha256((OLD_REPORT/'contract.json').read_bytes()).hexdigest(),
         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         release_gate='No reduction of fixed/mild stable or clean counts; nominal clean+moving; lower paired angular acceleration; lower max reference curvature',
         lighting='No light robustness claim for privileged-state teacher; image-only student remains untrained'))
    track=Track();rim=Vehicle(PRESETS['field_estimate'],np.zeros(3)).footprint(2);rows=[]
    for key in KEYS:
        name='_'.join(key);dst=OUT/'paths'/name;dst.mkdir(parents=True,exist_ok=True)
        old=np.load(OLD/'paths'/name/'path.npz');p=old['points'];h=old['heading']
        s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))];k=np.gradient(np.unwrap(h),s)
        np.savez_compressed(dst/'old.npz',points=p,heading=h,curvature=k,s=s)
        for target in [.5,1.]:
            def objective(x):
                p,h,k,s=path_for(key,x);m=clearance(p,h,rim,track)
                shape=shape_metrics(k,s)
                violations=max(0,target-float(m.min()))
                # Once required clearance is satisfied, prefer a broad and gradual turn.
                return (1000*violations**2+12*shape['max_abs_curvature_cm_inv']
                        +60*shape['curvature_derivative_energy']+.6*shape['curvature_variation'])
            opt=differential_evolution(objective,[(6,30),(6,30),(1,13),(1,13),(-2,2),(-2,2)],
                seed=314159,popsize=7,maxiter=70,tol=.001,polish=False)
            p,h,k,s=path_for(key,opt.x,True)
            m=clearance(p,h,Vehicle(PRESETS['field_estimate'],np.zeros(3)).footprint(.5),track)
            # Uniform arc length for identical nearest-reference index semantics.
            grid=np.linspace(0,s[-1],int(np.ceil(s[-1]/.25))+1)
            pp=np.c_[np.interp(grid,s,p[:,0]),np.interp(grid,s,p[:,1])]
            hh=np.interp(grid,s,np.unwrap(h));kk=np.interp(grid,s,k)
            label='wide' if target==.5 else 'reserve'
            np.savez_compressed(dst/(label+'.npz'),points=pp,heading=hh,curvature=kk,s=grid)
            row=dict(key=key,candidate=label,clearance_target_cm=target,static_min_margin_cm=float(m.min()),
                     parameters=opt.x,evaluations=opt.nfev,old_shape=shape_metrics(np.gradient(np.unwrap(old['heading']),
                         np.r_[0,np.cumsum(np.linalg.norm(np.diff(old['points'],axis=0),axis=1))]),
                         np.r_[0,np.cumsum(np.linalg.norm(np.diff(old['points'],axis=0),axis=1))]),**shape_metrics(k,s))
            rows.append(row);dump(dst/(label+'.json'),row)
            print(name,label,'radius',round(row['min_radius_cm'],2),'margin',round(m.min(),2),flush=True)
    dump(REPORT/'geometry.json',rows)


def speed_profile(k,s,top):
    target=np.minimum(top,np.minimum(.45/np.maximum(abs(k),1e-6),np.sqrt(1.5/np.maximum(abs(k),1e-6))))
    for i in range(len(target)-2,-1,-1):target[i]=min(target[i],np.sqrt(target[i+1]**2+2*(s[i+1]-s[i])))
    for i in range(1,len(target)):target[i]=min(target[i],np.sqrt(target[i-1]**2+2*(s[i]-s[i-1])))
    return target


def run(key,candidate,mode,preset,level,seed,tag,centered=False):
    dst=OUT/tag/('_'.join(key)+f'_{candidate}_{mode}_{preset}_{level}_{seed}'+('_centered' if centered else ''))
    if (dst/'result.json').exists():return json.loads((dst/'result.json').read_text())
    z=np.load(OUT/'paths'/'_'.join(key)/(candidate+'.npz'));pts=z['points'];h=z['heading'];k=z['curvature'];s=z['s']
    speed=old_speed(key);profile=speed_profile(k,s,speed)
    p=PRESETS[preset];c,di,do,arm=movement(*key);track=Track();rng=np.random.default_rng(seed)
    pose=np.r_[pts[0]+rng.uniform(-1,1)*np.array([di[1],-di[0]])+rng.uniform(-2,2)*di,heading_of(di)+np.radians(rng.uniform(-3,3))]
    if centered:pose=np.r_[pts[0],heading_of(di)]
    car=Vehicle(p,pose,replace(VARIATIONS[level],seed=seed));rim=car.footprint()
    alloc=SignedAllocator([p.start_duty_left+(0 if level=='fixed' else .06),p.start_duty_right+(0 if level=='fixed' else .06)])
    events=events_for(seed,level);samples=[];commands=[];t=0.;capture=1/30;outcome='time_limit';since=None;idx=0;exposure=0.;planned_v=0.
    def step_to(target):
        nonlocal t,since,outcome,exposure
        while t<target-1e-10:
            dt=min(.02,target-t);scale=1.
            for e in events:
                if e['kind']=='power' and e['t0']<=t<e['t0']+e['duration']:scale*=e['gain']
            car.step(t,dt,scale)
            for e in events:
                if e['kind']!='push':continue
                overlap=max(0,min(t+dt,e['t0']+e['duration'])-max(t,e['t0']))
                if overlap:
                    fraction=overlap/e['duration'];yaw=car.pose[2];exposure+=overlap
                    car.pose[:2]+=e['lateral_cm']*fraction*np.array([np.cos(yaw),-np.sin(yaw)])
                    car.pose[2]=wrap(car.pose[2]+np.radians(e['yaw_deg'])*fraction)
            t+=dt;clear=track.clearance(to_world(rim,car.pose));margin=float(clear.min());rel=car.pose[:2]-c
            samples.append([t,*car.pose,car.v,car.omega,margin])
            good=rel@do>=40 and margin>=0 and abs(rel@np.array([do[1],-do[0]]))<=1 and abs(wrap(car.pose[2]-heading_of(do)))<=np.radians(5)
            since=t if good and since is None else since if good else None
            if margin < -12 or clear.max()<0:outcome='severe_departure';return
            if since is not None and t-since>=1:outcome='stable_exit';return
            for other in track.data['junctions'][key[0]]['arms']:
                dd=np.asarray(ARM[other])
                if other!=arm and rel@dd>=40 and abs(rel@np.array([dd[1],-dd[0]]))<15 and abs(wrap(car.pose[2]-heading_of(dd)))<np.radians(30):
                    outcome='wrong_exit';return
    while capture<40:
        step_to(capture)
        if outcome!='time_limit':break
        observed=car.pose.copy();v=car.v;w=car.omega;done=capture+.036
        step_to(done)
        if outcome!='time_limit':break
        predicted=advance(observed,v,w,.036+p.actuation_delay_s+p.time_constant_s)
        near=idx+int(np.argmin(np.linalg.norm(pts[idx:min(len(pts),idx+100)]-predicted[:2],axis=1)));idx=max(idx-2,near)
        kff=float(np.interp(s[idx],s,k));error=predicted[:2]-pts[idx]
        lat=float(error@np.array([np.cos(h[idx]),-np.sin(h[idx])]))
        herr=wrap(h[idx]-predicted[2])
        if mode=='legacy':vv=speed/(1+5*abs(kff))
        else:
            # Spatial backward pass starts braking BEFORE the turn. Ramp start only;
            # no temporal low-pass filter on steering or PWM.
            planned_v=min(float(profile[idx]),planned_v+.036)
            vv=planned_v
        omega=vv*kff+1.3*herr-.16*lat
        eff=p.turn_eff_left if omega<0 else p.turn_eff_right;diff=omega*p.track_cm/(2*eff)
        desired=np.array([(vv+diff)/p.cm_s_per_duty_left,(vv-diff)/p.cm_s_per_duty_right])
        if max(abs(desired))>1:desired/=max(abs(desired))
        applied=alloc.apply(desired);car.command(done,applied);commands.append([capture,done,*desired,*applied])
        capture=max(capture+1/30,done)
    a=np.asarray(samples);cmd=np.asarray(commands)
    result=metrics(a[:,0],a[:,1:4],a[:,4],a[:,5],a[:,6],c,do,[e for e in events if e['kind']=='push'],reference=np.c_[pts,h])
    grid=np.arange(1,t,.1);vg=np.interp(grid,a[:,0],a[:,4]);wg=np.interp(grid,a[:,0],a[:,5])
    result.update(key=key,candidate=candidate,mode=mode,preset=preset,level=level,seed=seed,outcome=outcome,centered=centered,
                  simulated_s=t,initial_margin_cm=float(a[0,6]),push_exposure_s=exposure,commands=len(cmd),allocator_calls=alloc.calls,
                  opposing_wheel_fraction=float(np.mean(cmd[:,4]*cmd[:,5]<0)),
                  actual_peak_curvature_cm_inv=float(np.max(abs(wg[abs(vg)>1]/vg[abs(vg)>1]))),
                  path_shape=shape_metrics(k,s),artifact=str(dst))
    dst.mkdir(parents=True,exist_ok=False);dump(dst/'result.json',result);dump(dst/'events.json',events)
    np.savez_compressed(dst/'trace.npz',samples=a,commands=cmd)
    return result


def select():
    rows=[];trials=[]
    for key in KEYS:
        specs=[(p,l) for p in ['field_estimate','sluggish'] for l in ['fixed','mild']]
        baseline=[run(key,'old','legacy',p,l,1303,'selection') for p,l in specs]
        trials+=baseline;candidates=[]
        for path in ['wide','reserve']:
            for mode in ['legacy','planned']:
                rs=[run(key,path,mode,p,l,1303,'selection') for p,l in specs];trials+=rs
                # Rank safety before smoothness; preserve even rejected candidates.
                candidates.append((path,mode,rs))
        best=min(candidates,key=lambda z:(sum(not r['stable_success'] for r in z[2]),sum(not r['clean_stable'] for r in z[2]),
                    np.mean([r['angular_accel_rms_rad_s2'] for r in z[2]]),np.mean([r['simulated_s'] for r in z[2]])))
        row=dict(key=key,candidate=best[0],mode=best[1],selection_stable=sum(r['stable_success'] for r in best[2]),
                 selection_clean=sum(r['clean_stable'] for r in best[2]),baseline_clean=sum(r['clean_stable'] for r in baseline))
        rows.append(row);print('_'.join(key),best[0],best[1],row['selection_clean'],flush=True)
    dump(REPORT/'selection_trials.json',trials);dump(REPORT/'selection.json',rows)


def sweep():
    import concurrent.futures
    jobs=[];rows=[];counts={}
    for config in json.loads((REPORT/'selection.json').read_text()):
        key=tuple(config['key'])
        counts[key]=0
        for candidate,mode in [('old','legacy'),(config['candidate'],config['mode'])]:
            for preset in PRESETS_USED:
                for level in ['fixed','mild','harsh']:
                    for seed in DEV:jobs.append((key,candidate,mode,preset,level,seed,'validation'))
            jobs.append((key,candidate,mode,'field_estimate','fixed',1303,'nominal',True))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(run,*job):job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            result=future.result();job=futures[future]
            if job[6]=='validation':rows.append(result)
            counts[job[0]]+=1
            if counts[job[0]]==74:print('_'.join(job[0]),'validated',flush=True)
    rows.sort(key=lambda r:(r['key'],r['candidate'],r['mode'],r['preset'],r['level'],r['seed']))
    dump(REPORT/'validation.json',rows)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('task',choices=['search','select','sweep']);a=ap.parse_args()
    {'search':search,'select':select,'sweep':sweep}[a.task]()
