"""Offline, privileged-state path feasibility search. Never a deployable policy."""
import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.optimize import differential_evolution

from bench.world import Track, to_world, wrap, advance
from bench.vehicle import Vehicle, PRESETS, VARIATIONS
from bench.junction import movement, heading_of, load_plan_cache
from cnn_finetune.junction_stage02 import OUT, REPORT, DEV, dump, metrics


def bezier(ctrl,n):
    u=np.linspace(0,1,n)[:,None];order=len(ctrl)-1
    return sum(math.comb(order,i)*u**i*(1-u)**(order-i)*p for i,p in enumerate(ctrl))


def spline(key,x,spacing=.5):
    c,di,do,_=movement(*key)
    a,b,h1,h2,oi,oo=x
    ni=np.array([di[1],-di[0]]);no=np.array([do[1],-do[0]])
    start=c-45*di;end=c+55*do;p0=c-a*di+oi*ni;p5=c+b*do+oo*no
    # Quintic core: collinear first/last three controls give zero endpoint curvature.
    core=bezier(np.array([p0,p0+h1*di,p0+2*h1*di,p5-2*h2*do,p5-h2*do,p5]),150)
    pre=bezier(np.array([start,start+(45-a)/3*di,p0-(45-a)/3*di,p0]),80)
    post=bezier(np.array([p5,p5+(55-b)/3*do,end-(55-b)/3*do,end]),80)
    raw=np.vstack([pre[:-1],core[:-1],post]);s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(raw,axis=0),axis=1))]
    q=np.linspace(0,s[-1],int(np.ceil(s[-1]/spacing))+1)
    return np.c_[np.interp(q,s,raw[:,0]),np.interp(q,s,raw[:,1])]


def geometry(points,rim,track):
    d=np.gradient(points,axis=0);h=np.arctan2(d[:,0],d[:,1])
    co=np.cos(h)[:,None];si=np.sin(h)[:,None]
    world=np.stack([points[:,0,None]+rim[None,:,0]*co+rim[None,:,1]*si,
                    points[:,1,None]-rim[None,:,0]*si+rim[None,:,1]*co],axis=-1)
    return track.clearance(world).min(axis=1),h


def search(key):
    track=Track();rim=Vehicle(PRESETS['field_estimate'],np.zeros(3)).footprint(2)
    cache=load_plan_cache();oldroute,plans,gen=cache[key]
    if key[2]=='straight':
        c,di,do,_=movement(*key);pts=np.linspace(c-45*di,c+55*do,401);x=None;budget=0
    else:
        def objective(x):
            p=spline(key,x,1);m,h=geometry(p,rim,track)
            ds=np.linalg.norm(np.diff(p,axis=0),axis=1)
            k=wrap(np.diff(h))/ds
            # Geometric clearance first; discourage pathological high curvature/loops.
            return -float(m.min())+.005*float(np.mean(k*k))+10*max(0,float(abs(k).max())-.5)**2
        opt=differential_evolution(objective,[(5,26),(5,26),(1,11),(1,11),(-2,2),(-2,2)],
                                   seed=2718,popsize=6,maxiter=45,tol=.002,polish=False)
        x=opt.x;pts=spline(key,x,.25);budget=opt.nfev
    m,h=geometry(pts,Vehicle(PRESETS['field_estimate'],np.zeros(3)).footprint(.5),track)
    name='_'.join(key);dst=OUT/'paths'/name;dst.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dst/'path.npz',points=pts,heading=h,margin=m)
    meta=dict(key=key,kind='straight' if x is None else 'quintic_variable_curvature',parameters=x,
              static_min_margin_cm=float(m.min()),evaluations=budget,
              legacy_arc=gen,legacy_pivot_margin_cm=None if plans is None else plans['pivot']['margin_cm'],
              legacy_split_margin_cm=None if plans is None else plans['split']['margin_cm'],
              mission_allowed=key[0]!='center' or key[2]=='right')
    dump(dst/'path.json',meta)
    print(name,'static',round(m.min(),3),'evals',budget,flush=True)
    return meta


class SignedAllocator:
    """Signed average wheel duty realised with bounded error diffusion at 36ms."""
    def __init__(self,floor):self.floor=np.asarray(floor);self.error=np.zeros(2);self.calls=0
    def apply(self,desired):
        self.calls+=1
        x=np.clip(np.asarray(desired,float),-1,1);out=x.copy()
        for i in range(2):
            if 0<abs(x[i])<self.floor[i]:
                self.error[i]+=abs(x[i])/self.floor[i]
                if self.error[i]>=.5:out[i]=np.sign(x[i])*self.floor[i];self.error[i]-=1
                else:out[i]=0
            else:self.error[i]=0
        return out


def events_for(seed,level):
    if level=='fixed':return []
    r=np.random.default_rng(seed*65537+19)
    # Junction-local exposure: events start 7--8s after start, repeated at 15--16s.
    # Fixed clock schedule, independent of candidate and outcome. Not real frequencies.
    harsh=level=='harsh';amp=(4,6) if harsh else (1,2);yaw=(6,10) if harsh else (1,3)
    events=[]
    for base in [7,15]:
        t=base+float(r.uniform(0,1))
        events.extend([dict(kind='push',t0=t,duration=float(r.uniform(.3,.7)),lateral_cm=float(r.choice([-1,1])*r.uniform(*amp)),
                            yaw_deg=float(r.choice([-1,1])*r.uniform(*yaw))),
                       dict(kind='power',t0=t,duration=float(r.uniform(2,5)),gain=float(1+r.choice([-1,1])*r.uniform(*((.2,.35) if harsh else (.05,.15)))))])
    return events


def simulate(key,preset,level,seed,speed=5.,lookahead=2.,save=True,centered=False,initial_pose=None):
    path=np.load(OUT/'paths'/'_'.join(key)/'path.npz');pts=path['points'];h=path['heading']
    ds=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))]
    curvature=np.gradient(np.unwrap(h),ds)
    p=PRESETS[preset];c,di,do,arm=movement(*key);track=Track()
    rng=np.random.default_rng(seed);pose=np.r_[pts[0]+rng.uniform(-1,1)*np.array([di[1],-di[0]])+rng.uniform(-2,2)*di,
                                             heading_of(di)+np.radians(rng.uniform(-3,3))]
    if centered:pose=np.r_[pts[0],heading_of(di)]
    if initial_pose is not None:pose=np.asarray(initial_pose,float).copy()
    car=Vehicle(p,pose,replace(VARIATIONS[level],seed=seed));rim=car.footprint()
    initial_margin=float(track.clearance(to_world(rim,pose)).min())
    alloc=SignedAllocator([p.start_duty_left+(0 if level=='fixed' else .06),p.start_duty_right+(0 if level=='fixed' else .06)])
    events=events_for(seed,level);samples=[];commands=[];t=0.;capture=1/30;outcome='time_limit';since=None;idx=0;exposure=0.
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
    while capture<40:
        step_to(capture)
        if outcome!='time_limit':break
        observed=car.pose.copy();v=car.v;w=car.omega
        done=capture+.036
        step_to(done)
        if outcome!='time_limit':break
        # Oracle capture-state prediction across processing + nominal actuator delay/lag.
        predicted=advance(observed,v,w,.036+p.actuation_delay_s+p.time_constant_s)
        near=idx+int(np.argmin(np.linalg.norm(pts[idx:min(len(pts),idx+100)]-predicted[:2],axis=1)))
        idx=max(idx-2,near)
        kff=float(np.interp(ds[idx]+lookahead,ds,curvature));href=float(np.interp(ds[idx]+lookahead,ds,np.unwrap(h)))
        error=predicted[:2]-pts[idx];right=np.array([np.cos(h[idx]),-np.sin(h[idx])])
        lat=float(error@right);herr=wrap(h[idx]-predicted[2])
        vv=speed/(1+5*abs(kff));omega=vv*kff+1.3*herr-.16*lat
        eff=p.turn_eff_left if omega<0 else p.turn_eff_right
        diff=omega*p.track_cm/(2*eff)
        desired=np.array([(vv+diff)/p.cm_s_per_duty_left,(vv-diff)/p.cm_s_per_duty_right])
        if max(abs(desired))>1:desired/=max(abs(desired))
        applied=alloc.apply(desired);car.command(done,applied)
        commands.append([capture,done,*desired,*applied])
        capture=max(capture+1/30,done)
    a=np.asarray(samples);cmd=np.asarray(commands)
    result=metrics(a[:,0],a[:,1:4],a[:,4],a[:,5],a[:,6],c,do,[e for e in events if e['kind']=='push'],reference=np.c_[pts,h])
    result.update(key=key,preset=preset,level=level,seed=seed,speed=speed,lookahead=lookahead,outcome=outcome,
                  simulated_s=t,opposing_wheel_fraction=float(np.mean(cmd[:,4]*cmd[:,5]<0)),
                  push_exposure_s=exposure,allocator_calls=alloc.calls,commands=len(cmd),
                  initial_margin_cm=initial_margin,initially_inside=initial_margin>=0,centered=centered,
                  initial_pose_override=initial_pose,
                  negative_body_velocity_fraction=float(np.mean(a[:,4]<-.5)))
    if save:
        dst=OUT/'dynamic'/('_'.join(key)+f'_{preset}_{level}_{seed}_v{speed}_l{lookahead}'+('_centered' if centered else '')+('_legacy_pose' if initial_pose is not None else ''))
        dst.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(dst/'trace.npz',samples=a,commands=cmd)
        dump(dst/'result.json',result);dump(dst/'events.json',events)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('task',choices=['search','probe','select','sweep']);ap.add_argument('--key');args=ap.parse_args()
    if args.task=='search':
        keys=[tuple(args.key.split(','))] if args.key else list(load_plan_cache())
        metas=[search(k) for k in keys];dump(REPORT/'paths.json',metas)
    elif args.task=='probe':
        keys=[tuple(args.key.split(','))] if args.key else [('center','N','right'),('south','N','right'),('north','S','right'),('east','S','left')]
        for k in keys:
            for speed,look in [(5.,0.),(5.,2.),(4.,1.)]:
                r=simulate(k,'field_estimate','fixed',1301,speed,look)
                print(k,speed,look,r['outcome'],round(r['min_margin_cm'],3),flush=True)
    elif args.task=='select':
        selected=[];trials=[]
        for k in load_plan_cache():
            runs=[simulate(k,'field_estimate','fixed',1303,speed,0.) for speed in [4.,5.,6.]]
            trials.extend(runs)
            best=min(runs,key=lambda r:(not r['clean_stable'],not r['stable_success'],-r['min_margin_cm'],r['angular_accel_rms_rad_s2']))
            selected.append(dict(key=k,speed=best['speed'],lookahead=0.))
            print(k,'selected',best['speed'],best['outcome'],round(best['min_margin_cm'],3),flush=True)
        dump(REPORT/'selection.json',selected);dump(REPORT/'selection_trials.json',trials)
    else:
        selection=json.loads((REPORT/'selection.json').read_text());results=[]
        for config in selection:
            k=tuple(config['key'])
            for preset in ['field_estimate','sluggish','unequal_sides','field_pivot_slip']:
                for level in ['fixed','mild','harsh']:
                    for seed in DEV:
                        existing=OUT/'dynamic'/('_'.join(k)+f"_{preset}_{level}_{seed}_v{config['speed']}_l{config['lookahead']}")
                        r=json.loads((existing/'result.json').read_text()) if existing.exists() else simulate(k,preset,level,seed,config['speed'],config['lookahead'])
                        results.append(r)
            nominal=simulate(k,'field_estimate','fixed',1303,config['speed'],config['lookahead'],centered=True)
            dump(OUT/'paths'/'_'.join(k)/'centered_result.json',nominal)
            print(k,'done',sum(r['clean_stable'] for r in results[-36:]),'/36',flush=True)
        dump(REPORT/'dynamic_results.json',results)


if __name__=='__main__':main()
