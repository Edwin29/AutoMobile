"""Seed-separated direct teacher data with DART and junction exclusions."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from bench.world import Track,Camera,Renderer
from bench.vehicle import Vehicle,random_params
from bench.realcam import RealCam,pose_from_history
from .direct_teacher import DirectTeacher
from .model import preprocess_bgr


def collect(seed,out,count=1000,corner_only=False):
    cv2.setNumThreads(1)
    rng=np.random.default_rng(seed);track=Track();camera=Camera();renderer=Renderer(track,camera)
    direction='cw' if seed%2 else 'ccw';teacher=DirectTeacher(direction,variant=3)
    outer_route=teacher.route.copy()
    inner_axis=seed%2
    inner_sign=1 if seed%4<2 else -1
    params=random_params(seed);teacher.sim_params=params
    profile='harsh' if seed%3==0 else 'typical';fx=RealCam(renderer,camera,profile,seed)
    junctions=np.array([v['xy'] for v in track.data['junctions'].values()])
    X=[];Y=[];meta=[];t=0.;episode=0;history=[];car=None;last_dart=-99.;dart_until=-99.;dart_sign=1.
    while len(X)<count:
        inner=not corner_only and len(X)<count//10
        if car is not None and (inner and not 36<abs(car.pose[inner_axis])<49 or not inner and getattr(teacher,'inner',False)):
            car=None
        teacher.inner=inner
        if inner:
            line=np.zeros((171,2));line[:,inner_axis]=np.linspace(-85,85,171)
            teacher.route=line if inner_sign>0 else line[::-1]
        else:teacher.route=outer_route
        if car is None or t-episode>(12 if corner_only else 120) or track.clearance(car.pose[None,:2])[0]<0:
            route=teacher.route;eligible=np.flatnonzero((np.abs(route[:,0])>65)&(np.abs(route[:,1])>65)) if corner_only else np.arange(len(route));i=int(rng.choice(eligible));delta=route[(i+1)%len(route)]-route[i]
            pose=np.r_[route[i]+rng.normal(0,.5,2),np.arctan2(*delta)+rng.normal(0,.03)]
            if inner:
                pose=np.zeros(3);pose[inner_axis]=inner_sign*38
                pose[1-inner_axis]=rng.normal(0,.3)
                pose[2]=np.arctan2(inner_sign if inner_axis==0 else 0,inner_sign if inner_axis==1 else 0)
            car=Vehicle(params,pose);history=[(t,pose.copy())];episode=t;teacher.last_pwm=np.zeros(2)
        if int((t-episode)/6)>int((t-episode-.1)/6):
            last_dart=t;dart_until=t+.35;dart_sign=float(rng.choice([-1,1]))
        pwm,info=teacher.step(car.pose)
        distance=float(np.linalg.norm(junctions-car.pose[:2],axis=1).min())
        if distance>=35 and (not corner_only or abs(info["k"])>.02):
            image=fx.frame(t,pose_from_history(history),raw=True)
            X.append(np.rint(preprocess_bgr(image)*255).astype(np.uint8));Y.append(info['label'])
            meta.append(dict(seed=seed,pose=car.pose.tolist(),k=info['k'],mean=info['used_mean'],
                             recovery=0<=t-last_dart<=2.35,junction_distance=distance,profile=profile,
                             direction=direction,road='inner' if inner else 'outer',vehicle=params.to_dict()))
        if t<dart_until:pwm=np.clip(pwm+dart_sign*np.array([.08,-.08]),0,.66)
        car.command(t+.036,pwm)
        for _ in range(5):
            car.step(t,.02);t+=.02;history.append((t,car.pose.copy()))
        history=history[-50:]
        if len(X) and len(X)%250==0: print(seed,len(X),flush=True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    suffix='_corner' if corner_only else ''
    np.savez_compressed(out/f'seed_{seed}{suffix}.npz',x=np.array(X),y=np.array(Y,np.float32),
                        corner=np.array([abs(m['k'])>.02 for m in meta]),recovery=np.array([m['recovery'] for m in meta]))
    (out/f'seed_{seed}{suffix}.json').write_text(json.dumps(meta),encoding='utf-8')
    return dict(seed=seed,count=len(X),corner=sum(abs(m['k'])>.02 for m in meta),recovery=sum(m['recovery'] for m in meta),minimum_junction_distance=min(m['junction_distance'] for m in meta))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--out',required=True);ap.add_argument('--count',type=int,default=1000);ap.add_argument('--corner-only',action='store_true');a=ap.parse_args()
    print(json.dumps(collect(a.seed,a.out,a.count,a.corner_only)),flush=True)
