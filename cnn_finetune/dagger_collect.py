"""Seed-separated direct teacher data with DART and junction exclusions."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from bench.world import Track,Camera,Renderer,to_world,wrap
from bench.vehicle import Vehicle,random_params
from bench.realcam import RealCam,pose_from_history
from .direct_teacher import DirectTeacher
from .model import preprocess_bgr


def collect(seed,out,model,iteration,count=1000):
    corner_only=False
    beta={1:.5,2:.25,3:0.}[iteration]
    net=cv2.dnn.readNetFromONNX(str(model))
    applied=np.zeros(2);policy="teacher";segment=-1
    from .teacher import MotorModel,load_config
    allocator=MotorModel(dict(load_config()["motor"]))
    cv2.setNumThreads(1)
    rng=np.random.default_rng(seed);track=Track();camera=Camera();renderer=Renderer(track,camera)
    direction='cw' if seed%2 else 'ccw';teacher=DirectTeacher(direction,variant=3)
    allocator=teacher.motor
    outer_route=teacher.route.copy()
    inner_axis=seed%2
    inner_sign=1 if seed%4<2 else -1
    params=random_params(seed);teacher.sim_params=params
    profile='harsh' if seed%3==0 else 'typical';fx=RealCam(renderer,camera,profile,seed)
    junctions=np.array([v['xy'] for v in track.data['junctions'].values()])
    episode_count=0
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
        margin=100. if car is None else float(track.clearance(to_world(car.footprint(),car.pose)).min())
        error=0.
        if car is not None:
            ri=int(np.argmin(np.sum((teacher.route-car.pose[:2])**2,axis=1)))
            tangent=teacher.route[(ri+1)%len(teacher.route)]-teacher.route[ri]
            error=abs(wrap(car.pose[2]-np.arctan2(*tangent)))
        if car is None or t-episode>(12 if corner_only else 120) or margin < -3 or error > np.pi/3:
            corner_only=not inner and episode_count%4!=3
            episode_count+=1
            route=teacher.route;eligible=np.flatnonzero((np.abs(route[:,0])>65)&(np.abs(route[:,1])>65)) if corner_only else np.arange(len(route));i=int(rng.choice(eligible));delta=route[(i+1)%len(route)]-route[i]
            pose=np.r_[route[i]+rng.normal(0,.5,2),np.arctan2(*delta)+rng.normal(0,.03)]
            if inner:
                pose=np.zeros(3);pose[inner_axis]=inner_sign*38
                pose[1-inner_axis]=rng.normal(0,.3)
                pose[2]=np.arctan2(inner_sign if inner_axis==0 else 0,inner_sign if inner_axis==1 else 0)
            car=Vehicle(params,pose);history=[(t,pose.copy())];episode=t;teacher.last_pwm=np.zeros(2);applied=np.zeros(2);allocator.dither=0.;error=0.
        teacher.last_pwm=applied.copy()
        previous_dither=allocator.dither
        pwm,info=teacher.step(car.pose)
        image=fx.frame(t,pose_from_history(history),raw=True)
        if int(t/.5)!=segment:
            segment=int(t/.5);policy='teacher' if rng.random()<beta else 'student'
        if policy=='student':
            allocator.dither=previous_dither
            net.setInput(preprocess_bgr(image)[None]);prediction=net.forward()[0]
            mean=.66*float(np.clip(prediction[0],0,1));k=float(np.clip(-prediction[1]*.095,-.085,.085))
            pwm=allocator.mix(mean,k,.66)[0] if mean>=.05 else np.zeros(2)
        margin=float(track.clearance(to_world(car.footprint(),car.pose)).min())
        distance=float(np.linalg.norm(junctions-car.pose[:2],axis=1).min())
        if distance>=35 and margin>=-3:
            X.append(np.rint(preprocess_bgr(image)*255).astype(np.uint8));Y.append(info['label'])
            meta.append(dict(t=t,episode=episode_count,corner_episode=corner_only,applied_pwm=applied.tolist(),seed=seed,pose=car.pose.tolist(),k=info['k'],mean=info['used_mean'],
                             recovery=bool(error>np.radians(5)),iteration=iteration,policy=policy,beta=beta,margin=margin,junction_distance=distance,profile=profile,
                             direction=direction,road='inner' if inner else 'outer',vehicle=params.to_dict()))
        car.command(t+.036,pwm)
        for dt in [.02,.016]:
            car.step(t,dt);t+=dt;history.append((t,car.pose.copy()))
        applied=car._active(t,0).copy()
        history=history[-50:]
        if len(X) and len(X)%250==0: print(seed,len(X),flush=True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    suffix=''
    np.savez_compressed(out/f'seed_{seed}{suffix}.npz',x=np.array(X),y=np.array(Y,np.float32),
                        corner=np.array([abs(m['k'])>.02 for m in meta]),recovery=np.array([m['recovery'] for m in meta]),student=np.array([m['policy']=='student' for m in meta]))
    (out/f'seed_{seed}{suffix}.json').write_text(json.dumps(meta),encoding='utf-8')
    return dict(seed=seed,count=len(X),corner=sum(abs(m['k'])>.02 for m in meta),recovery=sum(m['recovery'] for m in meta),minimum_junction_distance=min(m['junction_distance'] for m in meta))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--out',required=True);ap.add_argument('--count',type=int,default=1000);ap.add_argument('--model',required=True);ap.add_argument('--iteration',type=int,required=True);a=ap.parse_args()
    print(json.dumps(collect(a.seed,a.out,a.model,a.iteration,a.count)),flush=True)
