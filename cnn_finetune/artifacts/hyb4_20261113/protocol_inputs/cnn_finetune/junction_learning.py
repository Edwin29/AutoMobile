"""Small, controlled visual recurrent-policy learning experiment. Offline only."""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch import nn

from bench.world import Track,Camera,Renderer,to_world,advance,wrap
from bench.vehicle import Vehicle,PRESETS,VARIATIONS
from bench.realcam import RealCam,pose_from_history
from bench.junction import movement,heading_of,ARM
from .model import load_a5b,preprocess_bgr
from .junction_cornering import REPORT as TEACHER_REPORT
from .junction_feasibility import SignedAllocator,events_for
from .junction_stage02 import ROOT,dump,metrics

OUT=Path('D:/autodrive_cnn/runs/junction_learning_20261113')
REPORT=ROOT/'cnn_finetune/results/junction_learning_20261113'
CHECKPOINT=Path('D:/autodrive_cnn/checkpoints/dagger_i1_v2/best.pt')
KEYS=[('east','W','left'),('east','W','right'),('south','N','right'),('center','N','right')]
CONDITIONS={'normal':('fixed','typical'),'mild':('mild','typical'),
            'mild_light':('mild','harsh_light'),'harsh':('harsh','harsh')}
TRAIN_SEEDS=list(range(2001,2007));DEV_SEEDS=[1301,1303];TEST_SEEDS=[3001,3007,3011]
REGISTRY={tuple(r['key']):r for r in json.loads((TEACHER_REPORT/'teacher_registry.json').read_text())}


def setup():
    cv2.setNumThreads(1);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False


class Encoder(nn.Module):
    def __init__(self):
        super().__init__();base=load_a5b();base.load_state_dict(torch.load(CHECKPOINT,weights_only=True,map_location='cpu'))
        self.net=base.net[:11].eval()
        for p in self.parameters():p.requires_grad_(False)
    def forward(self,x):return self.net(x)


class TemporalPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.Sequential(nn.LayerNorm(1152),nn.Linear(1152,128),nn.ELU())
        self.gru=nn.GRU(132,64,batch_first=True)
        self.head=nn.Sequential(nn.Linear(64,32),nn.ELU(),nn.Linear(32,2),nn.Tanh())
    def forward(self,features,aux,hidden=None):
        x=torch.cat([self.project(features),aux],dim=-1);z,hidden=self.gru(x,hidden)
        return self.head(z),hidden


def model_at(path):
    m=TemporalPolicy().cuda();m.load_state_dict(torch.load(path,weights_only=True,map_location='cuda'));return m.eval()


class Scene:
    def __init__(self,key,condition,seed,preset='field_estimate',encoder=None):
        self.key=tuple(key);self.condition=condition;self.seed=seed;self.preset=preset
        self.level,profile=CONDITIONS[condition];self.p=PRESETS[preset]
        self.c,self.di,self.do,self.arm=movement(*key);self.track=Track()
        target=REGISTRY[tuple(key)];assert target['status']=='accepted_dev'
        z=np.load(target['reference']);self.points=z['points'];self.h=z['heading'];self.k=z['curvature'];self.s=z['s'];self.speed=z['nominal_speed_cm_s']
        self.teacher_mode=target['mode'];self.idx=0;self.planned_v=0.
        rng=np.random.default_rng(seed)
        initial=np.r_[self.points[0]+rng.uniform(-1,1)*np.array([self.di[1],-self.di[0]])+rng.uniform(-2,2)*self.di,
                      heading_of(self.di)+np.radians(rng.uniform(-3,3))]
        self.car=Vehicle(self.p,initial,replace(VARIATIONS[self.level],seed=seed));self.rim=self.car.footprint()
        camera=Camera();renderer=Renderer(self.track,camera);self.camera=RealCam(renderer,camera,profile,seed)
        self.encoder=encoder if encoder is not None else Encoder().cuda().eval()
        self.alloc=SignedAllocator([.36,.36]);self.events=events_for(seed,self.level)
        self.t=0.;self.capture=1/30;self.previous=np.zeros(2);self.history=[(0.,initial.copy())]
        self.physics=[];self.records=[];self.hidden=None;self.since=None;self.outcome='time_limit';self.exposure=0.

    def clone(self):
        out=copy.copy(self)
        for name in ['car','alloc','history','previous']:setattr(out,name,copy.deepcopy(getattr(self,name)))
        out.physics=list(self.physics);out.records=list(self.records)
        out.hidden=None if self.hidden is None else self.hidden.detach().clone()
        out.camera=copy.copy(self.camera)
        for name in ['gain','awb','awb_target','last_t','meter_ref','rng','flags','stats']:
            setattr(out.camera,name,copy.deepcopy(getattr(self.camera,name)))
        out.camera.gen=torch.Generator(device=self.camera.dev);out.camera.gen.set_state(self.camera.gen.get_state().clone())
        return out

    def advance_to(self,target):
        while self.t<target-1e-10 and self.outcome=='time_limit':
            dt=min(.02,target-self.t);scale=1.
            for e in self.events:
                if e['kind']=='power' and e['t0']<=self.t<e['t0']+e['duration']:scale*=e['gain']
            self.car.step(self.t,dt,scale)
            for e in self.events:
                if e['kind']!='push':continue
                overlap=max(0,min(self.t+dt,e['t0']+e['duration'])-max(self.t,e['t0']))
                if overlap:
                    f=overlap/e['duration'];yaw=self.car.pose[2];self.exposure+=overlap
                    self.car.pose[:2]+=e['lateral_cm']*f*np.array([np.cos(yaw),-np.sin(yaw)])
                    self.car.pose[2]=wrap(self.car.pose[2]+np.radians(e['yaw_deg'])*f)
            self.t+=dt;pose=self.car.pose;clear=self.track.clearance(to_world(self.rim,pose));margin=float(clear.min());rel=pose[:2]-self.c
            self.physics.append([self.t,*pose,self.car.v,self.car.omega,margin]);self.history.append((self.t,pose.copy()))
            if len(self.history)>400:del self.history[:200]
            good=rel@self.do>=40 and margin>=0 and abs(rel@np.array([self.do[1],-self.do[0]]))<=1 and abs(wrap(pose[2]-heading_of(self.do)))<=np.radians(5)
            self.since=self.t if good and self.since is None else self.since if good else None
            if margin < -12 or clear.max()<0:self.outcome='severe_departure'
            elif self.since is not None and self.t-self.since>=1:self.outcome='stable_exit'
            else:
                for arm in self.track.data['junctions'][self.key[0]]['arms']:
                    d=np.asarray(ARM[arm])
                    if arm!=self.arm and rel@d>=40 and abs(rel@np.array([d[1],-d[0]]))<15 and abs(wrap(pose[2]-heading_of(d)))<np.radians(30):self.outcome='wrong_exit'

    def teacher(self,observed,v,w):
        predicted=advance(observed,v,w,.036+self.p.actuation_delay_s+self.p.time_constant_s)
        near=self.idx+int(np.argmin(np.linalg.norm(self.points[self.idx:min(len(self.points),self.idx+100)]-predicted[:2],axis=1)))
        self.idx=max(self.idx-2,near);k=float(self.k[self.idx]);h=self.h[self.idx]
        lat=float((predicted[:2]-self.points[self.idx])@np.array([np.cos(h),-np.sin(h)]));herr=wrap(h-predicted[2])
        if self.teacher_mode=='planned':self.planned_v=min(float(self.speed[self.idx]),self.planned_v+.036);v=self.planned_v
        else:v=float(self.speed[self.idx])
        omega=v*k+1.3*herr-.16*lat;eff=self.p.turn_eff_left if omega<0 else self.p.turn_eff_right;diff=omega*self.p.track_cm/(2*eff)
        desired=np.array([(v+diff)/self.p.cm_s_per_duty_left,(v-diff)/self.p.cm_s_per_duty_right])
        return desired/max(1,float(max(abs(desired))))

    @torch.no_grad()
    def step(self,policy=None,intervention=None):
        self.advance_to(self.capture)
        if self.outcome!='time_limit':return False
        image=self.camera.frame(self.capture,pose_from_history(self.history),raw=True)
        x=np.rint(preprocess_bgr(image)*255).astype(np.uint8)
        feat=self.encoder(torch.from_numpy(x[None]).cuda().float()/255).cpu().numpy()[0]
        aux=np.r_[self.previous,[1.,0.] if self.key[2]=='left' else [0.,1.]].astype(np.float32)
        label=self.teacher(self.car.pose.copy(),self.car.v,self.car.omega)
        done=self.capture+.036;self.advance_to(done)
        if self.outcome!='time_limit':return False
        if policy is None:proposed=label.copy();ref=np.zeros(2,dtype=np.float32)
        else:
            y,self.hidden=policy(torch.from_numpy(feat[None,None]).cuda(),torch.from_numpy(aux[None,None]).cuda(),self.hidden)
            proposed=ref=y[0,0].cpu().numpy()
        if intervention=='teacher':proposed=label.copy()
        elif intervention=='left':proposed=np.clip(proposed+np.array([-.12,.12]),-1,1)
        elif intervention=='right':proposed=np.clip(proposed+np.array([.12,-.12]),-1,1)
        applied=self.alloc.apply(proposed);self.car.command(done,applied)
        self.records.append(dict(x=x,f=feat,aux=aux,label=label.astype(np.float32),action=np.asarray(proposed,np.float32),reference=ref,
                                 applied=applied.copy(),capture=self.capture,done=done,margin=self.physics[-1][-1]))
        self.previous=applied.copy();self.capture=max(self.capture+1/30,done)
        return True

    def finish(self,policy=None,until=40.):
        while self.capture<until and self.outcome=='time_limit':
            if not self.step(policy):break
        return self

    def summary(self):
        a=np.asarray(self.physics);r=metrics(a[:,0],a[:,1:4],a[:,4],a[:,5],a[:,6],self.c,self.do,
            [e for e in self.events if e['kind']=='push'],reference=np.c_[self.points,self.h])
        return dict(r,key=self.key,condition=self.condition,seed=self.seed,preset=self.preset,outcome=self.outcome,
                    simulated_s=self.t,frames=len(self.records),push_exposure_s=self.exposure)

    def save(self,dst,supervise_from=0,save_images=True):
        dst=Path(dst);dst.mkdir(parents=True,exist_ok=False);r=self.summary();dump(dst/'result.json',r);dump(dst/'events.json',self.events)
        data={k:np.array([q[k] for q in self.records]) for k in ['f','aux','label','action','reference','applied','capture','done','margin']}
        data['supervise']=np.arange(len(self.records))>=supervise_from
        if save_images:data['x']=np.array([q['x'] for q in self.records])
        np.savez_compressed(dst/'sequence.npz',**data);np.savez_compressed(dst/'physics.npz',samples=np.asarray(self.physics))
        return dict(r,artifact=str(dst))


def freeze():
    REPORT.mkdir(parents=True,exist_ok=False);OUT.mkdir(parents=True,exist_ok=False)
    dump(REPORT/'protocol.json',dict(keys=KEYS,train_seeds=TRAIN_SEEDS,development=DEV_SEEDS,final_test=TEST_SEEDS,
        initialization_seed=17,training_replicates=1,model='frozen direct_v2 convolutional encoder -> LayerNorm/128 -> GRU64 -> signed LR head',
        student_inputs=['raw bottom-half YUV camera','previous applied signed PWM','left/right instruction'],
        forbidden_inputs=['truth pose','true velocity','junction identity','route progress','simulator parameters','time'],
        conditions=CONDITIONS,timing=dict(capture_period=1/30,latency=.036,physics_max_dt=.02),
        allocator=dict(fixed_floor=.36,updates_per_command=1),
        train=dict(warm_epochs=12,comparison_epochs=8,lr=.0005,batch_episodes=4,checkpoints=[4,8]),
        comparison='same warmstart, training data, order, optimizer and number of steps; BC vs BC+0.1 preference',
        preferences=dict(branch_times=[6.,10.],horizon_s=8.,intervention_steps=20,actions=['student','teacher','left','right'],
                         root_seeds=[2001,2002],conditions=['mild','mild_light'],sigma=[.15,.15],beta=1.,anchor=.1),
        checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        teacher_registry_sha256=hashlib.sha256((TEACHER_REPORT/'teacher_registry.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


def collect():
    encoder=Encoder().cuda().eval();results=[]
    for i,seed in enumerate(TRAIN_SEEDS):
        for key in KEYS:
            for condition in CONDITIONS:
                dst=OUT/'teacher_data'/('_'.join(key)+f'_{condition}_{seed}')
                if (dst/'result.json').exists():r=dict(json.loads((dst/'result.json').read_text()),artifact=str(dst))
                else:r=Scene(key,condition,seed,['field_estimate','sluggish','unequal_sides'][i%3],encoder).finish().save(dst)
                results.append(r)
        print('teacher seed',seed,'done',flush=True)
    dump(REPORT/'teacher_data.json',results)


def outcome_cost(scene,start,horizon=8.):
    from scipy.spatial import cKDTree
    a=np.asarray(scene.physics);a=a[a[:,0]>=start]
    dt=np.diff(a[:,0],prepend=start);duration=float(a[-1,0]-start);failed=scene.outcome in ['severe_departure','wrong_exit']
    missing=max(0.,horizon-duration);outside=float(np.sum(dt*(a[:,6]<0)))+(missing if failed else 0)
    risk=float(np.sum(dt*np.clip((1-a[:,6])/3,0,1)))+(missing if failed else 0)
    distance,idx=cKDTree(scene.points).query(a[:,1:3]);head=abs(wrap(a[:,3]-scene.h[idx]))
    aligned=(distance<=1)&(head<=np.radians(5))&(a[:,6]>=0)
    recent=a[:,0]>=a[-1,0]-1
    stable=bool(aligned[recent].all() and duration>=1)
    progress=float(scene.s[idx[-1]]-scene.s[idx[0]])
    remaining=max(1,scene.s[-1]-scene.s[idx[0]]-10)
    deficit=0 if scene.outcome=='stable_exit' else 1-np.clip(progress/min(4*horizon,remaining),0,1)
    omega=np.interp(np.arange(start,a[-1,0],.1),a[:,0],a[:,5]);alpha=np.diff(omega)/.1
    smooth=float(np.mean(alpha*alpha)) if len(alpha) else 0.
    parts=dict(failure=100.*failed,wrong_exit=50.*(scene.outcome=='wrong_exit'),
               unrecovered=20.*(not stable and scene.outcome!='stable_exit'),outside=20.*outside/horizon,
               risk=2.*risk/horizon,progress=5.*deficit,smooth=.1*min(smooth,10.))
    return dict(total=float(sum(parts.values())),components=parts,observed_s=duration,stable_end=stable)


def branches():
    encoder=Encoder().cuda().eval();model=model_at(OUT/'warmstart.pt');roots=[];pairs=[];datasets=[]
    for key in KEYS:
        for seed in [2001,2002]:
            for condition in ['mild','mild_light']:
                env=Scene(key,condition,seed,'field_estimate' if seed==2001 else 'sluggish',encoder)
                for target in [6.,10.]:
                    env.finish(model,target)
                    if env.outcome!='time_limit':
                        roots.append(dict(key=key,seed=seed,condition=condition,target=target,skipped=env.outcome));continue
                    start=env.t;prefix=len(env.records);root='_'.join(key)+f'_{seed}_{condition}_{target}'
                    candidates=[]
                    for action in ['student','teacher','left','right']:
                        branch=env.clone()
                        for _ in range(20):
                            if not branch.step(model,None if action=='student' else action):break
                        end_intervention=len(branch.records)
                        intervention_records=list(branch.records)
                        branch.finish(model,start+8.)
                        cost=outcome_cost(branch,start)
                        # Preference model sees causal prefix + own intervention, not future outcome observations.
                        future_records=branch.records;branch.records=intervention_records
                        dst=OUT/'branches'/root/action
                        r=branch.save(dst,supervise_from=prefix)
                        branch.records=future_records
                        dump(dst/'cost.json',cost)
                        candidates.append(dict(action=action,path=str(dst),cost=cost['total'],prefix=prefix,end=end_intervention))
                        datasets.append(dict(r,branch=True))
                    winner=min(candidates,key=lambda x:x['cost']);loser=max(candidates,key=lambda x:x['cost'])
                    rootrow=dict(root=root,prefix=prefix,start_s=start,candidates=candidates)
                    roots.append(rootrow)
                    if loser['cost']-winner['cost']>=.5 and winner['end']-prefix==20 and loser['end']-prefix==20:
                        pairs.append(dict(root=root,winner=winner['path'],loser=loser['path'],prefix=prefix,gap=loser['cost']-winner['cost']))
                print('branches',key,seed,condition,'pairs',len(pairs),flush=True)
    dump(REPORT/'branch_roots.json',roots);dump(REPORT/'preferences.json',pairs);dump(REPORT/'branch_data.json',datasets)


def evaluate(checkpoint,split,tag):
    encoder=Encoder().cuda().eval();model=model_at(checkpoint);results=[]
    seeds=DEV_SEEDS if split=='dev' else TEST_SEEDS
    conditions=['normal','mild_light'] if split=='dev' else list(CONDITIONS)
    for i,seed in enumerate(seeds):
        for key in KEYS:
            for condition in conditions:
                dst=OUT/'evaluations'/tag/('_'.join(key)+f'_{condition}_{seed}')
                if (dst/'result.json').exists():r=dict(json.loads((dst/'result.json').read_text()),artifact=str(dst))
                else:r=Scene(key,condition,seed,['field_estimate','sluggish','unequal_sides'][i%3],encoder).finish(model).save(dst,save_images=False)
                results.append(r)
        print('evaluation',tag,seed,'success',sum(r['stable_success'] for r in results),'/',len(results),flush=True)
    dump(REPORT/(tag+'.json'),dict(checkpoint=str(checkpoint),sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),split=split,results=results))


if __name__=='__main__':
    setup();a=argparse.ArgumentParser();a.add_argument('task',choices=['freeze','collect','branches','evaluate']);a.add_argument('--checkpoint');a.add_argument('--split',default='dev');a.add_argument('--tag');args=a.parse_args()
    if args.task=='freeze':freeze()
    elif args.task=='collect':collect()
    elif args.task=='branches':branches()
    else:evaluate(args.checkpoint,args.split,args.tag)
