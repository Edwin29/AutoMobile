"""Offline stages 0--2. No hardware imports or model training.

Evaluation and collection share bench.junction.run; collection is an observer.
Truth-based path search is explicitly an offline feasibility teacher, not a student.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('D:/autodrive_cnn/runs/junction_stages02_20261113')
REPORT = ROOT / 'cnn_finetune/results/junction_stages02_20261113'
DEV = [1301, 1303, 1307]


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False,
                               default=lambda x: x.tolist() if isinstance(x, np.ndarray) else bool(x) if isinstance(x,np.bool_) else float(x)), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract():
    from bench.vehicle import PRESETS, VARIATIONS
    from bench.junction import load_plan_cache
    from bench.controllers import VENDOR_V4
    sources = {}
    for folder in ['bench', 'drive_v4', 'cnn_finetune']:
        for p in (ROOT / folder).rglob('*'):
            if p.is_file() and p.suffix in {'.py', '.onnx', '.npz', '.json', '.yaml', '.txt', '.pkl'} and not {'results', '__pycache__'} & set(p.parts):
                sources[str(p.relative_to(ROOT))] = sha(p)
    data = dict(schema=1, git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_hashes=sources, pinned_v4=str(VENDOR_V4.relative_to(ROOT)),
        baseline=dict(a5b_v4='bench/_vendor/tdt_models_2026-11-01/models/lane-v5b/policy.onnx',
                      direct_v2='cnn_finetune/models/direct_v2/policy.onnx', junction='hyb2j', mission='hyb2m'),
        seed_registry=dict(train=list(range(2001, 2021)), development=DEV, final_test=[3001,3007,3011,3019,3023,3037],
                           final_test_used=False, observed_old=[41,42,104,109,118,123,106,111,126,129,601,607,613]),
        priority=['severe departure / wrong exit / task failure', 'clean stable success', 'stable recovery', 'smoothness'],
        stable_exit=dict(along_cm=40, margin_cm=0, lateral_abs_cm=1, heading_abs_deg=5, hold_s=1),
        recovery=dict(same_bounds=True, hold_s=1, nonrecovery='null time plus explicit unrecovered count; never drop failures',
                      restricted_time='end minus event time for censored events; report alongside recovery rate'),
        smoothness=dict(grid_s=.1, acceleration='diff(interpolated actual body v)/0.1 cm/s^2',
                        angular_acceleration='diff(interpolated actual omega)/0.1 rad/s^2',
                        stop='abs(v)<0.5 cm/s AND abs(omega)<0.05 rad/s for >=0.3s; first 1s startup excluded',
                        mandatory_stop='mask excluded from unnecessary stop time/count; physical metrics retained'),
        action=dict(version='junction_signed_lr_v1', order=['left','right'], range=[-1,1], positive='forward',
                    low_duty='signed error diffusion below floor, once per decision; modelled lag, no extra smoothing',
                    direct_exterior='unchanged: mean duty / .66, left-positive curvature / .095, cap .085'),
        timing=dict(capture_period_s=1/30, latency_s=.036, physics_max_step_s=.02, previous_command_during_processing=True),
        search=dict(family='cubic Bezier with tangent straight approach and exit, varying curvature; existing arc/pivot/split references',
                    oracle='simulated pose and nominal vehicle parameters; OFFLINE ONLY',
                    lighting='does not validate a truth-based teacher; image-policy controls evaluated separately'),
        movements=[dict(junction=k[0], came_from=k[1], exit=k[2], mission_allowed=k[0]!='center' or k[2]=='right') for k in load_plan_cache()],
        vehicles={k:v.to_dict() for k,v in PRESETS.items()}, variations={k:v.to_dict() for k,v in VARIATIONS.items()})
    checkpoint=Path('D:/autodrive_cnn/checkpoints/dagger_i1_v2/best.pt')
    if checkpoint.exists(): data['checkpoint_sha256']=sha(checkpoint)
    dst=REPORT/'contract.json'
    if dst.exists(): raise FileExistsError('Frozen contract already exists')
    dump(dst,data)
    print('Contract frozen:',dst,flush=True)


def metrics(t, poses, velocity, omega, margins, centre, dout, events=(), mandatory=None, reference=None):
    """All samples are actual physics, not PWM-derived estimates."""
    from bench.world import wrap
    t, poses, margins = np.asarray(t), np.asarray(poses), np.asarray(margins)
    rel=poses[:,:2]-centre
    lateral=np.abs(rel @ np.array([dout[1],-dout[0]]))
    heading=np.abs(np.degrees(wrap(poses[:,2]-np.arctan2(dout[0],dout[1]))))
    aligned=(margins>=0)&(lateral<=1)&(heading<=5)
    stable=(rel@dout>=40)&aligned
    if reference is not None:
        from scipy.spatial import cKDTree
        reference=np.asarray(reference)
        distance,nearest=cKDTree(reference[:,:2]).query(poses[:,:2])
        aligned=(margins>=0)&(distance<=1)&(np.abs(np.degrees(wrap(poses[:,2]-reference[nearest,2])))<=5)
    def first_hold(mask,start=0):
        since=None
        for ti,ok in zip(t,mask):
            if ti<start: continue
            since=ti if ok and since is None else since if ok else None
            if since is not None and ti-since>=1-1e-9:return float(ti)
        return None
    exit_t=first_hold(stable)
    recovery=[]
    for event in events:
        start=event['t0']+event['duration']
        if event['t0']>=t[-1]:continue
        if start>t[-1]:
            recovery.append(dict(event_end_s=start,recovered=False,recovery_s=None,restricted_s=0.,
                                 interrupted_during_event=True,exposure_s=float(t[-1]-event['t0'])))
            continue
        rt=first_hold(aligned,start)
        recovery.append(dict(event_end_s=start, recovered=rt is not None,
                             recovery_s=None if rt is None else rt-start,
                             restricted_s=(t[-1] if rt is None else rt)-start))
    grid=np.arange(0,float(t[-1]),.1)
    v=np.interp(grid,t,velocity);w=np.interp(grid,t,omega)
    acc=np.diff(v)/.1;alpha=np.diff(w)/.1
    stopped=(abs(v)<.5)&(abs(w)<.05)&(grid>=1)
    translation_pause=(abs(v)<.5)&(grid>=1)
    if mandatory is not None: stopped &= ~np.interp(grid,t,np.asarray(mandatory,float)).astype(bool)
    if mandatory is not None: translation_pause &= ~np.interp(grid,t,np.asarray(mandatory,float)).astype(bool)
    edges=np.diff(np.r_[False,stopped,False].astype(int)); lengths=(np.where(edges==-1)[0]-np.where(edges==1)[0])*.1
    edges2=np.diff(np.r_[False,translation_pause,False].astype(int)); lengths2=(np.where(edges2==-1)[0]-np.where(edges2==1)[0])*.1
    return dict(stable_success=exit_t is not None, stable_exit_s=exit_t, clean_stable=exit_t is not None and margins.min()>=0,
                min_margin_cm=float(margins.min()), final_lateral_cm=float(lateral[-1]), final_heading_deg=float(heading[-1]),
                outside_s=float(np.sum(np.diff(t,prepend=0)*(margins<0))),
                accel_rms_cm_s2=float(np.sqrt(np.mean(acc*acc))), angular_accel_rms_rad_s2=float(np.sqrt(np.mean(alpha*alpha))),
                unnecessary_stops=int(sum(lengths>=.3-1e-9)), unnecessary_stop_s=float(sum(lengths[lengths>=.3-1e-9])),
                translation_pause_s=float(sum(lengths2[lengths2>=.3-1e-9])),
                recovery=recovery, unrecovered=sum(not x['recovered'] for x in recovery))


def bench_run(spec, out):
    """Observer adapter. Original engine performs every capture, update and decision."""
    os.environ['BENCH_CNN_MIXER']=spec.get('mixer','v4')
    if spec.get('mixer')=='direct':os.environ['BENCH_CNN_ONNX']=str(ROOT/'cnn_finetune/models/direct_v2/policy.onnx')
    else:os.environ.pop('BENCH_CNN_ONNX',None)
    import bench.junction as engine
    from bench.controllers import CONTROLLERS
    from bench.vehicle import Vehicle, PRESETS
    from bench.world import to_world
    cache=engine.load_plan_cache()
    # Cache changes computation time only. Default cache validated independently below.
    engine.make_route=lambda j,c,e,rim,track=None,radius=None:(cache[(j,c,e)][0].copy(),cache[(j,c,e)][2].copy())
    engine.turn_plans=lambda j,c,e,rim,track=None:cache[(j,c,e)][1]
    physical=[]; frames=[]; instances=[]
    from cnn_finetune.junction_feasibility import events_for
    events=events_for(spec['seed'],spec.get('composite','fixed'))
    class ObservedVehicle(Vehicle):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);instances.append(self)
        def step(self,t,dt,speed_scale=1):
            for e in events:
                if e['kind']=='power' and e['t0']<=t<e['t0']+e['duration']:speed_scale*=e['gain']
            result=super().step(t,dt,speed_scale)
            for e in events:
                if e['kind']!='push':continue
                overlap=max(0,min(t+dt,e['t0']+e['duration'])-max(t,e['t0']))
                if overlap:
                    fraction=overlap/e['duration'];yaw=self.pose[2]
                    self.pose[:2]+=e['lateral_cm']*fraction*np.array([np.cos(yaw),-np.sin(yaw)])
                    self.pose[2]=engine.wrap(self.pose[2]+np.radians(e['yaw_deg'])*fraction)
            physical.append([t+dt,*self.pose,self.v,self.omega])
            return result
    engine.Vehicle=ObservedVehicle
    Original=CONTROLLERS['hyb2j']
    class ObservedController(Original):
        def step(self,image,capture,done):
            pwm,info=super().step(image,capture,done)
            if spec.get('collect'):
                frames.append(dict(capture_s=capture,apply_s=done,image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                                   proposed_pwm=np.asarray(pwm).tolist(),applied_pwm=np.asarray(pwm).tolist(),
                                   n=self.n, map_pending=len(self.pending), map_pose=info.get('est'),
                                   allocator_state={k:v for k,v in vars(self.cnn_alloc).items() if isinstance(v,(float,int))}))
            return pwm,info
    if spec.get('collect'):CONTROLLERS['hyb2j']=ObservedController
    # The new scoring protocol permits the body to settle along the correct exit.
    if spec.get('stable'):
        BaseScorer=engine.JunctionScorer
        class StableScorer(BaseScorer):
            def sample(self,t,dt,pose,pwm):
                super().sample(t,dt,pose,pwm)
                rel=pose[:2]-self.centre;d=np.asarray(engine.ARM[self.exit_arm])
                good=(rel@d>=40 and self.track.clearance(to_world(self.rim,pose)).min()>=0 and abs(rel@np.array([d[1],-d[0]]))<=1
                      and abs(engine.wrap(pose[2]-engine.heading_of(d)))<=np.radians(5))
                self.stable_since=t if good and getattr(self,'stable_since',None) is None else getattr(self,'stable_since',None) if good else None
                if self.end=='correct_exit':
                    if self.stable_since is None or t-self.stable_since<1:self.end=None
        engine.JunctionScorer=StableScorer
    result=engine.run('hyb2j',PRESETS[spec.get('vehicle','field_estimate')],spec.get('effects',['realcam']),
                      spec['junction'],spec['came_from'],spec['exit'],'own',spec['seed'],spec.get('seconds',35),out,
                      variation=spec.get('variation','fixed'),video=spec.get('video',False))
    p=np.asarray(physical);c,di,do,arm=engine.movement(spec['junction'],spec['came_from'],spec['exit'])
    car=instances[-1];track=engine.Track()
    margins=np.array([track.clearance(to_world(car.footprint(),pose)).min() for pose in p[:,1:4]])
    route=cache[(spec['junction'],spec['came_from'],spec['exit'])][0];d=np.gradient(route,axis=0)
    result['new_metrics']=metrics(p[:,0],p[:,1:4],p[:,4],p[:,5],margins,c,do,[e for e in events if e['kind']=='push'],reference=np.c_[route,np.arctan2(d[:,0],d[:,1])])
    result['initial_margin_cm']=float(track.clearance(to_world(car.footprint(),result['initial_truth'])).min())
    dump(Path(out)/'events.json',events)
    np.savez_compressed(Path(out)/'physics.npz',samples=p,margins=margins)
    if spec.get('collect'):dump(Path(out)/'collection.json',frames)
    dump(Path(out)/'result.json',dict(spec=spec,**result))
    print(json.dumps(dict(out=str(out),outcome=result['outcome'],margin=result['min_margin_cm'],stable=result['new_metrics']['stable_success'])),flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('task',choices=['contract','bench']);a.add_argument('--spec');a.add_argument('--out');args=a.parse_args()
    if args.task=='contract':contract()
    else:bench_run(json.loads(args.spec),Path(args.out))


if __name__=='__main__':main()
