"""Bounded, offline hidden-response feasibility. No controller adoption/training."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from bench.world import Camera, Track, Renderer, axes, wrap
from bench.realcam import RealCam, pose_from_history
from bench.vehicle import Vehicle, PRESETS
from .goal_image_response import FloorMotion, aggregate, fit_bank, predict, NOMINAL

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'cnn_finetune/results/goal_hidden_response_20261113'
OUT = Path('D:/autodrive_cnn/runs/goal_hidden_response_20261113')
OLD = ROOT/'cnn_finetune/results/goal_path_following_pilot_20261113/vision_normal.json'
SOURCE = Path(__file__).with_name('goal_image_response.py')
PROBE_SEEDS = [24001, 24002]


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibration(cam):
    return {k:getattr(cam,k) for k in ['size','K','D','rect_K','M','ppcm','lat_min','fwd_max','lens_ahead']}


def setup():
    cv2.setNumThreads(1); cv2.setRNGSeed(24000); torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def preflight():
    setup(); cam=Camera(); renderer=Renderer(Track(),cam)
    rows=[]
    for shift in ([0,.4,0],[.3,0,0],[0,0,.01],[.1,.3,-.01]):
        observer=FloorMotion(calibration(cam)); p=np.array([0.,-40.,0.])
        observer.observe(renderer.render_raw(p),0.)
        r=observer.observe(renderer.render_raw(p+shift),.1)
        assert r['valid'], r
        error=np.asarray(r['delta'])-shift
        assert np.linalg.norm(error[:2])<.12 and abs(error[2])<.003,(shift,r)
        rows.append(dict(expected=shift,measured=r['delta']))
    commands=np.array([[0,0,0],[.2,.4,.4],[1,.6,.4],[2,.4,.6],[3,0,0],[3.5,.5,.5],[4.4,.4,.6]])
    intervals=[dict(start=t,end=t+.2,usable=True) for t in np.arange(.2,5,.2)]
    exact=dict(delay=.15,tau=.25,coefficients=[16,0,16*.6/13,0])
    values=predict(commands,intervals,exact)
    for r,(v,w) in zip(intervals,values):r.update(v=float(v),w=float(w))
    fit=fit_bank(commands,intervals,5.2)
    assert abs(fit['best']['delay']-.15)<.026 and abs(fit['best']['tau']-.25)<.051,fit['best']
    # Future data cannot affect an estimate at the same cutoff.
    early=fit_bank(commands,intervals,3.)
    changed=commands.copy();changed[changed[:,0]>3,1:]=-.9
    assert early==fit_bank(changed,intervals+[dict(start=8,end=9,usable=True,v=999,w=999)],3.)
    print(json.dumps(dict(geometry=rows,synthetic_identification=fit['best'],causal_fit=True)),flush=True)


def freeze():
    assert not REPORT.exists()
    REPORT.mkdir(); OUT.mkdir()
    deps=[Path(__file__),SOURCE,ROOT/'bench/world.py',ROOT/'bench/vehicle.py',ROOT/'bench/realcam.py',
          ROOT/'bench/world/camera_track.npz',ROOT/'bench/world/track.json',OLD]
    dump(REPORT/'protocol.json',dict(hashes={str(p):sha(p) for p in deps},
        purpose='Image+actual PWM only response identification, before any large training or adaptive closed-loop control',
        boundaries='Observer accepts raw images, capture timestamps, fixed camera calibration. Fitter accepts image increments and applied PWM timestamps only. Labels/physics isolated in scorer files.',
        natural_replay='24 existing vision runs, first12s re-rendered15Hz from saved physics. No map estimates or pose labels used in identification. Exploratory existing development seeds22001/22002.',
        fresh_probe='6 hidden configurations x2new seeds24001/24002 x typical/harsh_light=24; extra nominal/sluggish x2seeds shove=4. 12s fixed open-loop PWM excitation; diagnostic only, not a safe driving maneuver.',
        hidden=['nominal','low_gain','delay_only','lag_only','sluggish','unequal_sides'],
        camera='known fixed planar calibration; 15Hz; unknown actual exposure, fixed28ms midpoint approximation;70ms processing availability',
        fit='72 delay/lag candidates; robust regularized fit of forward and yaw coefficients; snapshots at2,4,6s. Six-second snapshot frozen for later scored interval.',
        prediction='6.2-12s held-time forced-response validation using subsequently issued actual PWM; no future PWM in parameter fitting. Not a future-command-free trajectory forecast.',
        metrics='visual increment fidelity; accepted coverage; delay/tau/gain errors and plausible sets; frozen response RMSE in forward cm/s, yaw rad/s and sqrt(mean(v_error^2+(13*w_error)^2)); before first sustained turn time',
        decision_gate={'normal_probe':'At least10/12 individual combined RMSE improvements; >=25% median reduction; <=10% nominal regression; >=90% visual interval coverage. Delay+lag sum absolute error<=.075s in at least10/12.',
                       'natural':'At least20% median sluggish improvement, no >10% median nominal regression; all fit cutoffs before corner entry needed for before-turn claim.',
                       'nuisance':'Report all lighting/shove regressions and false effective-lag changes>.1s or gain changes>15%; no automatic adoption even if identification gate passes.'},
        current_goal='paused',new_training=False,final_exam_used=False))
    for p in [Path(__file__),SOURCE]:(OUT/p.name).write_bytes(p.read_bytes())
    print('protocol frozen',flush=True)


def params():
    p=PRESETS['field_estimate']
    return dict(nominal=p,low_gain=replace(p,cm_s_per_duty_left=13.5,cm_s_per_duty_right=13.5),
                delay_only=replace(p,actuation_delay_s=.15),lag_only=replace(p,time_constant_s=.25),
                sluggish=PRESETS['sluggish'],unequal_sides=PRESETS['unequal_sides'])


def probe_commands():
    return np.array([[0,0,0],[.2,.38,.38],[.8,.55,.55],[1.5,.38,.38],[2.1,.50,.40],
        [2.8,.40,.50],[3.5,0,0],[3.9,.50,.50],[4.5,.38,.48],[5.2,.48,.38],
        [6,.45,.45],[7,.52,.36],[8,.36,.52],[9,.40,.40],[10,0,0],[11,.5,.5]])


def probe_physics(label,seed,condition):
    p=params()[label];rng=np.random.default_rng(seed)
    start=np.array([rng.uniform(-.7,.7),-45+rng.uniform(-1,1),rng.uniform(-.025,.025)])
    car=Vehicle(p,start); commands=probe_commands();i=0;records=[[0,*start,0,0]]
    history=[(0,start.copy())]
    # All parameters and true state stay on the data-generation/scoring side.
    for t in np.arange(0,12,.005):
        while i<len(commands) and commands[i,0]<=t+1e-9:
            car.command(commands[i,0],commands[i,1:]);i+=1
        car.step(float(t),.005)
        if condition=='shove' and abs(t-4.8)<1e-8:
            car.pose[:2]+=np.array([1.5,0.])@axes(car.pose[2]);car.pose[2]+=.035
        history.append((float(t+.005),car.pose.copy()))
        records.append([float(t+.005),*car.pose,float(car.v),float(car.omega)])
    return commands,history,np.asarray(records),p


def collect_one(name,commands,history,truth,seed,profile,metadata):
    dst=OUT/name
    if (dst/'complete.json').exists():return json.loads((dst/'complete.json').read_text())
    dst.mkdir(exist_ok=True)
    cam=Camera();camera=RealCam(Renderer(Track(),cam),cam,profile,seed)
    observer=FloorMotion(calibration(cam)); at=pose_from_history(history)
    rows=[]; images=[];captures=[]; started=time.perf_counter()
    for capture in np.arange(0,12+1e-8,1/15):
        image=camera.frame(float(capture),at,raw=True)
        r=observer.observe(image,float(capture))
        if r is not None:rows.append(r)
        images.append(image);captures.append(capture)
    # A self-contained, blinded input bundle. Scorer metadata is separate.
    observations=aggregate(rows)
    dump(dst/'input.json',dict(commands=commands.tolist(),motion=rows,observations=observations))
    np.savez_compressed(dst/'video.npz',images=np.asarray(images),capture=np.asarray(captures))
    np.savez_compressed(dst/'scorer.npz',truth=truth)
    dump(dst/'scorer.json',metadata)
    fits=[]
    for cutoff in [2.,4.,6.]:
        count=sum(r['end']+.070<=cutoff and r['usable'] and r['start']>=.2 for r in observations)
        fits.append(fit_bank(commands,observations,cutoff) if count>=8 else
                    dict(cutoff=cutoff,n=count,best=NOMINAL,insufficient_observations=True))
    dump(dst/'fits.json',fits)
    result=dict(name=name,artifact=str(dst),visual_frame_fraction=float(np.mean([r['valid'] for r in rows])),
                usable_intervals=len([r for r in observations if r['usable']]),seconds=time.perf_counter()-started)
    dump(dst/'complete.json',result)
    return result


def collect(group):
    setup();protocol=json.loads((REPORT/'protocol.json').read_text())
    for f,h in protocol['hashes'].items():assert sha(f)==h,f
    results=[]
    if group=='natural':
        for i,row in enumerate(json.loads(OLD.read_text())):
            artifact=Path(row['artifact']);records=json.loads((artifact/'records.json').read_text())
            commands=np.array([[r['done'],*r['applied']] for r in records])
            z=np.load(artifact/'physics.npz');truth=np.vstack([np.r_[0,z['initial'],0,0,0],z['samples']])
            history=[(r[0],r[1:4]) for r in truth]
            p=PRESETS[row['preset']]
            # Turn entry is an offline scorer fact, never fitter input.
            turn=next((r['done'] for r in records if abs(r['heading_rad'])>.12),None)
            metadata=dict(group=group,seed=row['seed'],vehicle=row['preset'],condition='normal',key=row['key'],
                          parameters=p.to_dict(),turn_proxy_s=turn,source=str(artifact))
            result=collect_one(f'natural_{i:02d}',commands,history,truth,row['seed'],'typical',metadata)
            results.append(result);print(group,len(results),'/24',result,flush=True)
            dump(REPORT/(group+'_progress.json'),results)
    else:
        cases=[(label,seed,condition) for seed in PROBE_SEEDS for label in params() for condition in ['normal','light']]
        cases += [(label,seed,'shove') for seed in PROBE_SEEDS for label in ['nominal','sluggish']]
        for i,(label,seed,condition) in enumerate(cases):
            commands,history,truth,p=probe_physics(label,seed,condition)
            metadata=dict(group=group,seed=seed,vehicle=label,condition=condition,parameters=p.to_dict())
            result=collect_one(f'probe_{i:02d}',commands,history,truth,seed,'harsh_light' if condition=='light' else 'typical',metadata)
            results.append(result);print(group,len(results),'/28',label,condition,result,flush=True)
            dump(REPORT/(group+'_progress.json'),results)
    dump(REPORT/(group+'.json'),results)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('task',choices=['preflight','freeze','collect'])
    parser.add_argument('--group',choices=['natural','probe']);args=parser.parse_args()
    if args.task=='preflight':preflight()
    elif args.task=='freeze':freeze()
    else:collect(args.group)
