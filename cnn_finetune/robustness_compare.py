"""Frozen-model paired robustness study; additive external-force test adapter only.

Run with the desktop venv. Raw runs stay outside Git. Never invokes vehicle hardware.
"""
import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [601, 607, 613]
ARMS = ['a5b_v4', 'direct_v2']


def conditions():
    out = []
    def add(name, group, vehicle='field_estimate', camera='realcam', variation='fixed', push='none', power='none', extra=None):
        out.append(dict(condition=name, group=group, vehicle=vehicle, effects=[camera]+(extra or []), variation=variation, push=push, power=power))
    for v in ['field_estimate', 'weak_left_turn', 'sluggish', 'unequal_sides']:
        add('normal_'+v, 'normal', vehicle=v)
    add('mild_push', 'single', push='mild')
    add('mild_power', 'single', power='mild')
    add('strong_light', 'single', camera='realcam_light')
    add('mild_vehicle', 'single', variation='mild')
    add('mild_combined', 'combined', variation='mild', push='mild', power='mild')
    for v in ['field_estimate', 'sluggish', 'unequal_sides']:
        add('mild_combined_bright_'+v, 'combined', vehicle=v, camera='realcam_light', variation='mild', push='mild', power='mild')
    for f in ['shadow', 'exposure', 'blur', 'optics']:
        add('camera_'+f, 'camera_factor', camera='realcam_'+f)
    add('harsh_camera', 'harsh', camera='realcam_harsh')
    add('harsh_vehicle', 'harsh', variation='harsh')
    add('harsh_combined', 'harsh', camera='realcam_harsh', variation='harsh', push='harsh', power='harsh')
    add('whiteout_3s', 'harsh', extra=['whiteout_midturn'])
    return out


def schedule(seed, push, power):
    import numpy as np
    rng = np.random.default_rng(seed*65537+19)
    events=[]
    # Independent of controller, trajectory and outcomes. Both arms see identical draws.
    for kind, level in [('push',push),('power',power)]:
        if level=='none':continue
        t=12.+float(rng.uniform(0,5))
        while t<195:
            if kind=='push':
                amplitude=(1.,2.) if level=='mild' else (4.,6.)
                angle=(1.,3.) if level=='mild' else (6.,10.)
                event=dict(kind=kind,t0=t,duration=float(rng.uniform(.3,.7)),
                           lateral_cm=float(rng.choice([-1,1])*rng.uniform(*amplitude)),
                           yaw_deg=float(rng.choice([-1,1])*rng.uniform(*angle)))
            else:
                amplitude=(.05,.15) if level=='mild' else (.20,.35)
                event=dict(kind=kind,t0=t,duration=float(rng.uniform(2,5)),
                           gain=float(1+rng.choice([-1,1])*rng.uniform(*amplitude)))
            events.append(event);t+=float(rng.uniform(15,25))
    return sorted(events,key=lambda e:e['t0'])


def worker(spec, dst):
    import numpy as np
    import bench.run as engine
    from bench.vehicle import Vehicle as OriginalVehicle
    from bench.world import wrap
    events=schedule(spec['seed'],spec['push'],spec['power'])
    applied=[]
    class ForcedVehicle(OriginalVehicle):
        def step(self,t,dt,speed_scale=1.):
            scale=1.
            for e in events:
                if e['kind']=='power' and e['t0']<=t<e['t0']+e['duration']:scale*=e['gain']
            super().step(t,dt,speed_scale*scale)
            for e in events:
                if e['kind']!='push':continue
                overlap=max(0.,min(t+dt,e['t0']+e['duration'])-max(t,e['t0']))
                if overlap:
                    # Distributed external body-local side slip, plus yaw kick.
                    # It is an imposed disturbance, not a tire/contact physics model.
                    fraction=overlap/e['duration'];yaw=float(self.pose[2])
                    self.pose[:2]+=e['lateral_cm']*fraction*np.array([np.cos(yaw),-np.sin(yaw)])
                    self.pose[2]=wrap(self.pose[2]+np.radians(e['yaw_deg'])*fraction)
                    applied.append(dict(t=t,dt=overlap,lateral_cm=e['lateral_cm']*fraction,yaw_deg=e['yaw_deg']*fraction))
            return self.pose
    engine.Vehicle=ForcedVehicle
    result=engine.run('hyb2',engine.vehicle_from_args(spec['vehicle'],None),spec['effects'],spec['direction'],
                      'own',spec['seed'],200,dst,variation=spec['variation'])
    (dst/'external_events.json').write_text(json.dumps(dict(scheduled=events,applied_push=applied),indent=2))
    from cnn_finetune.compare_metrics import extra
    result['extra']=extra(dst,spec['direction'])
    result['spec']=spec
    (dst/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(lap=result['verified_lap'],margin=result['min_margin_cm'])),flush=True)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def resource_ok():
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ['total','available','tp','ap','tv','av','ex']]
    m=Memory();m.length=ctypes.sizeof(m);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.available>=2*1024**3 and shutil.disk_usage('D:/').free>=50*1024**3


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=6);ap.add_argument('--spec');a=ap.parse_args()
    out=Path(a.out)
    if a.spec:
        worker(json.loads(a.spec),out);return
    out.mkdir(parents=True,exist_ok=True)
    specs=[]
    for c in conditions():
        for seed in SEEDS:
            for direction in ['cw','ccw']:
                for arm in ARMS:specs.append(dict(c,seed=seed,direction=direction,arm=arm))
    sources=[p for folder in ['bench','drive_v4','cnn_finetune'] for p in (ROOT/folder).rglob('*') if p.is_file() and (p.suffix in {'.py','.onnx','.npz','.json','.yaml','.txt','.jpg'} and '__pycache__' not in str(p) and 'results' not in p.parts)]
    protocol=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  seeds=SEEDS,conditions=conditions(),runs=len(specs),controller='hyb2',timing='own/36ms',seconds=200,
                  source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},
                  external_disturbance='Synthetic sensitivity assumptions, not measured field frequencies: mild 1-2cm lateral/1-3deg yaw over 0.3-0.7s, output +/-5-15% over 2-5s; harsh 4-6cm/6-10deg and +/-20-35%. Repeat gaps 15-25s. Schedules start 12-17s.',
                  selection='No tuning or retraining. Seeds excluded from prior training/development; world, scoring and controllers unchanged.',
                  metrics='Clean success requires verified_lap AND min_margin_cm>=0. Recovery computed with existing compare_metrics; also report unrecovered_at_end and failed runs separately.')
    pp=out/'protocol.json'
    if pp.exists():
        assert json.loads(pp.read_text())==protocol,'Protocol changed; use a new output directory.'
    else:pp.write_text(json.dumps(protocol,indent=2))
    (out/'specs.json').write_text(json.dumps(specs,indent=2))
    print(f'PROTOCOL {len(specs)} runs, {a.workers} workers',flush=True)
    def one(sp):
        if not resource_ok():return dict(spec=sp,error='Resource gate; not launched')
        slug=f"{sp['condition']}__s{sp['seed']}__{sp['direction']}__{sp['arm']}"
        dst=out/slug
        if (dst/'result.json').exists():return dict(spec=sp,path=str(dst),cached=True)
        if dst.exists():return dict(spec=sp,error='Partial directory preserved; explicit retry needed')
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONIOENCODING='utf-8',BENCH_CNN_SPEED='1.0',BENCH_CNN_MIXER='v4' if sp['arm']=='a5b_v4' else 'direct')
        env.pop('BENCH_CNN_ONNX',None)
        if sp['arm']=='direct_v2':env['BENCH_CNN_ONNX']=str(ROOT/'cnn_finetune/models/direct_v2/policy.onnx')
        with (out/(slug+'.log')).open('w',encoding='utf-8') as f:
            r=subprocess.run([sys.executable,'-m','cnn_finetune.robustness_compare','--out',str(dst),'--spec',json.dumps(sp)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        return dict(spec=sp,path=str(dst),returncode=r.returncode)
    completed=[];start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(one,sp) for sp in specs]
        for future in concurrent.futures.as_completed(futures):
            row=future.result();completed.append(row)
            (out/'progress.json').write_text(json.dumps(dict(completed=completed,total=len(specs),elapsed_s=time.monotonic()-start),indent=2))
            sp=row['spec'];p=Path(row.get('path',''))/'result.json'
            if p.exists():
                s=json.loads(p.read_text());print(f"{len(completed)}/{len(specs)} {sp['condition']} s{sp['seed']} {sp['direction']} {sp['arm']} lap={s['verified_lap']} margin={s['min_margin_cm']:.3f}",flush=True)
            else:print('ERROR '+json.dumps(row),flush=True)
    failures=[r for r in completed if r.get('error') or r.get('returncode',0)]
    print(f'FINISHED failures={len(failures)} elapsed_s={time.monotonic()-start:.0f}',flush=True)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
