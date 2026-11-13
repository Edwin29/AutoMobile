"""Replay exact failure prefixes, then intervene without resetting any runtime state.

Truth is available ONLY to the labelled diagnostic teacher and offline scoring.
Student deployment code and frozen weights are not modified.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    dict(name='bright_sluggish_cw', condition='mild_combined_bright_sluggish', seed=613, direction='cw', start=94.),
    dict(name='bright_sluggish_ccw', condition='mild_combined_bright_sluggish', seed=613, direction='ccw', start=57.),
    dict(name='mild_combined_ccw', condition='mild_combined', seed=601, direction='ccw', start=87.),
]
MODES = ['student', 'teacher', 'teacher_burst', 'hold', 'slow', 'left', 'right']
BURST = .72
HORIZON = 15.


def run_case(spec, out):
    import copy
    from collections import deque
    import numpy as np
    import bench.run as engine
    from bench.controllers import CONTROLLERS, HybridCNN
    from bench.vehicle import Vehicle
    from bench.world import wrap, to_world, Track
    from bench.realcam import pose_from_history
    from .direct_teacher import DirectTeacher
    from .model import preprocess_bgr
    from .robustness_compare import conditions, schedule

    out = Path(out)
    c = next(c for c in conditions() if c['condition'] == spec['condition'])
    events = schedule(spec['seed'], c['push'], c['power'])
    holder = {}
    samples = []
    actions = []
    references = []
    observations = []
    window = []
    telemetry = []
    start = spec['start']
    track = Track()
    prefix = hashlib.sha256()

    class InstrumentedVehicle(Vehicle):
        def __init__(self, *args):
            super().__init__(*args)
            holder['car'] = self
            self.history = deque([(0., self.pose.copy())], maxlen=100)

        def step(self, t, dt, speed_scale=1.):
            scale = 1.
            for e in events:
                if e['kind'] == 'power' and e['t0'] <= t < e['t0'] + e['duration']:
                    scale *= e['gain']
            super().step(t, dt, speed_scale * scale)
            for e in events:
                if e['kind'] != 'push':
                    continue
                overlap = max(0., min(t+dt, e['t0']+e['duration'])-max(t, e['t0']))
                if overlap:
                    fraction = overlap/e['duration']
                    yaw = float(self.pose[2])
                    self.pose[:2] += e['lateral_cm']*fraction*np.array([np.cos(yaw), -np.sin(yaw)])
                    self.pose[2] = wrap(self.pose[2] + np.radians(e['yaw_deg'])*fraction)
            self.history.append((t+dt, self.pose.copy()))
            if t+dt >= start:
                window.append(dict(t=t+dt, dt=dt,
                    margin=float(track.clearance(to_world(self.footprint(), self.pose)).min()),
                    pose=self.pose.tolist(), speed=self.speed.tolist()))
            return self.pose

    class DiagnosticHybrid(HybridCNN):
        def __init__(self, direction, hint, timing):
            super().__init__(direction, hint, timing)
            self.teacher = DirectTeacher(direction, variant=3)
            self.teacher.sim_params = holder['car'].p
            self.teacher_started = False
            self.initial = None
            self.history_images = deque(maxlen=12)
            self.base_step = self.cnn.step
            self.cnn.step = self.intercept

        def intercept(self, image, capture, done):
            pwm, info = self.base_step(image, capture, done)
            car = holder['car']
            ref = np.array([info['linear'], info['angular']], dtype=float)
            chosen = ref.copy()
            x = np.rint(preprocess_bgr(image)*255).astype(np.uint8)
            self.history_images.append(x)
            active = capture >= start
            if not active:
                prefix.update(np.asarray([capture, *car.pose, *car.speed], dtype='<f8').tobytes())
                prefix.update(image.tobytes())
                prefix.update(ref.astype('<f8').tobytes())
            else:
                if self.initial is None:
                    self.initial = ref.copy()
                    # Runtime state is preserved. This is evidence, not a restart checkpoint.
                    state = dict(capture=capture, done=done, pose=car.pose.tolist(), speed=car.speed.tolist(),
                        command_queue=[(t, p.tolist()) for t, p in car.commands], wander=car.wander,
                        rng=car.rng.bit_generator.state, prefix_sha256=prefix.hexdigest(),
                        initial_action=ref.tolist(), previous_pwm=car._active(capture, 0).tolist())
                    (out/'intervention_state.json').write_text(json.dumps(state, indent=2))
                    np.savez_compressed(out/'history.npz', x=np.array(self.history_images))
                    # Same pose/image can conceal different wheel-speed histories.
                    # Counterfactual physical-state audit, not a trained temporal-policy evaluation.
                    hidden=[]
                    for factor in [.6, 1., 1.4]:
                        clone=copy.deepcopy(car)
                        clone.speed=clone.speed*factor
                        before=clone.pose.copy()
                        for n in range(25):
                            Vehicle.step(clone, done+n*.02, .02)
                        hidden.append(dict(speed_factor=factor, displacement_cm=(clone.pose[:2]-before[:2]).tolist(),
                            yaw_change_deg=float(np.degrees(wrap(clone.pose[2]-before[2])))))
                    (out/'hidden_dynamics.json').write_text(json.dumps(hidden, indent=2))
                burst = capture < start+BURST
                mode = spec['mode']
                use_teacher = mode == 'teacher' or (mode == 'teacher_burst' and burst)
                teacher_label = None
                if use_teacher:
                    if not self.teacher_started:
                        self.teacher.motor.dither = self.cnn_alloc.dither
                        self.teacher_started = True
                    self.teacher.last_pwm = car._active(capture, 0).copy()
                    _, ti = self.teacher.step(pose_from_history(list(car.history))(capture))
                    chosen = np.array(ti['label'])
                    teacher_label = chosen.tolist()
                elif burst:
                    offsets = {'hold': [0.,0.], 'slow': [-.1,0.], 'left': [0.,.15], 'right': [0.,-.15]}
                    if mode in offsets:
                        chosen = self.initial + offsets[mode]
                        chosen = np.clip(chosen, [0.,-.085/.095], [1.,.085/.095])
                if burst:
                    samples.append(x); actions.append(chosen.copy()); references.append(ref.copy())
                    observations.append(dict(capture=capture, done=done, pose=car.pose.tolist(), speed=car.speed.tolist()))
                telemetry.append(dict(capture=capture, reference=ref.tolist(), chosen=chosen.tolist(), teacher=teacher_label))
            info['linear'], info['angular'] = map(float, chosen)
            return pwm, info

    engine.Vehicle = InstrumentedVehicle
    CONTROLLERS['hyb2'] = DiagnosticHybrid
    summary = engine.run('hyb2', engine.vehicle_from_args(c['vehicle'], None), c['effects'],
        spec['direction'], 'own', spec['seed'], start+HORIZON, out, variation=c['variation'])
    np.savez_compressed(out/'segment.npz', x=np.array(samples), actions=np.array(actions,np.float32),
        reference=np.array(references,np.float32))
    (out/'window.json').write_text(json.dumps(window))
    (out/'telemetry.json').write_text(json.dumps(telemetry))
    (out/'observations.json').write_text(json.dumps(observations))
    summary.update(spec=spec, prefix_sha256=prefix.hexdigest(), intervention_samples=len(samples),
                   scheduled_events=events, diagnostic_only_truth_teacher=True)
    (out/'result.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k:summary[k] for k in ['spec','outcome','verified_lap','min_margin_cm']}), flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out', required=True);p.add_argument('--spec');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();out=Path(a.out)
    if a.spec:
        run_case(json.loads(a.spec),out);return
    out.mkdir(parents=True,exist_ok=False)
    specs=[dict(c,mode=m) for c in CASES for m in MODES]
    protocol=dict(cases=CASES,modes=MODES,burst_s=BURST,horizon_s=HORIZON,
        description='Exact replay prefix, constant initial action candidates for .72s then frozen student continuation; teacher has truth pose but same allocator and guard.',
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),ROOT/'cnn_finetune/models/direct_v2/policy.onnx',ROOT/'bench/run.py',ROOT/'bench/controllers.py',ROOT/'bench/vehicle.py']})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    def one(sp):
        name=sp['name']+'__'+sp['mode'];dst=out/name
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONIOENCODING='utf-8',
            BENCH_CNN_SPEED='1.0',BENCH_CNN_MIXER='direct',BENCH_CNN_ONNX=str(ROOT/'cnn_finetune/models/direct_v2/policy.onnx'))
        with (out/(name+'.log')).open('w',encoding='utf-8') as log:
            proc=subprocess.run([sys.executable,'-m','cnn_finetune.recovery_diagnosis','--out',str(dst),'--spec',json.dumps(sp)],
                                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        print(json.dumps(dict(name=name,returncode=proc.returncode)),flush=True)
        return dict(name=name,returncode=proc.returncode)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        results=list(pool.map(one,specs))
    (out/'execution.json').write_text(json.dumps(results,indent=2))
    if any(r['returncode'] for r in results):raise SystemExit(1)


if __name__=='__main__':main()
