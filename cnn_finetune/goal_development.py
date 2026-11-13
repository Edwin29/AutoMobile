"""Development diagnostics for the user-approved bench goal; never a final exam.

Keep the first pilot immutable. End each local diagnostic at the first 55cm
exit gate, before the next junction, and retain the old 1cm metric separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from bench.junction import ARM, heading_of
from bench.world import to_world, wrap
from .junction_learning import Scene, Encoder, model_at, setup, KEYS, ROOT, OUT as OLD_OUT
from .junction_stage02 import dump

OUT = Path('D:/autodrive_cnn/runs/goal_development_20261113')
REPORT = ROOT/'cnn_finetune/results/goal_development_20261113'
SEEDS = [4101, 4102]
PRESETS = ['field_estimate', 'sluggish', 'unequal_sides']
CONDITIONS = ['normal', 'mild', 'mild_light']
MODELS = {'bc': OLD_OUT/'models/bc/epoch_4.pt', 'outcome': OLD_OUT/'models/outcome/epoch_8.pt'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def terminal_at_gate(forward, lateral, heading_deg, margin, held_s):
    """Pure scoring rule. Called at each physics sample, never passed to policy."""
    if forward < 55:
        return None
    if abs(lateral) >= 15:
        return 'missed_exit'
    return 'stable_exit' if margin >= 0 and abs(heading_deg) <= 10 and held_s >= 1 else 'unstable_exit'


class LocalScene(Scene):
    def advance_to(self, target):
        while self.t < target-1e-10 and self.outcome == 'time_limit':
            dt = min(.02, target-self.t)
            scale = 1.
            for e in self.events:
                if e['kind'] == 'power' and e['t0'] <= self.t < e['t0']+e['duration']:
                    scale *= e['gain']
            self.car.step(self.t, dt, scale)
            for e in self.events:
                if e['kind'] != 'push':
                    continue
                overlap = max(0, min(self.t+dt, e['t0']+e['duration'])-max(self.t, e['t0']))
                if overlap:
                    f = overlap/e['duration']; yaw = self.car.pose[2]; self.exposure += overlap
                    self.car.pose[:2] += e['lateral_cm']*f*np.array([np.cos(yaw), -np.sin(yaw)])
                    self.car.pose[2] = wrap(self.car.pose[2]+np.radians(e['yaw_deg'])*f)
            self.t += dt
            pose = self.car.pose
            clear = self.track.clearance(to_world(self.rim, pose)); margin = float(clear.min())
            rel = pose[:2]-self.c
            forward = float(rel@self.do)
            lateral = float(rel@np.array([self.do[1], -self.do[0]]))
            heading = float(np.degrees(wrap(pose[2]-heading_of(self.do))))
            self.physics.append([self.t, *pose, self.car.v, self.car.omega, margin])
            self.history.append((self.t, pose.copy()))
            if len(self.history) > 400:
                del self.history[:200]
            good = forward >= 40 and abs(lateral) < 15 and margin >= 0 and abs(heading) <= 10
            self.since = self.t if good and self.since is None else self.since if good else None
            held = 0. if self.since is None else self.t-self.since
            terminal = terminal_at_gate(forward, lateral, heading, margin, held)
            if margin < -12 or clear.max() < 0:
                self.outcome = 'severe_departure'
            elif terminal:
                self.outcome = terminal
            else:
                for arm in self.track.data['junctions'][self.key[0]]['arms']:
                    d = np.asarray(ARM[arm])
                    if arm != self.arm and rel@d >= 40 and abs(rel@np.array([d[1], -d[0]])) < 15 and abs(wrap(pose[2]-heading_of(d))) < np.radians(30):
                        self.outcome = 'wrong_exit'

    def summary(self):
        r = super().summary()
        r['old_strict_stable'] = r['stable_success']
        r['old_strict_clean'] = r['clean_stable']
        r['stable_success'] = self.outcome == 'stable_exit'
        initial_margin = float(self.track.clearance(to_world(self.rim, self.initial_pose)).min())
        r['initial_margin_cm'] = initial_margin
        r['initial_valid'] = initial_margin >= 0
        r['clean_stable'] = r['stable_success'] and min(r['min_margin_cm'], initial_margin) >= 0
        r['local_gate_cm'] = 55
        return r

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_pose = self.car.pose.copy()


def freeze():
    REPORT.mkdir(parents=True, exist_ok=False); OUT.mkdir(parents=True, exist_ok=False)
    thresholds = dict(normal_clean_mission=.99, mild_compound_clean_mission=.95,
                      harsh_feasible_clean_mission=.80, mild_each_required_movement=.90,
                      normal_mild_wrong_exit_count=0, normal_mild_severe_count=0,
                      normal_mild_nonpivot_nonstop_turn_fraction=.95,
                      min_training_initializations=3, min_final_missions_per_condition=100)
    dump(REPORT/'acceptance.json', dict(schema=1, status='thresholds_locked_full_mission_manifest_pending',
         scope='bench-only development; not physical deployment', thresholds=thresholds,
         student_inputs=['camera history', 'previous sent PWM', 'direction command'],
         forbidden_student_inputs=['truth pose', 'truth velocity', 'junction ID', 'route progress', 'simulator parameters'],
         final_reserved_seed_range=[900000, 999999], final_runs_executed=0,
         safeguards=['No final seed used for training or selection', 'No removal of hard cases after observing candidate results',
                     'Freeze route manifest, feasible harsh envelope and time limits before final exam',
                     'Report full mission and each required movement, not four-movement pilot as completion',
                     'Report confidence intervals and training seed variation',
                     'Do not replace existing driving model or deploy to Pi'],
         smoothness_gate='paired angular acceleration lower than legacy; no safety or travel-time regression; mandatory stops excluded',
         user_decision_gate='Report before changing acceptance targets or fundamentally changing task/model input contract'))
    protocol = dict(schema=1, purpose='development diagnosis, NOT final acceptance test', seeds=SEEDS,
         keys=KEYS, presets=PRESETS, conditions=CONDITIONS, policies=['teacher', *MODELS],
         design='full cross of seed, vehicle, movement, condition; paired initial states and disturbances',
         local_metric=dict(first_exit_gate_cm=55, stable_from_cm=40, heading_deg=10, hold_s=1,
                           margin_cm=0, max_s=40, lateral='body inside road; 15cm arm assignment bound only',
                           old_metric_retained=True, initial_invalid='retain and report; cannot count as clean success'),
         no_training_in_this_phase=True, checkpoint_sha256={k: sha(v) for k,v in MODELS.items()},
         source_sha256=sha(__file__))
    dump(REPORT/'protocol.json', protocol)
    snap=OUT/'source_snapshot'; snap.mkdir()
    for name in ['goal_development.py','junction_learning.py','junction_learning_train.py','trajectory_loss.py','junction_stage02.py','junction_feasibility.py']:
        shutil.copy2(ROOT/'cnn_finetune'/name, snap/name)
    print('diagnostic protocol frozen', flush=True)


def evaluate(arm):
    setup(); protocol=json.loads((REPORT/'protocol.json').read_text())
    assert sha(__file__)==protocol['source_sha256'], 'Freeze a new revision if diagnostic semantics change'
    encoder=Encoder().cuda().eval(); model=None if arm=='teacher' else model_at(MODELS[arm])
    if model is not None:
        assert sha(MODELS[arm]) == protocol['checkpoint_sha256'][arm]
    rows=[]
    for seed in SEEDS:
        for preset in PRESETS:
            for key in KEYS:
                for condition in CONDITIONS:
                    dst=OUT/'rollouts'/arm/('_'.join(key)+f'_{condition}_{seed}_{preset}')
                    if (dst/'result.json').exists():
                        r=dict(json.loads((dst/'result.json').read_text()), artifact=str(dst))
                    else:
                        env=LocalScene(key,condition,seed,preset,encoder).finish(model)
                        r=env.save(dst,save_images=arm=='teacher')
                    rows.append(r)
            print(arm,seed,preset,'clean',sum(r['clean_stable'] for r in rows),'/',len(rows),flush=True)
            dump(REPORT/f'{arm}_progress.json',rows)
    dump(REPORT/f'{arm}.json',rows)


def analyze():
    from collections import Counter
    data={a:json.loads((REPORT/f'{a}.json').read_text()) for a in ['teacher',*MODELS]}
    def stats(rows):
        return dict(n=len(rows),stable=sum(r['stable_success'] for r in rows),clean=sum(r['clean_stable'] for r in rows),
                    initial_invalid=sum(not r['initial_valid'] for r in rows),outcomes=dict(Counter(r['outcome'] for r in rows)))
    result={a:dict(overall=stats(rs),conditions={c:stats([r for r in rs if r['condition']==c]) for c in CONDITIONS},
                  movements={'_'.join(k):stats([r for r in rs if r['key']==list(k)]) for k in KEYS},
                  presets={p:stats([r for r in rs if r['preset']==p]) for p in PRESETS}) for a,rs in data.items()}
    for i,t in enumerate(data['teacher']):
        for arm in MODELS:
            r=data[arm][i]
            assert (r['key'],r['seed'],r['preset'],r['condition']) == (t['key'],t['seed'],t['preset'],t['condition'])
            ap=np.load(Path(r['artifact'])/'physics.npz')['samples'];bp=np.load(Path(t['artifact'])/'physics.npz')['samples']
            np.testing.assert_array_equal(ap[0,1:4],bp[0,1:4])
            assert json.loads((Path(r['artifact'])/'events.json').read_text())==json.loads((Path(t['artifact'])/'events.json').read_text())
    dump(REPORT/'summary.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['freeze','evaluate','analyze']);p.add_argument('--arm',choices=['teacher',*MODELS]);a=p.parse_args()
    if a.task=='freeze':freeze()
    elif a.task=='evaluate':evaluate(a.arm)
    else:analyze()
