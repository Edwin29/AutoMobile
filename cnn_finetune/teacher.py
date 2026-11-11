"""Truth-pose teacher prototype; simulation only, clockwise curvature positive."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.optimize import least_squares
from bench.controllers import TeamCNN, HybridCNN, VENDOR_CNN, VENDOR_V4
from bench.world import Camera, axes
sys.path.insert(0, str(VENDOR_V4))
from drive_v4.core import MotorModel, load_config


class Teacher:
    def __init__(self, direction, lookahead=18.0):
        route = np.asarray(Camera().route)
        self.route = route if direction == 'cw' else route[::-1]
        self.lookahead = lookahead
        self.motor = MotorModel(dict(load_config()['motor']))
        self.team = TeamCNN.__new__(TeamCNN)
        self.team.mix = json.loads((VENDOR_CNN / 'configs/real_car.json').read_text(encoding='utf-8'))['duty_mixer']
        self.adapter = HybridCNN.__new__(HybridCNN)
        self.adapter.driver = SimpleNamespace(motor=self.motor)
        self.adapter.cnn_alloc = MotorModel(dict(self.motor.c))

    def target(self, pose):
        r = self.route
        i = int(np.argmin(np.sum((r - pose[:2])**2, axis=1)))
        distance = 0.0
        j = i
        while distance < self.lookahead:
            nxt = (j + 1) % len(r)
            distance += np.linalg.norm(r[nxt] - r[j])
            j = nxt
        delta = (r[j] - pose[:2]) @ axes(pose[2]).T
        k = float(2 * delta[0] / max(np.dot(delta, delta), 1e-9))
        return float(np.clip(k, -0.055, 0.055)), 0.43

    def team_pwm(self, label):
        linear, angular = label
        m = self.team.mix
        sides = np.array([linear - m['turn_mix']*angular, linear + m['turn_mix']*angular])
        sides /= max(1.0, float(np.abs(sides).max()))
        return np.array([self.team._side(sides[0], m['trim_left']), self.team._side(sides[1], m['trim_right'])])

    def inverse(self, k, mean):
        k = float(np.clip(k, -0.055, 0.055))
        m = self.team.mix
        assert m['trim_left'] == m['trim_right'] == 1 and m['deadband'] == 0
        ratio = k*self.motor.c['track_cm']/(2*self.motor.yaw_gain(k))
        factors = np.array([1+ratio, 1-ratio])
        # Exact solution of the constrained scalar quadratic: project desired
        # mean onto the feasible interval at fixed curvature. Negative/zero
        # wheel branches cannot express positive speed at |k| <= .055.
        lo = float(np.max((m['min_duty']+1e-10)/factors))
        hi = float(np.min(m['max_duty']/factors))
        if lo > hi:
            raise ValueError('Infeasible curvature')
        used = float(np.clip(mean, lo, hi))
        sides = (used*factors-m['min_duty'])/(m['max_duty']-m['min_duty'])
        label = np.array([sides.mean(), (sides[1]-sides[0])/(2*m['turn_mix'])])
        assert np.max(np.abs(label)) <= 1+1e-9
        return label

    def step(self, pose):
        k, mean = self.target(pose)
        label = self.inverse(k, mean)
        pwm = self.adapter._remap_cnn(self.team_pwm(label))
        return pwm, dict(k=k, requested_mean=mean, used_mean=float(self.team_pwm(label).mean()), label=label.tolist())


def check_inverse():
    teacher = Teacher('cw')
    rows = []
    for k in [-0.085, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.085]:
        requested_k = k
        k = float(np.clip(k, -0.055, 0.055))
        label = teacher.inverse(k, 0.43)
        pre = teacher.team_pwm(label)
        pre_mean = float(pre.mean())
        raw = float((pre[0]-pre[1])/teacher.motor.c['track_cm'])
        pre_k = raw*teacher.motor.yaw_gain(raw)/max(pre_mean, 1e-9)
        teacher.adapter.cnn_alloc.dither = 0.0
        pwm = np.array([teacher.adapter._remap_cnn(teacher.team_pwm(label)) for _ in range(1000)])
        velocities = np.array([teacher.motor.velocity(p) for p in pwm])
        actual_k = float(velocities[:,1].mean()/velocities[:,0].mean())
        actual_mean = float(pwm.mean())
        rows.append(dict(requested_k=requested_k, clipped_normal=requested_k != k, k=k, requested_mean=0.43, label=label.tolist(),
                         pre_k=pre_k, pre_mean=pre_mean, team_pwm=pre.tolist(), actual_k=actual_k, actual_mean=actual_mean,
                         passed=abs(pre_k-k)<1e-4 and abs(actual_k-k)<=max(abs(k)*0.05, 1e-8)))
    return dict(ok=all(r['passed'] for r in rows), samples=rows,
                note='Actual TeamCNN side conversion and HybridCNN remap, 1000-frame dither averages')

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-out', required=True)
    args = ap.parse_args()
    result = check_inverse()
    Path(args.check_out).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['ok'] else 1)

