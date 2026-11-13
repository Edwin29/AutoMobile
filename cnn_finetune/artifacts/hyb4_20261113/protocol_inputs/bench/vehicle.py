"""The VARIABLE part: how the car physically responds to PWM.

Every parameter is set by the experiment, because the real values are uncertain
(and change with battery, floor, tyres). Field evidence for reference only:
  * all four wheels start at PWM 0.30 (2026-11-10 threshold test)
  * achieved turn ~0.58-0.60 of an ideal differential drive (v4 live 195808),
    v2 learned left ~0.57-0.68, right 0.62 (2026-11-06)
  * straight progress ~1.05x of 16 cm/s per duty (v4 live 195808)
The drivers never see these values; they only see camera images.
"""
from dataclasses import asdict, dataclass, field

import numpy as np

from .world import advance


@dataclass
class VehicleParams:
    # Output range: a side's wheels do not turn below start_duty; above max_duty
    # the extra command gives no extra speed (saturation).
    start_duty_left: float = 0.30
    start_duty_right: float = 0.30
    max_duty: float = 1.0
    # Speed: cm/s per unit duty, per side (unequal sides = steady drift).
    cm_s_per_duty_left: float = 16.0
    cm_s_per_duty_right: float = 16.0
    # Turn efficiency: fraction of the ideal differential-drive yaw that is achieved,
    # separately for left (counter-clockwise) and right turns (skid-steer slip).
    turn_eff_left: float = 1.0
    turn_eff_right: float = 1.0
    # Constant yaw bias while moving, rad/s at 5 cm/s, scaled by speed (+ = right).
    drift_rad_s: float = 0.0
    # Response: pure command delay, then first-order wheel-speed lag.
    actuation_delay_s: float = 0.0
    time_constant_s: float = 0.0
    # Geometry (measured).
    track_cm: float = 13.0
    front_cm: float = 14.5
    rear_cm: float = 10.0
    half_width_cm: float = 8.0
    # Turning in place (wheels opposite): the body slides sideways by this much per
    # radian turned, in one direction per car (field: pivoting works, with some wheel
    # slip / body slide; size not measured). 0 = ideal pivot.
    pivot_slide_cm_per_rad: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass
class Variation:
    """How the car's personality CHANGES during a run (all zero = fixed car).

    wander_*   slow random wander around the base value (Ornstein-Uhlenbeck, time
               constant wander_tau_s): wheel/floor condition. Relative for speed,
               absolute for the rest.
    trend_*    linear change per minute (battery sag), relative to the base value.
    patches    number of floor patches on the outer road; inside one, speed and turn
               efficiency are multiplied by values drawn from patch_speed / patch_turn.
    breakaway  extra PWM needed to start a side from standstill (static friction);
               disappears once that side is rolling.
    """
    wander_tau_s: float = 8.0
    wander_turn_eff: float = 0.0
    wander_speed_rel: float = 0.0
    wander_start_duty: float = 0.0
    wander_max_duty: float = 0.0
    wander_delay_s: float = 0.0
    trend_speed_rel_per_min: float = 0.0
    trend_max_duty_rel_per_min: float = 0.0
    patches: int = 0
    patch_speed: tuple = (1.0, 1.0)
    patch_turn: tuple = (1.0, 1.0)
    patch_size_cm: tuple = (20.0, 40.0)
    breakaway_duty: float = 0.0
    seed: int = 0

    def to_dict(self):
        return asdict(self)


VARIATIONS = {
    "fixed": Variation(),
    "mild": Variation(wander_turn_eff=0.05, wander_speed_rel=0.04, wander_start_duty=0.01, wander_max_duty=0.03,
                      wander_delay_s=0.01, trend_speed_rel_per_min=-0.02, trend_max_duty_rel_per_min=-0.02,
                      patches=3, patch_speed=(0.9, 1.0), patch_turn=(0.85, 1.05), breakaway_duty=0.02),
    "harsh": Variation(wander_turn_eff=0.12, wander_speed_rel=0.10, wander_start_duty=0.025, wander_max_duty=0.08,
                       wander_delay_s=0.03, trend_speed_rel_per_min=-0.06, trend_max_duty_rel_per_min=-0.06,
                       patches=6, patch_speed=(0.75, 1.0), patch_turn=(0.65, 1.1), breakaway_duty=0.06),
}

WANDER_KEYS = ("turn_left", "turn_right", "speed_left", "speed_right", "start_left", "start_right", "max_duty", "delay")


def _patches(var, rng):
    """Floor patches placed on the outer ring road (fixed per variation seed)."""
    out = []
    for _ in range(var.patches):
        side = rng.integers(4)
        along = rng.uniform(-80, 80)
        across = rng.uniform(78, 96)
        x, y = [(across, along), (-across, along), (along, across), (along, -across)][side]
        half = rng.uniform(*var.patch_size_cm) / 2
        out.append(dict(box=[x - half, x + half, y - half, y + half], speed=float(rng.uniform(*var.patch_speed)),
                        turn=float(rng.uniform(*var.patch_turn))))
    return out


class Vehicle:
    def __init__(self, params, pose, variation=None):
        self.p = params
        self.var = variation or Variation()
        self.rng = np.random.default_rng(self.var.seed)
        self.patches = _patches(self.var, self.rng)
        self.wander = dict.fromkeys(WANDER_KEYS, 0.0)
        self.pose = np.asarray(pose, float).copy()
        self.commands = [(-1e9, np.zeros(2))]   # (time applied by the driver, pwm)
        self.speed = np.zeros(2)                  # actual wheel surface speed, cm/s
        self.v = self.omega = 0.0
        self.now_params = {}

    def _update_wander(self, dt):
        v, a = self.var, np.exp(-dt / max(self.var.wander_tau_s, 1e-6))
        sd = dict(turn_left=v.wander_turn_eff, turn_right=v.wander_turn_eff, speed_left=v.wander_speed_rel,
                  speed_right=v.wander_speed_rel, start_left=v.wander_start_duty, start_right=v.wander_start_duty,
                  max_duty=v.wander_max_duty, delay=v.wander_delay_s)
        for k in WANDER_KEYS:   # stationary std = sd[k]
            self.wander[k] = a * self.wander[k] + sd[k] * np.sqrt(1 - a * a) * self.rng.standard_normal()

    def effective(self, t):
        """The car's personality at time t and the current position."""
        p, v, w = self.p, self.var, self.wander
        minutes = t / 60.0
        speed_trend = 1 + v.trend_speed_rel_per_min * minutes
        patch_speed = patch_turn = 1.0
        for q in self.patches:
            x0, x1, y0, y1 = q["box"]
            if x0 <= self.pose[0] <= x1 and y0 <= self.pose[1] <= y1:
                patch_speed *= q["speed"]
                patch_turn *= q["turn"]
        return dict(
            start=np.array([p.start_duty_left + w["start_left"], p.start_duty_right + w["start_right"]]),
            max_duty=max(0.2, p.max_duty * (1 + v.trend_max_duty_rel_per_min * minutes) + w["max_duty"]),
            gain=np.array([p.cm_s_per_duty_left * (1 + w["speed_left"]), p.cm_s_per_duty_right * (1 + w["speed_right"])])
            * speed_trend * patch_speed,
            turn_left=float(np.clip((p.turn_eff_left + w["turn_left"]) * patch_turn, 0.2, 1.2)),
            turn_right=float(np.clip((p.turn_eff_right + w["turn_right"]) * patch_turn, 0.2, 1.2)),
            delay=max(0.0, p.actuation_delay_s + w["delay"]),
        )

    def command(self, t, pwm):
        self.commands.append((float(t), np.clip(np.asarray(pwm, float), -1.0, 1.0)))   # negative = reverse
        if len(self.commands) > 400:
            self.commands = self.commands[-200:]

    def _active(self, t, delay):
        due = t - delay
        cmd = self.commands[0][1]
        for tc, pwm in self.commands:
            if tc <= due:
                cmd = pwm
            else:
                break
        return cmd

    def step(self, t, dt, speed_scale=1.0):
        p = self.p
        self._update_wander(dt)
        e = self.effective(t)
        self.now_params = e
        pwm = self._active(t, e["delay"])
        start = e["start"] + np.where(np.abs(self.speed) < 0.2, self.var.breakaway_duty, 0.0)   # static friction at rest
        mag = np.abs(pwm)
        duty = np.sign(pwm) * np.where(mag >= start - 1e-9, np.minimum(mag, e["max_duty"]), 0.0)
        target = duty * e["gain"] * speed_scale
        if p.time_constant_s > 0:
            self.speed += (target - self.speed) * (1 - np.exp(-dt / p.time_constant_s))
        else:
            self.speed = target
        v = float(self.speed.mean())
        raw = float((self.speed[0] - self.speed[1]) / p.track_cm)
        omega = raw * (e["turn_left"] if raw < 0 else e["turn_right"]) + p.drift_rad_s * (v / 5.0)
        self.v, self.omega = v, omega
        self.pose = advance(self.pose, v, omega, dt)
        if p.pivot_slide_cm_per_rad > 0 and self.speed[0] * self.speed[1] < 0:
            if not hasattr(self, "_slide_dir"):     # own generator: other random draws unchanged
                a = np.random.default_rng(self.var.seed + 7919).uniform(-np.pi, np.pi)
                self._slide_dir = np.array([np.sin(a), np.cos(a)])
            c, s_ = np.cos(self.pose[2]), np.sin(self.pose[2])
            world = self._slide_dir @ np.array([[c, -s_], [s_, c]])
            self.pose[:2] += p.pivot_slide_cm_per_rad * abs(omega * dt) * world
        return self.pose

    def footprint(self, step=1.0):
        p = self.p
        w, f, r = p.half_width_cm, p.front_cm, p.rear_cm
        corners = np.array([[-w, -r], [w, -r], [w, f], [-w, f]])
        rim = [np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b - a) / step))), endpoint=False)
               for a, b in zip(corners, np.roll(corners, -1, axis=0))]
        return np.vstack(rim)


# ---- presets and random environments --------------------------------------
PRESETS = {
    "ideal": VehicleParams(),
    "field_estimate": VehicleParams(cm_s_per_duty_left=16.8, cm_s_per_duty_right=16.8,
                                    turn_eff_left=0.6, turn_eff_right=0.6,
                                    actuation_delay_s=0.05, time_constant_s=0.10),
    # field_estimate + 2 cm sideways body slide per radian of pivot (assumed size;
    # field: pivoting works with some slip, not measured)
    "field_pivot_slip": VehicleParams(cm_s_per_duty_left=16.8, cm_s_per_duty_right=16.8,
                                      turn_eff_left=0.6, turn_eff_right=0.6,
                                      actuation_delay_s=0.05, time_constant_s=0.10, pivot_slide_cm_per_rad=2.0),
    "weak_left_turn": VehicleParams(turn_eff_left=0.5, turn_eff_right=0.7, actuation_delay_s=0.05, time_constant_s=0.10),
    "sluggish": VehicleParams(turn_eff_left=0.6, turn_eff_right=0.6, actuation_delay_s=0.15, time_constant_s=0.25),
    "high_start_duty": VehicleParams(start_duty_left=0.34, start_duty_right=0.34, turn_eff_left=0.6, turn_eff_right=0.6,
                                     actuation_delay_s=0.05, time_constant_s=0.10),
    "low_saturation": VehicleParams(max_duty=0.5, turn_eff_left=0.6, turn_eff_right=0.6,
                                    actuation_delay_s=0.05, time_constant_s=0.10),
    "unequal_sides": VehicleParams(cm_s_per_duty_left=15.2, cm_s_per_duty_right=16.8, turn_eff_left=0.6,
                                   turn_eff_right=0.6, actuation_delay_s=0.05, time_constant_s=0.10),
}

RANDOM_RANGES = dict(
    start_duty=(0.26, 0.34),          # per side, independently
    max_duty=(0.50, 1.00),
    cm_s_per_duty=(14.0, 18.0),       # common value, then a per-side asymmetry
    side_asymmetry=(-0.06, 0.06),
    turn_eff=(0.45, 1.00),            # per direction, independently
    drift_rad_s=(-0.03, 0.03),
    actuation_delay_s=(0.0, 0.15),
    time_constant_s=(0.0, 0.25),
)


def random_params(seed, ranges=RANDOM_RANGES):
    r = np.random.default_rng(seed)
    u = lambda k: float(r.uniform(*ranges[k]))
    gain, asym = u("cm_s_per_duty"), u("side_asymmetry")
    return VehicleParams(start_duty_left=u("start_duty"), start_duty_right=u("start_duty"), max_duty=u("max_duty"),
                         cm_s_per_duty_left=gain * (1 - asym), cm_s_per_duty_right=gain * (1 + asym),
                         turn_eff_left=u("turn_eff"), turn_eff_right=u("turn_eff"), drift_rad_s=u("drift_rad_s"),
                         actuation_delay_s=u("actuation_delay_s"), time_constant_s=u("time_constant_s"))
