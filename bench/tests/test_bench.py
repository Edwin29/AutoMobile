"""Bench self-checks (no driver code).  python -m bench.tests.test_bench"""
import numpy as np

from bench.score import Scorer, corner_box
from bench.vehicle import VARIATIONS, Variation, Vehicle, VehicleParams
from bench.world import Camera, Track


def _drive_reference(direction, smooth=0):
    t, c = Track(), Camera()
    rim = Vehicle(VehicleParams(), [0, 0, 0]).footprint()
    r = c.route if direction == "cw" else c.route[::-1]
    start = t.start_pose(direction)
    i0 = int(np.argmin(np.sum((r - start[:2]) ** 2, axis=1)))
    loop = np.vstack((r[i0:], r[:i0], r[i0:i0 + 600]))
    if smooth:                                   # a car that turns in early and cuts the corner
        from scipy.ndimage import gaussian_filter1d
        loop = np.c_[gaussian_filter1d(loop[:, 0], smooth), gaussian_filter1d(loop[:, 1], smooth)]
    h = np.arctan2(np.diff(loop[:, 0]), np.diff(loop[:, 1]))
    s = Scorer(t, rim, c.route, direction, np.r_[loop[0], h[0]])
    for k in range(1, len(loop) - 1):
        s.sample(0.02 * k, 0.02, np.r_[loop[k], h[k]], np.array([.4, .4]))
        if s.end:
            break
    return s, s.summary(s.end or "time_limit")


def test_reference_lap_scores_four_corners_both_directions():
    for d in ("ccw", "cw"):
        s, sm = _drive_reference(d)
        assert sm["outcome"] == "lap_complete", (d, sm["outcome"])
        assert sm["proper_corners"] == 4, (d, [(c["corner"], c["turn_deg"]) for c in sm["corners"]])


def test_start_zone_is_outside_every_corner_box_and_first_corner_is_ahead():
    t = Track()
    for d in ("ccw", "cw"):
        assert corner_box(*t.start_pose(d)[:2]) is None
    s_ccw, _ = _drive_reference("ccw")
    s_cw, _ = _drive_reference("cw")
    assert s_ccw.first_corner_progress > 80      # NE corner, far ahead
    assert -20 < s_cw.first_corner_progress < 40  # SE corner, right after the start zone


def test_early_turn_in_still_counts_every_corner():
    # A box-only count missed turns that began before the box (field: 'lap but 2/4 corners').
    for d in ("ccw", "cw"):
        _, sm = _drive_reference(d, smooth=120)
        assert sm["outcome"] == "lap_complete" and sm["proper_corners"] == 4, (d, [c["turn_deg"] for c in sm["corners"]])


def test_fixed_variation_is_a_fixed_car():
    a = Vehicle(VehicleParams(turn_eff_left=0.6), [0, 0, 0])
    b = Vehicle(VehicleParams(turn_eff_left=0.6), [0, 0, 0], VARIATIONS["fixed"])
    for car in (a, b):
        car.command(0.0, [0.3, 0.5])
    for k in range(500):
        a.step(k * 0.02, 0.02)
        b.step(k * 0.02, 0.02)
    assert np.allclose(a.pose, b.pose)


def test_harsh_variation_changes_the_car_during_a_run_but_stays_plausible():
    from dataclasses import replace
    car = Vehicle(VehicleParams(turn_eff_left=0.6, turn_eff_right=0.6), [87, -50, 0], replace(VARIATIONS["harsh"], seed=3))
    car.command(0.0, [0.4, 0.4])
    turns, maxd = [], []
    for k in range(6000):                       # 120 s
        car.step(k * 0.02, 0.02)
        turns.append(car.now_params["turn_left"]); maxd.append(car.now_params["max_duty"])
    assert np.std(turns) > 0.03                 # it really varies
    assert 0.2 <= min(turns) and max(turns) <= 1.2
    assert maxd[-1] < maxd[0] + 0.15            # battery trend pulls it down over two minutes


def test_breakaway_needs_extra_pwm_only_from_standstill():
    var = Variation(breakaway_duty=0.05)
    car = Vehicle(VehicleParams(), [0, 0, 0], var)
    car.command(0.0, [0.32, 0.32])              # above 0.30 but below 0.30 + 0.05
    for k in range(100):
        car.step(k * 0.02, 0.02)
    assert car.v == 0.0                          # stuck at rest
    car.command(2.0, [0.40, 0.40])
    for k in range(100, 150):
        car.step(k * 0.02, 0.02)
    assert car.v > 0
    car.command(3.0, [0.32, 0.32])              # once rolling, 0.32 keeps it moving
    for k in range(150, 200):
        car.step(k * 0.02, 0.02)
    assert car.v > 0


def test_floor_patches_lie_on_the_outer_road():
    t = Track()
    car = Vehicle(VehicleParams(), [0, 0, 0], Variation(patches=20, seed=1))
    for q in car.patches:
        x0, x1, y0, y1 = q["box"]
        assert t.clearance(np.array([[(x0 + x1) / 2, (y0 + y1) / 2]]))[0] > -2.0


def test_recurring_disturbances_happen_all_lap_and_are_reproducible():
    from bench.disturb import Disturbances, combine
    a = Disturbances(combine(["recurring_strong"]), "ccw", 31)
    b = Disturbances(combine(["recurring_strong"]), "ccw", 31)
    assert a.schedule == b.schedule
    starts = [e["t0"] for e in a.schedule if e["t0"] < 120]
    assert len(starts) >= 15                      # ~every 5 s through a whole lap
    assert max(starts) > 100 and min(starts) < 10
    assert any(a.speed_scale(t) < 0.9 for t in np.arange(0, 120, 0.5))


def test_old_matrix_specs_unchanged_by_new_groups():
    from bench.matrix import specs
    c = [x["slug"] for x in specs() if x["group"] == "C_random_vehicle_and_effects"]
    assert "C_random_vehicle_and_effects__v4_own__random200__glare__ccw" in c or len(c) == 72
    assert all("recurring" not in x for x in c)


if __name__ == "__main__":
    for name, f in sorted(globals().items()):
        if name.startswith("test_"):
            f()
            print("PASS", name)
