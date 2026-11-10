"""drive_v4 unit tests. No GPIO, no camera.  python -m drive_v4.tests.test_core"""
import math

import numpy as np

from drive_v4.core import MAP_PATH, Driver, FloorLearner, MapAdvisor, MotorModel, TurnGainLearner, load_config, pursuit_on_path
from drive_v3.map_model import TrackMap
from drive_v3.settings import load_settings

CFG = load_config()


def motor():
    return MotorModel(dict(CFG["motor"]))


def test_deadband_means_no_motion():
    m = motor()
    assert m.velocity([0.2, 0.2]) == (0.0, 0.0)          # v3 believed this moved the car
    v, w = m.velocity([0.3, 0.3])
    assert v > 0 and w == 0


def test_straight_and_gentle_curve_keep_both_wheels_above_floor():
    m = motor()
    floor = CFG["motor"]["pwm_floor"]
    for k in (0.0, 0.01, -0.02):
        pwm, note = m.mix(CFG["motor"]["cruise_duty"], k)
        assert note == "ratio" and pwm.min() >= floor - 1e-9
        _, omega = m.velocity(pwm)
        v, _ = m.velocity(pwm)
        assert math.isclose(omega / v, k, rel_tol=1e-6, abs_tol=1e-9)  # curvature preserved


def test_tight_corner_dithers_to_the_requested_average_curvature():
    m = motor()
    k = -0.083  # tightest outer-route corner, R ~12cm
    ws, vs = [], []
    for _ in range(200):
        pwm, note = m.mix(CFG["motor"]["cruise_duty"], k)
        assert note.startswith("dither")
        assert set(np.round(pwm, 6)) <= {0.0, CFG["motor"]["pwm_floor"], CFG["motor"]["pwm_max"]}
        v, w = m.velocity(pwm)
        ws.append(w); vs.append(v)
    # The PATH curvature is total turn / total distance (not the mean of per-frame curvatures).
    assert math.isclose(np.sum(ws) / np.sum(vs), k, rel_tol=0.05), np.sum(ws) / np.sum(vs)


def advisor():
    track = TrackMap(MAP_PATH)
    route = track.outer_route("ccw")
    v3, _ = load_settings()
    pose = np.r_[route.points[100], route.heading[100]]
    return MapAdvisor(track, route, v3["estimator"], pose, [1, 1, .03], CFG["map"]), route


def test_waiting_does_not_inflate_uncertainty():
    a, _ = advisor()
    before = np.sqrt(np.diag(a.est.primary.P))
    for _ in range(100):
        a.est.predict(0.0, 0.0, 0.1)  # 10 s parked
    after = np.sqrt(np.diag(a.est.primary.P))
    assert after[0] - before[0] < 0.1 and after[1] - before[1] < 0.1


def test_map_advice_ignores_lateral_offset_on_a_straight():
    a, route = advisor()
    s = route.s
    straight = int(np.argmin(np.abs(route.curvature) + (np.abs(route.curvature) > 1e-3) * 10))
    p = route.points[straight]
    yaw = route.heading[straight]
    right = np.array([math.cos(yaw), -math.sin(yaw)])
    shifted = np.r_[p + 4.0 * right, yaw]  # the 4-5cm bias seen in live_20261110_182520
    (k_ff, k_head, corner), info = a.advice(shifted, CFG["steer"])
    assert abs(k_ff + k_head) < 0.005
    # v2/v3-style pursuit onto the route would have asked for a hard turn here:
    target = route.sample(s[straight] + 10)[0]
    from drive_v3.geometry import to_body
    q = to_body([target], shifted)[0]
    assert abs(2 * q[0] / (q @ q)) > 0.05


def test_pursuit_on_path_signs():
    assert abs(pursuit_on_path([[0, 0], [0, 10], [0, 20]], 14)) < 1e-12
    assert pursuit_on_path([[0, 0], [2, 10], [5, 20]], 14) > 0     # path bends right -> k > 0
    assert pursuit_on_path([[0, 0]], 14) is None


def test_black_frames_hold_zero_and_never_raise():
    d = Driver(CFG)
    black = np.zeros((480, 640, 3), np.uint8)
    t = 0.0
    reasons = []
    for _ in range(15):
        t += 0.1
        pwm, rec = d.step(black, t - 0.02, t)
        d.applied(t, pwm)
        reasons.append(rec["reason"])
    assert reasons[-1] == "road_lost_hold"
    assert np.all(pwm == 0)


def _feed(learner, ratio, raw_step_deg, n, t0=0.0, usable=True, yaw0=0.0):
    t, yaw = t0, yaw0
    learner.observe(t, yaw, 0.0, usable)
    for _ in range(n):
        t += 0.067
        yaw += math.radians(raw_step_deg * ratio)
        learner.observe(t, yaw, raw_step_deg, usable)
    return t, yaw


def test_turn_gain_learning_moves_toward_measured_ratio_with_bounded_steps():
    m = motor()
    learner = TurnGainLearner(m, dict(CFG["yaw_learn"]))
    start = m.gain_left
    _feed(learner, 0.50, -2.5, 60)               # left turns, true efficiency 0.50
    assert learner.events, "expected at least one update"
    for e in learner.events:
        assert abs(e["after"] - e["before"]) <= CFG["yaw_learn"]["max_step"] + 1e-9
    assert m.gain_left < start and m.gain_left >= 0.5 - 1e-9
    assert m.gain_right == CFG["motor"]["yaw_gain_right"]   # right untouched by left evidence


def test_turn_gain_learning_ignores_unusable_and_mixed_evidence():
    m = motor()
    learner = TurnGainLearner(m, dict(CFG["yaw_learn"]))
    _feed(learner, 0.5, -2.5, 60, usable=False)  # relocalisation / low trust
    assert not learner.events
    t, yaw = 0.0, 0.0
    learner.observe(t, yaw, 0.0, True)
    for i in range(60):                          # alternating left/right never fills a window
        t += 0.067
        step = -2.5 if i % 2 else 2.5
        yaw += math.radians(step * 0.5)
        learner.observe(t, yaw, step, True)
    assert not learner.events


def _floor_feed(learner, steps, pred_step_cm, meas_step_cm, pred_yaw_deg, meas_yaw_deg, low_pwm, straight):
    pose = np.zeros(3)
    learner.observe(0.0, pose, np.zeros(3), low_pwm, straight, True)
    events = []
    for i in range(1, steps + 1):
        pose = pose + np.array([0.0, meas_step_cm, math.radians(meas_yaw_deg)])
        e = learner.observe(i * 0.067, pose, np.array([0.0, pred_step_cm, math.radians(pred_yaw_deg)]),
                            low_pwm, straight, True)
        if e:
            events.append(e)
    return events


def test_floor_rises_when_commanded_motion_does_not_happen():
    m = motor()
    f = FloorLearner(m, dict(CFG["floor_learn"]))
    events = _floor_feed(f, 30, pred_step_cm=0.4, meas_step_cm=0.0, pred_yaw_deg=0, meas_yaw_deg=0, low_pwm=0.32, straight=True)
    assert events and events[0]["reason"] == "no_motion"
    assert math.isclose(m.floor, CFG["motor"]["pwm_floor"] + CFG["floor_learn"]["step"])


def test_floor_rises_when_one_side_stalls_on_a_straight_command():
    m = motor()
    f = FloorLearner(m, dict(CFG["floor_learn"]))
    events = _floor_feed(f, 30, pred_step_cm=0.4, meas_step_cm=0.35, pred_yaw_deg=0, meas_yaw_deg=1.0, low_pwm=0.32, straight=True)
    assert events and events[0]["reason"] == "one_side_stalls"


def test_floor_stays_for_a_normal_car_and_is_capped():
    m = motor()
    f = FloorLearner(m, dict(CFG["floor_learn"]))
    assert not _floor_feed(f, 60, pred_step_cm=0.4, meas_step_cm=0.41, pred_yaw_deg=0, meas_yaw_deg=0.02, low_pwm=0.36, straight=True)
    assert m.floor == CFG["motor"]["pwm_floor"]
    g = FloorLearner(m, dict(CFG["floor_learn"], cooldown_s=0.0))
    _floor_feed(g, 2000, pred_step_cm=0.4, meas_step_cm=0.0, pred_yaw_deg=0, meas_yaw_deg=0, low_pwm=0.32, straight=True)
    assert m.floor <= CFG["motor"]["pwm_floor"] + CFG["floor_learn"]["max_raise"] + 1e-9


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, f in tests:
        try:
            f()
            print("PASS", name)
        except Exception as exc:
            failed += 1
            print("FAIL", name, type(exc).__name__, exc)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
