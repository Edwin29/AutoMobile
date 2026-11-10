"""Thin adapters. The bench calls only:  step(image, capture_s, done_s) -> pwm.

A driver receives a rectified camera image, the capture time, the time its command
will be applied, and a start hint (the start-zone centre, same for every driver).
It never receives truth or vehicle parameters.

Each driver runs in its own process because both trees ship a package named
drive_v3 (original vs. v3-recovery).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AUTOMOBILE = HERE.parent                      # drive_v3 (original) + drive_v4
VENDOR_V3R = HERE / "_vendor" / "v3r_8b41dd9"  # v3-recovery, frozen at commit 8b41dd9
# drive_v4 frozen by `python -m bench.pin_v4` (code+config hash in the folder name)
VENDOR_V4 = HERE / "_vendor" / (HERE / "_vendor" / "V4_PIN.txt").read_text(encoding="utf-8").strip()
TRACK_JSON = HERE / "world" / "track.json"


class V4:
    name = "v4"
    source = f"drive_v4 pinned {VENDOR_V4.name}"

    def __init__(self, direction, start_hint, timing):
        sys.path.insert(0, str(VENDOR_V4))
        from drive_v4.core import Driver, load_config
        self.driver = Driver(load_config(), direction=direction, pose=np.asarray(start_hint, float), clock="bench")

    def step(self, image, capture_s, done_s):
        pwm, rec = self.driver.step(image, capture_s, done_s, rotate180=False, rectified=True)
        self.driver.applied(done_s, pwm)
        g = rec.get("turn_gain") or {}
        est = self.driver.map.pose_at(done_s)          # v4's belief (map pose): logged, never scored
        m = rec.get("map") or {}
        return np.asarray(pwm, float), dict(reason=rec["reason"], source=rec["source"], w_map=rec["w_map"],
                                            map=rec["map_status"], gain_l=g.get("left"), gain_r=g.get("right"),
                                            est=[float(v) for v in est], ms=rec["timings_ms"]["total"],
                                            k=rec.get("k"), k_map=rec.get("k_map"), k_local=rec.get("k_local"),
                                            w_corner=rec.get("w_corner"), k_ff=m.get("k_ff"), k_head=m.get("k_head"),
                                            k_lat=m.get("k_lat"), bias=m.get("bias_cm"), mix=rec.get("mix"),
                                            push=(rec.get("guard") or {}).get("push_k"))


class V3R:
    name = "v3r"
    source = "v3-recovery commit 8b41dd9 (recovery_rollout policy)"

    def __init__(self, direction, start_hint, timing):
        sys.path.insert(0, str(VENDOR_V3R))
        from drive_v3.compare_recovery import settings
        from drive_v3.map_model import TrackMap
        from drive_v3.runtime import Runtime
        cfg = settings("recovery")
        # Same timing contract as its own map_delay_experiment.experiment_settings().
        period, worst = timing["controller_period_s"], timing["max_latency_s"]
        cfg["controller"]["watchdog_s"] = period + worst + 0.1
        cfg["planner"].update(replan_period_s=period, rollout_dt_s=period, exact_rollout_horizon=True,
                              path_valid_s=period + worst + 0.1)
        track = TrackMap(TRACK_JSON)
        self.rt = Runtime(np.asarray(start_hint, float), track.outer_route(direction), cfg, clock_id="bench", track=track)

    def step(self, image, capture_s, done_s):
        rec, _, _, _ = self.rt.step(image, capture_s, done_s, rectified=True)
        est = self.rt.estimator.primary.mean            # v3r's belief: logged, never scored
        return np.asarray(rec["pwm"], float), dict(reason=rec["path_reason"], control=rec["control_reason"],
                                                   map=(rec.get("estimate") or {}).get("status"),
                                                   est=[float(v) for v in est], ms=rec.get("total_ms"))


VENDOR_CNN = HERE / "_vendor" / "tdt_models_2026-11-01"


class TeamCNN:
    """Team CNN lane policy (gawbi/track-digital-twin, Release models-2026-11-01).

    Same contract as its run_vehicle.py: the raw 640x480 camera frame (upright after
    --flip180, fisheye NOT undistorted) -> bottom half -> YUV -> 3x3 blur -> 200x66 ->
    ONNX -> [linear, angular] in [-1, 1] -> x speed_scale 0.5 -> DutyMixer from
    configs/real_car.json (min_duty 0.175, max_duty 0.66, turn_mix 0.72), signed duty.
    Its realtime loop stops on results older than 0.25 s; the bench latency is 36 ms.
    Inference uses OpenCV's ONNX reader (onnxruntime if installed gives the same graph).
    """
    raw_camera = True
    model = "lane-real-C5b"
    # The team runs --speed-scale 0.5; with its DutyMixer that puts straight-road duty
    # at ~0.27-0.31, at/below the start threshold (bench 0.30, team note: <0.33 stops).
    # BENCH_CNN_SPEED overrides it (1.0 gives ~0.36-0.40, close to v4's 0.36 cruise).
    speed_scale = float(__import__("os").environ.get("BENCH_CNN_SPEED", "0.5"))

    def __init__(self, direction, start_hint, timing):
        import json
        import cv2
        self.cv2 = cv2
        folder = VENDOR_CNN / "models" / self.model
        self.meta = json.loads((folder / "policy.json").read_text(encoding="utf-8"))
        if self.meta["preprocess"] != "rgb_bottom_half_yuv_blur3_200x66_chw_div255":
            raise ValueError("unsupported preprocessing")
        # read bytes first: OpenCV cannot open paths with non-ASCII characters (Korean user folder)
        self.net = cv2.dnn.readNetFromONNX(np.frombuffer((folder / "policy.onnx").read_bytes(), np.uint8))
        mix = json.loads((VENDOR_CNN / "configs" / "real_car.json").read_text(encoding="utf-8"))["duty_mixer"]
        self.mix = mix
        self.source = f"team CNN {self.model} (track-digital-twin models-2026-11-01), speed_scale {self.speed_scale}"

    def _preprocess(self, rgb):
        cv2 = self.cv2
        image = rgb[rgb.shape[0] // 2:]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        image = cv2.resize(image, (200, 66))
        return np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)[None] / 255.0

    def _side(self, value, trim):
        m = self.mix
        value = max(-1.0, min(1.0, value))
        if abs(value) <= m["deadband"]:
            return 0.0
        magnitude = m["min_duty"] + (m["max_duty"] - m["min_duty"]) * abs(value)
        return max(-1.0, min(1.0, float(np.copysign(magnitude, value)) * trim))

    def step(self, image, capture_s, done_s):
        rgb = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)
        self.net.setInput(self._preprocess(rgb))
        out = self.net.forward()[0]
        linear, angular = (float(np.clip(out[0], -1, 1)) * self.speed_scale,
                           float(np.clip(out[1], -1, 1)) * self.speed_scale)
        left, right = linear - self.mix["turn_mix"] * angular, linear + self.mix["turn_mix"] * angular
        scale = max(1.0, abs(left), abs(right))
        pwm = np.array([self._side(left / scale, self.mix["trim_left"]), self._side(right / scale, self.mix["trim_right"])])
        return pwm, dict(reason="drive", linear=linear, angular=angular)


def _cnn(name):
    return type(f"TeamCNN_{name}", (TeamCNN,), {"model": name})


CONTROLLERS = {"v4": V4, "v3r": V3R, "cnnC5b": _cnn("lane-real-C5b"), "cnnA5b": _cnn("lane-v5b"),
               "cnnB": _cnn("lane-real-B"), "cnnC": _cnn("lane-real-C"), "cnnA": _cnn("lane-v4b")}
