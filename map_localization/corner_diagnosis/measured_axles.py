"""Measured axle geometry, archived C lane detector and steering, first CCW corner."""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "c_drive_test"))
from simulate_c import Plant, DT, FWD, drive, lane, env, rim
from study import nearest, ahead, curvature

FRONT_AXLE, REAR_AXLE = 8.5, 20.5  # cm behind the front end
AXLE_MID = (FRONT_AXLE + REAR_AXLE) / 2
OFFSET = AXLE_MID - 24.5/2

def trial(pivot_back, map_track=None, trace=None):
    car = Plant("ccw", pivot_behind_center_cm=OFFSET)
    road = lane.DEFAULT_ROAD
    filtered = 0.
    minimum = float("inf")
    for _ in range(round(45/DT)):
        frame = car.capture_array()
        if map_track is None:
            result, _ = lane.analyze(frame, car.world.M, car.world.size, None, "straight",
                                     road, 18., pivot_back, 2.5, 0., 0., None, 0.)
            if result is None:
                raise RuntimeError("Lane lost in the comparison")
            road = result["road"]
            k, track = result["curv_pp"], 18.
        else:
            s,c = math.sin(car.yaw),math.cos(car.yaw)
            forward = np.array([s,c])
            right = np.array([c,-s])
            pivot = car.pos - OFFSET*forward
            delta = ahead(nearest(pivot)[0],10.) - pivot
            k, track = curvature(np.array([delta@right,delta@forward])), map_track
        filtered = .3*filtered + .7*k
        target = drive.wheel_speeds(filtered, .3, track, 10.)
        car.set(*drive.slew(car.out, target, drive.ACCEL_PER_S*DT))
        margin = car.rows[-1][4]
        if trace is not None:
            trace.append(car.rows[-1])
        if margin < minimum:
            minimum = margin
            time_s = car.tick*DT
            c,s = math.cos(car.yaw), math.sin(car.yaw)
            xy = rim@np.array([[c,-s],[s,c]]) + car.pos
            outside = -env["rrect_sdf"](*xy.T,*env["OUTER"],env["OUTER_R"])
            islands = [env["rrect_sdf"](*xy.T,*b,env["BLOCK_R"]) for b in env["BLOCKS"]]
            side = "island" if np.min(islands) < outside.min() else "outer"
        if car.pos[1] > -45 and car.pos[0] > 50:
            return dict(controller="C" if map_track is None else "map", steering_track_cm=track,
                        pivot_back_cm=pivot_back, minimum_clearance_cm=minimum,
                        nearest_boundary=side, time_of_min_s=time_s, exit_s=car.tick*DT)
    raise RuntimeError("Did not reach the exit")

if __name__ == "__main__":
    cv2.setNumThreads(1)
    results = [trial(value) for value in [0., FWD, FWD+OFFSET]]
    results += [trial(FWD+OFFSET, map_track=track) for track in [13., 18.]]
    output = json.dumps(dict(front_axle_cm=FRONT_AXLE, rear_axle_cm=REAR_AXLE,
                             pivot_behind_body_center_cm=OFFSET, trials=results), indent=2)
    (HERE/"measured_axles.json").write_text(output)
    print(output)
