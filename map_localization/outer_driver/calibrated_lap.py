"""Compare the same map driver on a plant calibrated independently using first-turn real footage."""
import argparse
import csv
import json
import math
from pathlib import Path
import cv2
import numpy as np
from validate_sim import Plant, DT
from controller import Driver, CENTER_AHEAD, axes
from yaw_gain import GainTracker

HERE=Path(__file__).resolve().parent
OUT=HERE/"calibrated_runs"

from feedback_driver import FixedGain, CompensatedDriver

def calibrate():
    with (HERE/"paint_yaw.csv").open() as stream:
        first={int(row["frame"]):row for row in csv.DictReader(stream)}
    def change(key):
        return float(first[115][key])-float(first[175][key])
    raw=change("pwm_from_start_deg")
    visual=change("yaw_change_from_start_deg")
    fitted=visual/raw
    second=json.loads((HERE/"second_corner_yaw.json").read_text())[-1]
    return dict(left_gain=fitted,train_pwm=raw,train_visual=visual,train_frames=[115,175],
                test_frames=[second["start"],second["end"]],test_observed=second["measured_left_deg"],
                test_uncorrected=second["original_left_deg"],test_predicted=fitted*second["original_left_deg"])



def run(mode,scenario="baseline",direction="ccw",cw_policy=None,output_policy=None):
    cv2.setNumThreads(1)
    calibration=calibrate()
    right_response=1.
    right_prior=1.
    if cw_policy:
        # Train on corner 1; the simulated vehicle uses corner 2 from the separate CW recording.
        first,second=json.loads((OUT/"cw_calibration.json").read_text())
        right_response=second["gain"]
        right_prior={"current":1.,"mirror":13./18.,"measured":first["gain"]}[cw_policy]
    plant=Plant(direction,pivot_behind_center_cm=CENTER_AHEAD,left_yaw_response=calibration["left_gain"],
                right_yaw_response=right_response)
    if scenario=="second_response":
        plant.left_yaw_response=calibration["test_observed"]/calibration["test_uncorrected"]
    elif scenario=="ideal_response":
        plant.left_yaw_response=1.
    elif scenario=="offset_right":
        plant.pos+=axes(plant.yaw)[0]  # unannounced 1 cm physical start offset
    elif scenario in ("symmetric","symmetric_drift"):
        plant.right_yaw_response=plant.left_yaw_response
        if scenario=="symmetric_drift":
            plant.drift_deg_s_at_30=-1.  # sensitivity assumption, not measured from the rail footage
    stem=mode if scenario=="baseline" else mode+"_"+scenario
    if cw_policy or direction!="ccw":
        stem+=f"_{direction}_{cw_policy or 'ideal_right'}"
    cls=CompensatedDriver if mode.endswith("_comp") else Driver
    driver=cls(direction,adaptive_yaw=(mode!="original"))
    if scenario=="glare":
        # Same camera-fixed reflection profile as the archived visibility replay.
        w,h=driver.size
        yy,xx=np.mgrid[:h,:w]
        alpha=(.99*np.exp(-.5*(((xx-.68*w)/(.09*w))**2+((yy-.7*h)/(.23*h))**2))).astype(np.float32)
        alpha=cv2.merge((alpha,alpha,alpha))
    if mode.startswith("fixed"):
        driver.yaw_tracker=FixedGain()
    if cw_policy:
        if isinstance(driver,CompensatedDriver):
            driver.right_mix=right_prior
        if driver.yaw_tracker is not None:
            driver.yaw_tracker.right=right_prior
    if output_policy:
        if mode!="fixed_comp" or cw_policy!="measured":
            raise ValueError("Output-only comparison needs the same fixed estimator and measured right prior")
        if output_policy=="both_left":
            driver.right_mix=driver.left_mix
        elif output_policy=="both_right":
            driver.left_mix=driver.right_mix
        stem+="_"+output_policy
    minimum=math.inf
    positions=[]; angles=[]; crossings=0; updates=[]
    OUT.mkdir(exist_ok=True)
    with (OUT/(stem+".csv")).open("w",newline="") as stream:
        log=csv.writer(stream)
        log.writerow(["t","x_body","y_body","yaw_deg","estimated_x_pivot","estimated_y_pivot","estimated_yaw_deg","actual_margin_cm","forecast_margin_cm","L_next","R_next"])
        for tick in range(3500):
            frame=plant.capture_array()
            if scenario=="glare":
                frame=np.rint(frame+(255.-frame.astype(np.float32))*alpha).astype(np.uint8)
            cmd=driver.step(frame,DT,plant.out,rectified=True)
            plant.set(*cmd)
            margin=plant.rows[-1][4]
            minimum=min(minimum,margin)
            crossings+=int(margin<0)
            pivot=plant.pos-CENTER_AHEAD*axes(plant.yaw)[1]
            positions.append(np.linalg.norm(pivot-driver.pose[:2]))
            angles.append(abs(math.degrees(math.remainder(plant.yaw-driver.pose[2],2*math.pi))))
            log.writerow([round(plant.tick*DT,4),*plant.pos,math.degrees(plant.yaw),*driver.pose[:2],math.degrees(driver.pose[2]),margin,driver.margin,*cmd])
            if driver.yaw_tracker and driver.yaw_tracker.event:
                updates.append(dict(t=plant.tick*DT,**driver.yaw_tracker.event))
            if driver.reason:
                assert not np.any(cmd)
                break
    result=dict(mode=mode,scenario=scenario,direction=direction,cw_policy=cw_policy,right_prior=right_prior,
                calibration=calibration,left_plant_gain=plant.left_yaw_response,
                right_plant_gain=plant.right_yaw_response,drift_deg_s_at_30=plant.drift_deg_s_at_30,reason=driver.reason or "timeout",
                elapsed_s=plant.tick*DT,actual_laps=plant.travel/(2*math.pi),minimum_cm=minimum,
                crossing_frames=crossings,max_error_cm=max(positions),max_error_deg=max(angles),
                last_forecast_margin_cm=driver.margin,updates=updates)
    if output_policy:
        result.update(output_policy=output_policy,left_output_multiplier=1/driver.left_mix,
                      right_output_multiplier=1/driver.right_mix)
    (OUT/(stem+".json")).write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["original","fixed","adaptive","fixed_comp","adaptive_comp"])
    parser.add_argument("--scenario",choices=["baseline","second_response","ideal_response","offset_right","glare","symmetric","symmetric_drift"],default="baseline")
    parser.add_argument("--direction",choices=["cw","ccw"],default="ccw")
    parser.add_argument("--cw-policy",choices=["current","mirror","measured"])
    parser.add_argument("--output-policy",choices=["both_left","both_right","directional"])
    args=parser.parse_args()
    run(args.mode,args.scenario,args.direction,args.cw_policy,args.output_policy)
