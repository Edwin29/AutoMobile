"""End-to-end test with rendered camera frames. Plant pose is read only for grading."""
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from controller import Driver, CENTER_AHEAD, axes

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"c_drive_test"))
from simulate_c import Plant, env, DT
env["HERE"]=HERE.parents[1]/"junction_logic/visibility_review/source"

def run(direction,fault,offset,adaptive=False):
    cv2.setNumThreads(1)
    driver=Driver(direction,adaptive_yaw=adaptive)
    plant=Plant(direction,pivot_behind_center_cm=CENTER_AHEAD)
    plant.pos+=offset*axes(plant.yaw)[0]
    body_min=float("inf")
    error_max=0.
    for tick in range(3500):
        image=plant.capture_array()
        if fault and tick==60:
            image[:]=255
        command=driver.step(image,DT,plant.out,rectified=True)
        plant.set(*command)
        body_min=min(body_min,plant.rows[-1][4])
        actual=plant.pos-CENTER_AHEAD*axes(plant.yaw)[1]
        error_max=max(error_max,float(np.linalg.norm(driver.pose[:2]-actual)))
        if driver.reason:
            assert not np.any(command)
            break
    result=dict(direction=direction,fault=fault,start_lateral_offset_cm=offset,adaptive_yaw=adaptive,reason=driver.reason or "timeout",
                true_laps=plant.travel/(2*np.pi),elapsed_s=(tick+1)*DT,
                true_minimum_cm=body_min,max_position_error_cm=error_max,estimated_margin_cm=driver.margin)
    print(json.dumps(result))
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("direction",choices=["cw","ccw"])
    parser.add_argument("--fault",action="store_true")
    parser.add_argument("--offset",type=float,default=0.)
    parser.add_argument("--adaptive-yaw",action="store_true")
    args=parser.parse_args()
    run(args.direction,args.fault,args.offset,args.adaptive_yaw)
