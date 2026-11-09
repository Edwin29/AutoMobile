"""Known-pose desktop localization checks. Truth is used only to render and score."""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"junction_trial/t_branch"))
from trials import env,axes,SITES,surface
from controller import Driver,CENTER_AHEAD,LENS_AHEAD
from matcher import Matcher


def setup():
    env["HERE"]=HERE.parents[1]/"junction_logic/visibility_review/source"
    world=env["World"]()
    assets=np.load(HERE.parent/"outer_driver/track.npz")
    ground=dict(np.load(env["HERE"]/"ground.npz"))
    H=env["L"].ground_to_image(ground,assets["rect_K"])
    matcher=Matcher(world.img,env["PPC"],env["EXT"],np.diag([.5,.5,1])@H)
    return world,matcher


def scenes():
    rows=np.genfromtxt(HERE.parent/"junction_trial/t_branch/narrow_pivot_normal.csv",names=True,delimiter=",")
    result={}
    for name,deg in (("entry",18.),("mid_turn",40.),("exit_turn",70.)):
        row=rows[np.argmin(np.abs(np.degrees(rows["yaw"])-deg))]
        pos=np.array([row["x"],row["y"]])-CENTER_AHEAD*axes(row["yaw"])[1]
        result[name]=np.r_[pos,row["yaw"]]
    result["outer_corner"]=np.array([-84.8,-69.2,math.radians(155.)])
    return result


def altered(frame,condition):
    image=frame.copy()
    h,w=image.shape[:2]
    if condition=="occluded":
        image[int(h*.48):,int(w*.55):]=(62,75,48)
    elif condition=="false_edge":
        polygon=np.array([[int(w*.49),int(h*.42)],[int(w*.60),int(h*.42)],
                          [int(w*.85),h-1],[int(w*.67),h-1]],np.int32)
        cv2.fillConvexPoly(image,polygon,env["COL"]["green"])
    elif condition=="blank":
        image[:]=255
    return image


def pose_error(estimate,truth):
    if estimate is None:return None,None
    p=np.array(estimate)
    return float(np.linalg.norm(p[:2]-truth[:2])),abs(math.degrees(math.remainder(p[2]-truth[2],2*math.pi)))


def run(scene,condition,world,matcher):
    truth=scenes()[scene]
    prior=truth+np.array([2.3,-3.7,math.radians(8.3)])
    if condition=="outside":prior=truth+np.array([9.3,-3.7,math.radians(8.3)])
    camera=truth[:2]+LENS_AHEAD*axes(truth[2])[1]
    frame=world.render((*camera,truth[2]))
    frame=altered(frame,condition)
    result,diag=matcher.search(frame,prior)
    result.update(scene=scene,condition=condition,truth=truth.tolist(),prior=prior.tolist())
    result["position_error_cm"],result["heading_error_deg"]=pose_error(result["pose"],truth)
    driver=Driver("cw");driver.pose=prior.copy()
    first=driver.observe(frame,True)
    result["local_first_match"]=bool(first)
    result["local_first_position_cm"],result["local_first_heading_deg"]=pose_error(driver.pose,truth)
    for _ in range(19):driver.observe(frame,True)
    result["local_repeated_position_cm"],result["local_repeated_heading_deg"]=pose_error(driver.pose,truth)
    if diag is not None:
        result["truth_forward_score"]=float(matcher.forward(truth[None])[0][0])
        np.savez_compressed(HERE/f"{scene}_{condition}_scores.npz",**diag)
    cv2.imwrite(str(HERE/f"{scene}_{condition}.png"),frame)
    (HERE/f"{scene}_{condition}.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(scene,condition,result["accepted"],result.get("score"),result.get("gap"),
          result["position_error_cm"],result["heading_error_deg"],round(result["seconds"],3),flush=True)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--scene",nargs="+",choices=list(scenes()),default=list(scenes()))
    parser.add_argument("--conditions",nargs="+",default=["normal","occluded","false_edge"])
    args=parser.parse_args()
    cv2.setNumThreads(1)
    world,matcher=setup()
    for scene in args.scene:
        for condition in args.conditions:run(scene,condition,world,matcher)
