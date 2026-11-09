"""Compare footprints on the two frozen CCW field logs. Run locally, no GPIO."""
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from controller import advance, axes
from feedback_driver import Feedback

# Wheel centre estimates: rear right, rear left, front right, front left.
WHEELS = np.array([[225,307],[302,194],[328,375],[400,258]],float)
# Outside tire shoulders, rear wire loop and board, front nose and ribbon.
TRACE = np.array([
    [182,283],[185,269],[200,241],[238,199],[252,180],[262,182],
    [270,177],[278,164],[284,162],[294,164],[320,179],[334,193],
    [338,204],[336,217],[374,230],[385,228],[395,231],[422,245],
    [434,254],[437,267],[433,278],[444,288],[453,304],[456,316],
    [453,341],[447,366],[428,379],[402,396],[383,402],[368,403],
    [363,396],[355,401],[346,403],[331,399],[315,391],[301,381],
    [295,373],[295,359],[251,326],[244,334],[233,334],[214,326],
    [199,316],[191,306],[190,297],[185,299],[182,291]
],float)
ORIGIN = WHEELS.mean(axis=0)
FORWARD = (WHEELS[2:].mean(axis=0)-WHEELS[:2].mean(axis=0))/12.
BASIS = np.array([[-FORWARD[1],FORWARD[0]],FORWARD])
RAW = (TRACE-ORIGIN)@np.linalg.inv(BASIS)
LAT_LIMIT = max(abs(RAW[:,0]))
LONG_KNOTS = [RAW[:,1].min(),-6,6,RAW[:,1].max()]
PHYSICAL = [-10,-6,6,14.5]

def hull(points):
    return cv2.convexHull(np.asarray(points,np.float32)).reshape(-1,2)

def outline():
    p=RAW.copy()
    p[:,0]*=8/LAT_LIMIT
    p[:,1]=np.interp(p[:,1],LONG_KNOTS,PHYSICAL)
    # Use the larger visible extent of the two sides. Preserve cable overhangs.
    return hull(np.vstack([p,p*[-1,1]]))

def rim(vertices):
    return np.concatenate([np.linspace(a,b,math.ceil(np.linalg.norm(b-a)/.25),endpoint=False)
                           for a,b in zip(vertices,np.roll(vertices,-1,axis=0))])

def padded(vertices):
    a=np.radians(np.arange(0,360,5))
    return hull((vertices[:,None,:]+.5*np.c_[np.cos(a),np.sin(a)]).reshape(-1,2))

def replay(path):
    driver=Feedback("ccw")
    polygon=outline()
    rims=[driver.rim,rim(polygon),rim(padded(polygon)),rim(hull(RAW))]
    rows=[]
    with path.open() as stream:
        for i,row in enumerate(csv.DictReader(stream)):
            pose=np.array([float(row["x"]),float(row["y"]),math.radians(float(row["yaw_deg"]))])
            driver.pose=pose
            k=driver.nearest()
            ahead=(driver.arc[k]+10)%driver.arc[-1]
            target=np.array([np.interp(ahead,driver.arc,driver.route[:,j]) for j in range(2)])
            q=(target-pose[:2])@axes(pose[2]).T
            driver.filtered=.3*driver.filtered+.7*(2*q[0]/(q@q))
            cmd=np.array([float(row["applied_L"]),float(row["applied_R"])])
            if i<2 or row["reason"]=="footprint limit":
                speed=.3/(1+10*abs(driver.filtered))
                request=driver.wheel_targets(speed,driver.filtered)
                cmd=driver.out+np.clip(request-driver.out,-.6/15,.6/15)
            driver.out=np.array([float(row["applied_L"]),float(row["applied_R"])])
            driver.left_mix=1/float(row["next_left_output_gain"])
            driver.right_mix=1/float(row["next_right_output_gain"])
            gain=float(row["next_left_yaw_gain"] if cmd[0]<cmd[1] else row["next_right_yaw_gain"])
            poses=[advance(pose,cmd,dt,g) for dt in (0.,.15,.30) for g in (1.,gain)]
            margins=[min(driver.sample(driver.sdf,points@axes(p[2])+p[:2]).min() for p in poses)*10 for points in rims]
            rows.append([float(row["t"]),float(row["margin_cm"])*10,*margins])
    a=np.asarray(rows)
    np.savetxt(HERE/(path.stem+"_outline.csv"),a,delimiter=",",fmt="%.6f",
               header="t,logged_mm,rectangle_mm,photo_mm,photo_plus5mm_mm,raw_photo_mm",comments="")
    result={"frames":len(a),"max_baseline_error_mm":float(np.max(abs(a[:,1]-a[:,2])))}
    for j,name in enumerate(("rectangle","photo","photo_plus5mm","raw_photo"),2):
        v=a[:,j];idx=v.argmin()
        result[name]=dict(min_mm=float(v[idx]),time=float(a[idx,0]),below=int(sum(v<2.5)),last_mm=float(v[-1]))
    return result

if __name__=="__main__":
    result={p:replay(HERE/p) for p in ["ccw_1lap_20261106_165722.csv","ccw_20261106_165141.csv"]}
    result["photo_vertices_right_forward_cm"]=outline().tolist()
    (HERE/"outline_replay.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
