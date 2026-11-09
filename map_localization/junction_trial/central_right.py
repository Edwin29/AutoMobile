"""Standalone central-junction feasibility test; outer route and Pi files remain untouched."""
import json
import math
import sys
from pathlib import Path
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"outer_driver/output_feedback"))
from experiment import Feedback,FixedGain,calibrate,OUT,Plant,DT,CENTER_AHEAD,axes
from simulate_c import env,rim,corner

def route_for(radius):
    straight=np.arange(-55.,-radius,.25)
    a=np.linspace(0,np.pi/2,201)
    exit_x=np.arange(radius+.25,66.,.25)
    return np.vstack([np.c_[np.zeros(len(straight)),straight],
                      np.c_[radius*(1-np.cos(a)),radius*(np.sin(a)-1)],
                      np.c_[exit_x,np.zeros(len(exit_x))]])

def run(yaw,radius,world,second_gain,first_gain,left_gain):
    start_axes=axes(yaw)
    path=route_for(radius)@start_axes
    plant=Plant("ccw",pivot_behind_center_cm=CENTER_AHEAD,left_yaw_response=left_gain,right_yaw_response=second_gain)
    plant.world=world
    plant.yaw=yaw
    plant.pos=path[0]+CENTER_AHEAD*start_axes[1]
    driver=Feedback("ccw")
    driver.right_mix=first_gain
    driver.yaw_tracker=FixedGain()
    driver.yaw_tracker.right=first_gain
    driver.pose=np.r_[path[0],yaw]
    driver.route=path
    driver.arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    driver.last_index=0
    rows=[]
    reason="timeout"
    for tick in range(900):
        image=plant.capture_array()
        cmd=driver.step(image,DT,plant.out,rectified=True)
        plant.set(*cmd)
        local=plant.pos@start_axes.T
        err=abs(math.degrees(math.remainder(plant.yaw-yaw-math.pi/2,2*math.pi)))
        rows.append([plant.tick*DT,*plant.pos,plant.yaw,plant.rows[-1][4],*cmd])
        if driver.reason:
            reason=driver.reason;break
        if local[0]>=40:
            reason="right exit" if abs(local[1])<2 and err<5 else "off target";break
    rows=np.array(rows)
    return dict(result=reason,seconds=rows[-1,0],min_cm=rows[:,4].min(),crossed=int((rows[:,4]<0).sum()),
                final_outbound_cm=local[0],final_lateral_cm=local[1],heading_error_deg=err),rows,path

if __name__=="__main__":
    cv2.setNumThreads(1)
    # Choose one shared arc using only known map and footprint. Account for both axes by rotation.
    raw=env["BLOCKS"]
    # Maximize the worst clearance over all four entries using one shared radius.
    scores={}
    for r in range(12,29):
        path=route_for(r)
        fw=np.gradient(path,axis=0);fw/=np.linalg.norm(fw,axis=1)[:,None]
        rt=np.c_[fw[:,1],-fw[:,0]]
        pts=path[:,None,:]+rt[:,None,:]*rim[:,0,None]+fw[:,None,:]*(rim[:,1,None]+CENTER_AHEAD)
        scores[r]=min(float(env["rrect_sdf"](*(pts@axes(y)).transpose(2,0,1),*b,env["BLOCK_R"]).min())
                      for y in (0,np.pi/2,np.pi,3*np.pi/2) for b in raw)
    radius=max(scores,key=scores.get)
    world=env["World"]()
    c=calibrate();first,second=json.loads((OUT/"cw_calibration.json").read_text())
    results={}
    for name,yaw in [('left',0),('top',np.pi/2),('right',np.pi),('bottom',3*np.pi/2)]:
        stats,rows,path=run(yaw,radius,world,second["gain"],first["gain"],c["left_gain"])
        results[name]=stats
        np.savetxt(HERE/f"from_{name}.csv",rows,delimiter=",",header="t,x_body,y_body,yaw_rad,actual_margin_cm,left,right",comments="")
        print(name,radius,stats,flush=True)
    (HERE/"result.json").write_text(json.dumps(dict(radius_cm=radius,geometry_min_cm=scores[radius],radii=scores,approaches=results),indent=2))
