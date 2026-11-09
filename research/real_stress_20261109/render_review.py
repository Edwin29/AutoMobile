"""Re-render logged simulator poses near each failure; not a second controller run."""
import csv
import json
import math
import numpy as np
from experiment import OUT, Plant, DT, axes, CENTER_AHEAD, driver_for, add_world_seam
from experiment import render_at, exposure_frame, corrupt_image, render_video

names=['ccw_clean_20','ccw_forward_persistent_2','cw_glare_20',
       'cw_seam_20','ccw_prior_lateral_20','ccw_combined_20']
for name in names:
    path=OUT/(name+'.json')
    if not path.exists():continue
    result=json.loads(path.read_text())
    with (OUT/(name+'.csv')).open() as f:
        rows=[{k:float(v) if v else None for k,v in r.items()} for r in csv.DictReader(f)]
    p=Plant(result['direction'],pivot_behind_center_cm=CENTER_AHEAD)
    d=driver_for(result['direction']); fault=result['config']; onset=result['onset']
    end=result['end_s'] if result['reason']!='lap complete' else onset+6
    begin=max(0.,end-6)
    trace=[]; seam_added=False
    for i,row in enumerate(rows):
        t=row['t']
        if not begin<=t<=end:continue
        if i%3 and i!=len(rows)-1:continue
        if fault.get('seam') and t>=onset and not seam_added:
            prev=rows[max(0,math.ceil(onset/DT)-2)]
            p.yaw=prev['true_yaw']
            p.pos=np.array([prev['true_x'],prev['true_y']])+CENTER_AHEAD*axes(p.yaw)[1]
            add_world_seam(p);seam_added=True
        p.yaw=row['true_yaw']
        true=np.array([row['true_x'],row['true_y'],p.yaw])
        p.pos=true[:2]+CENTER_AHEAD*axes(p.yaw)[1]
        frame=render_at(p,p.pos,p.yaw)
        if onset<=t<onset+4:
            if fault.get('exposure'):
                prev=rows[max(0,i-1)]
                prevyaw=prev['true_yaw']
                prevpos=np.array([prev['true_x'],prev['true_y']])+CENTER_AHEAD*axes(prevyaw)[1]
                frame=exposure_frame(p,prevpos,prevyaw,frame,fault['exposure'])
            frame=corrupt_image(frame,fault,min(1.,(t-onset+DT)/.5))
        estimated=np.array([row['est_x'],row['est_y'],row['est_yaw']])
        trace.append((frame,row,estimated,true,result['reason'] if i==len(rows)-1 else ''))
    render_video(name+'_review',trace,p.world.img,d)
    print(name, len(trace), 'frames',flush=True)
