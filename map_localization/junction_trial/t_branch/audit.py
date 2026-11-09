"""Independent output-lane check using actual rendered pixels and the saved controller estimates."""
import json
import numpy as np
import cv2
from pivot import exit_edges
from trials import HERE, ShortGapDriver,SITES,FRAME_ON,FRAME_OFF,axes,CENTER_AHEAD,env,surface
from controller import LENS_AHEAD
def main():
    cv2.setNumThreads(1)
    d=ShortGapDriver();world=env['World']();clean=world.img;summary=[]
    for site,(center,yaw) in SITES.items():
        for move in ("straight","right","pivot"):
            for cond in ("normal","hidden","fake"):
                stem=f"{site}_{move}_{cond}"
                rows=np.genfromtxt(HERE/(stem+".csv"),names=True,delimiter=",")
                data=json.loads((HERE/(stem+".json")).read_text())
                world.img=clean
                fault=surface(world,site,cond)
                exit_confirm=[]; pos_errors=[]
                for i,r in enumerate(rows):
                    # Look for 3 consecutive valid fixes on the selected outbound lane after recovery.
                    local=(np.array([r['x'],r['y']])-center)@axes(yaw).T
                    along=local[1 if move=="straight" else 0]
                    eligible=(r['frame']>=FRAME_OFF and along>=12 and bool(r['matched']))
                    if not eligible or len(exit_confirm)>=3:continue
                    d.pose=np.array([r['est_x'],r['est_y'],r['est_yaw']])
                    cam=np.array([r['x'],r['y']])+(LENS_AHEAD-CENTER_AHEAD)*axes(r['yaw'])[1]
                    world.img=clean
                    matched=exit_edges(world.render((*cam,r['yaw'])),d,site,'straight' if move=='straight' else 'right')
                    if matched:
                        if exit_confirm and i!=exit_confirm[-1]+1:exit_confirm=[]
                        exit_confirm.append(i)
                    else:exit_confirm=[]
                # Confirm that the perturbation changed actual camera pixels visible on approach.
                idx=int(FRAME_ON)-1;r=rows[idx];cam=np.array([r['x'],r['y']])+(LENS_AHEAD-CENTER_AHEAD)*axes(r['yaw'])[1]
                world.img=clean;orig=world.render((*cam,r['yaw']))
                world.img=fault;altered=world.render((*cam,r['yaw']))
                changed=int(np.sum(np.max(np.abs(orig.astype(int)-altered),axis=2)>20))
                if cond!="normal":assert changed>0
                ok=rows[(rows['frame']>=FRAME_OFF)]
                settled=None
                for j in range(len(ok)-4):
                    group=ok[j:j+5]
                    if (np.all(group['matched']==1) and np.max(group['pos_error'])<=.5 and np.max(group['angle_error'])<=1):
                        settled=int(group['frame'][0]-FRAME_OFF);break
                data.update(maneuver=move,first_fault_image_changed_pixels=changed,exit_edges_confirmed=len(exit_confirm)==3,
                            confirmed_exit_time=(float(rows[exit_confirm[-1]]['t']) if len(exit_confirm)==3 else None),
                            settled_frames_after_restore=settled)
                # A rollout in which the estimator picked the wrong branch must not count as a pass.
                if data['outcome']=='pass' and len(exit_confirm)!=3:data['outcome']='physical exit only'
                summary.append(data)
    (HERE/'comparison.json').write_text(json.dumps(summary,indent=2))
    print([(r['site'],r['maneuver'],r['condition'],r['outcome'],round(r['min_cm'],2),r['settled_frames_after_restore']) for r in summary])
if __name__=='__main__':main()
