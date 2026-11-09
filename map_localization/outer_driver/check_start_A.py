"""Held-out lower-road crosswalk after manual identification of start mark A."""
import csv,json,sys
from pathlib import Path
import cv2
import numpy as np
from controller import Driver,axes,advance,LENS_AHEAD
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from replay_pose import RUN,SRC,lane,geometry

def main():
    cv2.setNumThreads(1)
    driver=Driver("ccw")
    cfg,_,_=geometry()
    with np.load(HERE/"track.npz") as assets:
        K,D,P=assets["K"],assets["D"],assets["rect_K"]
    H=lane.ground_to_image(dict(np.load(SRC/"ground.npz")),P)
    def ground(uv):
        uv=cv2.fisheye.undistortPoints(np.array([[uv]],float),K,D,P=P).ravel()
        q=np.linalg.solve(H,[*uv,1])
        return q[:2]/q[2]
    mark=next(a[0] for k,a,b in cfg["ROAD_MARKS"] if k=="pink" and a[0]==b[0] and a[0]>0 and a[1]<0)
    h=axes(driver.pose[2])
    shift=float(mark-(driver.pose[:2]+(ground([329,188])+[0,LENS_AHEAD])@h)[0])
    driver.pose[0]+=shift
    driver.last_index=driver.nearest()
    diag=Driver("ccw")
    diag.pose=driver.pose.copy()
    center=next((x,y) for x,y,_ in cfg["CROSSWALKS"] if x>75)
    # Near edge, positive-y travel. Paint spans +/-6 cm along road, 17.75 cm across.
    edge=np.column_stack([np.linspace(center[0]-8.875,center[0]+8.875,51),np.full(51,center[1]-6)])
    first=None; last=None; output={}; checks=[]
    reference={280:(364,135),305:(378,150),320:(386,161),340:(435,184)}
    tiles=[]
    with (RUN/"log.csv").open(encoding="utf8") as f: rows=list(csv.DictReader(f))
    for i,r in enumerate(rows):
        f=int(r["frame"]); t=float(r["t"]);dt=t-(float(rows[i-1]["t"]) if i else 0)
        u=tuple(float(rows[max(0,i-1)][k]) for k in ("left","right")) if i else (0.,0.)
        path=RUN/f"frame_{f:05d}.jpg"
        if not path.exists():
            diag.pose=advance(diag.pose,u,dt)
            if not driver.reason: driver.pose=advance(driver.pose,u,dt)
            continue
        raw=cv2.imread(str(path)); diag.pose=advance(diag.pose,u,dt)
        accepted=diag.observe(raw,False)
        if not driver.reason:
            driver.step(raw,dt,u)
            if driver.reason:
                first=dict(frame=f,t=t,reason=driver.reason,residual=driver.residual,margin=driver.margin)
        h=axes(diag.pose[2]);cam=diag.pose[:2]+LENS_AHEAD*h[1]
        local=(edge-cam)@h.T
        proj=np.column_stack([local,np.ones(len(local))])@H.T
        rect=proj[:,:2]/proj[:,2,None]
        norm=np.column_stack([rect,np.ones(len(rect))])@np.linalg.inv(P).T
        uv=cv2.fisheye.distortPoints(np.ascontiguousarray(norm[:,:2]).reshape(-1,1,2),K,D).reshape(-1,2)
        visible=(local[:,1]>0)&(proj[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<640)&(uv[:,1]>=0)&(uv[:,1]<480)
        if last is None and visible.any():
            last=dict(frame=f,t=t)
        if f==155:
            snap=raw.copy()
            ids=np.flatnonzero(visible)
            if len(ids)>1:
                cv2.polylines(snap,[np.rint(uv[ids]).astype(np.int32)],False,(0,180,255),3)
            cv2.imwrite(str(HERE/"crosswalk_at_stop.jpg"),snap)
            first["predicted_center_uv"]=uv[25].round(1).tolist()
        if f in reference:
            xy=reference[f]
            observed=(ground(xy)+[0,LENS_AHEAD])@h+cam
            rec=dict(frame=f,t=t,matched_edges=bool(accepted),predicted_uv=uv[25].round(1).tolist(),observed_uv=xy,forward_error_cm=round(float(observed[1]-edge[25,1]),2),visible=bool(visible.any()))
            checks.append(rec)
            ids=np.flatnonzero(visible)
            if len(ids)>1:cv2.polylines(raw,[np.rint(uv[ids]).astype(np.int32)],False,(0,180,255),3)
            cv2.drawMarker(raw,xy,(250,0,250),cv2.MARKER_CROSS,18,2)
            cv2.putText(raw,f"#{f}  {t:.1f}s",(10,28),0,.8,(255,255,255),2,cv2.LINE_AA)
            tiles.append(raw)
        if f>=340:break
    cv2.imwrite(str(HERE/"heldout_crosswalk.jpg"),np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])]))
    output=dict(start_shift_cm=shift,operational_stop=first,diagnostic_first_visible=last,observations=checks)
    print(json.dumps(output,indent=2))
    (HERE/"heldout_crosswalk.json").write_text(json.dumps(output,indent=2))
if __name__=="__main__":main()


