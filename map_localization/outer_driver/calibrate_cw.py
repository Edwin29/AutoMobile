"""Clockwise real yaw samples on two separate corners, Oct 31 camera calibration."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/"pi_TEST_captures_20261105/runs/20261031_151453"
ARCHIVE=ROOT/"junction_logic/review_video"
OUT=Path(__file__).resolve().parent/"calibrated_runs"
# Ordered along the same pink stripe within each pair; native pixels, excluding the chassis.
POINTS={440:[(383,222),(422,243),(466,268),(507,291),(548,316),(579,336)],
        480:[(123,319),(145,344),(177,376),(203,400),(238,430),(262,447)],
        920:[(425,233),(454,251),(489,272),(532,300),(578,331),(617,359)],
        980:[(110,338),(134,366),(161,395),(188,421),(213,445)]}

def main():
    lens=np.load(ARCHIVE/"calib_checkerboard.npz")
    g=np.load(ARCHIVE/"ground.npz")
    K,D=lens["K"],lens["D"]
    P=cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K,D,(640,480),np.eye(3),balance=0.)
    H=P@np.c_[g["e"],g["d"],-g["H"]*g["n"]]
    cap=cv2.VideoCapture(str(RUN/"video_raw.mp4"))
    with (RUN/"log.csv").open(encoding="utf8") as f: log=list(csv.DictReader(f))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(log)
    angles={}
    imgs=[]
    for frame,uv in POINTS.items():
        rect=cv2.fisheye.undistortPoints(np.array(uv,dtype=float).reshape(-1,1,2),K,D,P=P).reshape(-1,2)
        xyz=np.column_stack([rect,np.ones(len(rect))])@np.linalg.inv(H).T
        xy=xyz[:,:2]/xyz[:,2,None]
        _,_,v=np.linalg.svd(xy-xy.mean(0),full_matrices=False)
        angles[frame]=np.degrees(np.arctan2(v[0,1],v[0,0]))
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
        ok,img=cap.read(); assert ok
        for pt in uv:cv2.circle(img,pt,4,(0,240,255),1,cv2.LINE_AA)
        cv2.putText(img,f"#{frame}",(7,25),0,.7,(0,255,255),2)
        imgs.append(cv2.resize(img,(400,300)))
    cap.release()
    ts=np.array([float(r["t"]) for r in log])
    diff=np.array([float(r["left"])-float(r["right"]) for r in log])
    rows=[]
    for i,j in ((440,480),(920,980)):
        measured=(angles[j]-angles[i]+90)%180-90
        predicted=np.degrees(np.sum(diff[i:j]*np.diff(ts[i:j+1]))*16/13)
        rows.append(dict(start=i,end=j,seconds=ts[j]-ts[i],measured_deg=measured,ideal_deg=predicted,gain=measured/predicted))
    cv2.imwrite(str(OUT/"cw_samples.jpg"),np.vstack([np.hstack(imgs[:2]),np.hstack(imgs[2:])]))
    (OUT/"cw_calibration.json").write_text(json.dumps(rows,indent=2))
    print(rows)
if __name__=="__main__":main()
