"""Replay real camera recordings with the same CNN observer as the simulated driver."""
import csv
import io
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from sim_clean import CNNDriver, Driver, HERE, RoadNet

ROOT = HERE.parent
OUT = HERE / "real_drive_review"
ARCHIVE = ROOT / "junction_logic/review_video"
sys.path.insert(0, str(ARCHIVE))
from lane_pipeline import bev_transform


def mark(frame, observer, title, colour):
    y, x = observer.boundary_pixels_bev(frame)
    keep = (y >= 75) & (y <= 220)  # 6-35 cm forward, 5 pixels/cm
    xy = np.float32(np.column_stack([x[keep], y[keep]]))
    view = cv2.resize(frame, (320,240), interpolation=cv2.INTER_AREA)
    if len(xy):
        uv = cv2.perspectiveTransform(xy[:,None], np.linalg.inv(observer.M))[:,0]/2
        u,v = np.rint(uv).astype(int).T
        ok = (u>=0)&(u<320)&(v>=0)&(v<240)
        mask = np.zeros((240,320), np.uint8)
        mask[v[ok],u[ok]] = 1
        view[cv2.dilate(mask,np.ones((2,2),np.uint8))>0] = colour
    cv2.putText(view,title,(5,18),0,.48,(255,255,255),1,cv2.LINE_AA)
    return view,len(xy)


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    cnn = CNNDriver("ccw")
    cnn.net = RoadNet().eval()
    with zipfile.ZipFile(sys.argv[1]) as z:
        cnn.net.load_state_dict(torch.load(io.BytesIO(z.read("best.pt")),weights_only=True,map_location="cpu"))
    hsv = Driver("ccw")
    field = ROOT/"map_localization/outer_driver/field_runs"
    with (OUT/"pose_comparison.csv").open("w",newline="") as f:
        log = csv.writer(f)
        log.writerow(["run","frame","method","accepted","median_residual_cm","dx_cm","dy_cm","dyaw_deg"])
        for folder in sorted(field.glob("*_edges")):
            hsv.color_value_min = 135 if "v135" in folder.name or "forward127" in folder.name else 130
            for file in sorted(folder.glob("*.npz")):
                raw = cv2.imread(str(file.with_suffix(".png")))
                with np.load(file) as z:
                    prior = z["prior"].copy()
                for name,driver in [("HSV",hsv),("CNN",cnn)]:
                    driver.pose = prior.copy()
                    driver.residual = None
                    accepted = driver.observe(raw,False)
                    dx,dy,dt = driver.pose-prior
                    log.writerow([folder.name,file.stem,name,accepted,driver.residual,dx,dy,np.degrees(dt)])
    lens = np.load(ARCHIVE/"calib_checkerboard.npz")
    K,D = lens["K"],lens["D"]
    P = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K,D,(640,480),np.eye(3),balance=0)
    remap = cv2.fisheye.initUndistortRectifyMap(K,D,np.eye(3),P,(640,480),cv2.CV_16SC2)
    hsv.M,hsv.bev,_ = bev_transform(np.load(ARCHIVE/"ground.npz"),P,480)
    cnn.M,cnn.bev = hsv.M,hsv.bev
    hsv.color_value_min = 135
    with (OUT/"frames.csv").open("w",newline="") as f:
        log=csv.writer(f)
        log.writerow(["run","frame","log_t","left","right","HSV_points","CNN_points"])
        for path in sorted(ROOT.glob("pi_TEST*/runs/*/video_raw.mp4")):
            cap=cv2.VideoCapture(str(path))
            with (path.parent/"log.csv").open(encoding="utf-8-sig") as src:
                records=list(csv.DictReader(src))
            count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            aligned=len(records)==count and all(int(r.get("frame",i))==i for i,r in enumerate(records))
            dest=OUT/(path.parent.name+".mp4")
            with subprocess.Popen(["ffmpeg","-loglevel","error","-y","-f","rawvideo","-pix_fmt","bgr24","-s","960x240","-r",str(cap.get(5)),"-i","-","-c:v","libx264","-preset","veryfast","-crf","19","-pix_fmt","yuv420p","-movflags","+faststart",str(dest)],stdin=subprocess.PIPE) as enc:
                for i in range(count):
                    ok,raw=cap.read()
                    if not ok: raise IOError((path,i))
                    rect=cv2.remap(raw,*remap,cv2.INTER_LINEAR)
                    left,a=mark(rect,hsv,"HSV V135",(0,200,255))
                    right,b=mark(rect,cnn,"CNN clean",(255,255,0))
                    base=cv2.resize(rect,(320,240),interpolation=cv2.INTER_AREA)
                    cv2.putText(base,f"{path.parent.name[-6:]}  #{i}",(5,18),0,.48,(255,255,255),1)
                    enc.stdin.write(np.hstack([base,left,right]).tobytes())
                    r=records[i] if aligned else {}
                    log.writerow([path.parent.name,i,r.get("t"),r.get("left"),r.get("right"),a,b])
                enc.stdin.close()
                if enc.wait(): raise IOError(dest)
            cap.release()
            f.flush()
            print(dest,count,aligned,flush=True)


if __name__=="__main__":
    main()

