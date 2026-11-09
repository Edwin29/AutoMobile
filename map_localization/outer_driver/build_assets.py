"""Build the portable track/camera assets from this project's calibration and map."""
from pathlib import Path
import sys
import importlib.util
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"ideal_drive"))
import ideal_drive as geometry

def build():
    src=HERE.parents[1]/"junction_logic/visibility_review/source"
    spec=importlib.util.spec_from_file_location("archived_lane",src/"lane_pipeline.py")
    lane=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lane)
    c=np.load(src/"calib_checkerboard.npz")
    size=tuple(map(int,c["size"]))
    K=cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(c["K"],c["D"],size,np.eye(3),balance=0.)
    M,bev,_=lane.bev_transform(dict(np.load(src/"ground.npz")),K,size[1])
    x,y=np.meshgrid(np.arange(-130,130.25,.25),np.arange(-130,130.25,.25))
    sdf=geometry.clearance(y,x).astype(np.float32)
    np.savez_compressed(HERE/"track.npz",route=geometry.ref[:,::-1],sdf=sdf,K=c["K"],D=c["D"],
                        rect_K=K,size=size,bev=bev,M=M)
if __name__=="__main__":
    build()
