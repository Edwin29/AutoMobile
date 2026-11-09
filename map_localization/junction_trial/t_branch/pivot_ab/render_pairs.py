"""Render three paired tests from saved poses and commands at a shared scenario clock."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent))
from trials import axes, env, SITES, CENTER_AHEAD
from simulate_c import corner
from controller import LENS_AHEAD
from render_comparison import write

def load_run(case,mode):
    name=f"{case}_{mode}"
    rows=np.genfromtxt(OUT/f"{name}.csv",delimiter=",",names=True)
    with (OUT/f"{name}_gains.csv").open() as stream:
        gains=list(csv.DictReader(stream))
    info=json.loads((OUT/f"{name}.json").read_text())
    assert len(rows)==len(gains)
    return rows,gains,info

def main():
    cv2.setNumThreads(1)
    world=env["World"]()
    center,_=SITES["narrow"]
    yy,xx=np.indices((210,210),dtype=np.float32)
    mx,my=world.to_px(xx/2.3-28+center[0],45-yy/2.3+center[1])
    background=cv2.remap(world.img,mx,my,cv2.INTER_LINEAR)
    def pixels(xy):
        loc=np.atleast_2d(xy)-center
        return np.rint(np.c_[loc[:,0]+28,45-loc[:,1]]*2.3).astype(np.int32)
    cmd=["ffmpeg","-v","error","-y","-f","rawvideo","-pix_fmt","bgr24","-s","1500x432",
         "-r","15","-i","-","-c:v","libx264","-threads","2","-pix_fmt","yuv420p",
         "-movflags","+faststart",str(OUT/"comparison.mp4")]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as encoder:
        for case,title in [("normal","정상 출발"),("delay_slide","출발 지연 + 밀림"),
                           ("weak_blind","회전 반응 저하 + 0.6초 가림")]:
            pair=[load_run(case,mode) for mode in ("output","joint","joint_edges")]
            base=np.full((432,1500,3),(39,33,27),np.uint8)
            write(base,title,15,8,23,True)
            for x,label in [(12,"A   출력 보상만 학습"),(510,"B   출력 + 횡단보도 예측계수"),(1008,"C   출력 + 경계 관측 예측계수")]:
                write(base,label,x,54,20,True)
            write(base,"흰색: 실제 차체   ·   자홍색: 추정 차축 중심   ·   2배속",12,397,16)
            for tick in range(min(r[2]["first_spin"] for r in pair)-6,
                              max(len(r[0]) for r in pair)+14,2):
                canvas=base.copy()
                cv2.putText(canvas,f"{tick/15:.1f}s",(1380,34),0,.72,(240,240,240),2)
                for j,(rows,gains,info) in enumerate(pair):
                    k=min(tick,len(rows))-1; row=rows[k];g=gains[k];x=12+j*498
                    pos=np.array([row["x"],row["y"]]);rot=axes(row["yaw"])
                    cam=world.render((*(pos+(LENS_AHEAD-CENTER_AHEAD)*rot[1]),row["yaw"]))
                    if int(g["blind"]):cam[:]=255
                    canvas[105:291,x:x+248]=cv2.resize(cam,(248,186))
                    mini=background.copy()
                    cv2.polylines(mini,[pixels(corner@rot+pos)],True,(255,255,255),2)
                    cv2.drawMarker(mini,pixels([row["est_x"],row["est_y"]])[0],(190,70,245),cv2.MARKER_CROSS,9,2)
                    canvas[93:303,x+258:x+468]=mini
                    state=info["outcome"] if tick>=len(rows) else g["phase"]
                    lines=[f'{state}   |   margin {row["margin"]:.2f} cm   yaw err {row["angle_error"]:.2f} deg',
                           f'prediction {float(g["yaw_gain"]):.3f}   output x{1/float(g["output_response"]):.3f}']
                    for i,line in enumerate(lines):
                        cv2.putText(canvas,line,(x,332+31*i),0,.5,(228,235,238),1,cv2.LINE_AA)
                encoder.stdin.write(canvas.tobytes())
        encoder.stdin.close()
        if encoder.wait():raise RuntimeError("ffmpeg failed")

if __name__=="__main__":
    main()
