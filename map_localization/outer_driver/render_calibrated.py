"""Compare trajectories over the first corner; freeze stopped vehicles at their actual stop poses."""
import math
import subprocess
import cv2
import numpy as np
from calibrated_lap import OUT
from validate_sim import Plant
from simulate_c import env,corner
from render_comparison import write
from controller import axes

def pixels(points):
    return np.rint((np.atleast_2d(points)[:,::-1]-[-109,13])*4).astype(np.int32)

def render():
    world=env["World"]()
    yy,xx=np.indices((360,362))
    mx,my=world.to_px(yy/4+13,xx/4-109)
    bg=cv2.remap(world.img,mx,my,cv2.INTER_LINEAR)
    modes=["original","adaptive","adaptive_comp"]
    labels=["기본 예측","회전 예측 보정","예측 + 출력 차이 보상"]
    runs=[np.genfromtxt(OUT/(mode+".csv"),delimiter=",",names=True) for mode in modes]
    panels=[bg.copy() for _ in runs]
    cmd=["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24","-s","1120x510",
         "-r","15","-i","-","-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",str(OUT/"first_corner.mp4")]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as proc:
        for i in range(min(390,max(map(len,runs)))):
            canvas=np.full((510,1120,3),(37,33,29),np.uint8)
            write(canvas,"실측 회전 반응을 반영한 가상 차  |  반시계 첫 코너",12,4,22,True)
            for n,rows in enumerate(runs):
                j=min(i,len(rows)-1);r=rows[j]
                pos=np.array([r["x_body"],r["y_body"]])
                rot=axes(math.radians(r["yaw_deg"]))
                if i==j and j:
                    prev=rows[j-1]
                    cv2.line(panels[n],pixels([prev["x_body"],prev["y_body"]])[0],pixels(pos)[0],(235,235,230),2)
                panel=panels[n].copy()
                stopped=j==len(rows)-1
                col=(85,95,250) if stopped else (235,245,230)
                cv2.polylines(panel,[pixels(corner@rot+pos)],True,col,2,cv2.LINE_AA)
                cv2.arrowedLine(panel,pixels(pos)[0],pixels(pos+9*rot[1])[0],(36,230,255),2,tipLength=.38)
                offset=10+370*n
                canvas[78:438,offset:offset+362]=panel
                write(canvas,labels[n],offset,44,19,True)
                write(canvas,"정지" if stopped else "주행 중",offset,443,17,True,col)
                cv2.putText(canvas,f"{r['t']:.1f} s",(offset+104,466),0,.65,col,1,cv2.LINE_AA)
                write(canvas,"차체 여유",offset,477,17)
                cv2.putText(canvas,f"{r['actual_margin_cm']:+.2f} cm",(offset+104,497),0,.65,(240,240,240),1,cv2.LINE_AA)
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError("Encoding failed")
if __name__=="__main__":render()
