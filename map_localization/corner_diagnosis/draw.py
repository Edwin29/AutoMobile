import subprocess
import cv2
import numpy as np
from study import HERE, route, lane, DT
from simulate_c import ARCHIVE
from render_comparison import write

BLUE=(255,188,65)
PINK=(227,98,237)
YELLOW=(60,223,255)
INK=(235,244,240)

def px(xy):
    return np.rint((np.atleast_2d(xy)[:,::-1]-[-103,15])*5.4+[8,0]).astype(np.int32)

def render(rows,views,world):
    c=np.load(ARCHIVE/"calib_checkerboard.npz")
    k=cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(c["K"],c["D"],(640,480),np.eye(3),balance=0.)
    H=lane.ground_to_image(dict(np.load(ARCHIVE/"ground.npz")),k)
    yy,xx=np.indices((490,500))
    x=yy/5.4+15
    y=(xx-8)/5.4-103
    mx,my=world.to_px(x,y)
    background=cv2.remap(world.img,mx,my,cv2.INTER_LINEAR)
    cv2.polylines(background,[px(route)],True,BLUE,2,cv2.LINE_AA)
    inside=[i for i,r in enumerate(rows) if r["margin"]<0]
    worst=min(range(len(rows)),key=lambda i:rows[i]["margin"])
    cmd=["ffmpeg","-loglevel","error","-y","-f","rawvideo","-pixel_format","bgr24","-video_size","1152x670","-framerate","15","-i","-","-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",str(HERE/"corner.mp4")]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as proc:
        for i,(r,v) in enumerate(zip(rows,views)):
            image0,cam,rot,raw,target,map_target,body=v
            if i:
                cv2.line(background,px([[rows[i-1]["x"],rows[i-1]["y"]]])[0],px([[r["x"],r["y"]]])[0],INK,2)
            top=background.copy()
            cv2.polylines(top,[px(raw)],False,PINK,3,cv2.LINE_AA)
            cv2.polylines(top,[px(body)],True,(55,68,251) if r["margin"]<0 else INK,2,cv2.LINE_AA)
            cv2.circle(top,px(target)[0],6,YELLOW,-1,cv2.LINE_AA)
            cv2.drawMarker(top,px(map_target)[0],BLUE,cv2.MARKER_CROSS,16,2)
            image=image0.copy()
            def project(xy):
                local=(np.atleast_2d(xy)-cam)@rot.T
                q=np.column_stack([local,np.ones(len(local))])@H.T
                ok=(local[:,1]>=6)&(local[:,1]<=50)&(q[:,2]>0)
                uv=q[:,:2]/np.where(ok,q[:,2],1)[:,None]
                return np.rint(uv).astype(int),ok
            for pts,col in [(route,BLUE),(raw,PINK)]:
                uv,ok=project(pts)
                ids=np.flatnonzero(ok)
                for group in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1):
                    if len(group)>1:
                        cv2.polylines(image,[uv[group].astype(np.int32)],False,col,2,cv2.LINE_AA)
            uv,ok=project(target)
            if ok[0]:
                cv2.circle(image,uv[0],7,YELLOW,-1)
            frame=np.full((670,1152,3),(37,33,29),np.uint8)
            write(frame,"C안의 왼쪽 아래 코너 진입",14,6,25,True)
            write(frame,"가상 카메라 · 반시계",14,52,18)
            write(frame,"같은 순간의 지도와 차체",655,52,18)
            frame[88:568,:640]=image
            frame[84:574,650:1150]=top
            write(frame,"지도 중앙선  /  새 코드 목표점 (+)",12,580,18,True,BLUE)
            write(frame,"C안 추정 중앙선",386,580,18,True,PINK)
            write(frame,"C안 보정 목표점",597,580,18,True,YELLOW)
            write(frame,"흰 궤적  차체 중심",845,580,18)
            cv2.putText(frame,f'{r["t"]:.1f} s',(14,647),0,.8,INK,2)
            write(frame,"차체 경계 여유",153,620,18)
            cv2.putText(frame,f'{r["margin"]:+.2f} cm',(305,646),0,.75,INK,2)
            status="첫 경계 침범" if i==inside[0] else "최대 침범" if i==worst else ""
            if status:
                write(frame,status,555,619,20,True,(102,145,255))
                cv2.imwrite(str(HERE/("first.png" if i==inside[0] else "worst.png")),frame)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError("Video encoding failed")
