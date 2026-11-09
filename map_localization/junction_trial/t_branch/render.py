"""Render real closed-loop paths, estimated positions, and the exact synthetic controller inputs."""
import argparse
import json
import math
import subprocess
import cv2
import numpy as np
from trials import HERE,SITES,FRAME_ON,FRAME_OFF,surface,env,axes,CENTER_AHEAD
from simulate_c import corner
from controller import LENS_AHEAD
from render_comparison import write

def render(site):
    cv2.setNumThreads(1)
    world=env["World"]();clean=world.img
    center,yaw=SITES[site];rot=axes(yaw)
    def pixel(points):
        uv=(np.atleast_2d(points)-center)@rot.T
        return np.rint(np.c_[uv[:,0]+34,50-uv[:,1]]*2.1).astype(np.int32)
    y,x=np.indices((223,187))
    xy=np.stack((x/2.1-34,50-y/2.1),axis=-1)@rot+center
    mx,my=world.to_px(*xy.transpose(2,0,1))
    bg=cv2.remap(clean,mx,my,cv2.INTER_LINEAR)
    box_w,box_h=445,330
    canvas0=np.full((1072,1344,3),(36,33,28),np.uint8)
    name="왼쪽 삼거리 · 가로 내부 도로" if site=="wide" else "위쪽 삼거리 · 세로 내부 도로"
    write(canvas0,name,12,5,23,True)
    write(canvas0,"2배속   |   카메라 입력과 실제 지도   ·   흰색: 차체와 궤적   /   자홍색: 추정 차축 중점",12,46,17)
    panels=[]
    for j,(turn,condition) in enumerate([(t,c) for t in ("straight","right","pivot") for c in ("normal","hidden","fake")]):
        rows=np.genfromtxt(HERE/f"{site}_{turn}_{condition}.csv",delimiter=",",names=True)
        title={"normal":"정상 시야","hidden":"분기 쪽 가림","fake":"가짜 입구"}[condition]
        x0,y0=8+(j%3)*box_w,80+(j//3)*box_h
        label={"straight":"외곽 직진","right":"원호 우회전","pivot":"정지 후 우회전"}[turn]
        write(canvas0,label+"  |  "+title,x0,y0,19,True)
        world.img=clean
        passed=json.loads((HERE/f"{site}_{turn}_{condition}.json").read_text())["outcome"]=="pass"
        panels.append((rows,passed,surface(world,site,condition),x0,y0,bg.copy(),pixel(np.c_[rows["x"],rows["y"]])))
    end=max(len(s[0]) for s in panels)
    filename=HERE/f"compare_{site}.mp4"
    cmd=["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24","-s","1344x1072","-r","15","-i","-","-an","-c:v","libx264","-crf","20","-preset","veryfast","-threads","2","-pix_fmt","yuv420p","-movflags","+faststart",str(filename)]
    last=[0]*len(panels)
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as proc:
        for tick in range(1,end+40,2):
            canvas=canvas0.copy()
            active=FRAME_ON<=tick<FRAME_OFF
            phase="장애 적용" if active else "시야 복원" if tick>=FRAME_OFF else "정상"
            write(canvas,phase,1025,8,20,True)
            cv2.putText(canvas,f'{tick/15:.1f}s',(1196,32),0,.8,(235,235,235),2)
            for j,(data,passed,texture,x0,y0,tr,pixels) in enumerate(panels):
                k=min(tick-1,len(data)-1);r=data[k];p=np.array([r["x"],r["y"]]);rot2=axes(r["yaw"])
                if k>last[j]:
                    cv2.polylines(tr,[pixels[last[j]:k+1]],False,(250,250,250),2,cv2.LINE_AA)
                    last[j]=k
                stopped=not passed and k==len(data)-1
                state="정지" if stopped else "통과" if k==len(data)-1 else "주행"
                col=(60,60,255) if stopped else (227,244,235)
                view=tr.copy()
                cv2.polylines(view,[pixel(corner@rot2+p)],True,col,2,cv2.LINE_AA)
                est=np.array([r["est_x"],r["est_y"]]);cv2.drawMarker(view,pixel(est)[0],(200,100,248),cv2.MARKER_CROSS,10,2)
                cam=p+(LENS_AHEAD-CENTER_AHEAD)*rot2[1]
                world.img=texture if active else clean
                frame=world.render((*cam,r["yaw"]))
                canvas[y0+48:y0+228,x0:x0+240]=cv2.resize(frame,(240,180),interpolation=cv2.INTER_AREA)
                canvas[y0+32:y0+255,x0+246:x0+433]=view
                write(canvas,f"{state}   ·   경계 여유 {r['margin']:.2f}cm",x0,y0+260,17,True,col)
                write(canvas,f"자세 오차   {r['pos_error']:.2f}cm   /   {r['angle_error']:.2f}°",x0,y0+288,16)
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close()
        if proc.wait():raise RuntimeError("Video encoding failed")
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("site",choices=SITES);render(ap.parse_args().site)
