"""Synchronized viewers for the archived map-controller experiments (no new driving)."""
import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

from calibrated_lap import OUT
from controller import axes
from validate_sim import env
from simulate_c import FWD, corner
from render_comparison import write

FPS = 15
RATE = 4
TILE = 364
PITCH = TILE+18
IVORY = (233,242,241)
MUTED = (194,203,205)
RED = (104,113,255)
GREEN = (157,239,148)
BACKGROUND = (36,31,27)
GROUPS = {
    "directional_transfer": ("환경별 양방향 주행  /  좌회전 ×1.38 · 우회전 ×1.61 고정", 5, [
        ("fixed_comp"+scenario+"_"+direction+"_measured_directional",label+" · "+title)
        for direction,label in (("cw","시계"),("ccw","반시계"))
        for scenario,title in (("","기준"),("_second_response","약한 좌회전 반응"),("_glare","합성 반사광"),("_symmetric","좌우 반응 대칭"),
                               ("_symmetric_drift","대칭 + 왼쪽 쏠림 가정"))]),
    "directions": ("시계·반시계  /  방향별 회전 보상", 2, [
        ("adaptive_comp_cw_current","시계 · 우회전 보상 없음"),
        ("adaptive_comp_cw_mirror","시계 · 좌회전 값 복사  ×1.38"),
        ("adaptive_comp_cw_measured","시계 · 우회전 실측값  ×1.61"),
        ("adaptive_comp_ccw_measured","반시계 · 좌회전 실측 기반  ×1.38")]),
    "control": ("반시계  /  위치 예측과 바퀴 출력", 3, [
        ("original","기본 예측 · 출력 보상 없음"),
        ("adaptive","회전 예측만 보정"),
        ("adaptive_comp","회전 예측 + 출력 차이 보상")]),
    "conditions": ("반시계  /  같은 제어기로 환경만 변경", 3, [
        ("adaptive_comp","기준 · 좌회전 반응 0.693"),
        ("adaptive_comp_second_response","좌회전 반응 0.645"),
        ("adaptive_comp_ideal_response","좌회전 반응 1.000"),
        ("adaptive_comp_offset_right","출발 위치 오른쪽 1cm"),
        ("adaptive_comp_glare","가상 카메라에 반사광 추가")])
}

def pixels(points):
    return np.rint((np.atleast_2d(points)[:,::-1]+125)*TILE/250).astype(np.int32)

class Run:
    def __init__(self, stem, bg):
        self.rows=np.genfromtxt(OUT/(stem+".csv"),names=True,delimiter=",")
        self.summary=json.loads((OUT/(stem+".json")).read_text())
        self.xy=np.column_stack([self.rows["x_body"],self.rows["y_body"]])
        self.centers=pixels(self.xy)
        self.minima=np.minimum.accumulate(self.rows["actual_margin_cm"])
        assert abs(self.minima[-1]-self.summary["minimum_cm"])<1e-5
        assert abs(self.rows["t"][-1]-self.summary["elapsed_s"])<.001
        self.trail=bg.copy()
        self.last=0

    def draw(self, canvas, x, y, clock):
        i=np.searchsorted(self.rows["t"],clock+1e-5,side="right")-1
        i=int(np.clip(i,0,len(self.rows)-1))
        if i>self.last:
            cv2.polylines(self.trail,[self.centers[self.last:i+1]],False,IVORY,2,cv2.LINE_AA)
            self.last=i
        img=self.trail.copy()
        cv2.circle(img,self.centers[0],4,(180,140,60),-1,cv2.LINE_AA)
        row=self.rows[i]
        pos=self.xy[i]
        rot=axes(math.radians(row["yaw_deg"]))
        finished=i==len(self.rows)-1
        done=self.summary["reason"]=="lap complete"
        color=(GREEN if done else RED) if finished else IVORY
        cv2.polylines(img,[pixels(corner@rot+pos)],True,color,2,cv2.LINE_AA)
        cv2.arrowedLine(img,self.centers[i],pixels(pos+10*rot[1])[0],(42,225,255),2,cv2.LINE_AA,tipLength=.36)
        canvas[y:y+TILE,x:x+TILE]=img
        status=("완주" if done else "정지") if finished else "주행"
        write(canvas,status,x,y+TILE+1,18,True,color)
        cv2.putText(canvas,f"{row['t']:.1f} s",(x+55,y+TILE+24),0,.62,IVORY,1,cv2.LINE_AA)
        cv2.putText(canvas,f"{row['actual_margin_cm']:+.2f} / {self.minima[i]:.2f} cm",
                    (x+156,y+TILE+24),0,.58,IVORY,1,cv2.LINE_AA)
        return row

def render(group):
    cv2.setNumThreads(1)
    name,cols,panels=GROUPS[group]
    world=env["World"]()
    bg=cv2.rotate(cv2.resize(world.img,(TILE,TILE),interpolation=cv2.INTER_AREA),cv2.ROTATE_90_CLOCKWISE)
    runs=[Run(stem,bg) for stem,_ in panels]
    width=cols*PITCH
    cell_height=TILE+78
    height=78+math.ceil(len(panels)/cols)*cell_height
    base=np.full((height,width,3),BACKGROUND,np.uint8)
    write(base,name,10,4,24,True)
    write(base,"4배속  ·  궤적은 흰색  ·  하단 수치는 현재 / 누적 최소 경계 여유",10,42,16,color=MUTED)
    def where(j):
        return 9+(j%cols)*PITCH,78+(j//cols)*cell_height
    for j,(_,label) in enumerate(panels):
        x,y=where(j)
        write(base,label,x,y,18,True)
    if group=="conditions":
        x,y=where(5)
        write(base,"반사광 조건의 가상 카메라 입력",x,y,18,True)
        yy,xx=np.mgrid[:480,:640]
        alpha=(.99*np.exp(-.5*(((xx-.68*640)/(.09*640))**2+((yy-.7*480)/(.23*480))**2))).astype(np.float32)[:,:,None]
    end=max(run.rows["t"][-1] for run in runs)
    frames=math.ceil(end*FPS/RATE)+1+2*FPS
    path=OUT/(group+"_experiments.mp4")
    cmd=["ffmpeg","-loglevel","error","-y","-f","rawvideo","-pix_fmt","bgr24","-s",f"{width}x{height}",
         "-r",str(FPS),"-i","-","-an","-c:v","libx264","-preset","veryfast","-threads","2","-crf","20","-pix_fmt","yuv420p",
         "-movflags","+faststart",str(path)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as proc:
        for i in range(frames):
            canvas=base.copy()
            for j,run in enumerate(runs):
                x,y=where(j)
                row=run.draw(canvas,x,y+34,i*RATE/FPS)
            if group=="conditions":
                pos=np.array([row["x_body"],row["y_body"]])
                yaw=math.radians(row["yaw_deg"])
                eye=pos+FWD*axes(yaw)[1]
                view=world.render((*eye,yaw))
                view=np.rint(view+(255.-view.astype(np.float32))*alpha).astype(np.uint8)
                x,y=where(5)
                canvas[y+75:y+75+273,x:x+TILE]=cv2.resize(view,(TILE,273))
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close()
        if proc.wait(): raise RuntimeError("encoding failed")
    print(path,frames)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("group",choices=list(GROUPS))
    render(ap.parse_args().group)
