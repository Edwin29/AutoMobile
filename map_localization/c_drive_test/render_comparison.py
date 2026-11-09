"""Render the saved C-controller runs in the track photo's orientation."""
import math
import subprocess
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from simulate_c import HERE, corner, env, rim

SIZE = 456
FPS = 15
STEP = 4
RED = (60, 62, 245)
IVORY = (238, 245, 231)
YELLOW = (63, 225, 255)
CHAPTERS = [
    ("공통 코너 보정 2.5cm", "base", "base"),
    ("공통 코너 보정 끔", "no_bias", "no_bias"),
    ("공통 보정 + 출발 코너 지연 2cm", "start2", "start2"),
    ("반시계 전용 보정 추가", "start2", "extra"),
]


@lru_cache(maxsize=100)
def label(text, size, bold=False):
    font = ImageFont.truetype("C:/Windows/Fonts/malgun" + ("bd" if bold else "") + ".ttf", size)
    mask = Image.new("L", (math.ceil(font.getlength(text))+4, size*2))
    ImageDraw.Draw(mask).text((1,0),text,font=font,fill=255)
    return np.asarray(mask.crop((0,0,mask.width,mask.getbbox()[3]+1)))[:,:,None]/255


def write(canvas,text,x,y,size=17,bold=False,color=IVORY):
    mask=label(text,size,bold)
    h,w=mask.shape[:2]
    region=canvas[y:y+h,x:x+w]
    region[:]=(1-mask)*region+mask*np.asarray(color)


def pixels(points):
    return np.rint((np.asarray(points)[:,::-1]+125)*SIZE/250).astype(np.int32)


class Panel:
    def __init__(self,world,direction,profile):
        self.world=world
        self.rows=np.loadtxt(HERE/f"{direction}_{profile}.csv",skiprows=1,delimiter=",")
        self.track=cv2.rotate(cv2.resize(world.img,(SIZE,SIZE),interpolation=cv2.INTER_AREA),cv2.ROTATE_90_CLOCKWISE)
        self.centers=pixels(self.rows[:,1:3])
        self.last=0
        self.direction="시계방향" if direction=="cw" else "반시계방향"
        self.minima=np.minimum.accumulate(self.rows[:,4])

    def draw(self,canvas,x,frame,tag):
        i=min(frame,len(self.rows)-1)
        t,px,py,degrees,margin,*_=self.rows[i]
        for j in range(self.last+1,i+1):
            color=RED if self.rows[j,4]<0 else IVORY
            cv2.line(self.track,self.centers[j-1],self.centers[j],color,2,cv2.LINE_AA)
        self.last=i
        write(canvas,self.direction,x,59,22,True)
        write(canvas,tag,x+112,64,16)
        time=f"{t:.1f}s"
        cv2.putText(canvas,time,(x+SIZE-82,84),0,.58,IVORY,1,cv2.LINE_AA)
        view=self.track.copy()
        c,s=np.cos(np.radians(degrees)),np.sin(np.radians(degrees))
        rot=np.array([[c,-s],[s,c]])
        pos=np.array([px,py])
        body=corner@rot+pos
        color=RED if margin<0 else IVORY
        cv2.polylines(view,[pixels(body)],True,color,3,cv2.LINE_AA)
        nose=pixels(np.array([[0,4],[0,12.25]])@rot+pos)
        cv2.arrowedLine(view,nose[0],nose[1],YELLOW,2,cv2.LINE_AA,tipLength=.45)
        edge=rim@rot+pos
        contact=edge[self.world.sd_at(*edge.T).argmin()]
        cv2.circle(view,pixels([contact])[0],4,YELLOW,-1,cv2.LINE_AA)
        canvas[101:101+SIZE,x:x+SIZE]=view
        status="안쪽 침범" if margin<0 else "경계 여유"
        # All seven source runs crossed only the island boundary (comparison.json).
        write(canvas,status,x,568,21,True,color)
        cv2.putText(canvas,f"{abs(margin):.2f} cm",(x+118,594),0,.83,color,2,cv2.LINE_AA)
        write(canvas,"누적 최대 침범",x+268,571,15)
        cv2.putText(canvas,f"{max(0.,-self.minima[i]):.2f} cm",(x+376,592),0,.54,IVORY,1,cv2.LINE_AA)


def main():
    world=env["World"]()
    path=HERE/"c_drive_comparison.mp4"
    cmd=["ffmpeg","-loglevel","error","-y","-f","rawvideo","-pix_fmt","bgr24","-s","960x648",
         "-r",str(FPS),"-i","-","-an","-c:v","libx264","-preset","veryfast","-crf","19",
         "-pix_fmt","yuv420p","-movflags","+faststart",str(path)]
    frame_count=0
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as proc:
        for number,(title,left,right) in enumerate(CHAPTERS,1):
            panels=[Panel(world,"cw",left),Panel(world,"ccw",right)]
            print(number,frame_count/FPS,title,flush=True)
            static=np.full((648,960,3),(37,33,29),np.uint8)
            write(static,f"{number}/4   {title}",18,8,28,True)
            write(static,"가상 주행  |  4배속",771,18,18)
            write(static,"노란 점   경계에 가장 가까운 차체 위치",18,615,16,color=YELLOW)
            write(static,"빨간 궤적   차체가 경계를 침범한 구간",494,615,16,color=(143,150,255))
            end=max(len(p.rows) for p in panels)-1
            for i in [*range(0,end,STEP),end]:
                canvas=static.copy()
                for k,panel in enumerate(panels):
                    tag=("직전 설정 유지" if k==0 else "전용 보정 적용") if number==4 else "동일 설정"
                    panel.draw(canvas,16+k*472,i,tag)
                proc.stdin.write(canvas.tobytes())
                frame_count+=1
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError("Video encoding failed")
    print("frames",frame_count)

if __name__=="__main__":
    main()
