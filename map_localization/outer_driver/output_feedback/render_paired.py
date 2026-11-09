"""Render the four paired step trials at a shared 4x simulation clock."""
import json
import math
import subprocess
import cv2
import numpy as np
from experiment import HERE,axes
from validate_sim import env
from simulate_c import corner
from render_comparison import write

def main():
    cv2.setNumThreads(1)
    size=356
    def pixels(pts):
        return np.rint((np.atleast_2d(pts)[:,::-1]+125)*size/250).astype(np.int32)
    cases=[("cw","fixed"),("cw","dynamic"),("ccw","fixed"),("ccw","dynamic")]
    world=env["World"]()
    bg=cv2.rotate(cv2.resize(world.img,(size,size),interpolation=cv2.INTER_AREA),cv2.ROTATE_90_CLOCKWISE)
    panels=[]
    base=np.full((984,760,3),(37,32,29),np.uint8)
    write(base,"회전 반응이 약해진 뒤의 외곽 한 바퀴",12,5,22,True)
    write(base,"4배속   ·   흰 선 = 차체 중심 궤적   ·   표시 배수 = 주회전 방향 출력 차이",12,39,15)
    for n,(direction,mode) in enumerate(cases):
        stem=HERE/f"paired_step_{mode}_{direction}"
        rows=np.genfromtxt(stem.with_suffix(".csv"),names=True,delimiter=",")
        info=json.loads(stem.with_suffix(".json").read_text())
        assert abs(rows["actual_margin_cm"].min()-info["min_cm"])<1e-6
        label=("시계" if direction=="cw" else "반시계")+"  ·  "+("고정 출력" if mode=="fixed" else "동적 출력")
        x=12+376*(n%2);y=80+452*(n//2)
        write(base,label,x,y,19,True)
        panels.append((rows,info,bg.copy(),pixels(np.c_[rows["x_body"],rows["y_body"]]),x,y))
    frames=math.ceil(max(p[0]["t"][-1] for p in panels)*15/4)+31
    last=[0]*4
    with subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24","-s","760x984","-r","15","-i","-","-c:v","libx264","-threads","2","-pix_fmt","yuv420p","-movflags","+faststart",str(HERE/"paired_laps.mp4")],stdin=subprocess.PIPE) as proc:
        for i in range(frames):
            canvas=base.copy()
            write(canvas,"12초부터 양쪽 회전 반응 약 7% 감소" if i*4/15>=12 else "초기 회전 반응",398,8,17,True)
            for n,(rows,info,trail,centers,x,y) in enumerate(panels):
                j=min(i*4,len(rows)-1);r=rows[j]
                cv2.polylines(trail,[centers[last[n]:j+1]],False,(235,235,230),2,cv2.LINE_AA)
                last[n]=j
                view=trail.copy();pos=np.array([r["x_body"],r["y_body"]])
                rot=axes(math.radians(r["yaw_deg"]))
                end=j==len(rows)-1
                done=info["reason"]=="lap complete"
                color=((155,240,151) if done else (85,95,245)) if end else (235,245,230)
                cv2.polylines(view,[pixels(corner@rot+pos)],True,color,2,cv2.LINE_AA)
                cv2.arrowedLine(view,pixels(pos)[0],pixels(pos+9*rot[1])[0],(36,230,255),2,tipLength=.38)
                canvas[y+32:y+32+size,x:x+size]=view
                state=("완주" if done else "정지") if end else "주행"
                gain=r["right_output_gain_next" if n<2 else "left_output_gain_next"]
                write(canvas,state,x,y+391,17,True,color)
                cv2.putText(canvas,f"{r['t']:.1f}s     x{gain:.3f}",(x+78,y+414),0,.62,color,1,cv2.LINE_AA)
                write(canvas,f"차체 경계 여유   {r['actual_margin_cm']:.2f} cm",x,y+421,16)
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close()
        if proc.wait():raise RuntimeError("Encoding failed")
if __name__=="__main__":main()
