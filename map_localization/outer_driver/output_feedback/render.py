"""Two synchronized views of the first corner, with the actual logged output multipliers."""
import math
import subprocess
import cv2
import numpy as np
from experiment import HERE
from render_calibrated import pixels
from validate_sim import env
from simulate_c import corner
from controller import axes
from render_comparison import write

def render():
    cv2.setNumThreads(1)
    world=env["World"]()
    yy,xx=np.indices((360,362))
    mx,my=world.to_px(yy/4+13,xx/4-109)
    bg=cv2.remap(world.img,mx,my,cv2.INTER_LINEAR)
    runs=[np.genfromtxt(HERE/f"step_{m}.csv",delimiter=",",names=True) for m in ("fixed","dynamic")]
    panels=[bg.copy() for _ in runs]
    cmd=["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24","-s","772x522","-r","15","-i","-","-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",str(HERE/"step_comparison.mp4")]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as proc:
        for i in range(90,375):
            canvas=np.full((522,772,3),(37,33,29),np.uint8)
            write(canvas,"같은 좌회전 코너",12,4,23,True)
            write(canvas,"차량 반응  0.693" if i+1<180 else "차량 반응  0.645",478,7,19,True)
            for n,rows in enumerate(runs):
                j=min(i,len(rows)-1);r=rows[j]
                pos=np.array([r["x_body"],r["y_body"]])
                rot=axes(math.radians(r["yaw_deg"]))
                if j and j==i:
                    p=rows[j-1];cv2.line(panels[n],pixels([p["x_body"],p["y_body"]])[0],pixels(pos)[0],(235,235,230),2)
                panel=panels[n].copy()
                stopped=j==len(rows)-1
                color=(85,95,245) if stopped else (235,245,230)
                cv2.polylines(panel,[pixels(corner@rot+pos)],True,color,2,cv2.LINE_AA)
                cv2.arrowedLine(panel,pixels(pos)[0],pixels(pos+9*rot[1])[0],(36,230,255),2,tipLength=.38)
                x=12+n*384
                canvas[82:442,x:x+362]=panel
                write(canvas,("고정 보상" if n==0 else "동적 보상")+f"   ×{r['left_output_gain_next']:.3f}",x,43,20,True)
                write(canvas,"정지" if stopped else "주행 중",x,452,18,True,color)
                cv2.putText(canvas,f"{r['t']:.1f} s",(x+118,474),0,.67,color,1,cv2.LINE_AA)
                write(canvas,f"차체 여유  {r['actual_margin_cm']:.2f} cm",x,483,17)
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close()
        if proc.wait():raise RuntimeError("Video encoding failed")
if __name__=="__main__":render()
