"""Side-by-side measured axle runs, synchronized by simulation time."""
import json
import subprocess
import numpy as np
import cv2
from measured_axles import HERE, trial, env, FWD, OFFSET
from simulate_c import corner
from render_comparison import write

def pixels(points):
    return np.rint((np.atleast_2d(points)[:,::-1]-[-109,13])*4).astype(np.int32)

def main():
    cv2.setNumThreads(1)
    settings=[(0.,None),(FWD+OFFSET,None),(FWD+OFFSET,13.)]
    expected=json.loads((HERE/"measured_axles.json").read_text())["trials"]
    runs=[]
    for pb,track in settings:
        trace=[]
        result=trial(pb, track, trace)
        baseline=next(r for r in expected if (pb,track or 18.,result["controller"])==
                      (r["pivot_back_cm"],r["steering_track_cm"],r["controller"]))
        assert abs(result["minimum_clearance_cm"]-baseline["minimum_clearance_cm"])<1e-5
        runs.append(np.array(trace))
    world=env["World"]()
    yy,xx=np.indices((360,362))
    mx,my=world.to_px(yy/4+13,xx/4-109)
    panels=[cv2.remap(world.img,mx,my,cv2.INTER_LINEAR) for _ in runs]
    labels=["기존 C안 · 0cm","기존 C안 · 12.36cm","지도 조향 · 정답 위치"]
    filename=HERE/"measured_corner.mp4"
    with subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24","-s","1120x508","-r","15","-i","-","-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",str(filename)],stdin=subprocess.PIPE) as proc:
        for i in range(max(map(len,runs))+22):
            canvas=np.full((508,1120,3),(37,33,29),np.uint8)
            write(canvas,"실측 차축 중점 기준  |  왼쪽 아래 코너",12,4,22,True)
            for n,rows in enumerate(runs):
                j=min(i,len(rows)-1)
                t,x,y,deg,margin,*_=rows[j]
                col=(71,69,250) if margin<0 else (242,245,223)
                if j:
                    cv2.line(panels[n],pixels(rows[j-1,1:3])[0],pixels([x,y])[0],col,2)
                panel=panels[n].copy()
                c,s=np.cos(np.radians(deg)),np.sin(np.radians(deg))
                rot=np.array([[c,-s],[s,c]])
                cv2.polylines(panel,[pixels(corner@rot+[x,y])],True,col,2,cv2.LINE_AA)
                arrow=pixels(np.array([[0,0],[0,10]])@rot+[x,y])
                cv2.arrowedLine(panel,*arrow,(36,230,255),2,tipLength=.38)
                offset=10+370*n
                canvas[78:438,offset:offset+362]=panel
                write(canvas,labels[n],offset,44,19,True)
                cv2.putText(canvas,f"{t:.1f}s   {margin:+.2f} cm",(offset,467),0,.7,col,2,cv2.LINE_AA)
                write(canvas,"최소 여유",offset,478,15)
                cv2.putText(canvas,f"{rows[:j+1,4].min():+.2f} cm",(offset+90,499),0,.62,(240,240,240),1,cv2.LINE_AA)
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError("Encoding failed")
if __name__=="__main__":
    main()
