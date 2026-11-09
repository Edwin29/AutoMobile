"""Four synchronized camera feeds and independent ground truth, 30-frame faults."""
import csv
import subprocess
import cv2
import numpy as np
from experiment import new_run,textures,HERE,ON,SHOW,axes,CENTER_AHEAD,LENS_AHEAD
from simulate_c import corner
from render_comparison import write

W=1240
IVORY=(235,243,240); PINK=(231,125,237); FAIL=(98,127,253); OK=(150,230,166)
DARK=(36,32,29)
labels={"normal":"정상 영상","inner":"안쪽 경계 가림","both":"양쪽 경계 가림","near":"가짜 경계 · 1.65cm","ghost":"가짜 경계 · 3cm"}
xy=lambda pts:np.rint((np.atleast_2d(pts)[:,::-1]-[-109,11])*2.45).astype(np.int32)

def main(reflections=False):
    kinds=("near","ghost") if reflections else ("normal","inner","both","ghost")
    duration=90 if reflections else SHOW
    H=462 if reflections else 816
    name="reflection_distance" if reflections else "corner_visibility"
    cv2.setNumThreads(1)
    plant,_=new_run()
    world=plant.world
    maps=textures(world)
    yy,xx=np.indices((252,252))
    mx,my=world.to_px(yy/2.45+11,xx/2.45-109)
    bg=cv2.remap(maps["normal"],mx,my,cv2.INTER_LINEAR)
    base=np.full((H,W,3),DARK,np.uint8)
    runs=[]
    for j,k in enumerate(kinds):
        x=12+(j%2)*615;y=91+(j//2)*355
        write(base,labels[k],x,y,20,True)
        with (HERE/(k+("_90.csv" if reflections else ".csv"))).open() as f:runs.append({int(r["tick"]):r for r in csv.DictReader(f)})
    write(base,f"왼쪽 아래 코너  /  카메라 장애 {duration}프레임",14,6,24,True)
    write(base,"흰 차체 = 실제   ·   자홍색 십자 = 추정 차축 중점   ·   지도와 경계 여유는 정답 좌표로 채점",13,50,16,color=IVORY)
    for j in range(len(kinds)):
        x=12+(j%2)*615;y=91+(j//2)*355
        write(base,"카메라 입력",x,y+32,14,color=IVORY)
        write(base,"가상 트랙",x+343,y+32,14,color=IVORY)
    cmd=["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24","-s",f"{W}x{H}","-r","15","-i","-","-c:v","libx264","-crf","19","-preset","veryfast","-threads","2","-pix_fmt","yuv420p","-movflags","+faststart",str(HERE/(name+".mp4"))]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as proc:
        for tick in range(ON-45,min(ON+duration+91,max(runs[0])+1)):
            canvas=base.copy()
            active=ON<=tick<ON+duration
            elapsed=tick/15
            phase="정상 시야" if tick<ON else "장애 주입" if active else "시야 복원"
            write(canvas,phase,818,7,20,True,FAIL if active else OK)
            cv2.putText(canvas,f"{elapsed:.2f} s  |  0.5x",(940,32),0,.66,IVORY,1,cv2.LINE_AA)
            for j,(k,data) in enumerate(zip(kinds,runs)):
                r=data[tick]
                x=12+(j%2)*615;y=91+(j//2)*355
                pos=np.array([float(r["x"]),float(r["y"])])
                rot=axes(float(r["yaw"]))
                cam=pos+(LENS_AHEAD-CENTER_AHEAD)*rot[1]
                world.img=maps[k if active else "normal"]
                frame=world.render((*cam,float(r["yaw"])))
                canvas[y+57:y+309,x:x+336]=cv2.resize(frame,(336,252),interpolation=cv2.INTER_AREA)
                top=bg.copy()
                trail=xy([[float(v["x"]),float(v["y"])] for i,v in data.items() if i<=tick])
                cv2.polylines(top,[trail],False,IVORY,1,cv2.LINE_AA)
                cv2.polylines(top,[xy(corner@rot+pos)],True,IVORY,2,cv2.LINE_AA)
                est=np.array([float(r["px"]),float(r["py"])])
                cv2.drawMarker(top,xy(est)[0],PINK,cv2.MARKER_CROSS,11,2)
                cv2.arrowedLine(top,xy(est)[0],xy(est+9*axes(float(r["pyaw"]))[1])[0],PINK,1,cv2.LINE_AA,tipLength=.4)
                canvas[y+57:y+309,x+343:x+595]=top
                status="정지 유지" if r["reason"] else "주행 · 정합"
                write(canvas,status,x+225,y,16,True,FAIL if r["reason"] else OK)
                write(canvas,f"위치 오차 {float(r['error_cm']):.2f}cm    방향 오차 {float(r['error_deg']):.2f}°    경계 여유 {float(r['clearance']):.2f}cm",x,y+315,15,color=IVORY)
            for _ in range(2):proc.stdin.write(canvas.tobytes())
            if tick==ON+duration-1:
                cv2.imwrite(str(HERE/(name+"_fault.jpg")),canvas)
        proc.stdin.close()
        if proc.wait():raise RuntimeError("encoding failed")
if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--reflections",action="store_true")
    main(ap.parse_args().reflections)

