"""Render the measured closed-loop traces; compare identical holds with and without restored vision."""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent))
from trials import CENTER_AHEAD, SITES, axes, env, surface
from simulate_c import corner
from controller import LENS_AHEAD


def label(canvas, text, xy, color=(235, 235, 235)):
    cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, .55, color, 1, cv2.LINE_AA)


def panel(canvas, x, y, tick, rows, info, world, clean, hidden):
    row = rows[min(tick,len(rows))-1]
    center, yaw = SITES[info["site"]]
    body = np.array([row["x"],row["y"]])
    rot = axes(row["yaw"])
    blocked = info["fault_start"] <= tick < info["fault_end"]
    world.img = hidden if blocked else clean
    frame = world.render((*(body+(LENS_AHEAD-CENTER_AHEAD)*rot[1]),row["yaw"]))
    canvas[y+34:y+214,x:x+240] = cv2.resize(frame,(240,180))
    def pixels(pos):
        local=(np.atleast_2d(pos)-center)@axes(yaw).T
        return np.rint(np.c_[local[:,0]+30,36-local[:,1]]*2.5).astype(np.int32)
    yy,xx=np.indices((180,204),dtype=np.float32)
    xy=np.stack((xx/2.5-30,36-yy/2.5),axis=-1)@axes(yaw)+center
    mx,my=world.to_px(*xy.transpose(2,0,1))
    mini=cv2.remap(clean,mx,my,cv2.INTER_LINEAR)
    cv2.polylines(mini,[pixels(corner@rot+body)],True,(255,255,255),2)
    cv2.drawMarker(mini,pixels([row["est_x"],row["est_y"]])[0],(220,80,240),cv2.MARKER_CROSS,9,2)
    canvas[y+34:y+214,x+246:x+450]=mini
    wait=info["phase_ticks"].get("wait")
    resume=info["resume_frame"]
    state="moving"
    if wait and tick>=wait:
        state="stopped / checking map" if not resume or tick<resume else "resuming right turn"
    if tick>=info["phase_ticks"].get("exit",9999):state="outlet confirmed / exiting"
    if tick>=len(rows):state=info["outcome"]
    label(canvas, info["site"].upper()+"   |   "+("vision restored" if info["fault_end"]<800 else "persistent blockage"), (x,y+22))
    label(canvas,state,(x,y+237))
    label(canvas,"view: "+("obstructed" if blocked else "clear")+f'    PWM {row["L"]:+.2f} / {row["R"]:+.2f}',(x,y+261))


def main():
    cv2.setNumThreads(1)
    world=env["World"]()
    clean=world.img
    cases=[]
    for site in SITES:
        world.img=clean
        hidden=surface(world,site,"hidden")
        for variant in ("temporary","persistent"):
            base=OUT/f"{site}_hidden_+0_{variant}"
            cases.append((np.genfromtxt(base.with_suffix(".csv"),delimiter=",",names=True),
                          json.loads(base.with_suffix(".json").read_text()),hidden))
    cmd=["ffmpeg","-loglevel","error","-y","-f","rawvideo","-pixel_format","bgr24",
         "-video_size","940x616","-framerate","15","-i","-","-c:v","libx264","-pix_fmt","yuv420p",
         "-movflags","+faststart",str(OUT/"recovery_compare.mp4")]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as proc:
        for tick in range(215, max(len(c[0]) for c in cases)+15, 2):
            canvas=np.full((616,940,3),30,np.uint8)
            label(canvas,f"T-junction recovery   |   t = {tick/15:4.1f} s   |   2x   |   no injected slip",(12,32))
            for i,(rows,info,hidden) in enumerate(cases):
                panel(canvas,12+(i%2)*466,49+(i//2)*280,tick,rows,info,world,clean,hidden)
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close()
        if proc.wait():raise RuntimeError("ffmpeg failed")


if __name__=="__main__":
    main()
