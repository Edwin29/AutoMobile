"""Ideal differential drive on the outer loop. Coordinates match the track photo."""
import ast
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "junction_logic/visibility_review/source/sim_drive.py"
keys = {"OUTER", "BLOCKS", "OUTER_R", "BLOCK_R", "CROSSWALKS"}
cfg = {n.targets[0].id: ast.literal_eval(n.value)
       for n in ast.parse(SOURCE.read_text(encoding="utf8")).body
       if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
       and n.targets[0].id in keys}


def photo_box(box):
    x0, x1, y0, y1 = box
    return y0, y1, x0, x1


outer = photo_box(cfg["OUTER"])
blocks = [photo_box(b) for b in cfg["BLOCKS"]]
outer_r, inner_r = cfg["OUTER_R"], cfg["BLOCK_R"]
vehicle = json.loads((ROOT / "track_guard/measurements.photo.json").read_text())["vehicle"]
length, width = vehicle["full_length"], vehicle["full_width_including_wheels"]
speed, dt, base, lookahead = 12., .05, 13., 10.  # cm, s; ideal wheel separation assumed
labels = ["top-right", "bottom-right", "bottom-left", "top-left"]


def box_distance(x, y, box, r):
    a, b, c, d = box
    u = abs(x - (a+b)/2) - (b-a)/2 + r
    v = abs(y - (c+d)/2) - (d-c)/2 + r
    return np.hypot(np.maximum(0, u), np.maximum(0, v)) + np.minimum(np.maximum(u, v), 0) - r


def clearance(x, y):
    dist = -box_distance(x, y, outer, outer_r)
    for box in blocks:
        dist = np.minimum(dist, box_distance(x, y, box, inner_r))
    return dist


def reference():
    # Average corresponding circular boundaries at equal tangent angle.
    arcs = []
    for i, (sx, sy) in enumerate([(1,-1), (1,1), (-1,1), (-1,-1)]):
        block = max(blocks, key=lambda b: sx*(b[0]+b[1])+sy*(b[2]+b[3]))
        def center(b, r):
            return np.array([b[int(sx>0)]-sx*r, b[2+int(sy>0)]-sy*r])
        origin = (center(outer, outer_r)+center(block, inner_r))/2
        angles = np.linspace((i-1)*np.pi/2, i*np.pi/2, 341)
        arcs.append(origin+(outer_r+inner_r)/2*np.c_[np.cos(angles), np.sin(angles)])
    parts = []
    for i, arc in enumerate(arcs):
        end, start = arc[-1], arcs[(i+1)%4][0]
        parts.extend([arc[:-1], np.linspace(end, start, math.ceil(np.linalg.norm(end-start)/.1), endpoint=False)])
    return np.concatenate(parts)


ref = reference()
corners = np.array([[1,1],[1,-1],[-1,-1],[-1,1]]) * [length/2,width/2]
# Perimeter samples spaced at most 0.25 cm, including corners.
rim = np.concatenate([np.linspace(a,b,math.ceil(np.linalg.norm(b-a)/.25),endpoint=False)
                      for a,b in zip(corners,np.roll(corners,-1,axis=0))])


def run(clockwise):
    route = ref if clockwise else ref[::-1]
    xw,yw,_ = next(mark for mark in cfg["CROSSWALKS"] if mark[1]<-75)
    start = np.argmin(np.linalg.norm(route-[yw,xw],axis=1))
    route = np.roll(route,-start,axis=0)
    points = np.vstack([route,route[0]])
    s = np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    n = len(route)
    idx = steps = 0
    pos = route[0].copy()
    theta = math.atan2(*(route[1]-pos)[::-1])
    records = []
    for step in range(math.ceil(1.5*s[-1]/(speed*dt))):
        # Use only local progress; actual vehicle state is never snapped to route.
        near = (idx+np.arange(-5,26))%n
        move = np.argmin(np.linalg.norm(route[near]-pos,axis=1))-5
        idx = (idx+move)%n
        steps += move
        target_s = (s[idx]+lookahead)%s[-1]
        target = np.array([np.interp(target_s,s,points[:,k]) for k in range(2)])
        c,h = math.cos(theta), math.sin(theta)
        rot = np.array([[c,h],[-h,c]])
        body = rim@rot+pos
        margins = clearance(body[:,0],body[:,1])
        worst = margins.argmin()
        records.append((pos.copy(), corners@rot+pos, target, float(margins[worst]), body[worst], float(np.linalg.norm(pos-route[idx]))))
        if steps>=n:
            return records
        delta = target-pos
        # In photo coordinates a positive heading change turns right.
        curvature = 2*(-h*delta[0]+c*delta[1])/np.dot(delta,delta)
        left, right = speed*(1+base*curvature/2), speed*(1-base*curvature/2)
        v, omega = (left+right)/2, (left-right)/base
        angle = omega*dt
        pos += v*dt*np.sinc(angle/(2*np.pi))*np.array([math.cos(theta+angle/2), math.sin(theta+angle/2)])
        theta += angle
    raise RuntimeError("Lap did not finish")


def corner_name(pos):
    x,y = pos
    if min(abs(x),abs(y))<50:
        return "straight"
    return ("bottom" if y>0 else "top")+("-right" if x>0 else "-left")


def summarize(rows):
    margins = [r[3] for r in rows]
    return dict(lap_time_s=(len(rows)-1)*dt, min_cm=min(margins), max_center_error_cm=max(r[5] for r in rows),
                crossed_samples=sum(v<0 for v in margins),
                corner_min_cm={k:min(r[3] for r in rows if corner_name(r[0])==k) for k in labels})


def px(points):
    return np.rint(np.asarray(points)*2+[232,260]).astype(np.int32)


def render(runs):
    yy,xx = np.indices((512,464))
    road = clearance((xx-232)/2,(yy-260)/2)>=0
    bg = np.full((512,464,3),(55,91,60),np.uint8)
    bg[road] = (54,52,48)
    cv2.polylines(bg,[px(ref)],True,(215,161,74),1,cv2.LINE_AA)
    names = list(runs)
    worsts = {k:min(runs[k],key=lambda row:row[3]) for k in runs}
    def draw(name, row, elapsed):
        panel = bg.copy()
        pos,box,target,margin,contact,_ = row
        cv2.polylines(panel,[px(box)],True,(233,243,247),2,cv2.LINE_AA)
        cv2.line(panel,px(pos),px(target),(72,218,237),2)
        cv2.circle(panel,px(target),4,(72,218,237),-1)
        cv2.circle(panel,px(contact),4,(123,99,255),-1)
        for text,y in [(name+"   |   "+elapsed,28),(f"Body clearance   {margin:.2f} cm",496)]:
            cv2.putText(panel,text,(15,y),cv2.FONT_HERSHEY_SIMPLEX,.62,(247,250,246),1,cv2.LINE_AA)
        return panel
    still=np.hstack([draw(k,worsts[k],"closest approach") for k in names])
    cv2.imwrite(str(HERE/"closest.png"),still)
    with subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","bgr24","-s","928x512","-r","20","-i","-","-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",str(HERE/"ideal_laps.mp4")],stdin=subprocess.PIPE) as proc:
        for i in range(0,max(map(len,runs.values())),4):
            panels=[]
            for k in names:
                j=min(i,len(runs[k])-1)
                panels.append(draw(k,runs[k][j],f"{j*dt:.1f} s   (4x)"))
            proc.stdin.write(np.hstack(panels).tobytes())
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError("Video encoding failed")


if __name__=="__main__":
    runs={k:run(k=="CW") for k in ["CW","CCW"]}
    summary={k:summarize(v) for k,v in runs.items()}
    (HERE/"results.json").write_text(json.dumps(summary,indent=2))
    render(runs)
    print(json.dumps(summary,indent=2))
