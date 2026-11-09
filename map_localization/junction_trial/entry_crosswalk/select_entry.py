"""Manual-independent entry selection on handheld video with an operator-provided entry direction.
The four blue floor landmarks initialize the plane, never the held-out paint or detector boxes.
"""
import ast
import json
import sys
import cv2
import numpy as np
from clip import HERE,frames

ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/"road_marks"))
from detect_marks import Detector

# Pixel approximations of the four innermost blue floor disks, in order
# near left, near right, far left, far right. Some are occluded by props/glare.
ANCHORS=np.float32([[78,141],[425,143],[111,54],[327,55]])
def layouts():
    tree=ast.parse((ROOT/"junction_logic/visibility_review/source/sim_drive.py").read_text(encoding="utf8"))
    marks=next(ast.literal_eval(a.value) for a in tree.body if isinstance(a,ast.Assign) and getattr(a.targets[0],"id",None)=="CROSSWALKS")
    widths=json.loads((ROOT/"track_guard/measurements.photo.json").read_text())["road_widths"]
    layouts=[]
    center_marks=[(x,y,d) for x,y,d in marks if max(abs(x),abs(y))<35]
    for forward in ((0,1),(1,0),(0,-1),(-1,0)):
        # Express all FOUR individual map centres in approach coordinates. No rescaling to equal gaps.
        f=np.array(forward);right=np.array([f[1],-f[0]])
        lat,fwd=(20.5,19.25) if f[1] else (19.25,20.5)
        disk=np.float32([[-lat,-fwd],[lat,-fwd],[-lat,fwd],[lat,fwd]])
        H=cv2.getPerspectiveTransform(disk,ANCHORS)
        arms={}
        for x,y,axis in center_marks:
            xy=np.array([x,y])
            u,v=xy@right,xy@f
            along_view=(axis=="ns") == bool(f[1])
            width=widths["inner_horizontal_at_mark" if axis=="ns" else "inner_vertical_at_mark"]
            span=min(17.75,width)
            w,h=(span/2,6) if along_view else (6,span/2)
            role=("entry" if v<0 else "opposite") if along_view else ("left" if u<0 else "right")
            arms[role]=np.float32([[u-w,v-h],[u+w,v-h],[u+w,v+h],[u-w,v+h]])
        layouts.append((H,[arms[k] for k in ("entry","opposite","left","right")]))
    return layouts

def ground_motion(prev,cur):
    hsv=cv2.cvtColor(prev,cv2.COLOR_BGR2HSV)
    green=cv2.inRange(hsv,(31,42,35),(113,255,255))
    # Local features around green islands and blue disks; exclude the white crossings from initialization.
    mask=cv2.dilate(green,np.ones((9,9),np.uint8))
    p=cv2.goodFeaturesToTrack(cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY),450,.003,5,mask=mask)
    if p is None or len(p)<12:return None
    old=cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY);new=cv2.cvtColor(cur,cv2.COLOR_BGR2GRAY)
    q,ok,_=cv2.calcOpticalFlowPyrLK(old,new,p,None)
    back,ok2,_=cv2.calcOpticalFlowPyrLK(new,old,q,None)
    good=(ok.ravel()>0)&(ok2.ravel()>0)&(np.linalg.norm(back-p,axis=2).ravel()<1.)
    a,b=p[good,0],q[good,0]
    if len(a)<12:return None
    mat,inliers=cv2.findHomography(a,b,cv2.RANSAC,1.8)
    if mat is None or sum(inliers.ravel())<12 or inliers.mean()<.60:return None
    accepted=a[inliers.ravel()>0]
    if np.ptp(accepted[:,0])<75 or np.ptp(accepted[:,1])<30:return None
    return mat

def visible_rect(mat,poly):
    q=np.column_stack([poly,np.ones(len(poly))])@mat.T
    if np.any(q[:,2]<=1e-5):return None
    pts=np.float32(q[:,:2]/q[:,2,None])
    area,visible=cv2.intersectConvexConvex(pts,np.float32([[0,0],[640,0],[640,360],[0,360]]))
    if visible is None or area<100 or area/max(1.,cv2.contourArea(pts))<.05:return None
    v=visible.reshape(-1,2)
    return [*v.min(0),*v.max(0)]

def iou(a,b):
    x1,y1,x2,y2=a;u1,v1,u2,v2=b
    inter=max(0,min(x2,u2)-max(x1,u1))*max(0,min(y2,v2)-max(y1,v1))
    return inter/max(1,(x2-x1)*(y2-y1)+(u2-u1)*(v2-v1)-inter)

def pick(polys,boxes):
    if polys[0] is None:return None
    scores=[]
    for i,box in enumerate(boxes):
        score=iou(box,polys[0])
        rival=max([iou(box,b) for b in polys[1:] if b is not None]+[0])
        if score>=.22 and score>=rival+.12:scores.append((score,i))
    scores.sort(reverse=True)
    if not scores or len(scores)>1 and scores[0][0]-scores[1][0]<.12:return None
    return scores[0][1]

def main():
    cv2.setNumThreads(1);cv2.setRNGSeed(73)
    model=Detector();maps=layouts();prev=None;motion=np.eye(3);lost=False;rows=[]
    last_entry=None; passed=False
    for i,t,im in frames():
        if prev is not None and not lost:
            mat=ground_motion(prev,im)
            if mat is None:lost=True
            else:motion=mat@motion
        prev=im
        dets=[r for r in model.detect(im,confidence=.50) if r["kind"]=="cross"]
        boxes=[r["box"] for r in dets]
        projections=[[visible_rect(motion@H,z) for z in rects] for H,rects in maps] if not lost else []
        picks=[pick(p,boxes) for p in projections]
        chosen=picks[0] if picks and all(v==picks[0] for v in picks) else None
        method="map" if chosen is not None else "hold"
        # If map support disappears at the bottom edge, continue a confirmed identity only
        # while consecutive YOLO boxes overlap. Do not acquire another arm across a dropout.
        if not passed and chosen is None and last_entry is not None and last_entry[3]>=.95*360:
            consistent=[j for j,b in enumerate(boxes) if b[3]>=.95*360 and iou(last_entry,b)>=.50]
            if len(consistent)==1:
                chosen=consistent[0];method="continuity"
            elif not consistent and last_entry[1]>.78*360:
                passed=True
        if passed:
            chosen=None;method="entry_lost_at_bottom"
        last_entry=None if chosen is None else boxes[chosen]
        rows.append(dict(frame=i,t=t,boxes=boxes,selected=chosen,tracking=not lost,method=method,entry_boxes=[p[0] for p in projections]))
    (HERE/"predictions.json").write_text(json.dumps(rows,default=float))
    print('tracked',sum(r['tracking'] for r in rows),'selected',sum(r['selected'] is not None for r in rows))
if __name__=="__main__":main()
