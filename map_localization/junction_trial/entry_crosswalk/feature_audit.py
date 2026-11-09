"""Audit frame-to-frame support with original mask vs white features confined to crosswalk detections."""
import csv,json
import cv2
import numpy as np
from clip import HERE,frames

def audit(prev,cur,boxes,use_paint):
    old=cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
    new=cv2.cvtColor(cur,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(prev,cv2.COLOR_BGR2HSV)
    mask=cv2.dilate(cv2.inRange(hsv,(31,42,35),(113,255,255)),np.ones((9,9),np.uint8))
    if use_paint:
        area=np.zeros_like(mask)
        for x,y,u,v in boxes:
            cv2.rectangle(area,(round(x),round(y)),(round(u),round(v)),255,-1)
        white=(hsv[:,:,1]<85)&(hsv[:,:,2]>140)
        mask |= cv2.dilate(white.astype(np.uint8)*255,np.ones((5,5),np.uint8)) & area
    p=cv2.goodFeaturesToTrack(old,450,.003,5,mask=mask)
    counts=dict(corners=len(p) if p is not None else 0,tracked=0,inliers=0,ratio=0.,span_x=0.,span_y=0.)
    if counts["corners"]<12:return counts
    q,ok,_=cv2.calcOpticalFlowPyrLK(old,new,p,None)
    back,ok2,_=cv2.calcOpticalFlowPyrLK(new,old,q,None)
    good=(ok.ravel()>0)&(ok2.ravel()>0)&(np.linalg.norm(back-p,axis=2).ravel()<1.)
    a,b=p[good,0],q[good,0]
    counts["tracked"]=len(a)
    if len(a)<12:return counts
    mat,inliers=cv2.findHomography(a,b,cv2.RANSAC,1.8)
    if mat is None:return counts
    inside=a[inliers.ravel()>0]
    counts.update(inliers=len(inside),ratio=float(inliers.mean()),span_x=float(np.ptp(inside[:,0])),span_y=float(np.ptp(inside[:,1])))
    return counts

def main():
    cv2.setNumThreads(1);cv2.setRNGSeed(73)
    predictions={r["frame"]:r["boxes"] for r in json.loads((HERE/"predictions.json").read_text())}
    prior=None;rows=[]
    for i,t,im in frames():
        if prior is not None:
            for mode in ("base","paint"):
                d=audit(prior,im,predictions[i-1],mode=="paint")
                passed=d["inliers"]>=12 and d["ratio"]>=.6 and d["span_x"]>=75 and d["span_y"]>=30
                rows.append(dict(frame=i,t=t,mode=mode,passed=passed,**d))
        prior=im
    with (HERE/"feature_audit.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    for mode in ("base","paint"):
        group=[r for r in rows if r["mode"]==mode]
        print(mode,sum(r["passed"] for r in group),len(group),'first fail',next((r['frame'] for r in group if not r['passed']),None))
        print([r for r in group if r["frame"]==694])
        print("after fail",sum(r["passed"] for r in group if r["frame"]>694))
if __name__=="__main__":main()
