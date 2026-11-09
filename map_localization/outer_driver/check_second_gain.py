"""Validate the fixed yaw gain on the SAME near red stripe at the second corner."""
import csv
import json
import cv2
import numpy as np
from check_corner_yaw import HERE,RUN,Ground,TRACK_CM,WHEEL_SPEED
from check_paint_yaw import direction
anchors={615:((2,339),(191,215)),645:((45,445),(366,259)),660:((296,437),(475,311))}
def main():
    ground=Ground()
    with (RUN/"log.csv").open(encoding="utf8") as f: rows=list(csv.DictReader(f))
    ts=np.array([float(r["t"]) for r in rows])
    diff=np.array([float(r["left"])-float(r["right"]) for r in rows])
    baseline=np.r_[0,np.cumsum(np.degrees(WHEEL_SPEED/TRACK_CM*diff[:-1]*np.diff(ts)))]
    ang={}; tiles=[]
    for i,((x0,y0),(x1,y1)) in anchors.items():
        im=cv2.imread(str(RUN/f"frame_{i:05}.jpg"))
        b,g,r=cv2.split(im.astype(float))
        red=(r>1.45*g)&(r>1.45*b)&(r>55)
        pts=[]
        for x in range(x0+4,x1-3,4):
            expected=np.interp(x,[x0,x1],[y0,y1])
            ys=np.where(red[:,x]&(np.abs(np.arange(len(red))-expected)<17))[0]
            if len(ys):pts.append((x,float(np.median(ys))))
        ang[i]=direction(ground,pts)
        for x,y in pts:cv2.circle(im,(x,round(y)),2,(0,255,255),-1)
        im=cv2.resize(im,(400,300));cv2.putText(im,f"#{i}  {ts[i]:.2f}s",(10,24),0,.65,(255,255,255),2)
        tiles.append(im)
    cv2.imwrite(str(HERE/"second_corner_yaw.jpg"),np.hstack(tiles))
    checks=[]
    for a,b in [(615,645),(645,660),(615,660)]:
        measured=ang[a]-ang[b];pred=baseline[a]-baseline[b]
        checks.append(dict(start=a,end=b,start_s=ts[a],end_s=ts[b],measured_left_deg=measured,
                           original_left_deg=pred,corrected_left_deg=pred*13/18))
    report=json.dumps(checks,indent=2);print(report)
    (HERE/"second_corner_yaw.json").write_text(report)
if __name__=="__main__":main()
