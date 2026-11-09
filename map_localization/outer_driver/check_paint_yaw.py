"""Manual paint-direction check independent of optical flow and the PWM fit."""
import csv
import math
import cv2
import numpy as np
from check_corner_yaw import Ground,HERE,RUN,WHEEL_SPEED,TRACK_CM

# Points read in the upright raw frames. A and B are transverse marks on perpendicular straights.
points={
0:("A",[(196,191),(247,189),(329,187),(404,186),(459,187)]),
20:("A",[(153,212),(222,205),(300,201),(371,199),(444,199)]),
50:("A",[(61,308),(163,307),(274,303),(393,295),(501,285)]),
115:("B",[(3,298),(17,286),(83,237),(98,228),(148,196)]),
145:("B",[(17,374),(65,341),(128,302),(194,263),(249,231),(279,214)]),
155:("B",[(36,414),(92,378),(167,332),(249,282),(341,231)]),
175:("B",[(303,430),(350,403),(396,375),(451,335),(505,287)]),
350:("cross",[(338,190),(382,192),(425,193),(463,194),(533,198)]),
370:("cross",[(295,237),(355,235),(415,234),(462,233),(514,232),(551,232)]),
385:("cross",[(199,272),(266,265),(333,258),(393,251),(452,244),(500,238)])}

def direction(ground, uv):
    g=ground.points(uv)
    _,_,v=np.linalg.svd(g-g.mean(0),full_matrices=False)
    ang=math.degrees(math.atan2(v[0,1],v[0,0]))
    return (ang+90)%180-90

def main():
    g=Ground()
    with (RUN/'log.csv').open(encoding='utf8') as f: log=list(csv.DictReader(f))
    ts=np.array([float(r['t']) for r in log]); diff=np.array([float(r['left'])-float(r['right']) for r in log])
    yaw=np.r_[0,np.cumsum(np.degrees(WHEEL_SPEED/TRACK_CM*diff[:-1]*np.diff(ts)))]
    start=direction(g,points[0][1])
    tiles=[]
    with (HERE/'paint_yaw.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['frame','t','mark','angle_in_vehicle_deg','yaw_change_from_start_deg','pwm_from_start_deg'])
        for frame,(kind,uv) in points.items():
            a=direction(g,uv)
            change=a-start-(0 if kind=='A' else 90)
            row=(frame,round(ts[frame],2),kind,round(a,1),round(change,1),round(yaw[frame],1));w.writerow(row);print(row)
            if frame in (0,50,115,155,175,370):
                img=cv2.imread(str(RUN/f'frame_{frame:05d}.jpg'))
                for xy in uv: cv2.circle(img,xy,5,(0,215,255),1,cv2.LINE_AA)
                img=cv2.resize(img,(400,300))
                cv2.rectangle(img,(0,0),(400,29),(35,34,32),-1)
                cv2.putText(img,f'#{frame}  {ts[frame]:.1f}s  |  {kind}',(9,20),0,.58,(245,245,245),1,cv2.LINE_AA)
                tiles.append(img)
    cv2.imwrite(str(HERE/'paint_yaw_evidence.jpg'),np.vstack([np.hstack(tiles[:3]),np.hstack(tiles[3:])]))
if __name__=='__main__':main()
