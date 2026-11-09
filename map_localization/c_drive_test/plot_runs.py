"""Plot body clearance by position in the track photo orientation."""
import cv2
import numpy as np

from simulate_c import HERE, env

world = env["World"]()
# Photo right is map north (+y) and photo down is map east (+x).
background = cv2.rotate(cv2.resize(world.img, (300,300)), cv2.ROTATE_90_CLOCKWISE)
def pixel(x,y):
    return np.rint(np.column_stack([y+125,x+125])*1.2).astype(np.int32)
profiles = ["no_bias","base","start2","extra"]
titles = ["Corner bias 0","Corner bias 2.5","Bias 2.5 + start delay 2","Previous + CCW package"]
canvas=np.full((714,1200,3),248,np.uint8)
for row,direction in enumerate(["cw","ccw"]):
    for col,profile in enumerate(profiles):
        x0,y0=col*300,40+row*335
        src=HERE/f"{direction}_{profile}.csv"
        label=titles[col]
        cv2.putText(canvas,direction.upper()+"  "+label,(x0+6,y0+19),0,.47,(55,47,40),1,cv2.LINE_AA)
        if not src.exists():
            cv2.putText(canvas,"Not applied",(x0+82,y0+165),0,.6,(110,110,110),1,cv2.LINE_AA)
            continue
        data=np.genfromtxt(src,delimiter=",",names=True)
        pts=pixel(data["x"],data["y"])
        view=background.copy()
        for i in range(1,len(pts)):
            c=(255,244,220) if data["body_clear_cm"][i]>=0 else (50,55,245)
            cv2.line(view,pts[i-1],pts[i],c,2,cv2.LINE_AA)
        j=data["body_clear_cm"].argmin()
        cv2.circle(view,pts[j],6,(0,255,255),2,cv2.LINE_AA)
        cv2.circle(view,pts[0],3,(255,200,60),-1,cv2.LINE_AA)
        canvas[y0+26:y0+326,x0:x0+300]=view
cv2.putText(canvas,"C controller | red = footprint crosses boundary     ring = least clearance     blue dot = start", (12,25),0,.67,(55,47,40),1,cv2.LINE_AA)
cv2.imwrite(str(HERE/"comparison.png"),canvas)
