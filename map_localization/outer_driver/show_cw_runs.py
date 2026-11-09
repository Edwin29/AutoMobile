"""Signed minimum margin and completed path for the three clockwise cases."""
import json
import cv2
import numpy as np
from calibrated_lap import OUT
from validate_sim import Plant
from simulate_c import env,corner
from controller import axes
from render_comparison import write

def main():
    size=344
    def pixels(xy):
        return np.rint((np.atleast_2d(xy)[:,::-1]+125)*size/250).astype(np.int32)
    world=env["World"]()
    bg=cv2.rotate(cv2.resize(world.img,(size,size),interpolation=cv2.INTER_AREA),cv2.ROTATE_90_CLOCKWISE)
    canvas=np.full((467,1080,3),(39,33,28),np.uint8)
    write(canvas,"같은 가상 차의 시계방향 주행  |  흰 선은 실제로 이동한 궤적",12,6,20,True)
    for i,(key,title) in enumerate([("current","우회전 보상 없음"),("mirror","좌회전 값 복사 · 1.38배"),("measured","우회전 실측값 · 1.61배")]):
        stem="adaptive_comp_cw_"+key
        result=json.loads((OUT/(stem+".json")).read_text())
        data=np.genfromtxt(OUT/(stem+".csv"),delimiter=",",names=True)
        pos=np.column_stack([data["x_body"],data["y_body"]])
        view=bg.copy()
        cv2.polylines(view,[pixels(pos)],False,(245,245,245),2,cv2.LINE_AA)
        cv2.circle(view,pixels(pos[0])[0],4,(255,180,60),-1)
        rot=axes(np.radians(data["yaw_deg"][-1]))
        done=result["reason"]=="lap complete"
        color=(130,255,200) if done else (90,100,250)
        cv2.polylines(view,[pixels(corner@rot+pos[-1])],True,color,2,cv2.LINE_AA)
        x=9+359*i
        canvas[77:77+size,x:x+size]=view
        write(canvas,title,x,44,18,True)
        status="완주" if done else "정지"
        write(canvas,f"{status}   ·   최소 여유 {result['minimum_cm']:.2f}cm",x,430,17,True,color)
    cv2.imwrite(str(OUT/"cw_outcomes.jpg"),canvas)
if __name__=="__main__":main()
