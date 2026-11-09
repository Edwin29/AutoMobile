"""Generate an offline schematic and decisions; does not connect to the car."""
import json
from pathlib import Path
import cv2
import numpy as np
from guard import TianGraph,demo_road,Supervisor,Vehicle,Sample

out=Path(__file__).parent/'output';out.mkdir(exist_ok=True)
graph=TianGraph();road=demo_road()
canvas=np.full((*road.mask.shape,3),(65,140,65),np.uint8)
canvas[road.mask]=(50,50,50)
coords=[.13,.9,1.67]
for n,(x,y) in graph.nodes.items():
    point=(int(coords[x]/road.resolution),int(coords[y]/road.resolution))
    for m in graph.adj[n]:
        xx,yy=graph.nodes[m]
        cv2.line(canvas,point,(int(coords[xx]/road.resolution),int(coords[yy]/road.resolution)),(180,180,180),2)
    cv2.circle(canvas,point,12,(0,210,255),-1)
canvas=cv2.flip(canvas,0)
for n,(x,y) in graph.nodes.items():
    cv2.putText(canvas,n,(int(coords[x]/road.resolution)+15,canvas.shape[0]-int(coords[y]/road.resolution)),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
cv2.putText(canvas,'FICTIONAL 1.8m x 1.8m DEMO - NOT A SURVEY',(15,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1)
cv2.putText(canvas,'00 -> 10 : STRAIGHT / LEFT only',(15,canvas.shape[0]-20),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
cv2.imwrite(str(out/'topology_demo.png'),canvas)
sup=Supervisor(road,Vehicle(.16,.22,.001))
path=[Sample(0,.9,.6,np.pi/2,0),Sample(.5,.9,.6,np.pi/2,0)]
base=dict(covariance=np.zeros((3,3)),now=1,last_camera=1,turn='straight',hypotheses=[('00','10')],min_deceleration=.3,latency=.1,tracking_error_bound=0,simulation=True)
cases={}
for name,kw in [('ideal_centered',{}),('wrong_right_turn',{'turn':'right'}),('camera_lost',{'last_camera':0}),('position_std_4mm',{'covariance':np.diag([.004**2,.004**2,0])}),('real_use_without_survey',{'simulation':False})]:
    decision=sup.evaluate(path,**(base|kw));cases[name]=vars(decision)
(out/'demo_results.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
print(json.dumps(cases,indent=2))
