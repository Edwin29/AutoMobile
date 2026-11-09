"""Fixed-prior map test with archived annotations, plus independent image-only spot checks."""
import csv
import io
import json
import sys
import zipfile
import cv2
import numpy as np
import torch
from sim_clean import CNNDriver,Driver,HERE,RoadNet
from controller import axes
from train import road_edge

# Visually traced on rectified 320x240 RGB before seeing predictions.
# positive, explicit negative, ignore polygons (never scored).
SPOTS={
29:([[(0,198),(30,166),(58,137),(83,112),(86,106),(81,101),(71,99)],[(246,102),(245,108),(249,115),(270,132),(296,152),(318,168)]],
    [[(116,120),(103,151),(88,195)]],[]),
85:([[(2,99),(44,99),(72,96),(99,92),(111,87)],[(198,86),(215,90),(244,92),(280,91),(315,90)]],
    [[(215,103),(217,119),(216,134)]],[[(204,40),(233,40),(227,81),(201,81)]]),
197:([[(4,88),(54,88),(102,89),(146,92),(177,97),(204,105),(239,116),(268,129),(298,145),(318,156)]],
     [[(12,116),(35,104),(54,96)]],[]),
239:([[(4,78),(41,87),(82,97),(123,107),(156,117),(190,130),(224,146),(255,164),(285,183),(317,205)]],
     [[(5,174),(42,155),(79,138),(112,125)]],[])}

def line_mask(shape,curves):
    m=np.zeros(shape,np.uint8)
    for c in curves: cv2.polylines(m,[np.array(c,np.int32)],False,1)
    return m

def main():
    cv2.setNumThreads(1);torch.set_num_threads(3)
    out=HERE/'real_drive_review'; field=HERE.parent/'map_localization/outer_driver/field_runs'
    hsv=Driver('ccw');cnn=CNNDriver('ccw');cnn.net=RoadNet().eval()
    with zipfile.ZipFile(sys.argv[1]) as z:cnn.net.load_state_dict(torch.load(io.BytesIO(z.read('best.pt')),weights_only=True,map_location='cpu'))
    annotations=json.loads((HERE/'kit/data/real_points.json').read_text())
    rows=[]; grid=[]
    for key,scene in [('173559','seam'),('173933','glare'),('174607','curve'),('184214','curve_repeat')]:
        folder=next(field.glob('*'+key+'_edges'))
        prior=np.load(folder/'23.npz')['prior'];raw=cv2.imread(str(folder/'23.png'))
        rect=cv2.remap(raw,*hsv.remap,cv2.INTER_LINEAR)
        hsv.color_value_min=135 if scene.startswith('curve') else 130
        bird=np.zeros(hsv.bev[::-1],np.uint8)
        for c in annotations[scene]['true_boundary']:
            pts=cv2.perspectiveTransform((2*np.array(c,np.float32))[:,None],hsv.M)
            cv2.polylines(bird,[np.rint(pts).astype(np.int32)],False,1)
        human=Driver('ccw');human.boundary_pixels_bev=lambda _:np.nonzero(bird)
        for name,d in [('HSV',hsv),('CNN',cnn),('reference',human)]:
            d.record_edges=True;d.pose=prior.copy()
            ok=d.observe(rect,True)
            points,res,_=d.edge_observation
            rows.append([scene,name,ok,len(res),round(np.mean(abs(res)<2.5),3),round(d.residual,3)])
        for along in np.arange(-5.,5.1,.5):
            for lat in np.arange(-2.,2.1,.5):
                for deg in [-4,-2,0,2,4]:
                    pose=prior.copy();pose[:2]+=np.array([lat,along])@axes(prior[2]);pose[2]+=np.deg2rad(deg)
                    res=human.sample(human.sdf,points@axes(pose[2])+pose[:2])
                    grid.append([scene,along,lat,deg,np.median(abs(res)),np.mean(abs(res)<2.5)])
    photos=HERE/'20261106_193448_316130/take_001'
    metrics=[];panels=[]
    for idx,(pos,neg,ignored) in SPOTS.items():
        raw=cv2.imread(str(photos/f'{idx:06}.png'));rect=cv2.remap(raw,*hsv.remap,cv2.INTER_LINEAR)
        im=cv2.resize(rect,(320,240),interpolation=cv2.INTER_AREA); canvas=im.copy()
        labels=cv2.imread(str(HERE/'clean_inference_257/labels/take_001'/f'{idx:06}.png'),0)
        h=cv2.cvtColor(rect,cv2.COLOR_BGR2HSV);color=cv2.inRange(h,(20,100,135),(87,255,255));dark=np.uint8(h[:,:,2]<130)
        e=(color>0)&(cv2.dilate(dark,np.ones((3,3),np.uint8))>0)
        hsv_edge=cv2.resize(np.uint8(e),(320,240),interpolation=cv2.INTER_AREA)
        masks=[line_mask(labels.shape,v) for v in (pos,neg)]
        for polygon in ignored:
            for m in masks:cv2.fillPoly(m,[np.array(polygon,np.int32)],0)
        for m,col in zip(masks,[(0,255,0),(0,0,255)]):canvas[m>0]=col
        cv2.putText(canvas,str(idx),(3,19),0,.6,(255,255,255),1)
        view=[canvas]
        for name,edge in [('HSV image',hsv_edge),('CNN',road_edge(labels))]:
            dist=cv2.distanceTransform(np.uint8(edge==0),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            metrics.append([idx,name,*[round(float(np.mean(dist[m>0]<=2)),3) for m in masks]])
            img=im.copy();img[edge>0]=(255,255,0);view.append(img)
        panels.append(np.hstack(view))
    cv2.imwrite(str(out/'oracle_spot_check.jpg'),np.vstack(panels))
    for file,header,data in [('oracle_gate',['scene','observer','accept','n','inlier','median_cm'],rows),('oracle_grid',['scene','forward_cm','right_cm','yaw_deg','median_cm','inlier'],grid),('oracle_spots',['frame','observer','positive_coverage','negative_hit'],metrics)]:
        with (out/(file+'.csv')).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(header);w.writerows(data)
    print(rows);print(metrics)
    for s in annotations:
        passed=[r[1] for r in grid if r[0]==s and r[2]==r[3]==0 and r[4]<=1.8 and r[5]>=.55]
        print(s,passed)

if __name__=='__main__':main()
