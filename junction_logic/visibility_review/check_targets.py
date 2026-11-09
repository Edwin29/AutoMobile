"""Offline outer-lap screening example. Flags uncertainty; never invents a new steering command."""
import csv
import json
import math
from pathlib import Path
import cv2
import numpy as np
from replay import ROOT, SRC, RUN, lane, limited_view


def reason(r):
    if not r:
        return 'no_target'
    if r['gap'] is not None and r['gap']<min(lane.ROAD_WIDTH_CM.values())-lane.PAIR_TOL_CM:
        return 'implausibly_close_pair'
    if math.hypot(r['x'],r['y'])>lane.LOOKAHEAD_CM+2*lane.TRACE_STEP_CM:
        return 'lookahead_overshoot'
    return None


def main():
    rows=json.loads((ROOT/'samples.json').read_text(encoding='utf-8'))
    report={}
    for key in ['base','before','glare','left_hidden','right_hidden']:
        pairs=[r for r in rows if r['base'] and r[key]]
        shifts=[r for r in pairs if abs(r[key]['x']-r['base']['x'])>4]
        flags=[r for r in pairs if reason(r[key])]
        report[key]=dict(returned_pair=len(pairs),large_shifts=len(shifts),flags=len(flags),
                         flagged_large=sum(bool(reason(r[key])) for r in shifts))
    (ROOT/'screen.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(report)
    c=np.load(SRC/'calib_checkerboard.npz')
    size=tuple(map(int,c['size']))
    K=cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(c['K'],c['D'],size,np.eye(3),balance=0)
    m1,m2=cv2.fisheye.initUndistortRectifyMap(c['K'],c['D'],np.eye(3),K,size,cv2.CV_16SC2)
    M,bev,_=lane.bev_transform(dict(np.load(SRC/'ground.npz')),K,size[1])
    with (RUN/'log.csv').open(encoding='utf-8') as f:
        logs={int(r['frame']):r for r in csv.DictReader(f)}
    yy,xx=np.mgrid[:size[1],:size[0]]
    tiles=[]
    for frame,key in [(515,'base'),(515,'right_hidden'),(520,'base'),(520,'glare')]:
        img=cv2.remap(cv2.imread(str(RUN/f'frame_{frame:05d}.jpg')),m1,m2,cv2.INTER_LINEAR)
        if key=='glare':
            a=.99*np.exp(-.5*(((xx-.68*size[0])/(.09*size[0]))**2+((yy-.7*size[1])/(.23*size[1]))**2))
            img=np.uint8(np.round(img+(255.-img)*a[:,:,None]))
        def analyze():
            return lane.analyze(img,M,bev,road_hint=logs[frame-1]['road'],corner_bias=2.5,
                                left_bias_extra=.8,left_delay=1.,keep_gain=.008)[0]
        if key=='right_hidden':
            keep=(cv2.warpPerspective(((xx<size[0]//2)*255).astype(np.uint8),M,bev)>250)
            with limited_view(keep.astype(np.uint8)):
                result=analyze()
            img[:,size[0]//2:]=55
        else:
            result=analyze()
        top=cv2.resize(img,(400,206))
        bottom=cv2.warpPerspective(img,M,bev)
        for det in result['det'].values():
            cv2.polylines(bottom,[det['edge'].astype(np.int32)],False,(0,185,255),2)
        cv2.polylines(bottom,[result['Ct'].astype(np.int32)],False,(200,90,255),2)
        target=tuple(result['Ct'][result['ia']].astype(int))
        cv2.circle(bottom,(max(4,min(bev[0]-5,target[0])),max(4,min(bev[1]-5,target[1]))),5,(0,255,255),2)
        lat,fwd=result['lookahead']
        strip=np.full((45,400,3),30,np.uint8)
        cv2.putText(strip,f'#{frame} {key}    target ({lat:+.1f}, {fwd:.1f}) cm',(5,17),0,.48,(250,250,250),1,cv2.LINE_AA)
        cv2.putText(strip,'orange: boundary   purple: path   circle: target',(5,36),0,.44,(215,215,215),1,cv2.LINE_AA)
        tiles.append(np.vstack([strip,top,bottom]))
    cv2.imwrite(str(ROOT/'paired_examples.jpg'),np.vstack([np.hstack(tiles[:2]),np.hstack(tiles[2:])]))


if __name__=='__main__':
    main()
