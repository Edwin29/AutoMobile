"""Paired replay on saved JPEGs only. No GPIO, motor, network or changes to Pi sources."""
import csv
import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'source'
RUN = ROOT / 'run_160024'
sys.path.insert(0, str(SRC))
import lane_pipeline as lane
import lane_pipeline_before_sameline as earlier


def snapshot(r):
    if r is None:
        return None
    return dict(x=float(r['lookahead'][0]), y=float(r['lookahead'][1]),
                k=float(r['curv_pp']), sides=len(r['det']), gap=r['gap'], road=r['road'],
                branches=list(r['branch_pts']), points=r['Ct'].tolist())


@contextmanager
def limited_view(mask):
    # Remove features AND road support after segmentation; do not paint a black
    # occluder that the HSV classifier would inadvertently classify as roadway.
    original = lane.edge_masks
    def observed(*args, **kwargs):
        e,road=original(*args, **kwargs)
        return e*mask, road*mask
    lane.edge_masks=observed
    try:
        yield
    finally:
        lane.edge_masks=original


def register(prev, cur, valid0, valid1):
    # Tracked floor texture estimates planar motion between consecutive saved images.
    pts=cv2.goodFeaturesToTrack(prev,160,.015,5,mask=valid0)
    if pts is None or len(pts)<10:
        return None
    moved,ok,_=cv2.calcOpticalFlowPyrLK(prev,cur,pts,None)
    back,ok2,_=cv2.calcOpticalFlowPyrLK(cur,prev,moved,None)
    h,w=cur.shape
    new=moved.reshape(-1,2)
    uv=np.rint(new).astype(int)
    accept=(ok.ravel()>0)&(ok2.ravel()>0)&(np.linalg.norm(back-pts,axis=2).ravel()<1.2)
    accept &= (uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    accept &= valid1[uv[:,1].clip(0,h-1),uv[:,0].clip(0,w-1)]>0
    if sum(accept)<10:
        return None
    src=pts.reshape(-1,2)[accept]
    mat,inliers=cv2.estimateAffinePartial2D(src,new[accept],method=cv2.RANSAC,ransacReprojThreshold=2.0)
    if mat is None or inliers.sum()<8 or inliers.mean()<.65:
        return None
    span=np.ptp(src[inliers.ravel()>0],axis=0)
    if min(span)<20 or not .95<=np.linalg.norm(mat[0,:2])<=1.05:
        return None
    return mat


def guarded_target(raw, prev, dt, prev_gray, gray, v0, v1, look):
    # One-step dropout experiment, with preceding unaltered observations.
    # If support collapses and no credible motion is available, abstain.
    if raw and raw['sides']==2 and raw['gap'] is not None and 16<=raw['gap']<=30:
        return raw['x'],'observed'
    if not (prev and 0<dt<=.45 and prev['sides']==2 and prev['gap'] is not None
            and 16<=prev['gap']<=30 and not prev['branches']):
        return None,'abstain'
    mat=register(prev_gray,gray,v0,v1)
    if mat is None:
        return None,'abstain'
    moved=np.array(prev['points'])@mat[:,:2].T+mat[:,2]
    cm=lane.to_cm(moved)
    d=np.linalg.norm(cm,axis=1)
    start=int(d.argmin())
    idx=np.flatnonzero((np.arange(len(d))>=start)&(cm[:,1]>0)&(d>=look))
    if not len(idx):
        return None,'abstain'
    x=float(cm[idx[0],0])
    if raw and abs(raw['x']-x)<=2.0:
        return raw['x'],'agrees'
    return x,'prior'


def summary(values):
    if not values:
        return None
    return dict(n=len(values),median=round(float(np.median(values)),2),
                p95=round(float(np.percentile(values,95)),2),maximum=round(max(values),2))


def main():
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    c=np.load(SRC/'calib_checkerboard.npz')
    g=dict(np.load(SRC/'ground.npz'))
    size=tuple(map(int,c['size']))
    kn=cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(c['K'],c['D'],size,np.eye(3),balance=0)
    m1,m2=cv2.fisheye.initUndistortRectifyMap(c['K'],c['D'],np.eye(3),kn,size,cv2.CV_16SC2)
    M,bev,fmin=lane.bev_transform(g,kn,size[1])
    w,h=size
    yy,xx=np.mgrid[:h,:w]
    alpha=.99*np.exp(-.5*(((xx-.68*w)/(.09*w))**2+((yy-.7*h)/(.23*h))**2))
    projected=lambda im:cv2.warpPerspective(im,M,bev)
    valid=projected(cv2.remap(np.full((h,w),255,np.uint8),m1,m2,cv2.INTER_NEAREST))>250
    validity={'left_hidden':valid&(projected((xx>=w//2).astype(np.uint8)*255)>250),
              'right_hidden':valid&(projected((xx<w//2).astype(np.uint8)*255)>250)}
    with (RUN/'log.csv').open(encoding='utf-8') as f:
        logs={int(row['frame']):row for row in csv.DictReader(f)}
    first_end=next((i for i,row in logs.items() if row['segment']=='S1'),-1)
    rows=[]; prev=None
    for fp in sorted(RUN.glob('frame_*.jpg')):
        frame=int(fp.stem.split('_')[1])
        rec=logs[frame]
        start_delay=2 if frame<first_end else 0
        hint=logs.get(frame-1,{}).get('road') or 'outer'
        def analyze(module,img,robust=False):
            r,notes=module.analyze(img,M,bev,road_hint=hint,corner_bias=2.5,corner_delay=start_delay,
                                   left_bias_extra=.8,left_delay=max(1,start_delay),keep_gain=.008,robust=robust)
            return snapshot(r),notes
        raw=cv2.imread(str(fp))
        und=cv2.remap(raw,m1,m2,cv2.INTER_LINEAR)
        glow=np.uint8(np.round(und+(255.-und)*alpha[:,:,None]))
        grey=lambda img:cv2.cvtColor(projected(img),cv2.COLOR_BGR2GRAY)
        base,notes=analyze(lane,und)
        old,_=analyze(earlier,und)
        robust,rnotes=analyze(lane,und,True)
        gl,gn=analyze(lane,glow)
        glrob,grn=analyze(lane,glow,True)
        cases={'base':base,'before':old,'robust':robust,'glare':gl,'glare_robust':glrob}
        for key,mask in validity.items():
            with limited_view(mask.astype(np.uint8)):
                cases[key]=analyze(lane,und)[0]
        targets={}
        for key in ['glare',*validity]:
            target,status=None,'abstain'
            if prev:
                mask=validity.get(key, valid&(projected(alpha.astype(np.float32))<.65))
                # Left/right loss is applied to tracked features as well as lane features.
                gray=grey(glow if key=='glare' else und)
                if key in validity:
                    gray[~mask]=0
                target,status=guarded_target(cases[key],prev[0],float(rec['t'])-prev[1],
                                             prev[2],gray,valid.astype(np.uint8),mask.astype(np.uint8),
                                             18-start_delay)
            targets[key]={'x':target,'status':status}
        prev=base,float(rec['t']),grey(und)
        rows.append(dict(frame=frame,t=float(rec['t']),segment=rec['segment'],
                         actual_x=rec['target_lat'],actual_ok=rec['ok'],guard=targets,
                         quality_reject=next((s for s in grn if '품질 낮음' in s),None),
                         same_line=any('두 번' in s for s in notes),
                         **{k:({a:v for a,v in r.items() if a!='points'} if r else None)
                            for k,r in cases.items()}))
    report=dict(n=len(rows),log_n=len(logs),log_lost=sum(r['ok']=='0' for r in logs.values()),
                model_min_forward_cm=float(fmin),height_cm=float(g['H']),pitch_deg=float(g['pitch']),
                same_line=sum(r['same_line'] for r in rows),conditions={},guard={})
    for key in cases:
        refkey='robust' if key=='glare_robust' else 'base'
        dif=[abs(r[key]['x']-r[refkey]['x']) for r in rows if r[key] and r[refkey]]
        report['conditions'][key]=dict(missing=sum(r[key] is None for r in rows),delta=summary(dif),
                                     over4=sum(v>4 for v in dif))
    for key in targets:
        shared=[r for r in rows if r['base'] and r[key] and r['guard'][key]['x'] is not None]
        report['guard'][key]=dict(counts=dict(Counter(r['guard'][key]['status'] for r in rows)),
            same_frames_raw=summary([abs(r[key]['x']-r['base']['x']) for r in shared]),
            same_frames_guard=summary([abs(r['guard'][key]['x']-r['base']['x']) for r in shared]))
    (ROOT/'samples.json').write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
    (ROOT/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':
    main()
