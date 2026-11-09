"""Plots and numeric audit of an offline candidate search."""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from validate_candidates import setup

ROOT=Path(__file__).resolve().parent


def main():
    world,matcher=setup()
    names=["mid_turn_normal","entry_false_edge","mid_turn_occluded"]
    figure=np.full((1400,1100,3),(30,28,24),np.uint8)
    def text(value,xy,scale=.55,color=(235,235,235)):
        cv2.putText(figure,value,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
    text("Candidate pose search: camera view + coarse position score", (20,30),.8)
    text("Magenta = prior   Yellow = selected projection   Green star = truth", (20,60),.6)
    for row,name in enumerate(names):
        info=json.loads((ROOT/f"{name}.json").read_text())
        frame=cv2.imread(str(ROOT/f"{name}.png"))
        prior=np.array(info["prior"]);truth=np.array(info["truth"]);best=np.array(info["pose"])
        matcher.prepare(frame,prior)
        canvas=cv2.resize(frame,matcher.size)
        for pose,color in [(prior,(220,60,220)),(best,(20,220,255))]:
            for points,directions in matcher.local:
                if not len(points):continue
                uv,_,valid=matcher.project(pose[None],points,directions)
                for u,v in np.rint(uv[0,valid[0]]).astype(int):
                    cv2.circle(canvas,(u,v),1,color,-1)
        status="ACCEPTED" if info["accepted"] else "HELD"
        top=95+row*440
        text(f'{info["scene"]} / {info["condition"]}: {status}',(20,top),.65)
        figure[top+14:top+374,20:500]=cv2.resize(canvas,(480,360))
        text(f'Error: {info["position_error_cm"]:.2f} cm / {info["heading_error_deg"]:.2f} deg',
             (20,top+395),.6)
        data=np.load(ROOT/f"{name}_scores.npz")
        p=data["coarse"];z=data["coarse_score"]
        xs=np.unique(p[:,0]);ys=np.unique(p[:,1])
        heat=np.empty((len(ys),len(xs)))
        for i,y in enumerate(ys):
            for j,x in enumerate(xs):
                heat[i,j]=z[(p[:,0]==x)&(p[:,1]==y)].min()
        heat=np.nan_to_num(heat,nan=1.,posinf=1.)
        normalized=np.uint8(np.clip((heat-heat.min())/max(heat.max()-heat.min(),1e-6)*255,0,255))
        colored=cv2.applyColorMap(normalized[::-1],cv2.COLORMAP_VIRIDIS)
        figure[top+14:top+364,650:1000]=cv2.resize(colored,(350,350),interpolation=cv2.INTER_NEAREST)
        def mark(point):
            return (round(650+(point[0]-xs[0]+1)/(xs[-1]-xs[0]+2)*350),
                    round(top+14+(ys[-1]+1-point[1])/(ys[-1]-ys[0]+2)*350))
        cv2.drawMarker(figure,mark(truth), (30,255,50),cv2.MARKER_STAR,22,2)
        cv2.drawMarker(figure,mark(prior), (230,60,230),cv2.MARKER_CROSS,18,2)
        cv2.circle(figure,mark(best),12,(20,200,255),2)
        text("Position score (best yaw); dark = low",(600,top),.57)
        for offset in (-6,-4,-2,0,2,4,6):
            ux,uy=mark(prior[:2]+[offset,0])
            text(str(offset),(ux-8,top+385),.45)
            ux,uy=mark(prior[:2]+[0,offset])
            text(str(offset),(620,uy+5),.45)
        text("X offset from prior (cm)",(710,top+402),.5)
    cv2.imwrite(str(ROOT/"candidate_comparison.png"),figure)
    summary=[]
    for file in sorted(ROOT.glob("*.json")):
        if file.name=="summary.json":continue
        result=json.loads(file.read_text())
        if "truth" not in result:continue
        original=cv2.imread(str(ROOT/f'{result["scene"]}_normal.png'))
        modified=cv2.imread(str(ROOT/f'{result["scene"]}_{result["condition"]}.png'))
        if original is not None and modified is not None:
            prior=np.array(result["prior"])
            matcher.prepare(original,prior)
            original_edges=[x[0].copy() for x in matcher.observed]
            matcher.prepare(modified,prior)
            changes=sum(np.count_nonzero(old^new[0]) for old,new in zip(original_edges,matcher.observed))
            result["changed_edge_pixels"]=int(changes)
        if result["pose"] is not None:
            predicted=np.array(result["pose"]);truth=np.array(result["truth"])
            assert np.isclose(np.linalg.norm(predicted[:2]-truth[:2]),result["position_error_cm"])
            assert np.max(np.abs(predicted[:2]-np.array(result["prior"])[:2]))<=6.001
        summary.append({key:result.get(key) for key in
                        ("scene","condition","accepted","reason","position_error_cm","heading_error_deg",
                         "score","gap","forward_coverage","reverse_coverage","local_first_match",
                         "local_repeated_position_cm","local_repeated_heading_deg","seconds","candidates",
                         "changed_edge_pixels")})
    (ROOT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
