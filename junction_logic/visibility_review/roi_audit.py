"""Compare full/cropped inference on the same 52 labelled test photos."""
import json
import sys
from pathlib import Path
import cv2

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'road_marks'))
from detect_marks import Detector,NAMES
from evaluate import DATA,iou


def main():
    net=Detector()
    kinds={'horn','slow','stop','right','green','red'}
    # Only count GT boxes entirely inside the chosen crop in the paired comparison.
    report={}
    for fraction in (0.5,0.35):
        counts=dict(total=0,entirely_left=0,partial=0,inside=0,full_tp_inside=0,crop_tp_inside=0)
        for f in sorted((DATA/'test/images').glob('*.jpg')):
            im=cv2.imread(str(f)); h,w=im.shape[:2]; cut=round(fraction*w)
            gt=[]
            for line in (DATA/'test/labels'/(f.stem+'.txt')).read_text().splitlines():
                c,x,y,bw,bh=map(float,line.split()); kind=NAMES[int(c)]
                if kind not in kinds: continue
                counts['total']+=1
                box=[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]
                if box[2]<=cut: counts['entirely_left']+=1
                elif box[0]<cut: counts['partial']+=1
                else:
                    counts['inside']+=1
                    gt.append((kind,box))
            for key,img,offset in [('full',im,0),('crop',im[:,cut:],cut)]:
                used=set()
                for pred in net.detect(img):
                    b=pred['box']; box=[b[0]+offset,b[1],b[2]+offset,b[3]]
                    pairs=[(iou(box,g),i) for i,(k,g) in enumerate(gt) if i not in used and pred['kind']==k]
                    overlap,i=max(pairs,default=(0,None))
                    if overlap>=.5: used.add(i)
                counts[key+'_tp_inside']+=len(used)
        report[str(fraction)]=counts
    print(report)
    (ROOT/'roi_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
