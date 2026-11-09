"""Compare input sizes on valid, choose by marking macro F1, then evaluate test once."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from detect_marks import ROOT, MARKS, NAMES, Detector

DATA = ROOT / "embedded_optimization_dataset_1"


def iou(a, b):
    area = lambda r: max(0, r[2]-r[0]) * max(0, r[3]-r[1])
    inter = area((max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])))
    return inter / max(1.e-8, area(a)+area(b)-inter)


def compare(detector, split):
    counts = {s: dict(tp=0,fp=0,fn=0) for s in MARKS}
    frames = []
    for path in sorted((DATA / split / "images").glob("*.jpg")):
        im = cv2.imread(str(path))
        h,w = im.shape[:2]
        targets = []
        for line in (DATA / split / "labels" / (path.stem+".txt")).read_text().splitlines():
            c,x,y,bw,bh = map(float,line.split())
            if NAMES[int(c)] in MARKS:
                targets.append(dict(kind=NAMES[int(c)],box=[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]))
        predictions = detector.detect(im)
        matched = set()
        for pred in predictions:
            kind = pred['kind']
            if kind not in MARKS:
                continue
            candidates = [(iou(pred['box'],gt['box']),j) for j,gt in enumerate(targets)
                          if gt['kind']==kind and j not in matched]
            overlap,j = max(candidates,default=(0,None))
            if overlap >= .5:
                matched.add(j)
                counts[kind]['tp'] += 1
            else:
                counts[kind]['fp'] += 1
        for j,gt in enumerate(targets):
            if j not in matched:
                counts[gt['kind']]['fn'] += 1
        frames.append(dict(file=path.name,predictions=predictions,targets=targets))
    if not frames:
        raise ValueError(f"No photos in {split}")
    for d in counts.values():
        d['f1'] = 2*d['tp']/max(1,2*d['tp']+d['fp']+d['fn'])
    return dict(images=len(frames),counts=counts,macro_f1=np.mean([d['f1'] for d in counts.values()])),frames


def make_detector(size):
    return Detector(ROOT/f"best_1_{size}_ncnn_model",size)


def main():
    summary = {'confidence':.25,'match_iou':.5,'nms_iou':.45,'validation':{}}
    for size in (320,416,640):
        report,_ = compare(make_detector(size),'valid')
        summary['validation'][size]=report
        print(size,report,flush=True)
    chosen=max(summary['validation'],key=lambda n:summary['validation'][n]['macro_f1'])
    summary['chosen']=chosen
    detector=make_detector(chosen)
    summary['test'],frames=compare(detector,'test')
    print('TEST',chosen,summary['test'],flush=True)
    def hashes(split):
        return {hashlib.sha256(p.read_bytes()).digest() for p in (DATA/split/'images').glob('*.jpg')}
    summary['identical_jpegs_between_valid_test']=len(hashes('valid') & hashes('test'))
    (ROOT/'results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (ROOT/'test_predictions.json').write_text(json.dumps(frames,indent=2),encoding='utf-8')
    groups=sorted((ROOT.parent/'camera_sampled_v2_all_segments').glob('*'))
    photos=[]
    for group in groups:
        for p in sorted(group.glob('*.jpg')):
            photos.append(dict(group=group.name,file=p.name,predictions=detector.detect(cv2.imread(str(p)))))
    (ROOT/'recorded_predictions.json').write_text(json.dumps(photos,indent=2),encoding='utf-8')
    print('Recorded photos',len(photos),flush=True)


if __name__=='__main__':
    main()
