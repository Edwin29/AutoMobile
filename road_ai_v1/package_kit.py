"""Audit generated pairs and build the self-contained Colab upload."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
KIT=HERE/"kit"
sys.path.insert(0,str(KIT))
from lighting import PROBES,lighting

def main():
    data=KIT/"data"
    with (data/"manifest.csv").open() as stream:entries=list(csv.DictReader(stream))
    splits={};hashes={};counts={};pixels=np.zeros(3,dtype=np.int64)
    for r in entries:
        site,split,uid=r["site"],r["split"],r["id"]
        assert splits.setdefault(site,split)==split
        img=data/"images"/(uid+".jpg")
        key=hashlib.sha256(img.read_bytes()).hexdigest()
        assert hashes.setdefault(key,split)==split
        mask=cv2.imread(str(data/"masks"/(uid+".png")),0)
        assert mask.shape==(240,320) and mask.max()<=2 and (mask==1).any()
        assert cv2.imread(str(img)).shape==(240,320,3)
        pixels+=np.bincount(mask.ravel(),minlength=3)
        counts[split]=counts.get(split,0)+1
    assert len(set(r["id"] for r in entries))==len(entries)
    audit=dict(split_frames=counts,site_splits=splits,class_pixels=pixels.tolist(),image_size=[320,240],
               positive_road="Includes crosswalk stripes and all painted transverse markings",
               real_eval="4 sparse manually reviewed diagnostic frames, no dense real training labels")
    (data/"audit.json").write_text(json.dumps(audit,indent=2))
    demos=[]
    example=next(r for r in entries if r["site"]=="T_east" and
                 (cv2.imread(str(data/"masks"/(r["id"]+".png")),0)==2).sum()>50)
    image=cv2.imread(str(data/"images"/(example["id"]+".jpg")))[:,:,::-1].copy()
    truth=cv2.imread(str(data/"masks"/(example["id"]+".png")),0)
    for i,key in enumerate(PROBES):
        im,gt=lighting(image,truth,key,np.random.default_rng(106+i))
        tile=im[:,:,::-1].copy()
        cv2.putText(tile,key,(7,20),0,.50,(15,15,15),2,cv2.LINE_AA)
        demos.append(tile)
    colormap=np.array([[70,70,70],[90,200,90],[215,135,45]],np.uint8)
    tile=colormap[truth];cv2.putText(tile,"labels: road / occluder",(6,20),0,.5,(255,255,255),1)
    demos.append(tile)
    cv2.imwrite(str(HERE/"stage_preview.jpg"),np.vstack([np.hstack(demos[:3]),np.hstack(demos[3:])]))
    with zipfile.ZipFile(HERE/"road_ai_starter.zip","w",zipfile.ZIP_DEFLATED,compresslevel=2) as archive:
        for file in [KIT/"train.py",KIT/"lighting.py",*sorted(data.rglob("*"))]:
            if file.is_file():archive.write(file,"road_ai_kit/"+file.relative_to(KIT).as_posix())
    with zipfile.ZipFile(HERE/"road_ai_starter.zip") as z:assert z.testzip() is None
    print(audit)

if __name__=="__main__":main()
