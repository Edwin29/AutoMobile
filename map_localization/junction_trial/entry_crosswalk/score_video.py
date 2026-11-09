"""Render and score predictions only after the independent selector has finished."""
import csv
import json
import subprocess
from collections import Counter
from functools import lru_cache
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from clip import HERE,frames

@lru_cache(maxsize=16)
def font(size,bold):
    return ImageFont.truetype("C:/Windows/Fonts/malgun"+("bd" if bold else "")+".ttf",size)

def text(img,xy,string,size=19,color=(234,240,242),bold=False):
    ImageDraw.Draw(img).text(xy,string,font=font(size,bold),fill=color)

def box(im,rect,color,width):
    a=np.rint(rect).astype(int)
    cv2.rectangle(im,tuple(a[:2]),tuple(a[2:]),color,width,cv2.LINE_AA)

def overlap(a,b):
    inter=np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])))
    return float(inter/(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter))

def main():
    ref=json.loads((HERE/"reference.json").read_text(encoding="utf-8-sig"))
    pred={r["frame"]:r for r in json.loads((HERE/"predictions.json").read_text())}
    keys=np.array(list(map(int,ref["keyframes"])))
    coords=np.array(list(ref["keyframes"].values()))*2.
    scores=[];counts=Counter()
    cmd=["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24","-s","1280x552",
         "-r","30","-i","-","-c:v","libx264","-pix_fmt","yuv420p","-crf","18","-threads","2","-movflags","+faststart",str(HERE/"entry_comparison.mp4")]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as proc:
        for i,t,im in frames():
            r=pred[i];boxes=np.array(r["boxes"]);select=r["selected"]
            manual=np.array([np.interp(i,keys,coords[:,k]) for k in range(4)]) if i<=ref["last_visible"] else None
            iou=overlap(manual,boxes[select]) if manual is not None and select is not None else None
            if manual is None:status="absent" if select is None else "wrong"
            elif select is None:status="hold"
            else:status="match" if iou>=.30 else "wrong"
            counts[status]+=1
            scores.append(dict(frame=i,source_s=t,manual_visible=manual is not None,candidates=len(boxes),
                               selection=select,method=r["method"],status=status,iou=iou))
            a,b=im.copy(),im.copy()
            if manual is not None:box(a,manual,(54,231,156),3)
            for j,rect in enumerate(boxes):
                if j!=select:box(b,rect,(172,160,150),1)
            if select is not None:box(b,boxes[select],(255,200,45),3)
            sheet=Image.new("RGB",(1280,552),(24,33,40))
            text(sheet,(16,5),"중앙 사거리  |  처음 진입하는 도로의 횡단보도",25,bold=True)
            text(sheet,(981,11),f"원본 {t:.2f}초   ·   ¼배속",19)
            text(sheet,(16,54),"수동 표시",22,(156,231,54),True)
            text(sheet,(655,54),"코드 선택",22,(45,200,255),True)
            text(sheet,(835,60),"회색 박스 = 나머지 검출 후보",16)
            sheet.paste(Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB)),(0,91))
            sheet.paste(Image.fromarray(cv2.cvtColor(b,cv2.COLOR_BGR2RGB)),(640,91))
            text(sheet,(16,461),"진입로 표식 보임" if manual is not None else "진입로 표식이 화면 아래로 벗어남",19)
            method={"map":"지도 위치로 선택","continuity":"직전 표식과 연속 추적","hold":"판단 보류","entry_lost_at_bottom":"하단에서 추적 종료 · 선택 없음"}[r["method"]]
            text(sheet,(655,461),f"{method}   /   후보 {len(boxes)}개",18)
            state={"match":"일치","absent":"표식 없음 · 미선택","hold":"보이지만 미선택","wrong":"다른 위치 선택"}[status]
            text(sheet,(16,505),f"대조 결과   {state}",19,(245,208,112),True)
            data=sheet.tobytes()
            for _ in range(4):proc.stdin.write(data)
            if i==693:sheet.save(HERE/"three_candidates.jpg",quality=94)
        proc.stdin.close()
        if proc.wait():raise RuntimeError("video encoding failed")
    assert len(scores)==len(pred)==90
    with (HERE/"scores.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(scores[0]));w.writeheader();w.writerows(scores)
    print(counts,Counter(r["method"] for r in scores),[r['frame'] for r in scores if r["status"] in ("wrong","hold")])
if __name__=="__main__":main()
