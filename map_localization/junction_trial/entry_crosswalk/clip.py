from pathlib import Path
import subprocess,json,cv2
import numpy as np
HERE=Path(__file__).resolve().parent
SOURCE=next((Path.home()/'Documents').glob('*/KakaoTalk_20261031_165145372.mp4'))
def frames():
    raw=subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_entries","packet=pts_time","-of","json",str(SOURCE)])
    pts=sorted(float(row["pts_time"]) for row in json.loads(raw)["packets"])
    sel=[i for i,t in enumerate(pts) if 22<=t<25]
    cap=cv2.VideoCapture(str(SOURCE))
    cap.set(cv2.CAP_PROP_POS_FRAMES,sel[0])
    try:
        for i in sel:
            ok,im=cap.read()
            if not ok:raise RuntimeError(f"Missing source frame {i}")
            yield i,pts[i],cv2.resize(im,(640,360),interpolation=cv2.INTER_AREA)
    finally:cap.release()
if __name__=="__main__":
    allframes=list(frames())
    print('samples',len(allframes),'first,last',[(i,t) for i,t,im in (allframes[0],allframes[-1])])
    for group,subset in enumerate([allframes[:48:3],allframes[42:51]]):
        thumbs=[]
        for i,t,im in subset:
            im=cv2.resize(im,(320,180));cv2.putText(im,f'{i}  {t:.3f}',(3,13),0,.42,(0,255,255),1,cv2.LINE_AA);thumbs.append(im)
        while len(thumbs)%3:thumbs.append(np.zeros_like(thumbs[0]))
        cv2.imwrite(str(HERE/f'manual_review_{group}.jpg'),np.vstack([np.hstack(thumbs[k:k+3]) for k in range(0,len(thumbs),3)]))
    cv2.imwrite(str(HERE/'initial.jpg'),allframes[0][2])

