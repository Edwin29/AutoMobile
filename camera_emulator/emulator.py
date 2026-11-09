"""Laptop-only, synthetic OV5647 appearance. Does not control the vehicle."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


class Emulator:
    def __init__(self, calibration=None, seed=42):
        self.calibration = Path(calibration or ROOT/'calibration/ov5647_fisheye_640x480.npz')
        with np.load(self.calibration) as c:
            self.K, self.D = c['K'].copy(), c['D'].copy()
            self.size = tuple(int(v) for v in c['size'])
        w,h = self.size
        yy,xx = np.mgrid[:h,:w].astype(np.float64)
        pixels = np.stack([xx,yy],-1).reshape(-1,1,2)
        # Backward map: every distorted output pixel asks for its ideal input ray.
        rays = cv2.fisheye.undistortPoints(pixels,self.K,self.D).reshape(h,w,2)
        if not np.isfinite(rays).all():
            raise ValueError('Calibration produces invalid rays')
        f = min((w-1)/2/np.abs(rays[...,0]).max(),(h-1)/2/np.abs(rays[...,1]).max())
        self.virtual_K = np.array([[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1.]])
        self.mx = (rays[...,0]*f+(w-1)/2).astype(np.float32)
        self.my = (rays[...,1]*f+(h-1)/2).astype(np.float32)
        self.rng = np.random.default_rng(seed)

    def crop(self, frame):
        if frame is None or frame.ndim != 3:
            raise ValueError('Expected BGR image')
        h,w = frame.shape[:2];tw,th=self.size
        cw=min(w,round(h*tw/th));ch=min(h,round(w*th/tw))
        x,y=(w-cw)//2,(h-ch)//2
        cropped=frame[y:y+ch,x:x+cw]
        return cv2.resize(cropped,self.size,interpolation=cv2.INTER_AREA),[x,y,cw,ch]

    def apply(self, frame, *, blur=1.0, noise=2., quality=85, gain=1., distort=True):
        base,crop=self.crop(frame)
        result=cv2.remap(base,self.mx,self.my,cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE) if distort else base.copy()
        if blur>0:result=cv2.GaussianBlur(result,(0,0),blur)
        pixels=result.astype(np.float32)*gain
        if noise>0:pixels+=self.rng.normal(0,noise,pixels.shape).astype(np.float32)
        result=np.clip(pixels,0,255).astype(np.uint8)
        if quality<100:
            ok,encoded=cv2.imencode('.jpg',result,[cv2.IMWRITE_JPEG_QUALITY,quality])
            if not ok:raise RuntimeError('JPEG encode failed')
            result=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
        return base,result,crop

    def metadata(self):
        return {'calibration':str(self.calibration.resolve()),
                'calibration_sha256':hashlib.sha256(self.calibration.read_bytes()).hexdigest(),
                'target_K':self.K.tolist(),'target_D':self.D.tolist(),
                'virtual_input_K':self.virtual_K.tolist(),'output_size':self.size,
                'limitations':'Synthetic degradation, not validated camera matching. Input assumed rectilinear; actual source FOV unknown.'}


def settings(args):
    return dict(blur=args.blur,noise=args.noise,quality=args.quality,gain=args.gain,distort=not args.no_distort)


def batch(args):
    source=Path(args.input).resolve();out=Path(args.output).resolve()
    if not source.exists():raise ValueError('Input does not exist')
    if out==source or (source.is_dir() and source in out.parents):
        raise ValueError('Output must be separate from input, not inside an input folder')
    out.mkdir(parents=True,exist_ok=False)  # Never overwrite a prior dataset.
    engine=Emulator(args.calibration,args.seed)
    meta={**engine.metadata(),'arguments':vars(args),'source':str(source)}
    (out/'config.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    image_ext={'.png','.jpg','.jpeg','.bmp','.webp'}
    video_ext={'.mp4','.avi','.mov','.mkv'}
    files=sorted(p for p in source.rglob('*') if p.suffix.lower() in image_ext|video_ext) if source.is_dir() else [source]
    count=0
    with (out/'manifest.jsonl').open('w',encoding='utf-8') as manifest:
        def save(frame,path,index,timestamp):
            nonlocal count
            params=settings(args)
            if args.randomize:
                params.update(blur=float(engine.rng.uniform(.4,1.8)),noise=float(engine.rng.uniform(0,5)),
                              quality=int(engine.rng.integers(65,96)),gain=float(engine.rng.uniform(.85,1.15)))
            _,result,crop=engine.apply(frame,**params)
            dest=out/f'{count:07d}.png'
            if not cv2.imwrite(str(dest),result):raise RuntimeError('Image write failed')
            manifest.write(json.dumps({'output':dest.name,'source':str(path),'source_frame':index,
                'source_time_s':timestamp,'crop_xywh':crop,'source_size':[frame.shape[1],frame.shape[0]],
                'parameters':params,'group_id':str(path)},ensure_ascii=False)+'\n');manifest.flush()
            count+=1
        for path in files:
            if args.limit and count>=args.limit:break
            if path.suffix.lower() in image_ext:
                frame=cv2.imread(str(path))
                if frame is None:raise ValueError(f'Cannot read {path}')
                save(frame,path,None,None)
            elif path.suffix.lower() in video_ext:
                cap=cv2.VideoCapture(str(path))
                if not cap.isOpened():raise ValueError(f'Cannot open {path}')
                i=0;next_time=0.;fps=cap.get(cv2.CAP_PROP_FPS)
                if fps<=0:cap.release();raise ValueError('Video FPS unavailable')
                try:
                    while not args.limit or count<args.limit:
                        ok,frame=cap.read()
                        if not ok:break
                        timestamp=i/fps
                        if timestamp+1e-6>=next_time:
                            save(frame,path,i,timestamp);next_time+=1/args.sample_fps
                        i+=1
                finally:cap.release()
            else:raise ValueError(f'Unsupported input: {path}')
    print(f'Saved {count} images: {out}')


def live(args):
    engine=Emulator(args.calibration,args.seed)
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps({**engine.metadata(),'arguments':vars(args)},indent=2),encoding='utf-8')
    cap=cv2.VideoCapture(args.camera,cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError('Webcam unavailable; check index/other camera apps; use a new output folder on retry')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,960)
    window='Webcam | synthetic car camera - Q quit, S snapshot'
    cv2.namedWindow(window,cv2.WINDOW_NORMAL)
    cv2.createTrackbar('blur x10',window,round(args.blur*10),30,lambda _:None)
    cv2.createTrackbar('noise',window,round(args.noise),15,lambda _:None)
    cv2.createTrackbar('JPEG quality',window,args.quality,100,lambda _:None)
    cv2.createTrackbar('gain x100',window,round(args.gain*100),200,lambda _:None)
    index=0
    try:
        while True:
            t=time.perf_counter();ok,frame=cap.read()
            if not ok:break
            params=dict(blur=cv2.getTrackbarPos('blur x10',window)/10,noise=cv2.getTrackbarPos('noise',window),
                        quality=max(1,cv2.getTrackbarPos('JPEG quality',window)),
                        gain=max(.01,cv2.getTrackbarPos('gain x100',window)/100),distort=not args.no_distort)
            t_process=time.perf_counter();base,result,crop=engine.apply(frame,**params)
            ms=(time.perf_counter()-t_process)*1000
            preview=np.hstack([base,result]);cv2.putText(preview,f'processing {ms:.1f}ms | capture+process {1000*(time.perf_counter()-t):.1f}ms',
                                                       (10,24),0,.55,(0,0,255),1)
            cv2.imshow(window,preview);key=cv2.waitKey(1)&255
            if key in (ord('q'),27) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
            if key==ord('s'):
                cv2.imwrite(str(out/f'{index:05d}_source.png'),frame)
                cv2.imwrite(str(out/f'{index:05d}_synthetic.png'),result)
                (out/f'{index:05d}.json').write_text(json.dumps({'crop_xywh':crop,'parameters':params}),encoding='utf-8')
                index+=1
    finally:cap.release();cv2.destroyAllWindows()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['live','batch'])
    p.add_argument('--input');p.add_argument('--output',required=True)
    p.add_argument('--camera',type=int,default=0);p.add_argument('--calibration')
    p.add_argument('--blur',type=float,default=1.);p.add_argument('--noise',type=float,default=2.)
    p.add_argument('--quality',type=int,default=85);p.add_argument('--gain',type=float,default=1.)
    p.add_argument('--no-distort',action='store_true');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--randomize',action='store_true');p.add_argument('--sample-fps',type=float,default=2.)
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    if not (0<=args.blur<=3 and 0<=args.noise<=15 and 1<=args.quality<=100 and .01<=args.gain<=2
            and 0<args.sample_fps<=60 and args.limit>=0):p.error('Parameter outside supported range')
    if args.mode=='batch' and not args.input:p.error('--input required')
    if args.mode=='live' and args.randomize:p.error('--randomize is for batch only')
    (live if args.mode=='live' else batch)(args)


if __name__=='__main__':main()
