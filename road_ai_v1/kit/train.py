"""Colab training/evaluation for visible road segmentation (RGB, rectified 320x240)."""
import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from lighting import PROBES,lighting

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"

def read_rgb(path):
    with path.open("rb") as stream:raw=np.frombuffer(stream.read(),np.uint8)
    return cv2.cvtColor(cv2.imdecode(raw,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)

class Frames(Dataset):
    def __init__(self,split,level="clean",probe="clean"):
        with (DATA/"manifest.csv").open() as f:self.entries=[r for r in csv.DictReader(f) if r["split"]==split]
        self.level,self.probe,self.split,self.epoch=level,probe,split,0
    def __len__(self):return len(self.entries)
    def __getitem__(self,i):
        r=self.entries[i];name=r["id"]
        im=read_rgb(DATA/"images"/(name+".jpg"))
        gt=cv2.imread(str(DATA/"masks"/(name+".png")),cv2.IMREAD_GRAYSCALE)
        seed=int(r["seed"])
        if self.split=="train":
            rng=np.random.default_rng(seed+100000*self.epoch)
            options={"clean":["clean"],"mild":["clean","clean","warm_bright","shadow"],
                     "hard":["clean","warm_bright","shadow","seam","hard_mix"]}[self.level]
            condition=rng.choice(options)
        else:
            condition=self.probe
            rng=np.random.default_rng(seed+971*PROBES.index(condition))
        im,gt=lighting(im,gt,condition,rng)
        return torch.from_numpy(im.transpose(2,0,1).copy()).float()/255,torch.from_numpy(gt.astype(np.int64))

def block(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU(),
                         nn.Conv2d(b,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU())

class RoadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1=block(3,12);self.enc2=block(12,24);self.center=block(24,48)
        self.dec2=block(72,24);self.dec1=block(36,12);self.out=nn.Conv2d(12,3,1)
    def forward(self,x):
        a=self.enc1(x);b=self.enc2(F.max_pool2d(a,2))
        c=self.center(F.max_pool2d(b,2))
        y=self.dec2(torch.cat([b,F.interpolate(c,size=b.shape[2:],mode="nearest")],dim=1))
        return self.out(self.dec1(torch.cat([a,F.interpolate(y,size=a.shape[2:],mode="nearest")],dim=1)))

K3=np.ones((3,3),np.uint8)
TOL=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

def road_edge(labels):
    """Road adjacent to visible non-road; paint and road/occluder silhouettes are excluded."""
    return ((labels==1)&(cv2.dilate((labels==0).astype(np.uint8),K3)>0)).astype(np.uint8)

def boundary_counts(pred,truth,roi):
    valid=roi&(cv2.dilate(((truth==255)|(truth==2)).astype(np.uint8),TOL)==0)
    p=road_edge(pred)*valid;t=road_edge(truth)*valid
    return np.array([np.sum(p*cv2.dilate(t,TOL)),p.sum(),np.sum(t*cv2.dilate(p,TOL)),t.sum()],dtype=float)

def safe_ratio(a,b):return float(a/b) if b else None

@torch.no_grad()
def evaluate(net,loader,device,roi):
    net.eval(); counts=np.zeros(4);conf=np.zeros((3,3),int);seen=0
    for x,labels in loader:
        predicted=net(x.to(device)).argmax(1).cpu().numpy()
        for pred,gt in zip(predicted,labels.numpy()):
            valid=roi&(gt!=255)
            conf+=np.bincount(3*gt[valid]+pred[valid],minlength=9).reshape(3,3)
            counts+=boundary_counts(pred,gt,roi);seen+=roi.sum()
    prec=safe_ratio(counts[0],counts[1]) or 0.
    recall=safe_ratio(counts[2],counts[3]) or 0.
    return dict(boundary_f1_2px=2*prec*recall/max(1e-12,prec+recall),boundary_precision=prec,
                boundary_recall=recall,road_iou=safe_ratio(conf[1,1],conf[1].sum()+conf[:,1].sum()-conf[1,1]),
                occluder_iou=safe_ratio(conf[2,2],conf[2].sum()+conf[:,2].sum()-conf[2,2]),
                scored_pixel_fraction=safe_ratio(conf.sum(),seen))

@torch.no_grad()
def real_eval(net,device,output,roi):
    net.eval();report={}
    annotations=json.loads((DATA/"real_points.json").read_text())
    for name,sets in annotations.items():
        image=read_rgb(DATA/"real"/(name+".png"))
        x=torch.from_numpy(image.transpose(2,0,1).copy()).float().unsqueeze(0).to(device)/255
        cnn=road_edge(net(x).argmax(1)[0].cpu().numpy())
        hsv=cv2.imread(str(DATA/"real"/(name+"_hsv.png")),0)>0
        panels=[];report[name]={}
        for method,edge in [("V135",hsv),("CNN",cnn)]:
            edge=(edge.astype(bool)&roi).astype(np.uint8)
            near=cv2.dilate(edge,TOL)>0
            canvas=image.copy();canvas[edge>0]=[0,250,245]
            scores={}
            for key,curves in sets.items():
                mask=np.zeros(roi.shape,np.uint8)
                for curve in curves:cv2.polylines(mask,[np.array(curve,np.int32)],False,1,1)
                points=(mask>0)&roi
                scores[key+"_near_edge"]=safe_ratio((points&near).sum(),points.sum())
                scores[key+"_pixels"]=int(points.sum())
            report[name][method]=scores
            cv2.putText(canvas,method,(8,20),0,.55,(255,255,255),1,cv2.LINE_AA)
            panels.append(canvas)
        cv2.imwrite(str(output/f"real_{name}.jpg"),cv2.cvtColor(np.hstack(panels),cv2.COLOR_RGB2BGR))
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=("clean","mild","hard"),default="clean")
    parser.add_argument("--epochs",type=int,default=12)
    parser.add_argument("--batch",type=int,default=12)
    parser.add_argument("--final-test",action="store_true")
    args=parser.parse_args()
    if args.epochs<1 or args.batch<1:parser.error("epochs and batch must be positive")
    torch.manual_seed(762);random.seed(762);np.random.seed(762);cv2.setNumThreads(0)
    device="cuda" if torch.cuda.is_available() else "cpu"
    output=ROOT/"results"/args.stage;output.mkdir(parents=True,exist_ok=True)
    roi=cv2.imread(str(DATA/"evaluation_roi.png"),0)>0
    train=Frames("train",args.stage);loader=DataLoader(train,batch_size=args.batch,shuffle=True)
    val=DataLoader(Frames("val"),batch_size=args.batch)
    net=RoadNet().to(device);opt=torch.optim.AdamW(net.parameters(),lr=.001)
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor([1.,1.,4.],device=device),ignore_index=255)
    best=-1;history=[]
    for epoch in range(args.epochs):
        train.epoch=epoch;net.train();total=0
        for x,y in loader:
            opt.zero_grad();loss=loss_fn(net(x.to(device)),y.to(device));loss.backward();opt.step()
            total+=loss.item()*len(x)
        metrics=evaluate(net,val,device,roi)
        history.append(dict(epoch=epoch+1,loss=total/len(train),**metrics))
        print(f"{args.stage} {epoch+1}/{args.epochs} loss={total/len(train):.3f} val edge F1={metrics['boundary_f1_2px']:.3f}",flush=True)
        if metrics["boundary_f1_2px"]>best:
            best=metrics["boundary_f1_2px"];torch.save(net.state_dict(),output/"best.pt")
    net.load_state_dict(torch.load(output/"best.pt",map_location=device,weights_only=True))
    splits=["val","test"] if args.final_test else ["val"]
    paired={split:{probe:evaluate(net,DataLoader(Frames(split,probe=probe),batch_size=args.batch),device,roi)
                   for probe in PROBES} for split in splits}
    report=dict(stage=args.stage,seed=762,epochs=args.epochs,device=device,history=history,
                synthetic=paired,real_sparse=real_eval(net,device,output,roi))
    (output/"metrics.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print("Saved",output)

if __name__=="__main__":main()
