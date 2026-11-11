"""Fine-tune A5b with direct labels; split by seed, stream one shard at a time."""
import argparse,json,random
from pathlib import Path
import cv2,numpy as np,torch
from .model import load_a5b,export_onnx
from .teacher import Teacher

def baseline_direct(y):
    t=Teacher('cw');out=[]
    for label in y:
        p=t.team_pwm(label);mean=float(p.mean());raw=(p[0]-p[1])/13
        k=raw*t.motor.yaw_gain(raw)/max(mean,1e-6)
        out.append([mean/.66,-np.clip(k,-.085,.085)/.095])
    return np.array(out,np.float32)

@torch.no_grad()
def evaluate(model,files,baseline=False):
    sums={k:np.zeros(2) for k in ['all','corner','recovery']};counts=dict.fromkeys(sums,0)
    model.eval()
    for p in files:
        with np.load(p) as z:
            x=z['x'];y=z['y'];masks={'all':np.ones(len(y),bool),'corner':z['corner'],'recovery':z['recovery']}
            pred=[]
            for i in range(0,len(y),64):pred.append(model(torch.from_numpy(x[i:i+64]).cuda().float()/255).cpu().numpy())
            pred=np.concatenate(pred)
            if baseline:pred=baseline_direct(pred)
            error=np.abs(pred-y)
            for k,mask in masks.items():sums[k]+=error[mask].sum(axis=0);counts[k]+=int(mask.sum())
    return {k:dict(count=counts[k],mae=(sums[k]/max(counts[k],1)).tolist(),mean=float(sums[k].sum()/max(2*counts[k],1))) for k in sums}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--epochs',type=int,default=6);ap.add_argument('--lr',type=float,default=3e-5);a=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(20261111);np.random.seed(20261111);random.seed(20261111)
    files=sorted(Path(a.data).glob('seed_*.npz'));validation_seeds={104,109,118,123};test_seeds={106,111,126,129};train=[p for p in files if int(p.stem.split('_')[1]) not in validation_seeds|test_seeds];val=[p for p in files if int(p.stem.split('_')[1]) in validation_seeds];test=[p for p in files if int(p.stem.split('_')[1]) in test_seeds]
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    model=load_a5b().cuda();base=evaluate(model,val,True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4);history=[];best=float('inf')
    for epoch in range(a.epochs):
        model.train();random.shuffle(train);losses=[]
        for p in train:
            with np.load(p) as z:x=z['x'];y=z['y']
            order=np.random.permutation(len(y))
            for start in range(0,len(y),64):
                ids=order[start:start+64];xb=torch.from_numpy(x[ids]).cuda().float()/255;yb=torch.from_numpy(y[ids]).cuda()
                flip=torch.rand(len(ids),device='cuda')<.5;xb[flip]=xb[flip].flip(-1);yb[flip,1]*=-1
                xb[:,0]*=torch.empty(len(ids),1,1,device='cuda').uniform_(.85,1.15);xb.clamp_(0,1)
                pred=model(xb);loss=torch.nn.functional.smooth_l1_loss(pred,yb,beta=.1)
                optimizer.zero_grad();loss.backward();optimizer.step();losses.append(loss.item())
        metrics=evaluate(model,val);row=dict(epoch=epoch+1,loss=float(np.mean(losses)),validation=metrics);history.append(row);print(json.dumps(row),flush=True)
        if metrics['all']['mean']<best:best=metrics['all']['mean'];torch.save(model.cpu().state_dict(),out/'best.pt');model.cuda()
        (out/'metrics.json').write_text(json.dumps(dict(baseline_validation=base,history=history),indent=2))
    model.load_state_dict(torch.load(out/'best.pt',weights_only=True));final=evaluate(model,val);test_result=evaluate(model,test)
    export_onnx(model,out/'policy.onnx')
    with np.load(val[0]) as z:sample=z['x'][:100].astype(np.float32)/255
    net=cv2.dnn.readNetFromONNX(str(out/'policy.onnx'));diff=0.
    for x in sample:
        net.setInput(x[None]);reference=net.forward();pred=model(torch.from_numpy(x[None])).detach().numpy();diff=max(diff,float(np.max(np.abs(reference-pred))))
    result=dict(baseline_validation=base,validation=final,test=test_result,history=history,onnx_max_abs=diff,split={k:[p.stem for p in fs] for k,fs in [('train',train),('validation',val),('test',test)]})
    (out/'metrics.json').write_text(json.dumps(result,indent=2));assert diff<1e-4
