"""Matched recurrent-head training: BC versus BC + anchored trajectory preference."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .junction_learning import OUT,REPORT,TemporalPolicy,setup
from .junction_stage02 import dump
from .trajectory_loss import trajectory_preference_loss


def read_sequence(path):
    with np.load(Path(path)/'sequence.npz') as z:
        data={k:torch.from_numpy(z[k].astype(np.float32)) for k in ['f','aux','label','action','reference','supervise','margin']}
    # Do not imitate severely off-road teacher queries as reliable demonstrations.
    data['supervise']*= (data['margin']>=-3).float()
    return data


def batch(items):
    length=max(len(x['f']) for x in items);b=len(items)
    data={k:torch.zeros((b,length,*items[0][k].shape[1:]),device='cuda') for k in ['f','aux','label','action','reference','supervise']}
    for i,row in enumerate(items):
        for k in data:data[k][i,:len(row[k])]=row[k].cuda()
    return data


def bc_loss(model,items):
    b=batch(items);pred,_=model(b['f'],b['aux']);error=F.smooth_l1_loss(pred,b['label'],beta=.05,reduction='none').mean(-1)
    weight=b['supervise']*(1+2*(abs(b['label'][...,0]-b['label'][...,1])>.15).float())
    return (error*weight).sum()/weight.sum().clamp(min=1)


def preference(model,reference,pair):
    w,l=pair
    wb=batch([w]);lb=batch([l]);wp,_=model(wb['f'],wb['aux']);lp,_=model(lb['f'],lb['aux'])
    with torch.no_grad():wr,_=reference(wb['f'],wb['aux']);lr,_=reference(lb['f'],lb['aux'])
    # Mask is intervention only, but recurrent state is reconstructed from the full prefix.
    wm=wb['supervise'];lm=lb['supervise']
    return trajectory_preference_loss(wp,wb['action'],wr,wm,lp,lb['action'],lr,lm,
                                       sigma=(.15,.15),anchor_weight=.1)


def train(arm):
    setup();torch.set_grad_enabled(True);torch.manual_seed(17)
    model=TemporalPolicy().cuda();warm=arm=='warm'
    if not warm:model.load_state_dict(torch.load(OUT/'warmstart.pt',weights_only=True,map_location='cuda'))
    reference=TemporalPolicy().cuda();reference.load_state_dict(model.state_dict());reference.eval()
    for p in reference.parameters():p.requires_grad_(False)
    rows=json.loads((REPORT/'teacher_data.json').read_text())
    teacher=[r for r in rows if r['stable_success']]
    data=[read_sequence(r['artifact']) for r in teacher]
    pairs=[]
    if not warm:
        branches=json.loads((REPORT/'branch_data.json').read_text())
        # Both arms see every branch's intervention-state oracle queries. The
        # same margin mask filters extreme out-of-road labels in both arms.
        data += [read_sequence(r['artifact']) for r in branches]
        # All arms have the same observed branch frames and oracle labels above.
        for p in json.loads((REPORT/'preferences.json').read_text()):
            w=read_sequence(p['winner']);l=read_sequence(p['loser'])
            # Outcome preferences remain meaningful even when a branch crossed a boundary.
            for row in [w,l]:
                row['supervise'].zero_();row['supervise'][p['prefix']:]=1
            pairs.append((w,l))
        if not pairs:raise RuntimeError('No usable outcome preferences; do not claim outcome training')
    epochs=12 if warm else 8;optimizer=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0001)
    rng=np.random.default_rng(17);pair_rng=np.random.default_rng(117);history=[];out=OUT/'models'/arm;out.mkdir(parents=True,exist_ok=False)
    torch.save(model.state_dict(),out/'initial.pt')
    initial_hash=hashlib.sha256((out/'initial.pt').read_bytes()).hexdigest()
    started=time.perf_counter();updates=0
    for epoch in range(1,epochs+1):
        order=rng.permutation(len(data));losses=[];bcs=[];prefs=[];grads=[]
        model.train()
        for i in range(0,len(order),4):
            optimizer.zero_grad();base=bc_loss(model,[data[j] for j in order[i:i+4]])
            loss=base
            if arm=='outcome':
                p_loss,info=preference(model,reference,pairs[int(pair_rng.integers(len(pairs)))])
                loss=base+.1*p_loss;prefs.append(float(p_loss.detach()))
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(grad):raise RuntimeError('Non-finite gradient')
            optimizer.step();updates+=1;losses.append(float(loss.detach()));bcs.append(float(base.detach()));grads.append(float(grad))
        row=dict(epoch=epoch,loss=float(np.mean(losses)),bc=float(np.mean(bcs)),preference=float(np.mean(prefs)) if prefs else None,
                 max_grad=float(max(grads)),updates=updates,wall_s=time.perf_counter()-started)
        history.append(row);print(arm,json.dumps(row),flush=True)
        if epoch in [4,8,12]:torch.save(model.state_dict(),out/f'epoch_{epoch}.pt')
        dump(out/'training.json',dict(arm=arm,initial_sha256=initial_hash,history=history,teacher_episodes=len(teacher),
             all_teacher_episodes=len(rows),train_sequences=len(data),preference_pairs=len(pairs),trainable_parameters=sum(p.numel() for p in model.parameters()),
             encoder_frozen=True,seed=17,batch_size=4,lr=.0005,updates=updates))
    if warm:torch.save(model.state_dict(),OUT/'warmstart.pt')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('arm',choices=['warm','bc','outcome']);train(parser.parse_args().arm)
