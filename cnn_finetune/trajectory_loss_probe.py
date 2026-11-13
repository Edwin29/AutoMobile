"""Actual-CNN backward/update probe on diagnostic rollouts; never exports a policy.

This deliberately is NOT a generalization or driving-improvement experiment.
It verifies that measured future outcomes can change the existing model head.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .model import load_a5b
from .trajectory_loss import trajectory_preference_loss


def probe(preferences,checkpoint,out,steps=12):
    torch.set_num_threads(2);torch.manual_seed(20261112)
    # OpenCV runs FP32; do not let CUDA TF32 hide checkpoint/input mismatches.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    pairs=json.loads(preferences.read_text())
    if not pairs:raise ValueError('No outcome-labelled preference pairs')
    selected=[]
    for case in dict.fromkeys(p['case'] for p in pairs):
        selected.append(max([p for p in pairs if p['case']==case],key=lambda p:p['gap']))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=load_a5b().to(device)
    model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location=device))
    model.eval()  # Keep dropout disabled; gradients still work.
    for param in model.parameters():param.requires_grad_(False)
    head=[m for m in model.net if isinstance(m,torch.nn.Linear)][-1]
    for param in head.parameters():param.requires_grad_(True)
    data=[];reference_diff=0.
    for pair in selected:
        side=[]
        for key in ['winner','loser']:
            with np.load(Path(pair[key])/'segment.npz') as z:
                x=torch.tensor(z['x'],device=device,dtype=torch.float32)/255
                action=torch.tensor(z['actions'],device=device,dtype=torch.float32)[None]
                logged_ref=torch.tensor(z['reference'],device=device,dtype=torch.float32)[None]
            with torch.no_grad():ref=model(x)[None].detach()
            reference_diff=max(reference_diff,(ref-logged_ref).abs().max().item())
            side.append((x,action,ref,torch.ones(1,len(x),device=device)))
        data.append(side)
    if reference_diff>1e-4:raise ValueError(f'Checkpoint differs from rollout policy: {reference_diff}')
    def objective():
        losses=[];margins=[]
        for ((wx,wa,wr,wm),(lx,la,lr,lm)) in data:
            loss,info=trajectory_preference_loss(model(wx)[None],wa,wr,wm,model(lx)[None],la,lr,lm)
            losses.append(loss);margins.append(info['preference_margin'])
        return torch.stack(losses).mean(),torch.stack(margins).mean()
    optimizer=torch.optim.SGD(head.parameters(),lr=1e-4)
    history=[]
    for step in range(steps+1):
        optimizer.zero_grad();loss,margin=objective()
        history.append(dict(step=step,loss=loss.item(),preference_margin=margin.item()))
        if step==steps:break
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
        optimizer.step()
    assert history[-1]['loss']<history[0]['loss']
    result=dict(status='gradient_probe_only_not_driving_validation',steps=steps,selected_pairs=selected,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),reference_max_abs_diff=reference_diff,
        history=history,exported_weights=False,trainable='last linear layer only',device=device)
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preferences',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--out',required=True);a=p.parse_args()
    probe(Path(a.preferences),Path(a.checkpoint),Path(a.out))
