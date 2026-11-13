"""Reproducible all28 recurrent-policy baseline on clean recovery demonstrations.

This is supervised behavior cloning, not a claim of improved outcome loss.
Fixed-final checkpoints are assessed by actual closed-loop driving afterwards.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch.nn import functional as F
from .goal_all_movement_data import OUT,REPORT,CONDITIONS
from .goal_teacher_coverage import KEYS,ROOT
from .junction_learning import TemporalPolicy,setup,CHECKPOINT
from .goal_development import sha
from .junction_stage02 import dump

TRAINING_SEEDS=[17,29,43]
UPDATES=2400


def training_eligible(row):
    # The collector's clean flag measures safety. Avoid teaching its occasional
    # unnecessary translation pauses as the desired smooth-turn behavior.
    return bool(row['eligible'] and row['translation_pause_s']==0)


def read_data(row):
    assert training_eligible(row)
    with np.load(Path(row['artifact'])/'sequence.npz') as z:
        assert z['label_valid'].all() and z['supervise'].all()
        return {k:torch.from_numpy(z[k].astype(np.float32)) for k in ['f','aux','label','supervise']}


def padded(items):
    length=max(len(r['f']) for r in items)
    out={k:torch.zeros((len(items),length,*items[0][k].shape[1:]),device='cuda') for k in items[0]}
    for i,r in enumerate(items):
        for k in out:out[k][i,:len(r[k])]=r[k].cuda()
    return out


def loss(model,items):
    b=padded(items);prediction,_=model(b['f'],b['aux'])
    error=F.smooth_l1_loss(prediction,b['label'],beta=.05,reduction='none').mean(-1)
    weight=b['supervise']*(1+2*(abs(b['label'][...,0]-b['label'][...,1])>.15).float())
    assert torch.all(weight.sum(1)>0)
    return ((error*weight).sum(1)/weight.sum(1)).mean()


def check_loss():
    setup();torch.manual_seed(913);model=TemporalPolicy().cuda().train();rng=np.random.default_rng(913)
    one=dict(f=torch.tensor(rng.normal(size=(5,1152)),dtype=torch.float32),aux=torch.zeros((5,4)),
        label=torch.tensor(rng.uniform(-1,1,(5,2)),dtype=torch.float32),supervise=torch.ones(5))
    two={k:torch.cat([v,torch.zeros((8,*v.shape[1:]))],dim=0) for k,v in one.items()}
    a=loss(model,[one]);b=loss(model,[two]);torch.testing.assert_close(a,b,atol=1e-6,rtol=0)
    # Longer padding must also leave each real frame's gradient unchanged.
    model.zero_grad();a.backward();ga=[p.grad.clone() for p in model.parameters()]
    model.zero_grad();b.backward()
    for g,p in zip(ga,model.parameters()):torch.testing.assert_close(g,p.grad,atol=2e-6,rtol=1e-5)
    return dict(padding_loss_invariant=True,padding_gradient_invariant=True)


def freeze():
    checked=check_loss();data=json.loads((REPORT/'data.json').read_text());assert len(data['rows'])==504
    eligible=[r for r in data['rows'] if training_eligible(r)]
    assert all(any(tuple(r['key'])==k for r in eligible) for k in KEYS)
    path=REPORT/'training_protocol.json';assert not path.exists()
    dump(path,dict(source_sha256=sha(__file__),data_manifest_sha256=sha(REPORT/'data.json'),encoder_sha256=sha(CHECKPOINT),
        seeds=TRAINING_SEEDS,updates=UPDATES,model='unchanged frozen CNN -> LayerNorm/128 projection -> GRU64 -> signedPWM2 head',
        initial='independently randomized recurrent/project/output weights; common pretrained frozen CNN',
        optimizer='AdamW lr5e-4 weight_decay1e-4, cosine to5e-5, gradient clip1',
        batch='7 entire episodes, all28 movements visited once per4 updates; movement order and condition/episode samples use common sampler seed221',
        objective='equal-episode turn-weighted SmoothL1 beta.05; only whole-episode continuous-footprint-clean demonstrations',
        selection='fixed final update2400, no driving-based checkpoint selection; save600/1200/1800 for diagnosis only',
        evaluation='new6901/6902 local visual development; first nominal all28 screen before broader evaluation; no final9xxxxx use',
        stages='first seed17 prototype; remaining independent seeds29/43 remain required for goal completion and can follow after diagnosis of prototype',
        eligible_episodes=len(eligible),failed_episodes_retained=len(data['rows'])-len(eligible),checks=checked,
        training_eligibility='collector clean stable episode and zero translation_pause_s (no >=.3s interval with abs(v)<.5cm/s after first1s); collector flags remain unchanged',
        clean_but_pause_excluded=sum(r['eligible'] and not training_eligible(r) for r in data['rows']),
        no_outcome_loss_claim=True,final_exam_used=False))
    snap=OUT/'training_source_snapshot.py';shutil.copy2(__file__,snap);print('training protocol frozen',len(eligible),flush=True)


def train(seed):
    setup();torch.set_grad_enabled(True);p=json.loads((REPORT/'training_protocol.json').read_text())
    assert seed in p['seeds'] and sha(__file__)==p['source_sha256']
    assert sha(REPORT/'data.json')==p['data_manifest_sha256'] and sha(CHECKPOINT)==p['encoder_sha256']
    rows=[r for r in json.loads((REPORT/'data.json').read_text())['rows'] if training_eligible(r)]
    pools={k:{c:[] for c in CONDITIONS} for k in KEYS}
    for r in rows:
        assert sha(Path(r['artifact'])/'sequence.npz')==r['sequence_sha256']
        pools[tuple(r['key'])][r['condition']].append((r['artifact'],read_data(r)))
    dst=OUT/'models'/f'seed_{seed}';dst.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);model=TemporalPolicy().cuda();model.train();torch.save(model.state_dict(),dst/'initial.pt')
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=UPDATES,eta_min=5e-5)
    rng=np.random.default_rng(221);history=[];window=[];order=[];started=time.perf_counter();key_order=[]
    for update in range(1,UPDATES+1):
        if (update-1)%4==0:key_order=[KEYS[i] for i in rng.permutation(len(KEYS))]
        keys=key_order[((update-1)%4)*7:((update-1)%4+1)*7];items=[];paths=[]
        for key in keys:
            conditions=[c for c in CONDITIONS if pools[key][c]];condition=conditions[int(rng.integers(len(conditions)))]
            path,item=pools[key][condition][int(rng.integers(len(pools[key][condition])))];paths.append(path);items.append(item)
        order.append(paths);optimizer.zero_grad();value=loss(model,items);value.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(value) and torch.isfinite(grad)
        optimizer.step();scheduler.step();window.append((float(value.detach()),float(grad)))
        if update%100==0:
            row=dict(update=update,loss=float(np.mean([x[0] for x in window])),max_gradient=float(max(x[1] for x in window)),wall_s=time.perf_counter()-started)
            history.append(row);window=[];dump(dst/'progress.json',history);print('all28 training',seed,json.dumps(row),flush=True)
        if update%600==0:torch.save(model.state_dict(),dst/f'update_{update}.pt')
    dump(dst/'training.json',dict(seed=seed,updates=UPDATES,source_sha256=sha(__file__),protocol_sha256=sha(REPORT/'training_protocol.json'),
        initial_sha256=sha(dst/'initial.pt'),final_checkpoint=str(dst/f'update_{UPDATES}.pt'),final_sha256=sha(dst/f'update_{UPDATES}.pt'),
        sampler_order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),history=history,eligible_episodes=len(rows),
        trainable_parameters=sum(p.numel() for p in model.parameters()),encoder_frozen=True,device=torch.cuda.get_device_name(0),final_exam_used=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['check','freeze','train']);p.add_argument('--seed',type=int);a=p.parse_args()
    if a.task=='train':train(a.seed)
    elif a.task=='check':print(json.dumps(check_loss()),flush=True)
    else:freeze()
