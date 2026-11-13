"""Fixed-budget comparison of low recovery mixing with nominal preservation.

The dual variable penalizes drift from the initial actor on teacher histories.
This is an observation-space trust region, NOT a closed-loop safety guarantee.
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .goal_full_dagger import OUT as SOURCE,REPORT as SOURCE_REPORT,INITIAL,KEYS,ROOT,sha
from .junction_learning import setup,model_at
from .junction_learning_train import read_sequence,batch
from .junction_stage02 import dump

OUT=Path('D:/autodrive_cnn/runs/goal_nominal_constraint_20261113')
REPORT=ROOT/'cnn_finetune/results/goal_nominal_constraint_20261113'
EPS=.01
DUAL_LR=1e-4


def sources():
    teacher=json.loads((SOURCE_REPORT/'replay_data.json').read_text())
    teacher += [r for r in json.loads((SOURCE_REPORT/'teacher_data.json').read_text()) if r['clean_stable']]
    return teacher,json.loads((SOURCE_REPORT/'student_data.json').read_text())


def freeze():
    OUT.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    dump(REPORT/'protocol.json',dict(source_sha256=sha(__file__),initial_sha256=sha(INITIAL),
        hypothesis='Reducing recovery dominance and constraining nominal action drift can avoid entry collapse',
        arms=['low_mix','constrained'],source_protocol_sha256=sha(SOURCE_REPORT/'protocol.json'),
        data='identical 118 teacher and 72 student full episodes; same masks and sample RNG streams',
        base_loss='.9 teacher weighted SmoothL1 + .1 student weighted SmoothL1',
        constraint='mean over episodes of max(all supervised mean wheel squared drift, 3-6s supervised mean wheel squared drift) / .01^2 <= 1',
        reference='fixed initial actor on exactly the teacher observation and previous-command histories',
        constrained_loss='base + lambda*(normalized_drift-1)',
        dual_update='lambda=max(0,lambda+1e-4*(normalized_drift.detach()-1)); initial lambda=0',
        constraint_limit='stochastic training proxy; no per-episode or closed-loop guarantee',
        epsilon=EPS,dual_lr=DUAL_LR,updates=1200,lr=1e-4,min_lr=1e-5,seed=17,
        encoder_frozen=True,actor_inputs_unchanged=True,final_exam_used=False,
        selection='fixed final update 1200 before any driving; no outcome checkpoint selection',
        evaluation='same development matrix 4101/4102 x3 vehicles x4 movements x3 conditions'))
    snap=OUT/'source_snapshot';snap.mkdir()
    for name in ['goal_nominal_constraint.py','goal_full_dagger_train.py','junction_learning.py','junction_learning_train.py']:
        shutil.copy2(ROOT/'cnn_finetune'/name,snap/name)


def imitation(pred,b):
    error=F.smooth_l1_loss(pred,b['label'],beta=.05,reduction='none').mean(-1)
    weight=b['supervise']*(1+2*(abs(b['label'][...,0]-b['label'][...,1])>.15).float())
    return ((error*weight).sum(1)/weight.sum(1).clamp(min=1)).mean()


def nominal_drift(pred,items):
    values=[]
    for i,item in enumerate(items):
        length=len(item['f']);d=(pred[i,:length]-item['anchor'].cuda()).square().mean(-1)
        mask=item['supervise'].cuda();entry=item['entry'].cuda()*mask
        all_mse=(d*mask).sum()/mask.sum().clamp(min=1)
        entry_mse=(d*entry).sum()/entry.sum().clamp(min=1)
        values.append(torch.maximum(all_mse,entry_mse))
    return torch.stack(values).mean()/EPS**2


def train(arm):
    setup();torch.set_grad_enabled(True);torch.manual_seed(17)
    protocol=json.loads((REPORT/'protocol.json').read_text())
    assert sha(__file__)==protocol['source_sha256'];assert sha(INITIAL)==protocol['initial_sha256']
    teacher,student=sources();reference=model_at(INITIAL)
    def pool(rows,anchor=False):
        result={tuple(k):[] for k in KEYS}
        for r in rows:
            data=read_sequence(r['artifact'])
            if anchor:
                b=batch([data])
                with torch.no_grad():pred,_=reference(b['f'],b['aux'])
                data['anchor']=pred[0].cpu()
                with np.load(Path(r['artifact'])/'sequence.npz') as z:t=z['capture']
                data['entry']=torch.from_numpy(((t>=3)&(t<6)).astype(np.float32))
            if data['supervise'].sum()>0:result[tuple(r['key'])].append((r['artifact'],data))
        return result
    common=pool(teacher,True);second=pool(student);del reference
    model=model_at(INITIAL);model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=1200,eta_min=1e-5)
    dst=OUT/'models'/arm;dst.mkdir(parents=True,exist_ok=False)
    torch.save(model.state_dict(),dst/'initial.pt')
    rng=np.random.default_rng(17);other_rng=np.random.default_rng(117)
    dual=0.;history=[];values=[];order=[];start=time.perf_counter()
    for update in range(1,1201):
        a=[common[tuple(k)][int(rng.integers(len(common[tuple(k)])))] for k in KEYS]
        b=[second[tuple(k)][int(other_rng.integers(len(second[tuple(k)])))] for k in KEYS]
        order.append([x[0] for x in a]);ai=[x[1] for x in a];bi=[x[1] for x in b]
        ab=batch(ai);bb=batch(bi);optimizer.zero_grad()
        ap,_=model(ab['f'],ab['aux']);bp,_=model(bb['f'],bb['aux'])
        la=imitation(ap,ab);lb=imitation(bp,bb);drift=nominal_drift(ap,ai)
        loss=.9*la+.1*lb
        if arm=='constrained':loss=loss+dual*(drift-1)
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        assert torch.isfinite(grad);optimizer.step();scheduler.step()
        if arm=='constrained':dual=max(0.,dual+DUAL_LR*(float(drift.detach())-1))
        assert np.isfinite(dual) and dual<100,'dual divergence'
        values.append((la.item(),lb.item(),drift.item(),dual,float(grad)))
        if update%100==0:
            v=np.asarray(values);entry=dict(update=update,teacher_loss=float(v[:,0].mean()),student_loss=float(v[:,1].mean()),
                normalized_drift=float(v[:,2].mean()),dual=float(v[-1,3]),max_grad=float(v[:,4].max()),wall_s=time.perf_counter()-start)
            history.append(entry);values=[];print(arm,json.dumps(entry),flush=True)
        if update%400==0:torch.save(model.state_dict(),dst/f'update_{update}.pt')
    order_hash=hashlib.sha256(json.dumps(order).encode()).hexdigest()
    assert order_hash==json.loads((SOURCE/'models/full_dagger/training.json').read_text())['common_sample_order_sha256']
    # Full nominal pool audit at fixed checkpoint, with full recurrent histories.
    model.eval();audit=[]
    with torch.no_grad():
        for group in common.values():
            for path,item in group:
                b=batch([item]);pred,_=model(b['f'],b['aux'])
                audit.append(dict(artifact=path,normalized_drift=float(nominal_drift(pred,[item])),teacher_loss=float(imitation(pred,b))))
    dump(dst/'training.json',dict(history=history,updates=1200,initial_sha256=sha(INITIAL),source_sha256=sha(__file__),
        common_sample_order_sha256=order_hash,checkpoint=str(dst/'update_1200.pt'),checkpoint_sha256=sha(dst/'update_1200.pt'),
        encoder_frozen=True,training_initializations=1,final_exam_used=False,
        nominal_audit=dict(mean_normalized_drift=float(np.mean([x['normalized_drift'] for x in audit])),
            episodes_exceeding_budget=sum(x['normalized_drift']>1 for x in audit),episodes=len(audit),rows=audit)))


def check():
    # A known .01 wheel shift exhausts the nominal RMS budget. Padding must not
    # contribute, and a localized entry shift must not be hidden by long tails.
    item={'f':torch.zeros(100,1152),'anchor':torch.zeros(100,2),'supervise':torch.ones(100),
          'entry':torch.cat([torch.ones(10),torch.zeros(90)])}
    p=torch.zeros(1,120,2,device='cuda',requires_grad=True)
    with torch.no_grad():p[0,:10]=EPS;p[0,100:]=100
    value=nominal_drift(p,[item]);torch.testing.assert_close(value,torch.ones_like(value))
    value.backward();assert not p.grad[0,100:].any();assert p.grad[0,:10].sum()>0
    assert max(0.,0.+DUAL_LR*(2.-1.))>0
    assert max(0.,0.+DUAL_LR*(0.-1.))==0
    print('entry budget, padding exclusion, dual sign checks passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['freeze','train','check']);p.add_argument('--arm',choices=['low_mix','constrained'])
    a=p.parse_args();train(a.arm) if a.task=='train' else globals()[a.task]()
