"""Equal-budget teacher-only versus full-student-state supervised learning."""
import argparse
import hashlib
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from .goal_full_dagger import OUT,REPORT,INITIAL,KEYS,sha
from .junction_learning import model_at,setup
from .junction_learning_train import read_sequence,batch
from .junction_stage02 import dump


def loss_for(model,items):
    b=batch(items);prediction,_=model(b['f'],b['aux'])
    error=F.smooth_l1_loss(prediction,b['label'],beta=.05,reduction='none').mean(-1)
    weight=b['supervise']*(1+2*(abs(b['label'][...,0]-b['label'][...,1])>.15).float())
    assert torch.all(weight.sum(1)>0)
    return ((error*weight).sum(1)/weight.sum(1)).mean()


def main(arm):
    setup();torch.set_grad_enabled(True);torch.manual_seed(17)
    protocol=json.loads((REPORT/'protocol.json').read_text());assert sha(INITIAL)==protocol['initial_sha256']
    trows=json.loads((REPORT/'replay_data.json').read_text())+[r for r in json.loads((REPORT/'teacher_data.json').read_text()) if r['clean_stable']]
    srows=trows if arm=='teacher_only' else json.loads((REPORT/'student_data.json').read_text())
    def pool(rows):
        result={tuple(k):[] for k in KEYS}
        for r in rows:
            data=read_sequence(r['artifact'])
            if data['supervise'].sum()>0:result[tuple(r['key'])].append((r['artifact'],data))
        assert all(result.values());return result
    common=pool(trows);second=pool(srows)
    dst=OUT/'models'/arm;dst.mkdir(parents=True,exist_ok=False)
    dump(dst/'protocol.json',dict(source_sha256=sha(__file__),shared_protocol_sha256=sha(REPORT/'protocol.json'),
        initial_sha256=sha(INITIAL),arm=arm,common_episodes=len(trows),second_episodes=len(srows),
        common_sources=[r['artifact'] for r in trows],second_sources=[r['artifact'] for r in srows]))
    model=model_at(INITIAL);model.train();torch.save(model.state_dict(),dst/'initial.pt')
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=1200,eta_min=1e-5)
    rng=np.random.default_rng(17);other_rng=np.random.default_rng(117);history=[];common_order=[];values=[];started=time.perf_counter()
    for update in range(1,1201):
        one=[common[tuple(k)][int(rng.integers(len(common[tuple(k)])))] for k in KEYS]
        two=[second[tuple(k)][int(other_rng.integers(len(second[tuple(k)])))] for k in KEYS]
        common_order.append([x[0] for x in one]);optimizer.zero_grad()
        a=loss_for(model,[x[1] for x in one]);b=loss_for(model,[x[1] for x in two]);loss=.5*(a+b)
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(grad)
        optimizer.step();scheduler.step();values.append((float(a.detach()),float(b.detach()),float(grad)))
        if update%100==0:
            v=np.asarray(values);row=dict(update=update,common_loss=float(v[:,0].mean()),second_loss=float(v[:,1].mean()),max_grad=float(v[:,2].max()),wall_s=time.perf_counter()-started)
            history.append(row);values=[];print(arm,json.dumps(row),flush=True)
        if update%400==0:torch.save(model.state_dict(),dst/f'update_{update}.pt')
    dump(dst/'training.json',dict(history=history,updates=1200,trainable_parameters=sum(p.numel() for p in model.parameters()),
        source_sha256=sha(__file__),initial_sha256=sha(INITIAL),common_sample_order_sha256=hashlib.sha256(json.dumps(common_order).encode()).hexdigest(),
        final_checkpoint=str(dst/'update_1200.pt'),final_sha256=sha(dst/'update_1200.pt'),encoder_frozen=True,training_initializations=1,
        common_episodes=len(trows),second_episodes=len(srows),no_final_exam=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=['teacher_only','full_dagger']);a=p.parse_args();main(a.arm)
