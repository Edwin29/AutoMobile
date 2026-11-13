"""Bounded optimization diagnostic, not a new architecture or final model.

The common warmstart's training loss was still decreasing when stopped.
Continue the SAME 94 teacher sequences at lower LR, no extra branch/preference
data. Select on new development teacher-observation error, then test driving.
"""
import hashlib
import json
import time
import numpy as np
import torch

from .goal_development import ROOT,OUT,REPORT,OLD_OUT,sha
from .junction_learning import model_at,setup
from .junction_learning_train import read_sequence,bc_loss,batch
from .junction_stage02 import dump


@torch.no_grad()
def score(model,items):
    model.eval();values=[]
    for row in items:
        b=batch([row]);y,_=model(b['f'],b['aux']);mask=(row['margin']>=0).cuda()
        values.append(float((y[0]-b['label'][0]).abs()[mask].mean()))
    model.train();return float(np.mean(values))


def main():
    setup();torch.set_grad_enabled(True);torch.manual_seed(17)
    dst=OUT/'convergence_probe';dst.mkdir(exist_ok=False)
    old=ROOT/'cnn_finetune/results/junction_learning_20261113'
    teacher=[r for r in json.loads((old/'teacher_data.json').read_text()) if r['stable_success']]
    development=json.loads((REPORT/'teacher.json').read_text())
    dump(REPORT/'convergence_protocol.json',dict(hypothesis='Insufficient supervised optimization contributes to poor control; original warm loss had not plateaued',
        initial_checkpoint_sha256=sha(OLD_OUT/'warmstart.pt'),teacher_sequences=len(teacher),new_training_data=False,
        epochs=60,lr=1e-4,min_lr=1e-5,batch_size=4,seed=17,optimizer='AdamW reset; original optimizer state unavailable',
        encoder_frozen=True,architecture_changed=False,check_every=10,
        selection='lowest mean per-episode on-road wheel MAE on development teacher replay; freeze before closed-loop comparison',
        final_exam_used=False,source_sha256=sha(__file__)))
    train=[read_sequence(r['artifact']) for r in teacher];dev=[read_sequence(r['artifact']) for r in development]
    model=model_at(OLD_OUT/'warmstart.pt');model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=60,eta_min=1e-5)
    rng=np.random.default_rng(17);history=[];best=score(model,dev);selected=0
    torch.save(model.state_dict(),dst/'best.pt');history.append(dict(epoch=0,development_mae=best))
    started=time.perf_counter()
    for epoch in range(1,61):
        order=rng.permutation(len(train));losses=[]
        for i in range(0,len(order),4):
            optimizer.zero_grad();loss=bc_loss(model,[train[j] for j in order[i:i+4]]);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert torch.isfinite(norm);optimizer.step();losses.append(float(loss.detach()))
        schedule.step();entry=dict(epoch=epoch,train_loss=float(np.mean(losses)),lr=optimizer.param_groups[0]['lr'])
        if epoch%10==0:
            value=score(model,dev);entry['development_mae']=value
            torch.save(model.state_dict(),dst/f'epoch_{epoch}.pt')
            if value<best:best=value;selected=epoch;torch.save(model.state_dict(),dst/'best.pt')
            print('convergence',epoch,'train',entry['train_loss'],'dev',value,flush=True)
        history.append(entry)
    dump(REPORT/'convergence_training.json',dict(history=history,selected_epoch=selected,selected_checkpoint=str(dst/'best.pt'),
        selected_sha256=sha(dst/'best.pt'),wall_s=time.perf_counter()-started,updates=60*((len(train)+3)//4),
        frozen_before_closed_loop=True,scope='optimization diagnostic; not proof that supervision or fixed encoder is sufficient for driving'))


if __name__=='__main__':main()
