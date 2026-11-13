"""No-update audit: are teacher, branch imitation and preference gradients aligned?"""
import json
from pathlib import Path
import numpy as np
import torch
from .goal_development import REPORT, OLD_OUT
from .junction_learning import ROOT, model_at, setup
from .junction_learning_train import read_sequence, bc_loss, preference
from .junction_stage02 import dump


def main():
    setup();torch.set_grad_enabled(True)
    old=ROOT/'cnn_finetune/results/junction_learning_20261113'
    teachers=[read_sequence(r['artifact']) for r in json.loads((old/'teacher_data.json').read_text()) if r['stable_success']]
    branches=[read_sequence(r['artifact']) for r in json.loads((old/'branch_data.json').read_text())]
    pairs=[]
    for r in json.loads((old/'preferences.json').read_text()):
        pair=[read_sequence(r[k]) for k in ['winner','loser']]
        for x in pair:x['supervise'].zero_();x['supervise'][r['prefix']:]=1
        pairs.append(pair)
    reference=model_at(OLD_OUT/'warmstart.pt')
    for p in reference.parameters():p.requires_grad_(False)
    def weight(x):return float((x['supervise']*(1+2*(abs(x['label'][:,0]-x['label'][:,1])>.15).float())).sum())
    result={}
    for arm,checkpoint in [('warm',OLD_OUT/'warmstart.pt'),('bc4',OLD_OUT/'models/bc/epoch_4.pt'),('outcome8',OLD_OUT/'models/outcome/epoch_8.pt')]:
        model=model_at(checkpoint);model.train();params=list(model.parameters());vectors={};losses={}
        for group,items in [('teacher',teachers),('branches',branches)]:
            total=sum(map(weight,items));grad=torch.zeros(sum(p.numel() for p in params),device='cuda');avg=0.
            for i in range(0,len(items),4):
                batch=items[i:i+4];w=sum(map(weight,batch))/total;loss=bc_loss(model,batch)
                g=torch.autograd.grad(loss,params);grad+=w*torch.cat([v.reshape(-1) for v in g]);avg+=w*float(loss.detach())
            vectors[group]=grad;losses[group]=avg
        grad=torch.zeros_like(vectors['teacher']);avg=0.
        for pair in pairs:
            loss,_=preference(model,reference,pair);g=torch.autograd.grad(.1*loss,params)
            grad+=torch.cat([v.reshape(-1) for v in g])/len(pairs);avg+=float(.1*loss.detach())/len(pairs)
        vectors['weighted_preference']=grad;losses['weighted_preference']=avg
        result[arm]=dict(losses=losses,norms={k:float(v.norm()) for k,v in vectors.items()},
            cosine={a+'__'+b:float(torch.dot(vectors[a],vectors[b])/(vectors[a].norm()*vectors[b].norm()).clamp(min=1e-12))
                for a,b in [('teacher','branches'),('teacher','weighted_preference'),('branches','weighted_preference')]})
    dump(REPORT/'gradient_audit.json',dict(results=result,teacher_episodes=len(teachers),branch_sequences=len(branches),preference_pairs=len(pairs),
         aggregation='Frame/turn-weighted BC within each data source; equal pair preference mean with actual 0.1 mix. Pre-clipping gradients; not full minibatch trajectory.',
         no_updates=True,interpretation='Negative cosine is local objective conflict at this checkpoint; not proof of causal generalization failure.'))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
