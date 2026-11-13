"""Instruction intervention on identical pre-turn teacher observations/history.

This measures use of the direction command, not correctness of a counterfactual
trajectory. Only pre-turn entry observations are summarized.
"""
import json
from pathlib import Path
import numpy as np
import torch
from bench.junction import movement
from .goal_development import REPORT,OLD_OUT
from .junction_learning import model_at,setup
from .junction_stage02 import dump


def main():
    setup();rows=json.loads((REPORT/'teacher.json').read_text())
    models={a:model_at(p) for a,p in [('warm',OLD_OUT/'warmstart.pt'),('bc',OLD_OUT/'models/bc/epoch_4.pt'),('outcome',OLD_OUT/'models/outcome/epoch_8.pt')]}
    results=[]
    for r in rows:
        if r['key'][0]!='east':continue
        with np.load(Path(r['artifact'])/'sequence.npz') as z:
            f=torch.from_numpy(z['f'][None]).cuda();aux=torch.from_numpy(z['aux'][None]).cuda();times=z['capture']
        trace=np.load(Path(r['artifact'])/'physics.npz')['samples'];c,di,do,_=movement(*r['key'])
        xy=np.c_[np.interp(times,trace[:,0],trace[:,1]),np.interp(times,trace[:,0],trace[:,2])]
        progress=(xy-c)@di;mask=(progress>=-30)&(progress<=-20)
        left=aux.clone();left[:,:,2]=1;left[:,:,3]=0
        right=aux.clone();right[:,:,2]=0;right[:,:,3]=1
        for arm,model in models.items():
            with torch.no_grad():l,_=model(f,left);rr,_=model(f,right)
            l=l[0].cpu().numpy();rr=rr[0].cpu().numpy()
            # Positive delta is the expected command ordering: right request
            # has more positive clockwise yaw command than left request.
            delta=(rr[:,0]-rr[:,1])-(l[:,0]-l[:,1])
            results.append(dict(arm=arm,key=r['key'],seed=r['seed'],preset=r['preset'],condition=r['condition'],
                frames=int(mask.sum()),mean_right_minus_left=float(delta[mask].mean()),
                max_abs_difference=float(abs(delta[mask]).max()),ordered_fraction=float((delta[mask]>0).mean())))
    summary={a:dict(mean_right_minus_left=float(np.mean([r['mean_right_minus_left'] for r in results if r['arm']==a])),
                   mean_ordered_fraction=float(np.mean([r['ordered_fraction'] for r in results if r['arm']==a]))) for a in models}
    dump(REPORT/'direction_probe.json',dict(summary=summary,episodes=results,scope='Teacher-forced approach 20-30cm before center. Same images and prior PWM; only instruction changed from episode start. Not counterfactual driving success.'))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
