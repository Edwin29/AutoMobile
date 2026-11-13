"""Teacher-forced audit of common warmstart and both comparison checkpoints."""
import json
from pathlib import Path
import numpy as np
import torch
from .goal_development import OUT, REPORT, OLD_OUT
from .junction_learning import ROOT, model_at, setup
from .junction_stage02 import dump


def main():
    setup()
    models={'warm':OLD_OUT/'warmstart.pt','bc4':OLD_OUT/'models/bc/epoch_4.pt','bc8':OLD_OUT/'models/bc/epoch_8.pt',
            'outcome4':OLD_OUT/'models/outcome/epoch_4.pt','outcome8':OLD_OUT/'models/outcome/epoch_8.pt'}
    models={k:model_at(p) for k,p in models.items()}
    splits={'original_teacher_train':json.loads((ROOT/'cnn_finetune/results/junction_learning_20261113/teacher_data.json').read_text()),
            'new_development_teacher':json.loads((REPORT/'teacher.json').read_text())}
    output={}
    for split,rows in splits.items():
        records=[]
        for r in rows:
            if not r['stable_success']:continue
            with np.load(Path(r['artifact'])/'sequence.npz') as z:
                f=torch.from_numpy(z['f'][None]).cuda();aux=torch.from_numpy(z['aux'][None]).cuda();label=z['label'];mask=z['margin']>=0
            for arm,model in models.items():
                with torch.no_grad():y,_=model(f,aux)
                y=y[0].cpu().numpy();err=y-label
                records.append(dict(arm=arm,key=r['key'],condition=r['condition'],seed=r['seed'],preset=r['preset'],
                    wheel_mae=float(abs(err[mask]).mean()),diff_mae=float(abs(err[mask,0]-err[mask,1]).mean()),
                    mean_bias=float(err[mask].mean()),diff_bias=float((err[mask,0]-err[mask,1]).mean())))
        output[split]={'episodes':records,'aggregate':{arm:{condition:{k:float(np.mean([r[k] for r in records if r['arm']==arm and r['condition']==condition]))
            for k in ['wheel_mae','diff_mae','mean_bias','diff_bias']} for condition in sorted(set(r['condition'] for r in records))} for arm in models}}
    output['interpretation']='Teacher forcing, equal weight per episode, on-road labels only. Training episodes are not held-out. No new optimization performed.'
    dump(REPORT/'optimization_audit.json',output)
    print(json.dumps({k:v['aggregate'] for k,v in output.items() if isinstance(v,dict)}),flush=True)


if __name__=='__main__':main()
