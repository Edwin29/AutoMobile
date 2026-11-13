"""Teacher-forced fitting diagnosis, explicitly separate from driving success."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .goal_all_movement_data import REPORT as DATA_REPORT
from .goal_all_movement_train import training_eligible
from .goal_all_movement_eval import MODELS,REPORT
from .junction_learning import model_at,setup
from .goal_development import sha
from .junction_stage02 import dump


@torch.no_grad()
def run(arm):
    setup();data=DATA_REPORT/'data.json';rows=json.loads(data.read_text())['rows'];assert len(rows)==504
    selected=[r for r in rows if training_eligible(r)];model=model_at(MODELS[arm]);out=[]
    for r in selected:
        file=Path(r['artifact'])/'sequence.npz';assert sha(file)==r['sequence_sha256']
        with np.load(file) as z:
            f=torch.from_numpy(z['f'].astype(np.float32))[None].cuda();aux=torch.from_numpy(z['aux'].astype(np.float32))[None].cuda()
            label=torch.from_numpy(z['label'].astype(np.float32)).cuda()
            prediction,_=model(f,aux);prediction=prediction[0]
            turn=(abs(label[:,0]-label[:,1])>.15);weight=1+2*turn.float()
            element=F.smooth_l1_loss(prediction,label,beta=.05,reduction='none').mean(-1)
            mae=(prediction-label).abs().mean(-1).cpu().numpy();t=z['capture']
            sections={name:dict(frames=int(mask.sum()),mae=float(mae[mask].mean()) if mask.any() else None)
                for name,mask in [('early_3s',t<3),('turn',turn.cpu().numpy()),('nonturn',~turn.cpu().numpy())]}
            out.append(dict(key=r['key'],condition=r['condition'],seed=r['seed'],preset=r['preset'],artifact=r['artifact'],
                frames=len(mae),mae=float(mae.mean()),weighted_huber=float((element*weight).sum()/weight.sum()),sections=sections))
    by={}
    for condition in ['normal','mild','mild_light']:
        a=[r for r in out if r['condition']==condition]
        if a:by[condition]=dict(episodes=len(a),episode_macro_mae=float(np.mean([r['mae'] for r in a])),episode_macro_weighted_huber=float(np.mean([r['weighted_huber'] for r in a])))
    target=REPORT/f'{arm}_fit_audit.json';assert not target.exists()
    dump(target,dict(source_sha256=sha(__file__),eligibility_source_sha256=sha(Path(__file__).with_name('goal_all_movement_train.py')),
        model_sha256=sha(MODELS[arm]),data_sha256=sha(data),conditions=by,rows=out,
        scope='Training-demonstration teacher-forced inputs, including expert previousPWM; not student-distribution MAE, independent validation or closed-loop driving success.',
        purpose='Distinguish failure to fit demonstrations from closed-loop drift despite small demonstration error; fixed final checkpoint only, no selection by this audit.',final_exam_used=False))
    print(json.dumps(by),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=list(MODELS),required=True);run(p.parse_args().arm)
