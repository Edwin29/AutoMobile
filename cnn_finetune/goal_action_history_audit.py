"""Can image-free causal action-history baselines obtain low demonstration MAE?

This tests a shortcut available to the loss, not whether a network uses it.
No baseline here is a deployed controller or a neural checkpoint selection rule.
"""
import json
from pathlib import Path
import numpy as np
from .goal_all_movement_data import REPORT as DATA_REPORT
from .goal_all_movement_train import training_eligible
from .goal_development import ROOT,sha
from .junction_stage02 import dump

REPORT=ROOT/'cnn_finetune/results/goal_action_history_loss_audit_20261113'
WINDOWS=[1,3,7,14,28]


def run():
    data=DATA_REPORT/'data.json';rows=[r for r in json.loads(data.read_text())['rows'] if training_eligible(r)];result=[]
    REPORT.mkdir(exist_ok=False)
    dump(REPORT/'protocol.json',dict(source_sha256=sha(__file__),data_sha256=sha(data),windows=WINDOWS,
        rule='Untrained causal moving average of previous actually sentPWM, excluding current target; no image, pose, current action, or future access',
        interpretation='Low error establishes that a shortcut is available in demonstrations, not that the learned policy actually exploits it. No driving or causal reward conclusion.',
        references=['https://proceedings.neurips.cc/paper_files/paper/2020/hash/1b113258af3968aaf3969ca67e744ff8-Abstract.html',
                    'https://proceedings.mlr.press/v139/wen21d/wen21d.pdf'],final_exam_used=False))
    for r in rows:
        file=Path(r['artifact'])/'sequence.npz';assert sha(file)==r['sequence_sha256']
        with np.load(file) as z:previous=z['aux'][:,:2].astype(np.float64);target=z['label'].astype(np.float64);time=z['capture']
        accum=np.vstack([np.zeros((1,2)),np.cumsum(previous,axis=0)]);end=np.arange(1,len(previous)+1);turn=abs(target[:,0]-target[:,1])>.15
        scores={}
        for n in WINDOWS:
            start=np.maximum(0,end-n);prediction=(accum[end]-accum[start])/(end-start)[:,None]
            absolute=abs(prediction-target);error=absolute.mean(-1);weight=1+2*turn
            huber=np.where(absolute<.05,.5*absolute**2/.05,absolute-.025).mean(-1)
            scores[str(n)]=dict(mae=float(error.mean()),weighted_huber=float(np.sum(huber*weight)/np.sum(weight)),
                early_3s_mae=float(error[time<3].mean()),turn_mae=float(error[turn].mean()) if turn.any() else None)
        result.append(dict(key=r['key'],seed=r['seed'],preset=r['preset'],condition=r['condition'],scores=scores))
    summary={};losses={}
    for condition in ['normal','mild','mild_light']:
        a=[r for r in result if r['condition']==condition]
        summary[condition]={str(n):float(np.mean([r['scores'][str(n)]['mae'] for r in a])) for n in WINDOWS}
        losses[condition]={str(n):float(np.mean([r['scores'][str(n)]['weighted_huber'] for r in a])) for n in WINDOWS}
    dump(REPORT/'results.json',dict(episode_macro_mae=summary,episode_macro_weighted_huber=losses,rows=result,neural_copycat_not_proven=True,final_exam_used=False))
    (REPORT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(dict(mae=summary,weighted_huber=losses)),flush=True)


if __name__=='__main__':run()
