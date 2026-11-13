"""Counterfactual replay input sensitivity, not physically executed rollouts."""
import json
from pathlib import Path
import numpy as np
import torch
from .goal_visual_holdout_fit import REPORT as HOLDOUT_REPORT
from .goal_all_movement_eval import MODELS
from .junction_learning import model_at,setup
from .goal_development import ROOT,sha
from .junction_stage02 import dump

REPORT=ROOT/'cnn_finetune/results/goal_input_reliance_audit_20261113'


@torch.no_grad()
def run():
    setup();data=HOLDOUT_REPORT/'results.json';rows=json.loads(data.read_text())['rows'];assert len(rows)==84
    REPORT.mkdir(exist_ok=False);models={k:model_at(MODELS[k]) for k in ['reference','seed_17']}
    dump(REPORT/'protocol.json',dict(source_sha256=sha(__file__),holdout_results_sha256=sha(data),
        checkpoints={k:sha(MODELS[k]) for k in models},conditions=['factual','frozen_first_image','zero_previous_pwm'],
        scope='Replay the same held-development expert trajectories, retaining factual direction and resetting hidden state per episode. No new driving or training.',
        interpretation='These interventions can be off-distribution and inconsistent with image history. They measure prediction sensitivity, not causal attribution of driving reward or proof that visual input is ignored.',final_exam_used=False))
    results=[]
    for row in rows:
        path=Path(row['artifact']);assert sha(path/'sequence.npz')==row['sequence_sha256']
        with np.load(path/'sequence.npz') as z:
            f=torch.tensor(z['f'],device='cuda').float()[None];aux=torch.tensor(z['aux'],device='cuda').float()[None]
            label=z['label'];previous_zero=aux.clone();previous_zero[...,:2]=0
            frozen=f[:,:1].expand_as(f).contiguous();arms={}
            for name,model in models.items():
                factual,_=model(f,aux);base=factual[0].cpu().numpy();metrics={}
                for condition,features,inputs in [('factual',f,aux),('frozen_first_image',frozen,aux),('zero_previous_pwm',f,previous_zero)]:
                    prediction,_=model(features,inputs);prediction=prediction[0].cpu().numpy()
                    metrics[condition]=dict(mae=float(abs(prediction-label).mean()),prediction_change_mae=float(abs(prediction-base).mean()))
                arms[name]=metrics
        results.append(dict(key=row['key'],preset=row['preset'],teacher_clean=row['teacher_clean'],arms=arms))
    summary={name:{condition:dict(episode_macro_mae=float(np.mean([r['arms'][name][condition]['mae'] for r in results])),
            prediction_change_mae=float(np.mean([r['arms'][name][condition]['prediction_change_mae'] for r in results])))
        for condition in ['factual','frozen_first_image','zero_previous_pwm']} for name in models}
    dump(REPORT/'results.json',dict(summary=summary,rows=results,no_training=True,no_physical_counterfactual=True,final_exam_used=False))
    (REPORT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(summary),flush=True)


if __name__=='__main__':run()
