"""Held-development teacher-forced visual diagnosis, never training data.

Both fixed checkpoints see identical new6901 teacher trajectories. This separates
new-image generalization error from additional student-trajectory feedback.
"""
import argparse
import itertools
import json
from pathlib import Path
import time
import numpy as np
import torch
from .goal_visual_recovery import VisualRecoveryScene
from .goal_all_movement_data import REPORT as DATA_REPORT
from .goal_all_movement_eval import MODELS,REPORT as EVAL_REPORT
from .goal_visual_nominal_teacher_audit import REPORT as PHYSICS_REPORT
from .goal_teacher_coverage import ROOT,KEYS,PRESETS
from .goal_continuous_clearance import continuous_clearance
from .junction_learning import Encoder,model_at,setup
from .goal_development import sha
from .junction_stage02 import dump

OUT=Path('D:/autodrive_cnn/runs/goal_visual_holdout_fit_20261113')
REPORT=ROOT/'cnn_finetune/results/goal_visual_holdout_fit_20261113'
ARMS=['reference','seed_17']


def freeze():
    parent=json.loads((DATA_REPORT/'protocol.json').read_text())
    for f,h in parent['hashes'].items():assert sha(f)==h,f
    OUT.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    dump(REPORT/'protocol.json',dict(source_sha256=sha(__file__),data_protocol_sha256=sha(DATA_REPORT/'protocol.json'),
        models={k:dict(path=str(MODELS[k]),sha256=sha(MODELS[k])) for k in ARMS},
        keys=KEYS,presets=PRESETS,seed=6901,condition='normal',expected=84,
        purpose='Diagnose poor nominal student rollouts despite low training-demonstration error; same images, expert previousPWM, full recurrent history for both fixed models',
        no_training_use=True,final_exam_used=False,
        caveat='Teacher labels are actual teacher commands, not universally safe targets. Report all84 and separately the70continuous-clean teacher trajectories. Teacher-forced error is not closed-loop success.'))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print('held-development fit diagnosis frozen84episodes',flush=True)


@torch.no_grad()
def run():
    setup();p=json.loads((REPORT/'protocol.json').read_text());assert sha(__file__)==p['source_sha256']
    assert sha(DATA_REPORT/'protocol.json')==p['data_protocol_sha256']
    parent=json.loads((DATA_REPORT/'protocol.json').read_text())
    for f,h in parent['hashes'].items():assert sha(f)==h,f
    for spec in p['models'].values():assert sha(spec['path'])==spec['sha256']
    encoder=Encoder().cuda().eval();models={k:model_at(v['path']) for k,v in p['models'].items()}
    physics={(tuple(r['key']),r['preset']):r for r in json.loads((PHYSICS_REPORT/'results.json').read_text())['rows']}
    rows=[];started=time.perf_counter()
    for key,preset in itertools.product(KEYS,PRESETS):
        env=VisualRecoveryScene(key,'normal',6901,preset,encoder).finish()
        trace=np.asarray(env.physics);expected=np.load(Path(physics[key,preset]['artifact'])/'physics.npz')['physics']
        np.testing.assert_array_equal(trace,expected)
        dst=OUT/('_'.join(key)+'_'+preset);result=env.save(dst,save_images=False)
        with np.load(dst/'sequence.npz') as z:
            assert z['label_valid'].all();f=torch.tensor(z['f'],device='cuda').float()[None]
            aux=torch.tensor(z['aux'],device='cuda').float()[None];label=z['label'];t=z['capture']
            turn=abs(label[:,0]-label[:,1])>.15;predictions={};scores={}
            for arm,model in models.items():
                prediction,_=model(f,aux);pred=prediction[0].cpu().numpy();mae=abs(pred-label).mean(-1);predictions[arm]=pred
                scores[arm]=dict(mae=float(mae.mean()),p95_frame_mae=float(np.quantile(mae,.95)),
                    early_3s_mae=float(mae[t<3].mean()),turn_mae=float(mae[turn].mean()) if turn.any() else None)
        np.savez_compressed(dst/'predictions.npz',**predictions)
        margin=continuous_clearance(trace[:,1:4],env.p,env.track)
        row=dict(key=key,preset=preset,artifact=str(dst),frames=len(label),teacher_clean=bool(result['stable_success'] and result['initial_valid'] and margin.min()>=-1e-9),
            scores=scores,sequence_sha256=sha(dst/'sequence.npz'),no_training_use=True,physics_exact=True)
        dump(dst/'complete.json',row);rows.append(row)
        if len(rows)%12==0:
            dump(REPORT/'progress.json',rows);print('held-development fit',len(rows),'/84','elapsed',round(time.perf_counter()-started,1),flush=True)
    summary={}
    for arm in ARMS:
        summary[arm]={}
        for scope in ['all','teacher_clean']:
            selected=[r for r in rows if scope=='all' or r['teacher_clean']]
            summary[arm][scope]=dict(n=len(selected),episode_macro_mae=float(np.mean([r['scores'][arm]['mae'] for r in selected])),
                early_3s_macro_mae=float(np.mean([r['scores'][arm]['early_3s_mae'] for r in selected])))
    dump(REPORT/'results.json',dict(summary=summary,rows=rows,no_training_use=True,final_exam_used=False));print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['freeze','run']);a=p.parse_args();globals()[a.task]()
