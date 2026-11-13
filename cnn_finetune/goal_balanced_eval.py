"""Same visual scene/scoring for balanced data arms and two fixed controls."""
import argparse
import itertools
import json
from pathlib import Path
import time
import numpy as np
from .goal_visual_recovery import VisualRecoveryScene
from .goal_all_movement_eval import MODELS as OLD_MODELS,REPORT as OLD_EVAL_REPORT,forbidden_teacher
from .goal_all_movement_data import REPORT as OLD_DATA_REPORT
from .goal_balanced_training import OUT as TRAIN_OUT,REPORT as TRAIN_REPORT
from .goal_teacher_coverage import ROOT,KEYS,PRESETS
from .junction_learning import Encoder,model_at,setup
from .goal_continuous_clearance import continuous_clearance
from .goal_development import sha
from .junction_stage02 import dump

OUT=Path('D:/autodrive_cnn/runs/goal_balanced_eval_20261113')
REPORT=ROOT/'cnn_finetune/results/goal_balanced_eval_20261113'
ARMS=['reference','unbalanced','balanced','balanced_light']


def checkpoint(arm,training_seed):
    if arm in ['reference','unbalanced']:
        assert training_seed==17
        return OLD_MODELS['reference' if arm=='reference' else 'seed_17']
    return TRAIN_OUT/arm/f'seed_{training_seed}'/'update_2400.pt'


def tag(arm,training_seed):return f'{arm}_s{training_seed}'


def freeze():
    OUT.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    dump(REPORT/'protocol.json',dict(source_sha256=sha(__file__),training_protocol_sha256=sha(TRAIN_REPORT/'protocol.json'),
        parent_scene_protocol_sha256=sha(OLD_DATA_REPORT/'protocol.json'),
        old_evaluation_source_sha256=sha(ROOT/'cnn_finetune/goal_all_movement_eval.py'),
        control_checkpoints={a:dict(path=str(checkpoint(a,17)),sha256=sha(checkpoint(a,17))) for a in ['reference','unbalanced']},
        arms=ARMS,training_seeds=[17,29,43],evaluation_seeds=[6901,6902],conditions=['normal','mild','mild_light'],
        keys=KEYS,presets=PRESETS,n_per_condition_seed=84,
        order='6902normal fresh-development comparison first;6901normal is diagnosed regression set. Compound conditions and other initializations after prototype safety diagnosis.',
        scoring='unchanged stable55cm exit,valid start,continuous full-body margin>=0 at every physics sample; report severe/wrong exit as separate safety gates',
        reuse='For controls6901normal only, import earlier84frozen results after checkpoint and evaluator-source hash verification; do not rerun or alter them.',
        scope='Local visual development; not actual A5b+v4 legacy comparison or full missions. No final seeds.',final_exam_used=False))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print('balanced evaluation frozen',flush=True)


def evaluate(arm,training_seed,seed,condition):
    setup();p=json.loads((REPORT/'protocol.json').read_text());assert sha(__file__)==p['source_sha256']
    assert sha(TRAIN_REPORT/'protocol.json')==p['training_protocol_sha256'] and sha(OLD_DATA_REPORT/'protocol.json')==p['parent_scene_protocol_sha256']
    parent=json.loads((OLD_DATA_REPORT/'protocol.json').read_text())
    for f,h in parent['hashes'].items():assert sha(f)==h,f
    assert [str(f) for f in sorted((ROOT/'bench/world/realcam').glob('bg_*.jpg'))]==parent['background_files']
    assert seed in p['evaluation_seeds'] and condition in p['conditions'] and training_seed in p['training_seeds']
    path=checkpoint(arm,training_seed);model_hash=sha(path);name=tag(arm,training_seed)
    if arm in p['control_checkpoints']:assert model_hash==p['control_checkpoints'][arm]['sha256']
    else:
        trained=json.loads((path.parent/'training.json').read_text())
        assert trained['final_sha256']==model_hash and trained['updates']==2400 and trained['seed']==training_seed
        assert trained['protocol_sha256']==p['training_protocol_sha256']
    registration=REPORT/f'{name}_checkpoint.json'
    if registration.exists():assert json.loads(registration.read_text())['sha256']==model_hash
    else:dump(registration,dict(path=str(path),sha256=model_hash))
    target=REPORT/f'{name}_{condition}_{seed}.json'
    if target.exists():
        existing=json.loads(target.read_text());assert len(existing)==84 and all(r['checkpoint_sha256']==model_hash for r in existing)
        print('evaluation already complete',target.name,flush=True);return
    if arm in ['reference','unbalanced'] and seed==6901 and condition=='normal':
        assert sha(ROOT/'cnn_finetune/goal_all_movement_eval.py')==p['old_evaluation_source_sha256']
        old_arm='reference' if arm=='reference' else 'seed_17';source=OLD_EVAL_REPORT/f'{old_arm}_nominal.json'
        rows=json.loads(source.read_text());assert len(rows)==84 and all(r['checkpoint_sha256']==model_hash for r in rows)
        for r in rows:assert sha(Path(r['artifact'])/'physics.npz')==r['physics_sha256']
        dump(target,[dict(r,evaluation_role=name,imported_from=str(source),imported_sha256=sha(source)) for r in rows])
        print('reused frozen84control cases',name,flush=True);return
    encoder=Encoder().cuda().eval();model=model_at(path);rows=[];started=time.perf_counter()
    for key,preset in itertools.product(KEYS,PRESETS):
        episode='_'.join(key)+f'_{seed}_{preset}_{condition}';base=OUT/name/episode
        candidates=[base]+sorted(base.parent.glob(episode+'_attempt_*')) if base.parent.exists() else [base]
        done=[d for d in candidates if (d/'complete.json').exists()]
        if done:
            r=json.loads((done[-1]/'complete.json').read_text());assert r['checkpoint_sha256']==model_hash and r['source_sha256']==p['source_sha256']
            assert sha(Path(r['artifact'])/'physics.npz')==r['physics_sha256']
        else:
            dst=base;attempt=1
            while dst.exists():dst=base.with_name(episode+f'_attempt_{attempt}');attempt+=1
            env=VisualRecoveryScene(key,condition,seed,preset,encoder);env.teacher=forbidden_teacher;env.finish(model)
            assert env.actor_evaluation and not env.decisions
            trace=np.asarray(env.physics);margin=continuous_clearance(trace[:,1:4],env.p,env.track)
            r=env.save(dst,save_images=False)
            with np.load(dst/'sequence.npz') as z:assert not z['label_valid'].any() and not z['supervise'].any()
            r.update(evaluation_role=name,checkpoint_sha256=model_hash,source_sha256=p['source_sha256'],
                physics_sha256=sha(dst/'physics.npz'),continuous_min_clearance_cm=float(margin.min()),
                continuous_clean=bool(r['stable_success'] and r['initial_valid'] and margin.min()>=-1e-9))
            dump(dst/'complete.json',r)
        rows.append(r)
        if len(rows)%12==0:
            dump(REPORT/f'{name}_{condition}_{seed}_progress.json',rows)
            print('balanced eval',name,condition,seed,len(rows),'/84','clean',sum(r['continuous_clean'] for r in rows),'elapsed',round(time.perf_counter()-started,1),flush=True)
    dump(target,rows);print('balanced evaluation complete',name,condition,seed,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['freeze','evaluate']);p.add_argument('--arm',choices=ARMS)
    p.add_argument('--training-seed',type=int,default=17);p.add_argument('--seed',type=int,choices=[6901,6902]);p.add_argument('--condition',choices=['normal','mild','mild_light'])
    a=p.parse_args();freeze() if a.task=='freeze' else evaluate(a.arm,a.training_seed,a.seed,a.condition)
