"""Frozen visual closed-loop development comparison, with no teacher queries."""
import argparse
import itertools
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .goal_visual_recovery import VisualRecoveryScene
from .goal_all_movement_data import REPORT as DATA_REPORT,OUT as DATA_OUT
from .goal_teacher_coverage import ROOT,KEYS,PRESETS
from .junction_learning import Encoder,model_at,setup,CHECKPOINT
from .goal_continuous_clearance import continuous_clearance
from .goal_development import sha
from .junction_stage02 import dump

OUT=Path('D:/autodrive_cnn/runs/goal_all_movement_eval_20261113')
REPORT=ROOT/'cnn_finetune/results/goal_all_movement_eval_20261113'
SEEDS=[6901,6902]
CONDITIONS=['normal','mild','mild_light']
MODELS={'reference':Path('D:/autodrive_cnn/runs/goal_development_20261113/convergence_probe/best.pt'),
        **{f'seed_{s}':DATA_OUT/'models'/f'seed_{s}'/'update_2400.pt' for s in [17,29,43]}}


def freeze():
    p=json.loads((DATA_REPORT/'protocol.json').read_text())
    for f,h in p['hashes'].items():assert sha(f)==h,f
    OUT.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    dump(REPORT/'protocol.json',dict(source_sha256=sha(__file__),data_protocol_sha256=sha(DATA_REPORT/'protocol.json'),
        models={k:str(v) for k,v in MODELS.items()},reference_sha256=sha(MODELS['reference']),
        keys=KEYS,seeds=SEEDS,presets=PRESETS,conditions=CONDITIONS,expected_per_model=504,
        stages={'nominal':'all28 movements x3presets on6901,normal;84runs',
                'remaining':'remaining420cases, independently resumable; no subset may be represented as complete504'},
        primary='stable55cm exit + legal start + continuous full-body clearance>=0 for all physics samples',
        auxiliary='wrong exit, severe departure, missed exit, timeout, stop/pivot and angular acceleration; all failures retained',
        teacher_access='teacher method raises if called during actor evaluation; labels invalid and masked; actor sees image features,previousPWM,direction only',
        comparison='reference is frozenCNN+GRU trained on four movements, not A5b+v4 or direct_v2; report the four-movement subset separately',
        interpretation='local visual development, not full missions, final examination or deployment; success-only data selection does not remove any evaluation cases',
        checkpoint_registration='hash each specified checkpoint once on its first run, verify unchanged thereafter; fixed update2400 only',
        final_exam_used=False))
    shutil.copy2(__file__,OUT/'source_snapshot.py');print('all28 visual evaluation frozen',flush=True)


def forbidden_teacher(*args,**kwargs):
    raise AssertionError('Actor evaluation must not query the privileged teacher')


def evaluate(arm,stage):
    setup();p=json.loads((REPORT/'protocol.json').read_text());assert p['source_sha256']==sha(__file__)
    assert p['data_protocol_sha256']==sha(DATA_REPORT/'protocol.json')
    inherited=json.loads((DATA_REPORT/'protocol.json').read_text())
    for f,h in inherited['hashes'].items():assert sha(f)==h,f
    assert [str(f) for f in sorted((ROOT/'bench/world/realcam').glob('bg_*.jpg'))]==inherited['background_files']
    checkpoint=Path(p['models'][arm]);model_hash=sha(checkpoint)
    if arm=='reference':assert model_hash==p['reference_sha256']
    else:
        training=json.loads((checkpoint.parent/'training.json').read_text())
        assert training['final_sha256']==model_hash and training['updates']==2400
    registration=REPORT/f'{arm}_checkpoint.json'
    if registration.exists():assert json.loads(registration.read_text())['sha256']==model_hash
    else:dump(registration,dict(path=str(checkpoint),sha256=model_hash))
    encoder=Encoder().cuda().eval();policy=model_at(checkpoint);rows=[];start=time.perf_counter()
    cases=list(itertools.product(KEYS,SEEDS,PRESETS,CONDITIONS))
    cases=[c for c in cases if ((c[1]==6901 and c[3]=='normal')==(stage=='nominal'))]
    for key,seed,preset,condition in cases:
        name='_'.join(key)+f'_{seed}_{preset}_{condition}';base=OUT/arm/name
        candidates=[base]+sorted(base.parent.glob(name+'_attempt_*')) if base.parent.exists() else [base]
        completed=[d for d in candidates if (d/'complete.json').exists()]
        if completed:
            r=json.loads((completed[-1]/'complete.json').read_text());assert r['checkpoint_sha256']==model_hash
            assert r['source_sha256']==p['source_sha256']
            assert sha(Path(r['artifact'])/'physics.npz')==r['physics_sha256']
        else:
            dst=base;attempt=1
            while dst.exists():dst=base.with_name(name+f'_attempt_{attempt}');attempt+=1
            env=VisualRecoveryScene(key,condition,seed,preset,encoder);env.teacher=forbidden_teacher
            env.finish(policy);assert env.actor_evaluation and not env.decisions
            trace=np.asarray(env.physics);margin=continuous_clearance(trace[:,1:4],env.p,env.track)
            r=env.save(dst,save_images=False)
            with np.load(dst/'sequence.npz') as z:
                assert not z['label_valid'].any() and not z['supervise'].any()
                assert np.isfinite(z['action']).all()
            r.update(arm=arm,stage=stage,artifact=str(dst),checkpoint_sha256=model_hash,source_sha256=p['source_sha256'],
                physics_sha256=sha(dst/'physics.npz'),continuous_min_clearance_cm=float(margin.min()),
                continuous_clean=bool(r['stable_success'] and r['initial_valid'] and margin.min()>=-1e-9))
            dump(dst/'complete.json',r)
        rows.append(r)
        if len(rows)%12==0:
            dump(REPORT/f'{arm}_{stage}_progress.json',rows)
            print('visual evaluation',arm,stage,len(rows),'/',len(cases),'clean',sum(r['continuous_clean'] for r in rows),'elapsed',round(time.perf_counter()-start,1),flush=True)
    dump(REPORT/f'{arm}_{stage}.json',rows);print('visual evaluation complete',arm,stage,len(rows),flush=True)


def summarize():
    summaries={};pilot={('east','W','left'),('east','W','right'),('south','N','right'),('center','N','right')}
    for arm in MODELS:
        rows=[]
        for stage,n in [('nominal',84),('remaining',420)]:
            f=REPORT/f'{arm}_{stage}.json'
            if f.exists():
                a=json.loads(f.read_text());assert len(a)==n;rows+=a
        if not rows:continue
        assert len({(tuple(r['key']),r['seed'],r['preset'],r['condition']) for r in rows})==len(rows)
        by={}
        for condition in CONDITIONS:
            for scope in ['all28','old_four']:
                a=[r for r in rows if r['condition']==condition and (scope=='all28' or tuple(r['key']) in pilot)]
                if not a:continue
                by[condition+'_'+scope]=dict(n=len(a),clean=sum(r['continuous_clean'] for r in a),
                    stable=sum(r['stable_success'] for r in a),pauses=sum(r['translation_pause_s']>0 for r in a),
                    outcomes={v:sum(r['outcome']==v for r in a) for v in sorted({r['outcome'] for r in a})})
        summaries[arm]=dict(n=len(rows),all504_complete=len(rows)==504,groups=by)
    dump(REPORT/'summary.json',dict(models=summaries,final_exam_used=False,full_mission=False));print(json.dumps(summaries))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('task',choices=['freeze','evaluate','summarize'])
    parser.add_argument('--arm',choices=list(MODELS));parser.add_argument('--stage',choices=['nominal','remaining'])
    a=parser.parse_args()
    evaluate(a.arm,a.stage) if a.task=='evaluate' else globals()[a.task]()
