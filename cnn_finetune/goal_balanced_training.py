"""Controlled data expansion with unchanged recurrent architecture and BC loss.

Two arms share physical-episode sampling, initialization, updates and loss.
The light arm substitutes a matched strong-light image sequence on half of new
normal-physics draws. It does not reduce the compound-disturbance sampling rate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .goal_all_movement_train import loss,read_data,training_eligible
from .goal_all_movement_data import REPORT as OLD_REPORT,OUT as OLD_OUT
from .goal_balanced_visual_data import REPORT as NEW_REPORT
from .goal_teacher_coverage import KEYS,ROOT
from .junction_learning import TemporalPolicy,setup,CHECKPOINT
from .goal_development import sha
from .junction_stage02 import dump

OUT=Path('D:/autodrive_cnn/runs/goal_balanced_training_20261113')
REPORT=ROOT/'cnn_finetune/results/goal_balanced_training_20261113'
ARMS=['balanced','balanced_light'];FAMILIES=['normal','mild','mild_light'];UPDATES=2400
LOSS_SOURCE=ROOT/'cnn_finetune/goal_all_movement_train.py'


def plan():
    old=json.loads((OLD_REPORT/'training_protocol.json').read_text())
    assert sha(LOSS_SOURCE)==old['source_sha256']
    assert old['checks']['padding_loss_invariant'] and old['checks']['padding_gradient_invariant']
    OUT.mkdir(exist_ok=False);REPORT.mkdir(exist_ok=False)
    dump(REPORT/'protocol.json',dict(source_sha256=sha(__file__),loss_source_sha256=sha(LOSS_SOURCE),
        old_data_sha256=sha(OLD_REPORT/'data.json'),new_collection_protocol_sha256=sha(NEW_REPORT/'protocol.json'),
        encoder_sha256=sha(CHECKPOINT),old_seed17_initial_sha256=sha(OLD_OUT/'models/seed_17/initial.pt'),
        arms=ARMS,seeds=[17,29,43],first_seed=17,updates=UPDATES,
        model='same frozenCNN features -> LayerNorm/128 -> GRU64 ->32 -> signedPWM2;190050trainable parameters',
        optimizer='unchanged AdamW5e-4 weight_decay1e-4,cosine to5e-5,gradient clip1',
        loss='unchanged equal-episode turn-weighted SmoothL1 beta.05,turnweight3 when abs(labelL-labelR)>.15',
        sampling='7entire episodes,28moves visited once/4updates. Existing three physical/visual families normal,mild,mild_light sampled uniformly among nonempty pools; primary RNG221.',
        normal_pool='eligible old normal + all new normal. Existing mild/mild_light pools unchanged.',
        lighting_ablation='balanced_light replaces new normal sequence by its exact-physics normal_light partner with probability.5 using separate RNG224; balanced uses typical images. Identical physical sample order across arms verified.',
        expected_raw_episodes=1848,expected_eligible=1786,selection='fixed final update2400; no driving-selected checkpoint; intermediates600/1200/1800diagnostic only',
        evaluation='6902nominal fresh-development comparison first,6901regression separate; weak compound/strong-light and3initializations still required before goal completion',
        limitations='Dataset expansion changes initial-pose and camera-seed coverage together. Lighting-pair comparison isolates that augmentation within the expanded data; no claim of new outcome loss.',
        final_exam_used=False))
    (OUT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print('balanced training plan frozen; data not yet bound',flush=True)


def bind():
    p=json.loads((REPORT/'protocol.json').read_text());assert sha(__file__)==p['source_sha256']
    assert sha(OLD_REPORT/'data.json')==p['old_data_sha256'] and sha(NEW_REPORT/'protocol.json')==p['new_collection_protocol_sha256']
    old=json.loads((OLD_REPORT/'data.json').read_text());new=json.loads((NEW_REPORT/'data.json').read_text())
    assert len(old['rows'])==504 and new['summary']['n']==1344 and new['summary']['physical_lighting_pairs_exact']==672
    assert new['summary']['eligible']==1344 and all(training_eligible(r) for r in new['rows'])
    assert not any(r['seed'] in [6901,6902] or 900000<=r['seed']<=999999 for r in old['rows']+new['rows'])
    target=REPORT/'data_binding.json';assert not target.exists()
    dump(target,dict(new_data_sha256=sha(NEW_REPORT/'data.json'),old_data_sha256=sha(OLD_REPORT/'data.json'),
        new_eligible=1344,old_eligible=sum(training_eligible(r) for r in old['rows']),
        paired_source_rows=672,final_exam_used=False))
    print('balanced training data bound',flush=True)


def train(arm,seed):
    setup();torch.set_grad_enabled(True);p=json.loads((REPORT/'protocol.json').read_text());binding=json.loads((REPORT/'data_binding.json').read_text())
    assert arm in ARMS and seed in p['seeds'] and sha(__file__)==p['source_sha256'] and sha(LOSS_SOURCE)==p['loss_source_sha256']
    assert sha(CHECKPOINT)==p['encoder_sha256'] and sha(OLD_REPORT/'data.json')==binding['old_data_sha256'] and sha(NEW_REPORT/'data.json')==binding['new_data_sha256']
    old=[r for r in json.loads((OLD_REPORT/'data.json').read_text())['rows'] if training_eligible(r)]
    new=json.loads((NEW_REPORT/'data.json').read_text())['rows']
    partners={(tuple(r['key']),r['seed'],r['preset']):r for r in new if r['condition']=='normal_light'}
    physical=old+[r for r in new if r['condition']=='normal'];pools={k:{c:[] for c in FAMILIES} for k in KEYS};cache={}
    needed=physical+list(partners.values()) if arm=='balanced_light' else physical
    for r in needed:
        assert sha(Path(r['artifact'])/'sequence.npz')==r['sequence_sha256'];cache[r['artifact']]=read_data(r)
    for r in physical:pools[tuple(r['key'])][r['condition']].append(r)
    dst=OUT/arm/f'seed_{seed}';dst.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);model=TemporalPolicy().cuda().train()
    if seed==17:
        initial=OLD_OUT/'models/seed_17/initial.pt';assert sha(initial)==p['old_seed17_initial_sha256']
        old_state=torch.load(initial,weights_only=True,map_location='cpu')
        for k,v in model.state_dict().items():torch.testing.assert_close(v.cpu(),old_state[k],atol=0,rtol=0)
    torch.save(model.state_dict(),dst/'initial.pt')
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=UPDATES,eta_min=5e-5)
    rng=np.random.default_rng(221);light_rng=np.random.default_rng(224);history=[];window=[];order=[];visual_order=[];started=time.perf_counter()
    for update in range(1,UPDATES+1):
        if (update-1)%4==0:key_order=[KEYS[i] for i in rng.permutation(len(KEYS))]
        keys=key_order[((update-1)%4)*7:((update-1)%4+1)*7];items=[];paths=[];visual=[]
        for key in keys:
            families=[c for c in FAMILIES if pools[key][c]];family=families[int(rng.integers(len(families)))]
            row=pools[key][family][int(rng.integers(len(pools[key][family])))];paths.append(row['artifact']);chosen=row
            coin=light_rng.random();partner=partners.get((tuple(row['key']),row['seed'],row['preset'])) if family=='normal' else None
            if arm=='balanced_light' and partner is not None and coin<.5:chosen=partner
            visual.append(chosen['artifact']);items.append(cache[chosen['artifact']])
        order.append(paths);visual_order.append(visual);optimizer.zero_grad();value=loss(model,items);value.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(value) and torch.isfinite(grad)
        optimizer.step();scheduler.step();window.append(float(value.detach()))
        if update%100==0:
            row=dict(update=update,loss=float(np.mean(window)),wall_s=time.perf_counter()-started);history.append(row);window=[]
            dump(dst/'progress.json',history);print('balanced training',arm,seed,json.dumps(row),flush=True)
        if update%600==0:torch.save(model.state_dict(),dst/f'update_{update}.pt')
    physical_hash=hashlib.sha256(json.dumps(order).encode()).hexdigest()
    other=OUT/('balanced_light' if arm=='balanced' else 'balanced')/f'seed_{seed}'/'training.json'
    if other.exists():assert json.loads(other.read_text())['physical_sample_order_sha256']==physical_hash
    dump(dst/'training.json',dict(arm=arm,seed=seed,updates=UPDATES,source_sha256=sha(__file__),protocol_sha256=sha(REPORT/'protocol.json'),
        data_binding_sha256=sha(REPORT/'data_binding.json'),initial_sha256=sha(dst/'initial.pt'),
        final_checkpoint=str(dst/f'update_{UPDATES}.pt'),final_sha256=sha(dst/f'update_{UPDATES}.pt'),
        physical_sample_order_sha256=physical_hash,visual_sample_order_sha256=hashlib.sha256(json.dumps(visual_order).encode()).hexdigest(),
        history=history,trainable_parameters=sum(x.numel() for x in model.parameters()),encoder_frozen=True,
        device=torch.cuda.get_device_name(0),final_exam_used=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['plan','bind','train']);p.add_argument('--arm',choices=ARMS);p.add_argument('--seed',type=int);a=p.parse_args()
    train(a.arm,a.seed) if a.task=='train' else globals()[a.task]()
