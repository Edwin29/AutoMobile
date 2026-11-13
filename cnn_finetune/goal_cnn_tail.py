"""Paired diagnostic: learn the last two CNN convolutions on fixed data.

Same actor inputs, recurrent architecture, samples and 1200 updates as the
frozen-encoder source comparison. Never substitute a frozen encoder at test.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .goal_full_dagger import OUT as SOURCE, REPORT as SOURCE_REPORT, INITIAL, KEYS, ROOT, sha
from .goal_development import SEEDS, PRESETS, CONDITIONS, LocalScene
from .junction_learning import Encoder, model_at, setup
from .junction_learning_train import read_sequence
from .junction_stage02 import dump

OUT=Path('D:/autodrive_cnn/runs/goal_cnn_tail_20261113')
REPORT=ROOT/'cnn_finetune/results/goal_cnn_tail_20261113'


def source_rows():
    common=json.loads((SOURCE_REPORT/'replay_data.json').read_text())
    common += [r for r in json.loads((SOURCE_REPORT/'teacher_data.json').read_text()) if r['clean_stable']]
    second=json.loads((SOURCE_REPORT/'student_data.json').read_text())
    return common,second


def freeze():
    REPORT.mkdir(exist_ok=False);OUT.mkdir(exist_ok=False)
    dump(REPORT/'protocol.json',dict(
        hypothesis='Frozen visual representation may limit fitting recoveries; change only trainable CNN scope',
        source_protocol_sha256=sha(SOURCE_REPORT/'protocol.json'),source_sha256=sha(__file__),
        initial_sha256=sha(INITIAL),arms=['teacher_only','full_dagger'],
        same_as_source=['episode pools','initial CNN and recurrent weights','common and second RNG streams',
                        'objective and masks','1200 updates','head lr and scheduler','development matrix'],
        change='last two convolution layers trainable at 0.1 times recurrent learning rate; first three fixed',
        head_lr=1e-4,cnn_lr=1e-5,updates=1200,seed=17,
        checkpoint='fixed final update; no driving-based selection',
        input_contract='camera history + prior sent PWM + left/right command; no privileged actor inputs',
        evaluation=dict(seeds=SEEDS,presets=PRESETS,keys=KEYS,conditions=CONDITIONS),
        final_exam_used=False,scope='four-movement development diagnostic only'))
    snap=OUT/'source_snapshot';snap.mkdir()
    for name in ['goal_cnn_tail.py','goal_full_dagger.py','goal_full_dagger_train.py','junction_learning.py','goal_development.py']:
        shutil.copy2(ROOT/'cnn_finetune'/name,snap/name)


def cache():
    setup();encoder=Encoder().cuda().eval();common,second=source_rows()
    folder=OUT/'cache';folder.mkdir(exist_ok=True);manifest={};errors=[];frames=0
    for i,r in enumerate(common+second):
        path=r['artifact'];target=folder/(hashlib.sha256(path.encode()).hexdigest()+'.npy')
        with np.load(Path(path)/'sequence.npz') as z:
            x=z['x'];expected=z['f']
        if not target.exists():
            chunks=[]
            with torch.no_grad():
                for start in range(0,len(x),128):
                    chunks.append(encoder.net[:6](torch.from_numpy(x[start:start+128]).cuda().float()/255).cpu().numpy())
            np.save(target,np.concatenate(chunks))
        prefix=np.load(target)
        ids=np.linspace(0,len(x)-1,min(16,len(x))).astype(int)
        with torch.no_grad():
            actual=encoder.net[6:](torch.from_numpy(prefix[ids]).cuda()).cpu().numpy()
        error=float(np.max(abs(actual-expected[ids])));errors.append(error)
        np.testing.assert_allclose(actual,expected[ids],rtol=1e-5,atol=2e-5)
        manifest[path]=str(target);frames+=len(x)
        if (i+1)%24==0:print('cached',i+1,'episodes',frames,'frames',flush=True)
    dump(REPORT/'cache.json',dict(episodes=len(manifest),frames=frames,max_feature_error=max(errors),files=manifest))


def objective(model,tail,items):
    lengths=[len(x['prefix']) for x in items]
    encoded=tail(torch.cat([x['prefix'] for x in items]).cuda())
    features=pad_sequence(list(encoded.split(lengths)),batch_first=True)
    aux=pad_sequence([x['aux'] for x in items],batch_first=True).cuda()
    labels=pad_sequence([x['label'] for x in items],batch_first=True).cuda()
    mask=pad_sequence([x['supervise'] for x in items],batch_first=True).cuda()
    prediction,_=model(features,aux)
    error=F.smooth_l1_loss(prediction,labels,beta=.05,reduction='none').mean(-1)
    weight=mask*(1+2*(abs(labels[...,0]-labels[...,1])>.15).float())
    assert torch.all(weight.sum(1)>0)
    return ((error*weight).sum(1)/weight.sum(1)).mean()


def bundle_load(path):
    bundle=torch.load(path,weights_only=True,map_location='cuda')
    encoder=Encoder().cuda().eval();encoder.load_state_dict(bundle['encoder'])
    model=model_at(INITIAL);model.load_state_dict(bundle['policy']);model.eval()
    return model,encoder


def check():
    """Validate real-image cache parity, gradients and complete bundle reload."""
    setup();torch.set_grad_enabled(True);torch.manual_seed(17)
    row=source_rows()[1][0]
    with np.load(Path(row['artifact'])/'sequence.npz') as z:
        x=torch.from_numpy(z['x'][:40]).cuda().float()/255
        expected=torch.from_numpy(z['f'][:40]).cuda()
        aux=torch.from_numpy(z['aux'][:40]).cuda()[None]
    encoder=Encoder().cuda();tail=encoder.net[6:]
    for p in tail.parameters():p.requires_grad_(True)
    model=model_at(INITIAL);model.train()
    with torch.no_grad():prefix=encoder.net[:6](x)
    torch.testing.assert_close(tail(prefix),expected,atol=2e-5,rtol=1e-5)
    initial={k:v.clone() for k,v in encoder.state_dict().items()}
    opt=torch.optim.AdamW(list(tail.parameters())+list(model.parameters()),lr=1e-5)
    pred,_=model(tail(prefix)[None],aux);pred.square().mean().backward();opt.step()
    assert all(p.grad is None for p in encoder.net[:6].parameters())
    assert any(not torch.equal(initial[k],v) for k,v in encoder.state_dict().items() if k.startswith(('net.6.','net.8.')))
    assert all(torch.equal(initial[k],v) for k,v in encoder.state_dict().items() if k.startswith(('net.0.','net.2.','net.4.')))
    path=OUT/'serialization_check.pt';torch.save(dict(policy=model.state_dict(),encoder=encoder.state_dict()),path)
    loaded,enc=bundle_load(path);model.eval()
    with torch.no_grad():
        a,_=model(encoder(x)[None],aux);b,_=loaded(enc(x)[None],aux)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    dump(REPORT/'checks.json',dict(real_image_features_match=True,first_three_layers_unchanged=True,
                                  cnn_tail_updated=True,full_bundle_predictions_exact=True))
    print('cache, gradient and complete bundle checks passed',flush=True)


def train(arm):
    setup();torch.set_grad_enabled(True);torch.manual_seed(17)
    protocol=json.loads((REPORT/'protocol.json').read_text())
    assert sha(__file__)==protocol['source_sha256'];assert sha(INITIAL)==protocol['initial_sha256']
    manifest=json.loads((REPORT/'cache.json').read_text())['files']
    trows,srows=source_rows();srows=trows if arm=='teacher_only' else srows
    loaded={}
    def pool(rows):
        result={tuple(k):[] for k in KEYS}
        for r in rows:
            path=r['artifact']
            if path not in loaded:
                data=read_sequence(path);data['prefix']=torch.from_numpy(np.load(manifest[path]));loaded[path]=data
            data=loaded[path]
            if data['supervise'].sum()>0:result[tuple(r['key'])].append((path,data))
        return result
    common=pool(trows);second=pool(srows)
    dst=OUT/'models'/arm;dst.mkdir(parents=True,exist_ok=False)
    encoder=Encoder().cuda();tail=encoder.net[6:]
    for p in tail.parameters():p.requires_grad_(True)
    model=model_at(INITIAL);model.train()
    opt=torch.optim.AdamW([{'params':model.parameters(),'lr':1e-4},{'params':tail.parameters(),'lr':1e-5}],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda u:.1+.9*(1+math.cos(math.pi*u/1200))/2)
    rng=np.random.default_rng(17);other_rng=np.random.default_rng(117)
    history=[];common_order=[];values=[];started=time.perf_counter()
    for update in range(1,1201):
        one=[common[tuple(k)][int(rng.integers(len(common[tuple(k)])))] for k in KEYS]
        two=[second[tuple(k)][int(other_rng.integers(len(second[tuple(k)])))] for k in KEYS]
        common_order.append([x[0] for x in one]);opt.zero_grad()
        a=objective(model,tail,[x[1] for x in one]);b=objective(model,tail,[x[1] for x in two])
        loss=.5*(a+b);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(tail.parameters()),1.)
        assert torch.isfinite(grad);opt.step();scheduler.step();values.append((a.item(),b.item(),float(grad)))
        if update%100==0:
            v=np.asarray(values);row=dict(update=update,common_loss=float(v[:,0].mean()),second_loss=float(v[:,1].mean()),
                max_grad=float(v[:,2].max()),wall_s=time.perf_counter()-started)
            history.append(row);values=[];print(arm,json.dumps(row),flush=True)
        if update%400==0:torch.save(dict(policy=model.state_dict(),encoder=encoder.state_dict()),dst/f'update_{update}.pt')
    order_hash=hashlib.sha256(json.dumps(common_order).encode()).hexdigest()
    baseline=json.loads((SOURCE/'models'/arm/'training.json').read_text())
    assert order_hash==baseline['common_sample_order_sha256']
    dump(dst/'training.json',dict(history=history,updates=1200,common_sample_order_sha256=order_hash,
        trainable_parameters=sum(p.numel() for p in model.parameters())+sum(p.numel() for p in tail.parameters()),
        source_sha256=sha(__file__),initial_sha256=sha(INITIAL),checkpoint=str(dst/'update_1200.pt'),
        checkpoint_sha256=sha(dst/'update_1200.pt'),training_initializations=1,final_exam_used=False))


def evaluate(arm):
    setup();checkpoint=OUT/'models'/arm/'update_1200.pt'
    dump(REPORT/(arm+'_evaluation_protocol.json'),dict(checkpoint_sha256=sha(checkpoint),
         checkpoint_selection='final update 1200, frozen before driving',source_sha256=sha(__file__),
         full_encoder_loaded=True,final_exam_used=False))
    model,encoder=bundle_load(checkpoint);rows=[]
    for seed in SEEDS:
        for preset in PRESETS:
            for key in KEYS:
                for condition in CONDITIONS:
                    dst=OUT/'rollouts'/arm/('_'.join(key)+f'_{condition}_{seed}_{preset}')
                    if (dst/'complete.json').exists():r=json.loads((dst/'complete.json').read_text())
                    else:
                        r=LocalScene(key,condition,seed,preset,encoder).finish(model).save(dst,save_images=False)
                        dump(dst/'complete.json',r)
                    rows.append(r)
            dump(REPORT/(arm+'_progress.json'),rows)
            print('tail eval',arm,seed,preset,sum(r['clean_stable'] for r in rows),'/',len(rows),flush=True)
    dump(REPORT/(arm+'_evaluation.json'),rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['freeze','check','cache','train','evaluate'])
    p.add_argument('--arm',choices=['teacher_only','full_dagger']);a=p.parse_args()
    if a.task in ['train','evaluate']:globals()[a.task](a.arm)
    else:globals()[a.task]()
