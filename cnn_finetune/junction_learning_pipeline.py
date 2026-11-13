"""Run the authorized matched experiment through frozen-checkpoint final evaluation."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from .junction_learning import OUT,REPORT,ROOT
from .junction_stage02 import dump


def launch(module,args,name):
    log=OUT/'logs'/(name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf-8') as f:
        result=subprocess.run([sys.executable,'-m',module,*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{name} failed; inspect {log}')
    print(name,'completed',flush=True)


def evaluate(item):
    arm,epoch,split=item
    checkpoint=OUT/'models'/arm/f'epoch_{epoch}.pt';tag=f'{split}_{arm}_{epoch}'
    if not (REPORT/(tag+'.json')).exists():
        launch('cnn_finetune.junction_learning',['evaluate','--checkpoint',str(checkpoint),'--split',split,'--tag',tag],tag)
    return tag


def main():
    while not (REPORT/'teacher_data.json').exists():time.sleep(2)
    if not (OUT/'warmstart.pt').exists():launch('cnn_finetune.junction_learning_train',['warm'],'warm')
    if not (REPORT/'preferences.json').exists():launch('cnn_finetune.junction_learning',['branches'],'branches')
    for arm in ['bc','outcome']:
        if not (OUT/'models'/arm/'epoch_8.pt').exists():launch('cnn_finetune.junction_learning_train',[arm],arm)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        tags=list(pool.map(evaluate,[(arm,epoch,'dev') for arm in ['bc','outcome'] for epoch in [4,8]]))
    selected=[]
    for arm in ['bc','outcome']:
        candidates=[json.loads((REPORT/(tag+'.json')).read_text()) for tag in tags if tag.startswith('dev_'+arm+'_')]
        def rank(candidate):
            rows=candidate['results']
            return (sum(r['stable_success'] for r in rows),sum(r['clean_stable'] for r in rows),
                    -sum(r['outcome']=='severe_departure' for r in rows),sum(r['min_margin_cm'] for r in rows)/len(rows))
        best=max(candidates,key=rank);epoch=int(Path(best['checkpoint']).stem.split('_')[-1])
        selected.append(dict(arm=arm,epoch=epoch,checkpoint=best['checkpoint'],sha256=best['sha256'],development_rank=list(rank(best))))
    frozen=REPORT/'selected_before_final.json'
    if frozen.exists():assert json.loads(frozen.read_text())==selected
    else:dump(frozen,selected)
    print('CHECKPOINTS FROZEN',selected,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(evaluate,[(r['arm'],r['epoch'],'final') for r in selected]))
    dump(REPORT/'pipeline_complete.json',dict(selected=selected,completed=True,selection_frozen_before_final=True))


if __name__=='__main__':main()
