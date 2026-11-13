"""Resume-safe orchestration of the fixed data-source comparison."""
import concurrent.futures
import json
import subprocess
import sys
import time
from .goal_full_dagger import ROOT,OUT,REPORT
from .junction_stage02 import dump


def run(module,args,tag):
    dst=OUT/'logs';dst.mkdir(exist_ok=True)
    with (dst/(tag+'.log')).open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-m',module,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(tag,'finished',flush=True)


def evaluate(arm):
    previous=ROOT/'cnn_finetune/results/goal_development_20261113'
    tag='source_'+arm
    if not (previous/(tag+'.json')).exists():
        run('cnn_finetune.goal_candidate_evaluation',['--checkpoint',str(OUT/'models'/arm/'update_1200.pt'),'--tag',tag],tag)
    return json.loads((previous/(tag+'.json')).read_text())


def main():
    while not all((REPORT/f'{name}_data.json').exists() for name in ['teacher','student','replay']):time.sleep(5)
    for arm in ['teacher_only','full_dagger']:
        if not (OUT/'models'/arm/'training.json').exists():run('cnn_finetune.goal_full_dagger_train',[arm],'train_'+arm)
    training={arm:json.loads((OUT/'models'/arm/'training.json').read_text()) for arm in ['teacher_only','full_dagger']}
    assert training['teacher_only']['common_sample_order_sha256']==training['full_dagger']['common_sample_order_sha256']
    dump(REPORT/'selected_before_driving.json',training)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows=dict(zip(['teacher_only','full_dagger'],pool.map(evaluate,['teacher_only','full_dagger'])))
    for arm,rs in rows.items():dump(REPORT/(arm+'_evaluation.json'),rs)
    dump(REPORT/'pipeline_complete.json',dict(completed=True,final_exam_used=False))
    for arm,rs in rows.items():print(arm,'clean',sum(r['clean_stable'] for r in rs),'/',len(rs),flush=True)


if __name__=='__main__':main()
