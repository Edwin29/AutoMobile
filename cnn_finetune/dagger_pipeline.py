"""Bounded DAgger execution, seed42-only development selection."""
import concurrent.futures,json,os,subprocess,sys
from pathlib import Path
import numpy as np
from .compare_metrics import extra
ROOT=Path('D:/autodrive_cnn');REPO=Path(__file__).resolve().parents[1];PY=str(ROOT/'venv/Scripts/python.exe')
RESULTS=REPO/'cnn_finetune/results';DECISIONS=RESULTS/'DECISIONS.md'
ENV=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TEMP=str(ROOT/'tmp'),TMP=str(ROOT/'tmp'))
def call(args,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf-8') as f:r=subprocess.run(args,cwd=REPO,env=ENV,stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError(f'Failed {log}')
def note(text):
    with DECISIONS.open('a',encoding='utf-8') as f:f.write('\n- 2026-11-12: '+text+'\n')
    print(text,flush=True)
def push(message):
    for cmd in [['git','add','cnn_finetune'],['git','commit','-m',message],['git','push','origin','cnn-finetune']]:subprocess.run(cmd,cwd=REPO,check=True)
def dev(onnx,name):
    out=ROOT/'runs'/name
    if not (out/'matrix.json').exists():call([PY,'-m','cnn_finetune.evaluate','--arms','new','--seed','42','--dev','--onnx',str(onnx),'--out',str(out)],ROOT/'logs'/f'{name}.log')
    rows=json.loads((out/'matrix.json').read_text());events=[]
    for r in rows:r['extra']=extra(r['path'],r['direction']);events+=r['extra']['post_corner_recovery_s']
    score=[sum(r['summary']['verified_lap'] and r['summary']['min_margin_cm']>=0 for r in rows),min(r['summary']['min_margin_cm'] for r in rows),-float(np.median(events)) if events else -1e9]
    return dict(score=score,rows=rows)
def resource():
    import ctypes,shutil
    class M(ctypes.Structure):_fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ['total','avail','tp','ap','tv','av','ex']]
    m=M();m.length=ctypes.sizeof(m);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    if m.avail<2*1024**3 or shutil.disk_usage(ROOT).free<50*1024**3:raise RuntimeError('Resource gate; no new workers started')
if __name__=='__main__':
    resource()
    gate=json.loads((ROOT/'runs/dagger_teacher_initial/teacher_metrics.json').read_text())
    recovery=float(np.median([v for r in gate for v in r['extra']['post_corner_recovery_s']]))
    assert recovery<=1.49, 'Teacher correction required before DAgger'
    current_onnx=REPO/'cnn_finetune/models/direct_v1/policy.onnx';current_pt=ROOT/'checkpoints/direct_run2/best.pt'
    baseline=dev(current_onnx,'dagger_dev_v1');history=[];datasets=[];retries=0
    state=dict(teacher_recovery=recovery,initial_dev=baseline,iterations=history)
    (RESULTS/'DAGGER_PROGRESS.json').write_text(json.dumps(state,indent=2))
    note(f"seed42 direct_v1 개발 점수 {baseline['score']}. 이 시드로만 모델을 선정한다.");push('Record DAgger development baseline')
    previous=baseline['score']
    for iteration in [1,2,3]:
        resource();data=ROOT/'data'/f'dagger_{iteration}_v2';data.mkdir(parents=True,exist_ok=False)
        seeds=list(range(200+20*iteration,212+20*iteration));assert not set(seeds)&{41,42,104,109,118,123,106,111,126,129}
        note(f'DAgger 반복 {iteration}: beta={ {1:.5,2:.25,3:0}[iteration]}, 새 seed {seeds[0]}..{seeds[-1]}, 12000장. 제어36ms, 0.5초 정책 구간, DART 없음, 외곽은 코너 시작 에피소드, 전체 에피소드 중 코너 시작 비율을 검사한다.')
        def collect(seed):call([PY,'-m','cnn_finetune.dagger_collect','--seed',str(seed),'--out',str(data),'--model',str(current_onnx),'--iteration',str(iteration)],data/f'seed_{seed}.log')
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(collect,seeds))
        audit=[]
        for p in data.glob('seed_*.json'):
            meta=json.loads(p.read_text());assert len(meta)==1000 and min(m['margin'] for m in meta)>=-3 and min(m['junction_distance'] for m in meta)>=35
            audit.append(dict(seed=meta[0]['seed'],n=len(meta),student=sum(m['policy']=='student' for m in meta),inner=sum(m['road']=='inner' for m in meta),negative_margin=sum(m['margin']<0 for m in meta),episodes=len({m['episode'] for m in meta}),corner_episodes=sum({m['episode']:m['corner_episode'] for m in meta}.values())))
        assert len(audit)==12
        assert sum(r["corner_episodes"] for r in audit)>=sum(r["episodes"] for r in audit)/2
        datasets.append(data);candidates=[]
        for retry in [False,True]:
            if retry and (tuple(candidates[0]['development']['score'])>tuple(previous) or retries>=2):break
            if retry:retries+=1;note(f'반복 {iteration} 개발점수 미개선: 수정 {retries}/2, 동일 데이터 보존, 8epoch/학생 프레임 가중치2로 재학습.')
            name=f'dagger_i{iteration}_v2'+('_retry' if retry else '');out=ROOT/'checkpoints'/name
            args=[PY,'-m','cnn_finetune.dagger_train','--data',str(ROOT/'data/v1'),'--extra',*[str(d) for d in datasets],'--initial',str(current_pt),'--out',str(out),'--epochs','8' if retry else '6','--student-weight','2' if retry else '1']
            call(args,ROOT/'logs'/f'{name}_train.log');resource();development=dev(out/'policy.onnx',name+'_dev')
            candidates.append(dict(name=name,checkpoint=str(out),development=development,metrics=json.loads((out/'metrics.json').read_text())))
        winner=max(candidates,key=lambda r:tuple(r['development']['score']));current_onnx=Path(winner['checkpoint'])/'policy.onnx';current_pt=Path(winner['checkpoint'])/'best.pt';previous=winner['development']['score']
        history.append(dict(iteration=iteration,data=str(data),audit=audit,candidates=candidates,winner=winner['name']))
        state['retries_used']=retries;(RESULTS/'DAGGER_PROGRESS.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
        note(f"반복 {iteration} 완료, 선택 {winner['name']}, 개발점수 {previous}. seed41은 선택에 사용하지 않았다.");push(f'Record DAgger iteration {iteration} development results')
    best=max([c for i in history for c in i['candidates']],key=lambda c:tuple(c['development']['score']))
    state['selected']=best['name'];state['selected_checkpoint']=best['checkpoint'];(RESULTS/'DAGGER_PROGRESS.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    print('SELECTED',best['name'],flush=True)
