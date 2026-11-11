"""Recompute sampled labels independently from stored pose/applied commands."""
import json,hashlib
from pathlib import Path
import numpy as np
from bench.vehicle import random_params
from .direct_teacher import DirectTeacher
if __name__=='__main__':
    results=[]
    for iteration in (1,2,3):
        folder=Path(f'D:/autodrive_cnn/data/dagger_{iteration}_v2')
        for p in sorted(folder.glob('seed_*.json')):
            meta=json.loads(p.read_text(encoding='utf-8'));seed=meta[0]['seed']
            with np.load(p.with_suffix('.npz')) as z: y=z['y']
            teacher=DirectTeacher(meta[0]['direction'],variant=3);outer=teacher.route.copy();teacher.sim_params=random_params(seed);errors=[]
            for i in np.linspace(0,len(meta)-1,10,dtype=int):
                m=meta[i];teacher.route=outer
                if m['road']=='inner':
                    line=np.zeros((171,2));line[:,seed%2]=np.linspace(-85,85,171)
                    teacher.route=line if seed%4<2 else line[::-1]
                teacher.last_pwm=np.array(m['applied_pwm']);_,info=teacher.step(np.array(m['pose']))
                errors.append(float(np.max(np.abs(np.array(info['label'])-y[i]))))
            assert max(errors)<1e-6
            results.append(dict(iteration=iteration,seed=seed,samples=len(errors),max_recomputed_label_error=max(errors),sha256=hashlib.sha256(p.with_suffix('.npz').read_bytes()).hexdigest()))
    assert len(results)==36
    Path('cnn_finetune/results/DAGGER_LABEL_AUDIT.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(len(results),max(r['max_recomputed_label_error'] for r in results))
