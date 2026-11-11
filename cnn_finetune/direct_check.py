"""Exercise production direct adapter against labels and allocator averaging."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from bench.controllers import HybridCNN
from .direct_teacher import DirectTeacher

def check():
    rows=[]
    for k in [-.085,-.06,-.04,0,.04,.06,.085]:
        teacher=DirectTeacher('cw')
        adapter=teacher.adapter
        adapter.cnn=SimpleNamespace(speed_scale=1.)
        label=[.36/.66,-k/.095]
        velocities=[]
        for _ in range(1000):
            pwm=adapter._direct_cnn(dict(linear=label[0],angular=label[1]))
            velocities.append(teacher.motor.velocity(pwm))
        v=np.array(velocities).mean(axis=0);observed=float(v[1]/v[0])
        rows.append(dict(k=k,observed=observed,label=label,passed=abs(observed-k)<=max(abs(k)*.05,1e-8)))
    return dict(ok=all(r['passed'] for r in rows),samples=rows)
if __name__=='__main__':
    result=check();Path('D:/autodrive_cnn/logs/direct_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));raise SystemExit(not result['ok'])
