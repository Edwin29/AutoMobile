"""Measure source-gradient conflict without changing any trained checkpoint."""
import json
import numpy as np
import torch
from .goal_full_dagger import OUT,REPORT,INITIAL,KEYS,sha
from .goal_full_dagger_train import loss_for
from .junction_learning import setup,model_at
from .junction_learning_train import read_sequence
from .junction_stage02 import dump


def main():
    setup();torch.set_grad_enabled(True)
    trows=json.loads((REPORT/'replay_data.json').read_text())+[r for r in json.loads((REPORT/'teacher_data.json').read_text()) if r['clean_stable']]
    srows=json.loads((REPORT/'student_data.json').read_text())
    def pool(rows):
        groups={tuple(k):[] for k in KEYS}
        for r in rows:groups[tuple(r['key'])].append(read_sequence(r['artifact']))
        return groups
    teacher,student=pool(trows),pool(srows);rng=np.random.default_rng(8170)
    draws=[([teacher[tuple(k)][int(rng.integers(len(teacher[tuple(k)])))] for k in KEYS],
            [student[tuple(k)][int(rng.integers(len(student[tuple(k)])))] for k in KEYS]) for _ in range(24)]
    rows=[]
    for name,path in [('initial',INITIAL),('teacher_only',OUT/'models/teacher_only/update_1200.pt'),('full_dagger',OUT/'models/full_dagger/update_1200.pt')]:
        model=model_at(path);model.train();params=list(model.parameters())
        for i,(a,b) in enumerate(draws):
            la=loss_for(model,a);ga=torch.cat([g.flatten() for g in torch.autograd.grad(la,params)])
            lb=loss_for(model,b);gb=torch.cat([g.flatten() for g in torch.autograd.grad(lb,params)])
            cosine=float(torch.dot(ga,gb)/(ga.norm()*gb.norm()).clamp(min=1e-12))
            rows.append(dict(model=name,batch=i,teacher_loss=float(la),student_loss=float(lb),
                teacher_gradient_norm=float(ga.norm()),student_gradient_norm=float(gb.norm()),cosine=cosine,
                first_order_teacher_change_sign=float(torch.dot(ga,-.5*(ga+gb)))))
    summary=[]
    for name in ['initial','teacher_only','full_dagger']:
        r=[x for x in rows if x['model']==name]
        summary.append(dict(model=name,batches=len(r),median_cosine=float(np.median([x['cosine'] for x in r])),
            opposing_batches=sum(x['cosine']<0 for x in r),
            median_student_teacher_norm_ratio=float(np.median([x['student_gradient_norm']/x['teacher_gradient_norm'] for x in r])),
            equal_mix_increases_teacher_first_order=sum(x['first_order_teacher_change_sign']>0 for x in r)))
    dump(REPORT/'source_gradient_audit.json',dict(source_sha256=sha(__file__),sampling_seed=8170,summary=summary,rows=rows,
        interpretation='Raw gradients at fixed checkpoints, not actual AdamW steps or proof of closed-loop causality. Same 24 balanced source batches for each model.'))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
