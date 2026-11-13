"""Summarize the frozen prerequisite runs, without tuning or new rollouts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .pilot import REPORT as ORACLE,KEYS,SEEDS
from .image_pilot import REPORT
from cnn_finetune.junction_stage02 import dump


def main():
    oracle=json.loads((ORACLE/'runs.json').read_text());image=json.loads((REPORT/'runs.json').read_text())
    groups={};failures=[]
    for name,rows in [('exact_capture_path',oracle),('image_map_path',image)]:
        groups[name]={p:dict(n=sum(r['preset']==p for r in rows),clean=sum(r['preset']==p and r['continuous_clean'] for r in rows))
                      for p in ['field_estimate','sluggish','unequal_sides']}
    for r in image:
        if r['continuous_clean']:continue
        rows=json.loads((Path(r['artifact'])/'records.json').read_text())
        before=[x for x in rows if x['done']<=r['first_incursion_s']]
        pose=np.array([x['truth'] for x in before]);estimate=np.array([x['estimate'] for x in before]);delta=estimate-pose
        lat=delta[:,0]*np.cos(pose[:,2])-delta[:,1]*np.sin(pose[:,2])
        along=delta[:,0]*np.sin(pose[:,2])+delta[:,1]*np.cos(pose[:,2])
        failures.append(dict(key=r['key'],seed=r['seed'],preset=r['preset'],first_incursion_s=r['first_incursion_s'],
            min_clearance_cm=r['continuous_min_clearance_cm'],max_lateral_error_before_cm=float(abs(lat).max()),
            max_longitudinal_error_before_cm=float(abs(along).max()),max_heading_error_before_deg=r['pre_incursion_heading_max_deg'],
            last_pre_incursion=dict(time=before[-1]['done'],lateral_error_cm=float(lat[-1]),longitudinal_error_cm=float(along[-1]),
                                   target_w=before[-1]['target_w'],map_status=before[-1]['status'],sxy=before[-1]['sxy'])))
    dump(REPORT/'diagnosis.json',dict(by_preset=groups,failures=failures,
        conclusion='3 image-arm incursions under sluggish response; upstream errors observed, individual cause not isolated',
        caution='Exact arm uses capture-frame path with36ms age, image arm uses predicted-now pose. Not a single-factor observer ablation.'))
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,key in zip(axs.flat,KEYS):
        for rows,label,color in [(oracle,'Exact capture-frame path','#167f72'),(image,'Image map + known path','#cb572e')]:
            r=next(r for r in rows if tuple(r['key'])==key and r['seed']==SEEDS[0] and r['preset']=='sluggish')
            data=np.load(Path(r['artifact'])/'physics.npz');samples=data['samples']
            ax.plot(np.r_[0,samples[:,0]],data['continuous_clearance'],label=label,color=color)
        ax.axhline(0,color='black',ls='--',lw=1);ax.set(title='_'.join(key),xlabel='Time (s)',ylabel='Minimum body clearance (cm)',ylim=(-.6,4))
        ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8)
    fig.suptitle('hyb4 prerequisite: fixed seed25001, sluggish response | simulation, no CNN training')
    fig.savefig(REPORT/'clearance.png',dpi=140);plt.close(fig)
    print(json.dumps(dict(by_preset=groups,failures=failures),indent=2))


if __name__=='__main__':main()
