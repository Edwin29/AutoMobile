"""Post-pilot diagnosis; never modify the frozen feasibility controller."""
import argparse
import json
from pathlib import Path
import numpy as np
from bench.world import wrap
from .goal_path_following_pilot import REPORT,OUT,KEYS,SEEDS,PRESETS,PATH_REPORT
from .goal_clearance_paths import ReserveTeacher
from .goal_continuous_clearance import continuous_clearance
from .goal_development import sha
from .junction_stage02 import dump


def privileged():
    spec=REPORT/'privileged_diagnostic_protocol.json';assert not spec.exists()
    dump(spec,dict(source_sha256=sha(__file__),purpose='Separate reference feasibility from fixed nominal motor compensation after the nominal controls failed',
        cases='same24nominal cases as pilot, all retained',teacher='unchanged ReserveTeacher,same_mode; true pose,velocity and actual preset; offline diagnostic only',
        selection='No new controller tuning; known earlier teacher method',final_exam_used=False))
    registry={tuple(r['key']):r for r in json.loads((PATH_REPORT/'geometry.json').read_text())}
    rows=[]
    for seed in SEEDS:
        for key in KEYS:
            for preset in PRESETS:
                env=ReserveTeacher(key,'normal',seed,preset,'same_mode',registry).finish()
                trace=np.asarray(env.physics);m=continuous_clearance(np.vstack([env.initial_pose,trace[:,1:4]]),env.p,env.track)
                row=env.summary();row.update(continuous_min_clearance_cm=float(m.min()),continuous_clean=bool(row['stable_success'] and m.min()>=-1e-9))
                dst=OUT/'privileged_teacher'/('_'.join(key)+f'_{seed}_{preset}_normal');dst.mkdir(parents=True)
                np.savez_compressed(dst/'physics.npz',samples=trace,initial=env.initial_pose,continuous_clearance=m)
                dump(dst/'complete.json',row);rows.append(dict(row,artifact=str(dst)))
    reference=json.loads((REPORT/'truth_now_normal.json').read_text())
    for a,b in zip(rows,reference):assert a['initial_sampling']==b['initial_sampling'] and a['key']==b['key']
    dump(REPORT/'privileged_teacher_normal.json',rows)
    print(json.dumps(dict(n=len(rows),clean=sum(r['continuous_clean'] for r in rows),stable=sum(r['stable_success'] for r in rows),
                         min_clearance=min(r['continuous_min_clearance_cm'] for r in rows))),flush=True)


def analyze():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arms=['privileged_teacher','truth_now','truth_delayed','vision']
    data={a:json.loads((REPORT/f'{a}_normal.json').read_text()) for a in arms}
    table={};failure_rows=[]
    for a,rows in data.items():
        table[a]={}
        for key in KEYS:
            group=[r for r in rows if tuple(r['key'])==key]
            table[a]['_'.join(key)]=dict(n=len(group),clean=sum(r['continuous_clean'] for r in group),stable=sum(r['stable_success'] for r in group),
                worst_clearance_cm=min(r['continuous_min_clearance_cm'] for r in group))
    for row in data['vision']:
        rec=json.loads((Path(row['artifact'])/'records.json').read_text())
        # Errors at/before first incursion only, never infer cause from post-failure pose divergence.
        first=row['first_incursion_s'];prefix=[x for x in rec if first is None or x['done']<=first]
        offsets=np.array([np.asarray(x['estimate'])-np.asarray(x['truth']) for x in prefix])
        true=np.array([x['truth'] for x in prefix]);lat=offsets[:,0]*np.cos(true[:,2])-offsets[:,1]*np.sin(true[:,2])
        along=offsets[:,0]*np.sin(true[:,2])+offsets[:,1]*np.cos(true[:,2])
        failure_rows.append(dict(key=row['key'],seed=row['seed'],preset=row['preset'],clean=row['continuous_clean'],outcome=row['outcome'],
            first_incursion_s=first,pre_incursion_abs_lateral_max_cm=float(abs(lat).max()),pre_incursion_abs_longitudinal_max_cm=float(abs(along).max()),
            pre_incursion_heading_max_deg=row['pre_incursion_heading_max_deg'],matched_frame_fraction=row['matched_frame_fraction']))
    dump(REPORT/'diagnosis.json',dict(by_movement=table,vision_before_incursion=failure_rows,new_training=False,final_exam_used=False))
    fig,axes=plt.subplots(2,4,figsize=(16,8))
    colors={'privileged_teacher':'#26956b','truth_now':'#3377bb','truth_delayed':'#9d65c9','vision':'#df6538'}
    registry={tuple(r['key']):r for r in json.loads((PATH_REPORT/'geometry.json').read_text())}
    for j,key in enumerate(KEYS):
        reference=np.load(registry[key]['reference'])['points'];ax=axes[0,j]
        ax.plot(reference[:,0],reference[:,1],color='black',linestyle='--',label='Known path')
        for a in arms:
            row=next(r for r in data[a] if tuple(r['key'])==key and r['seed']==SEEDS[0] and r['preset']=='field_estimate')
            trace=np.load(Path(row['artifact'])/'physics.npz')['samples']
            ax.plot(trace[:,1],trace[:,2],label=a,color=colors[a],alpha=.8)
            if a=='vision':
                rec=json.loads((Path(row['artifact'])/'records.json').read_text());est=np.array([x['estimate'] for x in rec]);truth=np.array([x['truth'] for x in rec]);t=np.array([x['done'] for x in rec])
                axes[1,j].plot(t,np.linalg.norm(est[:,:2]-truth[:,:2],axis=1),color=colors[a],label='Position error (cm)')
                if row['first_incursion_s'] is not None:axes[1,j].axvline(row['first_incursion_s'],color='red',linestyle='--',label='First body incursion')
        ax.set_title('_'.join(key));ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('World x (cm)');ax.set_ylabel('World y (cm)');ax.grid(alpha=.2)
        axes[1,j].set_xlabel('Time (s)');axes[1,j].set_ylabel('Position error (cm)');axes[1,j].grid(alpha=.2)
    axes[0,0].legend(fontsize=7);axes[1,0].legend(fontsize=7)
    fig.suptitle('Fixed examples: seed22001, field_estimate; all four predeclared movements')
    fig.tight_layout();fig.savefig(REPORT/'pilot_diagnosis.png',dpi=140);plt.close(fig)
    print(json.dumps(table),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['privileged','analyze']);a=p.parse_args();globals()[a.task]()
