"""Post-run input substitution; identical receipts, no counterfactual rollout."""
import json
from pathlib import Path
import numpy as np
from .control import PathTracker
from .pilot import oracle_packet,REFERENCES,sha
from .image_pilot import REPORT
from cnn_finetune.goal_clearance_paths import ReserveTeacher
from cnn_finetune.junction_stage02 import dump


def main():
    registry={tuple(r['key']):r for r in json.loads((REFERENCES/'geometry.json').read_text())}
    results=[]
    for row in json.loads((REPORT/'runs.json').read_text()):
        if row['continuous_clean']:continue
        env=ReserveTeacher(tuple(row['key']),'normal',row['seed'],row['preset'],'same_mode',registry)
        control=PathTracker();comparisons=[]
        for r in json.loads((Path(row['artifact'])/'records.json').read_text()):
            if r['done']>row['first_incursion_s']:break
            packet=oracle_packet(env.points,env.s,env.speed,np.array(r['truth']),r['done'])
            _,info=control.desired(packet,r['done']);control.acknowledge(r['done'],r['applied'])
            if r['done']>=row['first_incursion_s']-.5:
                comparisons.append(dict(t=r['done'],image_target_w=r['target_w'],exact_current_pose_target_w=info['target_w'],
                    image_k=r['curvature'],exact_current_pose_k=info['curvature']))
        results.append(dict(key=row['key'],seed=row['seed'],last=comparisons[-1],window=comparisons))
    dump(REPORT/'same_history_input_substitution.json',dict(source_sha256=sha(__file__),
        scope='Offline single-decision input substitution on identical actual PWM histories; not a rollout or guaranteed recovery',rows=results))
    print('same-history probe saved',len(results),'cases')


if __name__=='__main__':main()
