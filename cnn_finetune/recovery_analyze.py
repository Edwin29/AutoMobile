"""Audit replay parity, score fixed-horizon outcomes and build valid preference pairs."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from .recovery_diagnosis import CASES, MODES, BURST, HORIZON
from .trajectory_loss import CostConfig, rollout_cost


def analyze(root,out,reference):
    out.mkdir(parents=True,exist_ok=True)
    rows=[];pairs=[];audits=[]
    for case in CASES:
        group=[]
        prefix=None
        initial_state=None
        for mode in MODES:
            folder=root/(case['name']+'__'+mode)
            r=json.loads((folder/'result.json').read_text())
            state=json.loads((folder/'intervention_state.json').read_text())
            if initial_state is None:initial_state=state
            assert state==initial_state,(case,mode,'complete intervention state')
            if prefix is None:prefix=r['prefix_sha256']
            assert r['prefix_sha256']==prefix,(case,mode,'prefix')
            window=json.loads((folder/'window.json').read_text())
            logs=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
            before=[s for s in logs if s['t']<case['start']]
            after=[s for s in logs if s['t']>=case['start']]
            initial_progress=before[-1]['progress']
            stop=min(case['start']+HORIZON,r['simulated_s'])
            # bench advances through the processing interval of its last frame.
            window=[s|{'dt':max(0.,min(s['t'],stop)-max(s['t']-s['dt'],case['start']))} for s in window]
            window=[s for s in window if s['dt']>0]
            observed=sum(s['dt'] for s in window)
            outside=sum(s['dt'] for s in window if s['margin']<0)
            risk=sum(s['dt']*np.clip((1.-s['margin'])/3.,0.,1.) for s in window)
            completed=r['verified_lap']
            failed=r['outcome']=='severe_departure' or (r['outcome']=='lap_complete' and not completed)
            if r['outcome']=='held':failed=True
            progress=after[-1]['progress']-initial_progress if after else 0.
            cost=rollout_cost(horizon_s=HORIZON,observed_s=min(HORIZON,observed),outside_s=outside,
                risk_integral_s=risk,progress_cm=progress,remaining_cm=max(0.,r['lap_cm']-initial_progress),
                failed=failed,completed=completed,end_margin_cm=window[-1]['margin'])
            row=dict(case=case['name'],mode=mode,path=str(folder),outcome=r['outcome'],verified_lap=completed,
                global_min_margin_cm=r['min_margin_cm'],window_min_margin_cm=min(s['margin'] for s in window),
                window_end_margin_cm=window[-1]['margin'],window_outside_s=outside,observed_s=observed,
                local_progress_cm=progress,cost=cost,frames=r['intervention_samples'],prefix_sha256=prefix,
                initial_action=state['initial_action'])
            group.append(row);rows.append(row)
            if mode=='student':
                original=reference/f"{case['condition']}__s{case['seed']}__{case['direction']}__direct_v2"
                old=[json.loads(line) for line in (original/'steps.jsonl').read_text().splitlines()]
                assert len(logs)<=len(old)
                for current,prior in zip(logs,old):
                    for key in ['t','truth','pwm','progress','linear','angular']:
                        assert current[key]==prior[key],(case['name'],key,current['t'])
                audits.append(dict(case=case['name'],baseline_frames_matched=len(logs),prefix_sha256=prefix))
        # Continuous teacher changes the continuation policy and is diagnostic only.
        eligible=[r for r in group if r['mode']!='teacher']
        for a,b in itertools.combinations(eligible,2):
            winner,loser=sorted([a,b],key=lambda r:r['cost']['total'])
            gap=loser['cost']['total']-winner['cost']['total']
            if gap<.05:continue
            assert winner['frames']==loser['frames'] and winner['frames']>0
            pairs.append(dict(case=case['name'],winner=winner['path'],loser=loser['path'],gap=gap,
                winner_cost=winner['cost']['total'],loser_cost=loser['cost']['total'],weight=min(1.,gap),
                prefix_sha256=prefix,split='diagnostic_only'))
    summary=dict(runs=len(rows),cases=len(CASES),rows=rows,audits=audits,preference_pairs=len(pairs),
        protocol='15s fixed-horizon outcome; .72s intervention then frozen-student continuation; full teacher excluded from preferences',
        cost_config=vars(CostConfig()),warning='Three selected failure cases: debugging data, not held-out training validation')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'preferences.json').write_text(json.dumps(pairs,indent=2))
    (out/'protocol.json').write_bytes((root/'protocol.json').read_bytes())
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--reference',default='D:/autodrive_cnn/runs/robustness_20261112');a=p.parse_args()
    analyze(Path(a.root),Path(a.out),Path(a.reference))
