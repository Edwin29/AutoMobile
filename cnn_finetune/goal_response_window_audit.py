"""Replay teacher failures and locate the first physically possible response.

This is a forensic timing audit, never an online control feature. It separates
failure before a new observation can affect wheels from later controller error.
"""
import json
from pathlib import Path
import numpy as np
from .goal_teacher_coverage import CoverageTeacher,ROOT
from .goal_development import sha
from .junction_stage02 import dump

REPORT=ROOT/'cnn_finetune/results/goal_teacher_coverage_20261113'


def main():
    original=json.loads((REPORT/'coverage.json').read_text())['runs']
    cases=[r for r in original if r['variant']=='centered' and r['condition']=='mild' and not r['clean_stable']]
    results=[]
    for i,r in enumerate(cases):
        env=CoverageTeacher(tuple(r['key']),'mild',r['seed'],r['preset'],'centered')
        active=env.car._active;timing=[]
        def record(t,delay):
            timing.append((t,delay,t-delay));return active(t,delay)
        env.car._active=record;env.finish()
        stored=np.load(Path(r['artifact'])/'physics.npz')['samples'];np.testing.assert_array_equal(np.asarray(env.physics),stored)
        bad=np.flatnonzero(stored[:,6]<0);first=None if not len(bad) else float(stored[bad[0],0])
        preceding=[e for e in env.events if e['kind']=='push' and first is not None and e['t0']<=first]
        if not preceding:
            results.append(dict(key=r['key'],seed=r['seed'],preset=r['preset'],first_incursion_s=first,before_any_push=True));continue
        event=max(preceding,key=lambda e:e['t0'])
        detection=next(x for x in env.records if x['capture']>=event['t0'])
        stamp=detection['done']
        physical=next((x for x in timing if x[2]>=stamp),None)
        earliest=None if physical is None else physical[0]
        # At this start time the first altered command could influence the NEXT
        # integration sample. An incursion already sampled by then is too early.
        results.append(dict(key=r['key'],seed=r['seed'],preset=r['preset'],first_incursion_s=first,
            before_any_push=False,push_start_s=event['t0'],first_capture_s=detection['capture'],first_command_s=stamp,
            earliest_actuation_step_start_s=earliest,response_latency_s=None if earliest is None else earliest-event['t0'],
            incursion_before_possible_response=earliest is not None and first<=earliest+1e-10,
            margin_at_push=float(stored[np.searchsorted(stored[:,0],event['t0'])-1,6])))
        if (i+1)%20==0:print('response-window audit',i+1,'/',len(cases),flush=True)
    pushed=[r for r in results if not r['before_any_push']]
    summary=dict(failed_cases=len(cases),exact_replays=len(cases),before_push=sum(r['before_any_push'] for r in results),
        after_push=len(pushed),before_possible_response=sum(r['incursion_before_possible_response'] for r in pushed),
        latency_range_s=[min(r['response_latency_s'] for r in pushed),max(r['response_latency_s'] for r in pushed)])
    dump(REPORT/'response_window_audit.json',dict(source_sha256=sha(__file__),summary=summary,results=results,
        interpretation='Unavoidable from the realized onset state before new camera-driven actuation, not proof that a different earlier trajectory or the task is infeasible.'))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
