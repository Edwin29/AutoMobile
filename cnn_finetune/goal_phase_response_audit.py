"""Forensic response-window audit of phase-entry failures; no online inputs."""
import json
from pathlib import Path
import numpy as np
from bench.world import Track
from bench.vehicle import PRESETS
from .goal_phase_disturbance import PhaseTeacher,REPORT,registry
from .goal_continuous_clearance import continuous_clearance
from .goal_development import sha
from .junction_stage02 import dump


def main():
    cases=[]
    for arm in ['baseline','reserve']:
        cases += [r for r in json.loads((REPORT/(arm+'.json')).read_text()) if r['phase']=='entry' and not r['continuous_clean']]
    rows=[];reg=registry();track=Track()
    for r in cases:
        env=PhaseTeacher(tuple(r['key']),r['seed'],r['preset'],r['arm'],'entry',reg)
        active=env.car._active;timing=[]
        def record(t,delay):
            timing.append((t,delay,t-delay));return active(t,delay)
        env.car._active=record;env.finish()
        stored=np.load(Path(r['artifact'])/'physics.npz')['samples'];np.testing.assert_array_equal(env.physics,stored)
        margin=continuous_clearance(stored[:,1:4],PRESETS[r['preset']],track)
        i=np.flatnonzero(margin<0)[0];first=float(stored[i,0]);onset=env.trigger['time']
        # Capture exactly at onset contains no shove motion yet. Using the next
        # truth-pose observation is optimistic versus delayed visual estimation.
        detection=next(x for x in env.records if x['capture']>onset+1e-10)
        stamp=detection['done'];earliest=next(x[0] for x in timing if x[2]>=stamp)
        j=max(0,np.searchsorted(stored[:,0],onset)-1)
        rows.append(dict(arm=r['arm'],key=r['key'],seed=r['seed'],preset=r['preset'],
            push_start_s=onset,first_incursion_s=first,first_capture_s=detection['capture'],
            first_command_s=stamp,earliest_actuation_step_start_s=earliest,
            before_push=first<=onset,incursion_before_possible_response=first<=earliest+1e-10,
            response_latency_s=earliest-onset,onset_margin_cm=float(margin[j]),events=env.events))
        if len(rows)%40==0:print('entry response audit',len(rows),'/',len(cases),flush=True)
    summary={}
    for arm in ['baseline','reserve']:
        a=[r for r in rows if r['arm']==arm]
        summary[arm]=dict(failed_entry_cases=len(a),exact_replays=len(a),before_push=sum(r['before_push'] for r in a),
            before_possible_response=sum(r['incursion_before_possible_response'] for r in a),
            latency_range_s=[min(r['response_latency_s'] for r in a),max(r['response_latency_s'] for r in a)])
    dump(REPORT/'entry_response_audit.json',dict(source_sha256=sha(__file__),summary=summary,rows=rows,
        interpretation='Conditional impossibility of reactive correction from the realized onset state before any new command can affect wheels. Not proof of global infeasibility: different approach state/speed/preemptive control could change the result. Later failures are not automatically recoverable.',
        observer='optimistic current truth-pose capture, no additional visual perception delay; diagnostic only, no actor input change'))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
