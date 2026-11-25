"""Isolate planner policies on identical saved pose beliefs; never actuates."""
import argparse,copy,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .estimation import Belief
from .map_model import TrackMap
from .planning import LocalPlanner
from .settings import load_settings
from .camera import sha256_file
from .provenance import code_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rows=[json.loads(s) for s in (args.input/'steps.jsonl').read_text().splitlines()]
    if any(r['estimate'].get('status')!='observed' for r in rows):
        raise ValueError('This frozen-belief diagnostic requires observed timestamp on every frame')
    cfg,_=load_settings();track=TrackMap();route=track.outer_route('ccw');summaries={}
    for name,mode,distance in [('legacy_grid','legacy_full','grid_lower_bound'),
            ('exact_full','legacy_full','exact'),('bounded_prefix','bounded_prefix','exact')]:
        pcfg=copy.deepcopy(cfg['planner']);pcfg.update(acceptance_mode=mode,distance_mode=distance)
        planner=LocalPlanner(track,route,pcfg,cfg['vehicle']);results=[]
        for row in rows:
            est=SimpleNamespace(hypotheses=[Belief(np.array(b['pose']),np.array(b['P']),last_observed_s=row['capture_s']) for b in row['beliefs']],motion_distance=0.)
            begin=time.perf_counter()
            result=planner.plan(est,None,row['now_s'],row['clock'],row['mission']['hold'],row['mission']['speed_cap'])
            results.append(dict(index=row['index'],reason=result.reason,plan_ms=(time.perf_counter()-begin)*1000,
                speed_cm_s=result.speed_limit_cm_s,diagnostics=copy.deepcopy(planner.last_diagnostics)))
        (args.output/f'{name}.json').write_text(json.dumps(results,indent=2))
        counts={}
        for r in results:counts[r['reason']]=counts.get(r['reason'],0)+1
        summaries[name]=dict(counts=counts,warm_plan_ms=dict(zip(['p50','p95'],map(float,np.percentile([r['plan_ms'] for r in results[3:]],[50,95])))))
    report=dict(note='Frozen same observed beliefs and timestamps; no image re-estimation and no movement. Not a closed-loop test.',
        source_sha256=sha256_file(args.input/'steps.jsonl'),code=code_manifest(),results=summaries)
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
