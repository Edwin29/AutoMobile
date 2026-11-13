"""Aggregate paired frozen-model tests without tuning or censoring failures away."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ARMS=['a5b_v4','direct_v2']

def median(v):return float(np.median(v)) if v else None
def clean(r):return bool(r['verified_lap'] and r['min_margin_cm']>=0)
def stats(rows):
    laps=[r for r in rows if r['verified_lap']]
    recovery=[x for r in laps for x in r['extra']['post_corner_recovery_s']]
    return dict(n=len(rows),laps=len(laps),clean=sum(clean(r) for r in rows),
                outcomes=dict(collections.Counter(r['outcome'] for r in rows)),
                median_margin_cm=median([r['min_margin_cm'] for r in rows]),
                worst_margin_cm=min((r['min_margin_cm'] for r in rows),default=None),
                sub_mm_intrusion_runs=sum(-.1<r['min_margin_cm']<0 for r in rows),
                mean_outside_s=float(np.mean([r['outside_s'] for r in rows])) if rows else None,
                successful_lap_recovery_event_median_s=median(recovery),successful_lap_recovery_events=len(recovery),
                unrecovered_heading_at_end=sum(r['extra']['unrecovered_at_end'] for r in rows),
                unrecovered_departures=sum(r['unrecovered_episodes'] for r in rows),
                median_corner_low_pwm_fraction=median([r['extra']['inner_below_floor_fraction'] for r in rows if r['extra']['corner_frames']]))

def analyze(root,out,partial=False):
    protocol=json.loads((root/'protocol.json').read_text());specs=json.loads((root/'specs.json').read_text())
    rows=[];missing=[];by_pair={};pair_checks=[]
    for sp in specs:
        slug=f"{sp['condition']}__s{sp['seed']}__{sp['direction']}__{sp['arm']}"
        p=root/slug/'result.json'
        if not p.exists():missing.append(slug);continue
        r=json.loads(p.read_text());assert r['spec']==sp
        r['path']=str(p.parent);rows.append(r)
        key=(sp['condition'],sp['seed'],sp['direction']);by_pair.setdefault(key,{})[sp['arm']]=r
    if not partial:assert not missing, f'{len(missing)} missing runs'
    pairs=[]
    for key,arms in by_pair.items():
        if len(arms)!=2:continue
        a,b=[arms[k] for k in ARMS]
        for field in ['camera','vehicle','variation','floor_patches']:
            # Device belongs to camera but identical on this workstation.
            assert a[field]==b[field],(key,field)
        ae=json.loads((Path(a['path'])/'external_events.json').read_text())
        be=json.loads((Path(b['path'])/'external_events.json').read_text())
        assert ae['scheduled']==be['scheduled'],key
        checks=[]
        for r,e in [(a,ae),(b,be)]:
            expected=sum(x['lateral_cm']*max(0,min(r['simulated_s']-x['t0'],x['duration']))/x['duration'] for x in e['scheduled'] if x['kind']=='push')
            observed=sum(x['lateral_cm'] for x in e['applied_push'])
            assert abs(expected-observed)<1e-5,(key,expected,observed)
            checks.append(dict(arm=r['spec']['arm'],push_integral_error=abs(expected-observed),
                               push_events_started=sum(x['kind']=='push' and x['t0']<r['simulated_s'] for x in e['scheduled']),
                               power_events_started=sum(x['kind']=='power' and x['t0']<r['simulated_s'] for x in e['scheduled'])))
        pair_checks.append(dict(condition=key[0],seed=key[1],direction=key[2],checks=checks))
        am=median(a['extra']['post_corner_recovery_s']);bm=median(b['extra']['post_corner_recovery_s'])
        pairs.append(dict(condition=key[0],group=a['spec']['group'],seed=key[1],direction=key[2],
                          baseline_lap=a['verified_lap'],new_lap=b['verified_lap'],baseline_clean=clean(a),new_clean=clean(b),
                          baseline_margin_cm=a['min_margin_cm'],new_margin_cm=b['min_margin_cm'],margin_delta_cm=b['min_margin_cm']-a['min_margin_cm'],
                          outside_delta_s=b['outside_s']-a['outside_s'],
                          paired_success_recovery_delta_s=bm-am if a['verified_lap'] and b['verified_lap'] and am is not None and bm is not None and not a['extra']['unrecovered_at_end'] and not b['extra']['unrecovered_at_end'] else None))
    def paired_stats(ps):
        return dict(n=len(ps),clean_wins=sum(p['new_clean'] and not p['baseline_clean'] for p in ps),
                    clean_losses=sum(p['baseline_clean'] and not p['new_clean'] for p in ps),
                    lap_wins=sum(p['new_lap'] and not p['baseline_lap'] for p in ps),
                    lap_losses=sum(p['baseline_lap'] and not p['new_lap'] for p in ps),
                    margin_delta_median_cm=median([p['margin_delta_cm'] for p in ps]),
                    margin_regressions_over_0_5cm=sum(p['margin_delta_cm']<-.5 for p in ps),
                    paired_success_recovery_delta_median_s=median([p['paired_success_recovery_delta_s'] for p in ps if p['paired_success_recovery_delta_s'] is not None]),
                    paired_success_recovery_n=sum(p['paired_success_recovery_delta_s'] is not None for p in ps))
    result=dict(completed=len(rows),expected=len(specs),missing=missing,overall={a:stats([r for r in rows if r['spec']['arm']==a]) for a in ARMS},paired=paired_stats(pairs),groups={},conditions={},seeds={})
    for field,target in [('group','groups'),('condition','conditions'),('seed','seeds')]:
        for value in dict.fromkeys(s[field] for s in specs):
            rs=[r for r in rows if r['spec'][field]==value];ps=[p for p in pairs if p[field]==value]
            result[target][value]={a:stats([r for r in rs if r['spec']['arm']==a]) for a in ARMS}|{'paired':paired_stats(ps)}
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    (out/'pairs.json').write_text(json.dumps(pairs,indent=2))
    (out/'pair_checks.json').write_text(json.dumps(pair_checks,indent=2))
    if pairs:
        with (out/'pairs.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out',required=True);ap.add_argument('--partial',action='store_true');a=ap.parse_args()
    analyze(Path(a.root),Path(a.out),a.partial)
