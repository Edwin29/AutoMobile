"""Create a compact, non-actuating report from an existing trial log and manifest."""
import argparse,collections,hashlib,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('steps',type=Path);p.add_argument('manifest',type=Path)
    p.add_argument('--run-id',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    rows=[json.loads(s) for s in a.steps.read_text().splitlines()]
    rows=[r for r in rows if 'sequence' in r]
    manifest=json.loads(a.manifest.read_text())
    drive=[r for r in rows if r.get('phase')=='drive']
    powered=[r for r in drive if max(r.get('actual_pwm',[0,0]))>0]
    values=[v for r in powered for v in r['actual_pwm'] if v>0]
    report=dict(run_id=a.run_id,source_steps_sha256=hashlib.sha256(a.steps.read_bytes()).hexdigest(),
        source_code=manifest['code'],settings=manifest['settings'],frames=len(rows),drive_frames=len(drive),
        positive_pwm_frames=len(powered),path_reason_counts=dict(collections.Counter(r['path_reason'] for r in drive)),
        estimate_status_counts=dict(collections.Counter(r['estimate']['status'] for r in rows)),
        positive_command_min_duty=min(values) if values else None,
        positive_command_max_duty=max(values) if values else None,
        processing_ms=dict(zip(['median','p95','max'],map(float,np.percentile([r['total_ms'] for r in rows[3:]],[50,95,100])))),
        last_command=rows[-1]['actual_pwm'],model_route_progress_cm=rows[-1].get('estimated_lap_progress_cm'),
        limits=['Progress is PWM-predicted, not measured travel or lap ground truth.',
            'For local mode, topology transitions and map-shape path alignment were not implemented.',
            'Physical outcome and later code changes must be checked in the dated research record.'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_bytes((json.dumps(report,indent=2,ensure_ascii=False)+'\n').encode('utf-8'))
    print(json.dumps({k:report[k] for k in ('frames','positive_pwm_frames','path_reason_counts','processing_ms','last_command')}))


if __name__=='__main__':main()
