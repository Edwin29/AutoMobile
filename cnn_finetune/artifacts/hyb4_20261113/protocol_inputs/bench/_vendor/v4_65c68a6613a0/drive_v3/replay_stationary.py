"""Compare boundary weighting on the same saved stationary camera sequence.

Input: steps.jsonl plus NNN_raw.png for every row, from a no-GPIO probe.
No hardware imports. Both modes predict strictly zero applied movement.
"""
import argparse
import copy
import json
from pathlib import Path
import cv2
import numpy as np
from .runtime import Runtime, RecordingSink
from .settings import load_settings
from .map_model import TrackMap
from .camera import sha256_file
from .provenance import code_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1)
    rows=[json.loads(line) for line in (args.input/'steps.jsonl').read_text().splitlines()]
    if any(r.get('actual_pwm')!=[0,0] or r.get('gpio_enabled') is not False for r in rows):
        raise ValueError('Only declared stationary no-GPIO captures accepted')
    pose=json.loads((args.input/'manifest.json').read_text())['initial_pose']
    images=[cv2.imread(str(args.input/f"{r['index']:03d}_raw.png")) for r in rows]
    if any(image is None for image in images):raise ValueError('Every original frame is required')
    inputs={f"{r['index']:03d}_raw.png":sha256_file(args.input/f"{r['index']:03d}_raw.png") for r in rows}
    cfg,_=load_settings();route=TrackMap().outer_route('ccw');summaries={}
    for name,enabled in [('equal_fragment',False),('observed_length',True)]:
        settings=copy.deepcopy(cfg);settings['estimator']['length_weighted_boundaries']=enabled
        records=[];applied=[]
        with Runtime(pose,route,settings,start_s=0.,clock_id=rows[0]['clock'],sink=RecordingSink()) as rt:
            original=rt.timeline.command
            def command(now,v,w):
                if v!=0 or w!=0:raise AssertionError('Hypothetical movement entered stationary prediction')
                applied.append([v,w]);return original(now,v,w)
            rt.timeline.command=command
            for image,row in zip(images,rows):
                record,_,_,_=rt.step(image,row['capture_s'],row['now_s'],rotate180=True,shadow_pwm=[0,0])
                records.append(record)
        (args.output/f'{name}.jsonl').write_text('\n'.join(json.dumps(r,default=lambda a:a.tolist()) for r in records)+'\n')
        a=np.array([r['pose'] for r in records]);a[:,2]=np.unwrap(a[:,2]);tail=a[len(a)//2:]
        summary=dict(frames=len(records),observed=sum(r['estimate']['status']=='observed' for r in records),
            tail_half_span_x_y_cm=np.ptp(tail[:,:2],axis=0).tolist(),tail_half_span_yaw_deg=float(np.rad2deg(np.ptp(tail[:,2]))),
            last_pose=records[-1]['pose'],path_counts={},prediction_commands_all_zero=bool(np.all(np.array(applied)==0)))
        for r in records:summary['path_counts'][r['path_reason']]=summary['path_counts'].get(r['path_reason'],0)+1
        summaries[name]=summary
    report=dict(note='Same captured stationary frames/timestamps and initial pose. Pose variation is not absolute pose accuracy or driving performance.',
        steps_sha256=sha256_file(args.input/'steps.jsonl'),inputs=inputs,code=code_manifest(),results=summaries)
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
