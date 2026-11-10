"""Read-only counterfactual planner decisions on frozen real logged beliefs.

No virtual command is allowed to change the subsequent recorded camera/pose.
Uses original per-run configuration for old-vs-new algorithm comparison. This
does NOT show that newly commanded trajectories would have succeeded physically.
"""
import argparse,copy,json,hashlib,time
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from .map_model import TrackMap,HERE
from .planning import LocalPlanner
from .recovery_planning import RecoveryPlanner
from .control import Controller,ResponseModel
from .recovery_control import RollingController
from .estimation import Belief
from .camera import CameraGeometry
from .observation import Observer
from .contracts import FrameStamp


def run(root,output):
    cv2.setNumThreads(1);root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    track=TrackMap();route=track.outer_route('ccw');summaries={}
    for folder in sorted(root.glob('live_*')):
        manifest=json.loads((folder/'manifest.json').read_text());cfg=manifest['settings']
        rows=[json.loads(x) for x in (folder/'steps.jsonl').read_text().splitlines()]
        cached=output/f'{folder.name}.json'
        if cached.exists():
            meta=json.loads((output/f'{folder.name}_input.json').read_text())
            if meta['log_sha256']!=hashlib.sha256((folder/'steps.jsonl').read_bytes()).hexdigest():raise ValueError('Input changed')
            results=json.loads(cached.read_text());summary={}
            for policy in ['baseline','recovery']:
                counts={}
                for r in results:counts[r[policy]['control']]=counts.get(r[policy]['control'],0)+1
                summary[policy]=dict(nonzero_proposed_frames=sum(max(r[policy]['pwm'])>1e-8 for r in results),frames=len(results),reasons=counts)
            summaries[folder.name]=summary;continue
        models=[]
        for policy in ['baseline','recovery']:
            conf=copy.deepcopy(cfg);model=ResponseModel(conf['controller'],conf['vehicle'])
            if policy=='baseline':
                controller=Controller(model,conf['controller'],'pure_pursuit')
                planner=LocalPlanner(track,route,conf['planner'],conf['vehicle'])
            else:
                controller=RollingController(model,conf['controller'])
                conf['planner'].update(rollout_s=4.,rollout_dt_s=.2,hard_road_depth_cm=6.,coast_s=1.2)
                planner=RecoveryPlanner(track,route,conf['planner'],conf['vehicle'],controller)
            models.append((policy,conf,model,controller,planner))
        camera=CameraGeometry(HERE/'config/camera_legacy_track.json');observer=Observer(camera,HERE/'config/observation.json')
        observer.mark_cfg=cfg['marks'];results=[]
        previous=rows[0]['now_s']-.2;applied=[0.,0.];last_observed=-1e9
        for r in rows:
            if r['estimate'].get('status')=='observed':last_observed=r['capture_s']
            # Same frozen global belief for both. Additional startup fallback is
            # evaluated on fresh re-extracted images only; not claimed independent truth.
            observation=None
            if r['estimate'].get('status')!='observed':
                image=cv2.imread(str(folder/f"{r['index']:04d}_raw.jpg"))
                if image is not None:
                    observation,_=observer.process(image,FrameStamp(r['index'],r['capture_s'],r['clock']),rectified=False,rotate180=True)
            est=SimpleNamespace(hypotheses=[Belief(np.array(b['pose']),np.array(b['P']),last_observed_s=last_observed) for b in r['beliefs']],motion_distance=0.)
            est.primary=est.hypotheses[0]
            rec=dict(index=r['index'],phase=r['phase'],recorded_reason=r['control_reason'],recorded_actual_pwm=r['actual_pwm'])
            for policy,conf,model,controller,planner in models:
                controller.pwm=np.array(applied);controller.speed=model.forward_model(applied)[0]
                begin=time.perf_counter()
                path=planner.plan(est,observation,r['now_s'],r['clock'],r['mission']['hold'],r['mission']['speed_cap'])
                out=controller.step(path,r['now_s'],r['now_s']-previous,r['clock'],r['estimate'].get('status')=='observed')
                rec[policy]=dict(path=path.reason,control=out.reason,pwm=out.pwm.tolist(),ms=(time.perf_counter()-begin)*1000,
                    nominal_min_cm=planner.last_diagnostics.get('nominal_min_cm'),source=planner.last_diagnostics.get('source'))
            results.append(rec);previous=r['now_s'];applied=r['actual_pwm']
        summary={}
        for policy,*_ in models:
            counters={}
            for r in results:
                name=r[policy]['control'];counters[name]=counters.get(name,0)+1
            summary[policy]=dict(nonzero_proposed_frames=sum(max(r[policy]['pwm'])>1e-8 for r in results),
                frames=len(results),reasons=counters)
        summaries[folder.name]=summary
        (output/f'{folder.name}.json').write_text(json.dumps(results,indent=2))
        (output/f'{folder.name}_input.json').write_text(json.dumps(dict(settings=cfg,
            log_sha256=hashlib.sha256((folder/'steps.jsonl').read_bytes()).hexdigest(),
            image_note='JPEG raw, rotate180, calibrated remap; fallback only on non-observed frames'),indent=2))
    report=dict(results=summaries,limit='Frozen logged poses, actual PWM history and historical images. New commands not fed to future states. No physical corner/recovery ground truth.')
    (output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.input,a.output)
