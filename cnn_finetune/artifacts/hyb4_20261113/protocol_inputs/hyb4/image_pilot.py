"""Image-map controlled known-path gate; still no learned trajectory policy."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from .control import PathTracker
from .pilot import KEYS,SEEDS,PRESETS,ROOT,REFERENCES,sha,oracle_packet,REPORT as ORACLE_REPORT
from cnn_finetune.goal_clearance_paths import ReserveTeacher
from cnn_finetune.goal_continuous_clearance import continuous_clearance
from cnn_finetune.goal_mission_map import DelayedImageMap
from cnn_finetune.junction_learning import setup
from cnn_finetune.junction_stage02 import dump
from bench.world import Camera,Renderer,wrap
from bench.realcam import RealCam,pose_from_history
from bench.junction import heading_of

REPORT=ROOT/'cnn_finetune/results/hyb4_image_prerequisite_20261113'
OUT=Path('D:/autodrive_cnn/runs/hyb4_image_prerequisite_20261113')


def freeze():
    assert json.loads((ORACLE_REPORT/'summary.json').read_text())['gate_pass']
    REPORT.mkdir(exist_ok=False);OUT.mkdir(parents=True,exist_ok=False)
    files=set(Path(f) for f in json.loads((ORACLE_REPORT/'protocol.json').read_text())['hashes'])
    files.add(Path(__file__))
    for module in list(sys.modules.values()):
        f=getattr(module,'__file__',None)
        if f and f.endswith('.py') and Path(f).is_file() and Path(f).resolve().is_relative_to(ROOT):files.add(Path(f).resolve())
    files.update(p for p in (ROOT/'bench/_vendor/v4_65c68a6613a0').rglob('*') if p.suffix in ['.py','.json','.npz'])
    dump(REPORT/'protocol.json',dict(keys=KEYS,seeds=SEEDS,presets=PRESETS,condition='normal',n=24,
        hashes={str(f):sha(f) for f in sorted(files)},scope='Existing image map + known path; not CNN or full mission',
        gate='>=22/24 continuous-clean,24/24 stable,0wrong/severe,0 unnecessary pauses; otherwise stop training expansion and report',
        observer='pinned HSV map,70ms publication every3frames,nominal unperturbed path-start hint; signed actual PWM receipts',
        learning='yaw and floor auto-learning disabled in both map copies for entire experiment; signed-domain learner validity unverified',
        controller='same hyb4/control.py as oracle gate; path projected using estimated NOW pose,packet stamp NOW,not stale capture pose',
        uncertainty='known path and start hint available only in this diagnostic,not inputs to proposed CNN',
        final_exam_used=False,new_training=False))
    print('image protocol frozen',flush=True)


def run_one(key,seed,preset,registry):
    env=ReserveTeacher(key,'normal',seed,preset,'same_mode',registry);controller=PathTracker()
    hint=np.r_[env.points[0],heading_of(env.di)];observer=DelayedImageMap('cw',hint);observer.set_route(env.points)
    for d in [observer.visible,observer.working]:
        d.learner.c['enabled']=False;d.floor_learner.c['enabled']=False
    cam=Camera();camera=RealCam(Renderer(env.track,cam),cam,env.profile,seed);records=[]
    while env.capture<40 and env.outcome=='time_limit':
        env.advance_to(env.capture)
        if env.outcome!='time_limit':break
        capture=env.capture;done=capture+.036
        image=camera.frame(capture,pose_from_history(env.history),raw=True)
        env.advance_to(done)
        if env.outcome!='time_limit':break
        estimate=observer.observe(image,capture,done);pose=estimate['pose']
        # This projector accepts a pose; unlike oracle pilot it receives ONLY
        # the delayed image-map estimate, never env.car.pose or actual response.
        packet=oracle_packet(env.points,env.s,env.speed,pose,done)
        desired,info=controller.desired(packet,done);applied=controller.drive.allocate(desired,.036)
        env.car.command(done,applied);controller.acknowledge(done,applied);observer.on_applied(done,applied)
        records.append(dict(capture=capture,done=done,desired=desired,applied=applied,estimate=pose,
            truth=env.car.pose.copy(),sxy=estimate['sxy'],map_capture=estimate['map_capture'],
            status=observer.visible.map.status,**info))
        env.records.append(dict(capture=capture,done=done,action=desired,applied=applied))
        env.previous=applied.copy();env.capture=done
    physics=np.asarray(env.physics)
    clearance=continuous_clearance(np.vstack([env.initial_pose,physics[:,1:4]]),env.p,env.track)
    result=env.summary();estimated=np.asarray([r['estimate'] for r in records]);truth=np.asarray([r['truth'] for r in records])
    xy=np.linalg.norm(estimated[:,:2]-truth[:,:2],axis=1);yaw=abs(np.degrees(wrap(estimated[:,2]-truth[:,2])))
    result.update(continuous_min_clearance_cm=float(clearance.min()),
        continuous_clean=bool(result['stable_success'] and result['initial_valid'] and clearance.min()>=-1e-9),
        position_error_p95_cm=float(np.percentile(xy,95)),heading_error_p95_deg=float(np.percentile(yaw,95)),
        oracle_path=False,known_global_path=True,controller_true_preset=False)
    first=np.where(clearance<0)[0];first_t=None if not len(first) else float(0 if first[0]==0 else physics[first[0]-1,0])
    before=np.array([r['done'] for r in records])<=(first_t if first_t is not None else env.t)
    result.update(first_incursion_s=first_t,pre_incursion_position_max_cm=float(xy[before].max()) if before.any() else None,
        pre_incursion_heading_max_deg=float(yaw[before].max()) if before.any() else None)
    dst=OUT/('_'.join(key)+f'_{seed}_{preset}');dst.mkdir(exist_ok=False)
    np.savez_compressed(dst/'physics.npz',samples=physics,initial=env.initial_pose,continuous_clearance=clearance)
    dump(dst/'records.json',records);dump(dst/'events.json',env.events)
    result['artifact']=str(dst);dump(dst/'complete.json',result);return result


def run():
    setup();protocol=json.loads((REPORT/'protocol.json').read_text())
    for file,value in protocol['hashes'].items():assert sha(file)==value,file
    registry={tuple(r['key']):r for r in json.loads((REFERENCES/'geometry.json').read_text())}
    rows=[];started=time.perf_counter()
    for seed in SEEDS:
        for key in KEYS:
            for preset in PRESETS:
                dst=OUT/('_'.join(key)+f'_{seed}_{preset}')/'complete.json'
                r=json.loads(dst.read_text()) if dst.exists() else run_one(key,seed,preset,registry)
                rows.append(r);dump(REPORT/'progress.json',rows)
                print(len(rows),'/24',key,preset,r['outcome'],'clean',r['continuous_clean'],
                      'clearance',round(r['continuous_min_clearance_cm'],3),'seconds',round(time.perf_counter()-started,1),flush=True)
    result=dict(n=len(rows),clean=sum(r['continuous_clean'] for r in rows),stable=sum(r['stable_success'] for r in rows),
        wrong=sum(r['outcome']=='wrong_exit' for r in rows),severe=sum(r['outcome']=='severe_departure' for r in rows),
        pauses=sum(r['translation_pause_s']>0 for r in rows),new_training=False,final_exam_used=False)
    result['gate_pass']=result['clean']>=22 and result['stable']==24 and result['wrong']==result['severe']==result['pauses']==0
    oracle=json.loads((ORACLE_REPORT/'runs.json').read_text())
    for a,b in zip(oracle,rows):
        assert all(a[k]==b[k] for k in ['key','seed','preset','initial_sampling'])
        assert json.loads((Path(a['artifact'])/'events.json').read_text())==json.loads((Path(b['artifact'])/'events.json').read_text())
    result['paired_initials_events_exact']=True
    dump(REPORT/'runs.json',rows);dump(REPORT/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['freeze','run']);a=p.parse_args();globals()[a.task]()
