"""Paired closed-loop tests. Truth is used ONLY by renderer, plant and scoring.

python -m drive_v3.compare_recovery --output drive_v3/runs/recovery --policy recovery --scenario baseline --direction ccw
"""
import argparse,copy,json,time,hashlib
from pathlib import Path
import cv2
import numpy as np
from .simulation import Renderer,json_default
from .runtime import Runtime
from .map_model import TrackMap,HERE
from .camera import CameraGeometry
from .geometry import advance,to_world,footprint,wrap
from .settings import load_settings
from .provenance import code_manifest


def settings(policy):
    cfg,_=load_settings()
    # Same latest published FIELD global-mode settings in both arms.
    cfg['planner'].update(acceptance_mode='bounded_prefix',offsets_cm=[0,-1,1,-2,2,-3,3,-4,4],
        path_valid_s=.4,replan_period_s=.5,allowed_penetration_cm=3.,clearance_cm=0.,
        uncertain_penetration_cm=5.,recovery_motion_tolerance_cm=.02,
        normal_speed_cm_s=8.,degraded_speed_cm_s=8.)
    cfg['controller'].update(watchdog_s=.5,pwm_min=.30,pwm_max=.60,preferred_speed_cm_s=4.8,trim_enabled=False)
    if policy in ('noise_only','recovery','recovery_rescue'):cfg['estimator']['motion_scaled_noise']=True
    if policy in ('recovery','recovery_rescue'):cfg['planner'].update(mode='recovery_rollout',rollout_s=4.,rollout_dt_s=.2,
        rollout_offsets_cm=[-2.,0.,2.],rollout_lookaheads_cm=[10.,16.],hard_road_depth_cm=6.,coast_s=1.2)
    if policy=='recovery_rescue':cfg['planner'].update(allow_existing_deep_recovery=True,max_recovery_travel_cm=49.)
    return cfg


def run(output,policy='recovery',direction='ccw',scenario='baseline',seconds=180.,video=False):
    cv2.setNumThreads(1)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cfg=settings(policy);track=TrackMap();route=track.outer_route(direction)
    camera=CameraGeometry(HERE/'config/camera_legacy_track.json');renderer=Renderer(track,camera)
    index=route.nearest([-86.25,-20.25]);initial=np.r_[route.points[index],route.heading[index]]
    forward=(np.arange(index,index+len(route.points)-1)%(len(route.points)-1)).astype(int)
    ahead_arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(route.points[forward],axis=0),axis=1))]
    curved=np.flatnonzero(np.abs(route.curvature[forward])>.015)
    first_corner=float(ahead_arc[curved[0]]) if len(curved) else 45.
    truth=initial.copy()
    if scenario=='start_outside':
        # Outside the LEFT external road edge for either direction; no truth reset later.
        truth[0]-=6.;initial=truth.copy()
    rim=footprint(cfg['vehicle']);applied=np.zeros(2);dt=.2
    records=[];progress=0.;last_arc=route.s[index];zero_since=None;outcome='time_limit'
    injected=False;recovery_start=None;episodes=[];current_episode=None;settle=0;sensor_event_start=None
    turn_total=0.;previous_heading=truth[2];minimum=float('inf');start=time.perf_counter()
    (output/'settings.json').write_text(json.dumps(cfg,indent=2))
    (output/'code_manifest.json').write_text(json.dumps(code_manifest(),indent=2))
    writer=cv2.VideoWriter(str(output/'comparison_view.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),5.,(960,480)) if video else None
    with Runtime(initial,route,cfg,clock_id='sim') as rt,(output/'steps.jsonl').open('w') as stream:
        for step in range(1,int(seconds/dt)+1):
            now=step*dt
            wheel=np.where(applied>=.30-1e-9,applied,0.)*16.
            v=float(wheel.mean());omega=float((wheel[0]-wheel[1])/13.)
            if scenario=='bias':v*=1.27;omega=omega*.70-.035*(v>0)
            if scenario=='slip' and progress>=25 and not injected:
                # 6cm lateral shove, 8deg yaw; not disclosed to the estimator.
                truth[0]-=6.;truth[2]=wrap(truth[2]+np.deg2rad(8));injected=True;recovery_start=now
            truth=advance(truth,v,omega,dt)
            image=renderer.render(truth)
            # Sensor failures triggered by geometric progress only in test apparatus.
            if scenario in ('dropout','occlusion','seam') and progress>=first_corner and sensor_event_start is None:sensor_event_start=now
            event_age=now-sensor_event_start if sensor_event_start is not None else float('inf')
            if scenario=='dropout' and event_age<.8:image[:]=245
            if scenario=='occlusion' and event_age<3.:cv2.rectangle(image,(0,160),(250,450),(70,70,70),-1)
            if scenario=='seam' and event_age<3.:cv2.line(image,(30,440),(340,220),(220,220,220),7)
            rec,obs,debug,path=rt.step(image,now,now,rectified=True)
            applied=np.asarray(rec['pwm'])
            distances=track.clearance(to_world(rim,truth));margin=float(distances.min());minimum=min(minimum,margin)
            i=route.nearest(truth[:2]);arc=route.s[i];delta=(arc-last_arc+route.s[-1]/2)%route.s[-1]-route.s[-1]/2
            progress+=delta;last_arc=arc
            turn_total+=float(wrap(truth[2]-previous_heading));previous_heading=truth[2]
            # Recovery requires >=1s fully back inside, not a single positive sample.
            if margin<0:
                settle=0
                if current_episode is None:current_episode=dict(start_s=now,start_progress_cm=progress,min_cm=margin)
                current_episode['min_cm']=min(current_episode['min_cm'],margin)
            elif current_episode is not None:
                settle+=1
                if settle>=5:
                    current_episode.update(end_s=now,travel_cm=progress-current_episode['start_progress_cm'],recovered=True)
                    episodes.append(current_episode);current_episode=None;settle=0
            rec.update(truth=truth.tolist(),true_margin_cm=margin,progress_cm=progress,applied_pwm=applied.tolist(),
                pose_error_cm=float(np.linalg.norm(truth[:2]-rt.estimator.primary.mean[:2])),
                yaw_error_rad=float(wrap(truth[2]-rt.estimator.primary.mean[2])))
            stream.write(json.dumps(rec,default=json_default)+'\n');records.append(rec)
            if writer:
                canvas=np.zeros((480,960,3),np.uint8);canvas[:,:640]=image
                for row,label in enumerate([f'{policy} {direction} {scenario}',f't={now:.1f}  s={progress:.1f}cm',rec['path_reason'],
                    f'true margin {margin:.2f}cm',f'pose error {rec["pose_error_cm"]:.2f}cm',f'PWM {applied.round(2)}']):
                    cv2.putText(canvas,label,(645,35+row*30),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
                if len(path.points_body_cm)>2:
                    uv=camera.body_to_rectified(np.asarray(path.points_body_cm))
                    cv2.polylines(canvas,[np.clip(uv,-10000,10000).astype(np.int32)],False,(255,0,255),2)
                writer.write(canvas)
            if distances.max()<0 or margin < -12:outcome='severe_departure';break
            if progress>=route.s[-1]-.5:outcome='lap_complete';break
            zero=np.max(applied)<1e-6
            if zero and zero_since is None:zero_since=now
            if not zero:zero_since=None
            if zero_since is not None and now-zero_since>=5.:outcome='held';break
        if current_episode is not None:
            current_episode.update(end_s=records[-1]['now_s'],recovered=False);episodes.append(current_episode)
    if writer:writer.release()
    counts={}
    for r in records:
        key=r['control_reason'];counts[key]=counts.get(key,0)+1
    # True route progression and direction of heading change; not planner success flags.
    corner_passes=[]
    sequence=np.r_[np.arange(index,len(route.points)-1),np.arange(0,index+1)]
    running=0.;inside=False
    for a,b in zip(sequence[:-1],sequence[1:]):
        running+=np.linalg.norm(route.points[b]-route.points[a])
        curved=abs(route.curvature[b])>.012
        if inside and not curved and progress>=running:corner_passes.append(float(running))
        inside=curved
    summary=dict(policy=policy,direction=direction,scenario=scenario,outcome=outcome,frames=len(records),
        simulated_s=records[-1]['now_s'],progress_cm=progress,route_length_cm=float(route.s[-1]),
        minimum_true_margin_cm=minimum,penetration_episodes=episodes,corner_exits_passed=len(corner_passes),
        signed_turn_deg=float(np.rad2deg(turn_total)),nonzero_frames=sum(max(r['pwm'])>0 for r in records),
        reasons=counts,max_pose_error_cm=max(r['pose_error_cm'] for r in records),
        median_step_ms=float(np.median([r['total_ms'] for r in records])),p95_step_ms=float(np.percentile([r['total_ms'] for r in records],95)),
        wall_s=time.perf_counter()-start,map_hash=track.id,code_hash=code_manifest()['sha256'],
        sensor_event_start_s=sensor_event_start,first_corner_progress_cm=first_corner,
        limits='Synthetic planar RGB. Same map used for rendering/localization. Discrete footprint scoring, no physical validation; 30% floor is a test assumption, not measured minimum.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--policy',choices=['baseline','noise_only','recovery','recovery_rescue'],default='recovery')
    p.add_argument('--direction',choices=['cw','ccw'],default='ccw')
    p.add_argument('--scenario',choices=['baseline','start_outside','slip','bias','dropout','occlusion','seam'],default='baseline')
    p.add_argument('--seconds',type=float,default=180.);p.add_argument('--video',action='store_true')
    a=p.parse_args();run(a.output,a.policy,a.direction,a.scenario,a.seconds,a.video)
