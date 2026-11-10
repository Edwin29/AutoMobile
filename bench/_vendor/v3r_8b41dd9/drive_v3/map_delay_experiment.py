"""Offline map mismatch and capture-to-command latency experiment. No GPIO.

Truth is confined to rendering, plant integration and evaluation. The runtime
receives its own map/route, initial map-relative prior and timestamped images.
"""
import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .camera import CameraGeometry
from .compare_recovery import settings
from .geometry import advance, footprint, to_world, wrap
from .map_model import HERE, TrackMap
from .provenance import code_manifest
from .runtime import Runtime
from .simulation import Renderer, json_default

EXPERIMENT = HERE.parent / 'research/map_delay_comparison'
MAPS = {'old': EXPERIMENT/'inputs/map_old.json', 'corrected': EXPERIMENT/'inputs/map_corrected_candidate.json'}
COMBINATIONS = {'old_old': ('old','old'), 'corrected_corrected': ('corrected','corrected'),
                'corrected_old': ('corrected','old')}


def save(path, value):
    Path(path).write_bytes((json.dumps(value, indent=2, default=json_default)+'\n').encode('utf-8'))


def completion(capture, period, sequence, latency):
    """Single worker, fresh frame at start; no stale-frame queue or future input."""
    delay = 0. if not latency else latency[sequence % len(latency)]
    done = capture + delay
    return done, max(capture + period, done), delay


def experiment_settings(period, maximum_delay):
    cfg = settings('recovery')
    # Same controller/road acceptance/motor model in every arm. Only the timing
    # contract follows the tested update period; no hard-stop disable switch.
    cfg['controller']['watchdog_s'] = period + maximum_delay + .1
    cfg['planner'].update(replan_period_s=period, rollout_dt_s=period, exact_rollout_horizon=True,
                          path_valid_s=period+maximum_delay+.1)
    return cfg


def initial_for_map(track, route):
    # Map-relative start: left external-road centre at y=-20.25, not a truth
    # coordinate copied to the wrong-map estimator. Same photo/world frame.
    left_inner = min(b[0] for b in track.data['blocks'])
    x = .5*(track.data['outer'][0]+left_inner)
    index = route.nearest([x,-20.25])
    return np.r_[route.points[index],route.heading[index]], index


class Plant:
    def __init__(self, pose, track, route, rim, index):
        self.pose = pose.copy(); self.t = 0.; self.pwm = np.zeros(2)
        self.track = track; self.route = route; self.rim = rim
        self.last_arc = route.s[index]; self.progress = 0.; self.minimum = float('inf')
        self.rows = []; self.episode = None; self.episodes = []; self.inside_since = None
        self.outside_s = 0.; self.stopped_s = 0.; self.end = None
        self.sample(0.)

    def sample(self, dt):
        distances = self.track.clearance(to_world(self.rim,self.pose))
        margin = float(distances.min()); self.minimum = min(self.minimum,margin)
        arc = self.route.s[self.route.nearest(self.pose[:2])]; length = self.route.s[-1]
        self.progress += (arc-self.last_arc+length/2)%length-length/2; self.last_arc=arc
        if margin < 0:
            self.outside_s += dt; self.inside_since = None
            if self.episode is None:self.episode=dict(start_s=self.t,start_progress_cm=self.progress,min_cm=margin)
            self.episode['min_cm'] = min(self.episode['min_cm'],margin)
        elif self.episode is not None:
            if self.inside_since is None:self.inside_since=self.t
            if self.t-self.inside_since >= 1.-1e-9:
                self.episode.update(end_s=self.t,travel_cm=self.progress-self.episode['start_progress_cm'],recovered=True)
                self.episodes.append(self.episode); self.episode=None; self.inside_since=None
        if np.max(self.pwm)<1e-8:self.stopped_s+=dt
        self.rows.append(dict(now_s=self.t,truth=self.pose.tolist(),true_margin_cm=margin,progress_cm=self.progress))
        if distances.max()<0 or margin < -12:self.end='severe_departure'
        elif self.progress>=length-.5:self.end='lap_complete'

    def advance_to(self, target):
        # Commands are piecewise constant and applied ONLY at completion. Sample
        # true footprint every <=20ms, independently of controller frequency.
        while self.t < target-1e-10 and self.end is None:
            dt=min(.02,target-self.t)
            wheels=np.where(self.pwm>=.30-1e-9,self.pwm,0.)*16.
            self.pose=advance(self.pose,float(wheels.mean()),float((wheels[0]-wheels[1])/13.),dt)
            self.t+=dt; self.sample(dt)


def run(output, combination, direction, period=.2, delayed=False, seconds=180.):
    cv2.setNumThreads(1)
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    world_name,belief_name=COMBINATIONS[combination]
    world=TrackMap(MAPS[world_name]); belief=TrackMap(MAPS[belief_name])
    world_route=world.outer_route(direction); belief_route=belief.outer_route(direction)
    true_start,index=initial_for_map(world,world_route); prior,_=initial_for_map(belief,belief_route)
    trace=json.loads((EXPERIMENT/'inputs/processing_delay.json').read_text(encoding='utf-8'))
    delays=[v/1000.+trace['assumed_sensor_delivery_s'] for v in trace['total_ms']] if delayed else []
    cfg=experiment_settings(period,max(delays,default=0.))
    camera=CameraGeometry(HERE/'config/camera_legacy_track.json'); renderer=Renderer(world,camera)
    plant=Plant(true_start,world,world_route,footprint(cfg['vehicle']),index)
    manifest=code_manifest(); save(output/'code_manifest.json',manifest); save(output/'settings.json',cfg)
    save(output/'setup.json',dict(world_map=world_name,belief_map=belief_name,world_hash=world.id,belief_hash=belief.id,
        true_start=true_start,initial_prior=prior,world_route_min_cm=world_route.clearance,belief_route_min_cm=belief_route.clearance,
        latency_input_sha256=hashlib.sha256((EXPERIMENT/'inputs/processing_delay.json').read_bytes()).hexdigest()))
    np.savez_compressed(output/'routes.npz',world=world_route.points,belief=belief_route.points)
    records=[]; capture=period; sequence=0; zero_since=None; skipped_deadlines=0; start=time.perf_counter()
    outcome='time_limit'
    with Runtime(prior,belief_route,cfg,clock_id='map_delay_sim',track=belief) as rt, (output/'steps.jsonl').open('w',encoding='utf-8',newline='\n') as stream:
        while capture<seconds:
            plant.advance_to(capture)
            if plant.end:outcome=plant.end;break
            captured_pose=plant.pose.copy(); image=renderer.render(captured_pose)
            done,next_capture,delay=completion(capture,period,sequence,delays)
            if done>seconds:plant.advance_to(seconds);break
            before=plant.pwm.copy()
            plant.advance_to(done)  # old command continues during processing
            if plant.end:outcome=plant.end;break
            rec,obs,debug,path=rt.step(image,capture,done,rectified=True)
            # Runtime timeline and virtual actuators both receive this command at
            # done, never at capture. Nominal prediction sees no true pose/speed.
            plant.pwm=np.asarray(rec['pwm'])
            rec.update(truth=plant.pose.tolist(),captured_truth_for_evaluation=captured_pose.tolist(),
                previous_applied_pwm=before.tolist(),applied_at_s=done,
                true_margin_cm=plant.rows[-1]['true_margin_cm'],progress_cm=plant.progress,
                pose_error_cm=float(np.linalg.norm(plant.pose[:2]-rt.estimator.primary.mean[:2])),
                yaw_error_rad=float(wrap(plant.pose[2]-rt.estimator.primary.mean[2])),modeled_latency_s=delay,
                requested_period_s=period,control_interval_s=done-(records[-1]['now_s'] if records else 0.))
            stream.write(json.dumps(rec,default=json_default)+'\n');records.append(rec)
            if delay>period:skipped_deadlines+=1
            if np.max(plant.pwm)<1e-8:
                if zero_since is None:zero_since=done
                if done-zero_since>=5.:outcome='held';break
            else:zero_since=None
            capture=next_capture;sequence+=1
    if plant.episode:
        plant.episode.update(end_s=plant.t,recovered=False);plant.episodes.append(plant.episode)
    from research.recovery_comparison.summarize import corner_events
    corners=corner_events(plant.rows,direction)
    proper=sum(c['proper_exit'] for c in corners); unrecovered=sum(not e['recovered'] for e in plant.episodes)
    summary=dict(combination=combination,direction=direction,period_s=period,delayed=delayed,
        outcome=outcome,verified_lap=outcome=='lap_complete' and proper==4 and unrecovered==0,
        frames=len(records),simulated_s=plant.t,progress_cm=plant.progress,route_length_cm=float(world_route.s[-1]),
        minimum_true_margin_cm=plant.minimum,outside_s=plant.outside_s,stopped_s=plant.stopped_s,
        penetration_episodes=plant.episodes,proper_corner_exits=proper,truth_corner_events=corners,
        max_pose_error_cm=max((r['pose_error_cm'] for r in records),default=0.),
        mean_pose_error_cm=float(np.mean([r['pose_error_cm'] for r in records])),
        reasons=dict(Counter(r['control_reason'] for r in records)),
        path_reasons=dict(Counter(r['path_reason'] for r in records)),
        observation_status=dict(Counter(r['estimate'].get('status') for r in records)),
        latency_median_s=float(np.median([r['modeled_latency_s'] for r in records])),
        latency_max_s=max((r['modeled_latency_s'] for r in records),default=0.),
        interval_p95_s=float(np.percentile([r['control_interval_s'] for r in records],95)),
        missed_capture_deadlines=skipped_deadlines,
        wall_s=time.perf_counter()-start,world_hash=world.id,belief_hash=belief.id,code_hash=manifest['sha256'],
        code_unchanged=code_manifest()==manifest,
        limits='Synthetic flat road, nominal motors, no obstacles or new slip. Corrected map is a candidate, not surveyed truth. Laptop processing trace plus assumed 50ms sensor delivery; not measured Pi latency. Known map-relative starting prior. 20ms footprint samples are not continuous collision proof.')
    save(output/'summary.json',summary)
    with (output/'truth.jsonl').open('w',encoding='utf-8',newline='\n') as f:
        for r in plant.rows:f.write(json.dumps(r)+'\n')
    print(json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--combination',choices=COMBINATIONS,required=True)
    p.add_argument('--direction',choices=['cw','ccw'],required=True);p.add_argument('--period',type=float,default=.2)
    p.add_argument('--delayed',action='store_true');p.add_argument('--seconds',type=float,default=180.)
    a=p.parse_args();run(a.output,a.combination,a.direction,a.period,a.delayed,a.seconds)
