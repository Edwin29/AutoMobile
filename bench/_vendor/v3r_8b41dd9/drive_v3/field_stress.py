"""Corrected-map closed-loop disturbances, with camera vs PWM-only pose audit.

No motor, GPIO, network, or physical camera imports. Disturbances and truth stay
in the simulator. The unchanged recovery controller is not told the plant gains.
"""
import argparse,copy,hashlib,json,time
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from .map_delay_experiment import Plant,MAPS,EXPERIMENT,save,completion,experiment_settings,initial_for_map
from .map_model import TrackMap,HERE
from .camera import CameraGeometry
from .geometry import advance,wrap,to_world,footprint
from .runtime import Runtime
from .simulation import Renderer,json_default
from .provenance import code_manifest

STUDY=HERE.parent/'research/field_stress'
SCENARIOS={
 'baseline':{},
 'drift_left':{'drift':-.035},
 'turn_under':{'turn_proxy':True},
 'slip':{'slip_cm':6.,'slip_deg':8.},
 'low_power':{'sag':.65,'sag_s':12.},
 'wheel_stall':{'stall_s':2.5},
 'glare':{'glare_s':4.},
 'seam':{'seam':True},
 'occlusion':{'occlusion_s':3.},
 'dropout':{'dropout_s':1.2},
 'blur_warm':{'blur':True},
 'mixed_mild':{'drift':-.0174533,'yaw_scale':.85,'speed_scale':.9,'slip_cm':3.,'slip_deg':4.,'glare_s':4.,'occlusion_s':1.,'blur':True},
 'mixed_strong':{'drift':-.035,'turn_proxy':True,'speed_scale':.85,'sag':.5,'sag_s':12.,'stall_s':2.5,'slip_cm':6.,'slip_deg':8.,'glare_s':8.,'occlusion_s':3.,'dropout_s':2.4,'blur':True,'strong':True},
}


def velocity(pwm,profile,age,left_gain,right_gain,t=0.,seed=17):
    """Effective two-side sensitivity model, not a four-wheel traction solver."""
    wheels=np.where(np.asarray(pwm)>=.30-1e-9,pwm,0.)*16.
    scale=profile.get('speed_scale',1.)
    if 0<=age<profile.get('sag_s',0.):scale*=profile.get('sag',1.)
    wheels=wheels*scale
    if 0<=age<profile.get('stall_s',0.):wheels[0]*=.5  # rear-left stalls: assumed half-side effect
    v=float(wheels.mean());w=float((wheels[0]-wheels[1])/13.)
    w*=profile.get('yaw_scale',1.)
    if profile.get('turn_proxy'):w*=left_gain if w<0 else right_gain
    w+=profile.get('drift',0.)*(v/4.8)
    return v,w,wheels


class StressPlant(Plant):
    def __init__(self,pose,track,route,rim,index,profile,direction,seed,gains):
        self.shadow=pose.copy();self.profile=profile;self.direction=direction;self.seed=seed;self.gains=gains
        self.event_time=None;self.slipped=False;self.events=[];self.last_velocity=(0.,0.);self.last_wheels=np.zeros(2)
        seq=np.arange(index,index+len(route.points)-1)%(len(route.points)-1)
        dist=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(route.points[seq],axis=0),axis=1))]
        curved=np.flatnonzero(abs(route.curvature[seq])>.015)
        self.first_corner=float(dist[curved[0]]) if len(curved) else 45.
        self.event_progress=max(5.,self.first_corner-(30. if profile.get('seam') else 4.))
        super().__init__(pose,track,route,rim,index)

    def age(self,t=None):return (self.t if t is None else t)-self.event_time if self.event_time is not None else -1e9

    def advance_to(self,target):
        while self.t<target-1e-10 and self.end is None:
            dt=min(.02,target-self.t)
            if self.event_time is None and self.progress>=self.event_progress:
                self.event_time=self.t;self.events.append(dict(kind='first_corner_disturbance_window',t=self.t,progress_cm=self.progress))
            age=self.age()
            if not self.slipped and self.profile.get('slip_cm') and age>=2.:
                # Outward shove in world coordinates; yaw opposes the desired turn.
                before=self.pose.copy();unit=self.pose[:2]/np.linalg.norm(self.pose[:2])
                self.pose[:2]+=unit*self.profile['slip_cm']
                self.pose[2]=wrap(self.pose[2]+np.deg2rad(self.profile['slip_deg'])*(1 if self.direction=='ccw' else -1))
                self.slipped=True;self.events.append(dict(kind='unannounced_slip',t=self.t,before=before.tolist(),after=self.pose.tolist()))
            # Shadow receives the same actually applied PWM, not the hidden response.
            nominal=np.where(self.pwm>=.30-1e-9,self.pwm,0.)*16.
            self.shadow=advance(self.shadow,float(nominal.mean()),float((nominal[0]-nominal[1])/13.),dt)
            v,w,wheels=velocity(self.pwm,self.profile,age,*self.gains,self.t,self.seed)
            self.pose=advance(self.pose,v,w,dt);self.last_velocity=(v,w);self.last_wheels=wheels
            self.t+=dt;self.sample(dt)
            self.rows[-1].update(pwm_only_pose=self.shadow.tolist(),true_v=v,true_w=w,
                hidden_effective_wheels_cm_s=wheels.tolist(),event_age_s=self.age(),
                pwm_only_error_cm=float(np.linalg.norm(self.shadow[:2]-self.pose[:2])))


def render_input(renderer,plant,t,profile,seed):
    image=renderer.render(plant.pose);age=plant.age(t);flags=[]
    if profile.get('seam'):flags.append('fixed_ground_joint')
    if profile.get('blur'):
        # 66.513ms exposure observed in field metadata; integrate past truth only.
        rows=plant.rows;times=np.array([r['now_s'] for r in rows])
        frames=[image.astype(np.float32)]
        for lag in (.0332565,.066513):
            i=max(0,int(np.searchsorted(times,t-lag,side='right')-1))
            frames.append(renderer.render(np.array(rows[i]['truth'])).astype(np.float32))
        image=np.mean(frames,axis=0)
        image*=np.array([.85,1.02,1.12],np.float32)*(1+.10*np.sin(t*1.7+seed))
        kernel=11 if profile.get('strong') else 7
        image=cv2.GaussianBlur(image,(kernel,kernel),2.4 if profile.get('strong') else 1.4)
        rng=np.random.default_rng(seed*1000003+int(np.floor(t*10)))
        image=np.clip(image+rng.normal(0,6.,image.shape),0,255).astype(np.uint8);flags.append('exposure_blur_warm_noise')
    if 0<=age<profile.get('glare_s',0.):
        yy,xx=np.mgrid[:480,:640];jitter=(seed%13-6)*3
        width=1.4 if profile.get('strong') else 1.
        alpha=.99*np.exp(-.5*(((xx-(435+jitter))/(57.6*width))**2+((yy-336)/110.4)**2))
        image=np.clip(image.astype(float)+(255.-image)*alpha[:,:,None],0,255).astype(np.uint8);flags.append('glare')
    if 0<=age<profile.get('occlusion_s',0.):
        x1,x2=(0,250) if plant.direction=='ccw' else (390,639)
        cv2.rectangle(image,(x1,160),(x2,450),(70,70,70),-1);flags.append('inner_side_occlusion')
    dropout_start=2. if profile.get('strong') else 0.
    if dropout_start<=age<dropout_start+profile.get('dropout_s',0.):
        image[:]=245;flags.append('blank_camera')
    return image,flags


def run(output,scenario,direction,period,seed=17,seconds=240.):
    cv2.setNumThreads(1);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    profile=copy.deepcopy(SCENARIOS[scenario]);evidence=json.loads((STUDY/'inputs/field_evidence.json').read_text(encoding='utf-8'))
    track=TrackMap(MAPS['corrected']);route=track.outer_route(direction);initial,index=initial_for_map(track,route)
    trace=json.loads((EXPERIMENT/'inputs/processing_delay.json').read_text(encoding='utf-8'))
    delays=[v/1000+trace['assumed_sensor_delivery_s'] for v in trace['total_ms']]
    cfg=experiment_settings(period,max(delays))
    camera=CameraGeometry(HERE/'config/camera_legacy_track.json');renderer=Renderer(track,camera)
    plant=StressPlant(initial,track,route,footprint(cfg['vehicle']),index,profile,direction,seed,
        (evidence['left_turn_proxy'],evidence['right_turn_proxy']))
    if profile.get('seam'):
        # Physical ground joint fixed across left road; absent from estimator map.
        y=-50. if direction=='ccw' else 50.
        pixel=lambda p:tuple(np.rint((np.array(p)-renderer.origin)*renderer.ppcm).astype(int))
        cv2.line(renderer.texture,pixel([-103.,y]),pixel([-65.,y]),(180,190,190),3)
    manifest=code_manifest();save(output/'code_manifest.json',manifest);save(output/'settings.json',cfg)
    save(output/'setup.json',dict(scenario=scenario,profile=profile,seed=seed,map_hash=track.id,initial_pose=initial,
        gains_hidden_from_controller=plant.gains,event_progress_cm=plant.event_progress,
        evidence_sha256=hashlib.sha256((STUDY/'inputs/field_evidence.json').read_bytes()).hexdigest()))
    np.savez_compressed(output/'route.npz',route=route.points)
    writer=cv2.VideoWriter(str(output/'camera.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),1/period,(640,480))
    if not writer.isOpened():raise RuntimeError('camera trace writer failed')
    capture=period;seq=0;records=[];zero_since=None;outcome='time_limit';start=time.perf_counter()
    with Runtime(initial,route,cfg,clock_id='field_stress_sim',track=track) as rt,(output/'steps.jsonl').open('w',encoding='utf-8',newline='\n') as f:
        while capture<seconds:
            plant.advance_to(capture)
            if plant.end:outcome=plant.end;break
            image,flags=render_input(renderer,plant,capture,profile,seed)
            done,next_capture,delay=completion(capture,period,seq,delays)
            if done>seconds:plant.advance_to(seconds);break
            plant.advance_to(done)
            if plant.end:outcome=plant.end;break
            rec,obs,debug,path=rt.step(image,capture,done,rectified=True)
            plant.pwm=np.asarray(rec['pwm'])
            nominal=np.where(plant.pwm>=.30-1e-9,plant.pwm,0.)*16.
            if not np.allclose([rec['predicted_v'],rec['predicted_omega']],
                               [nominal.mean(),(nominal[0]-nominal[1])/13.],atol=1e-9,rtol=0):
                raise AssertionError('Command predictor differs from the declared nominal plant')
            rec.update(truth=plant.pose.tolist(),true_margin_cm=plant.rows[-1]['true_margin_cm'],progress_cm=plant.progress,
                pose_error_cm=float(np.linalg.norm(plant.pose[:2]-rt.estimator.primary.mean[:2])),
                yaw_error_rad=float(wrap(plant.pose[2]-rt.estimator.primary.mean[2])),
                pwm_only_pose=plant.shadow.tolist(),pwm_only_error_cm=float(np.linalg.norm(plant.pose[:2]-plant.shadow[:2])),
                pwm_only_yaw_error_rad=float(wrap(plant.pose[2]-plant.shadow[2])),
                camera_flags=flags,event_age_s=plant.age(),true_v=plant.last_velocity[0],true_w=plant.last_velocity[1],
                hidden_effective_wheels_cm_s=plant.last_wheels.tolist(),modeled_latency_s=delay)
            records.append(rec);f.write(json.dumps(rec,default=json_default)+'\n');writer.write(image)
            if np.max(plant.pwm)<1e-8:
                if zero_since is None:zero_since=done
                if done-zero_since>=5.:outcome='held';break
            else:zero_since=None
            capture=next_capture;seq+=1
    writer.release()
    if plant.episode:
        plant.episode.update(end_s=plant.t,recovered=False);plant.episodes.append(plant.episode)
    from research.recovery_comparison.summarize import corner_events
    corners=corner_events(plant.rows,direction);proper=sum(c['proper_exit'] for c in corners)
    errors=np.array([r['pose_error_cm'] for r in records]);shadow=np.array([r['pwm_only_error_cm'] for r in records])
    summary=dict(scenario=scenario,profile=profile,direction=direction,period_s=period,seed=seed,
        outcome=outcome,verified_lap=outcome=='lap_complete' and proper==4 and not any(not e['recovered'] for e in plant.episodes),
        frames=len(records),simulated_s=plant.t,progress_cm=plant.progress,proper_corner_exits=proper,truth_corner_events=corners,
        minimum_true_margin_cm=plant.minimum,outside_s=plant.outside_s,stopped_s=plant.stopped_s,penetration_episodes=plant.episodes,
        events=plant.events,event_time_s=plant.event_time,
        mean_pose_error_cm=float(errors.mean()),max_pose_error_cm=float(errors.max()),p95_pose_error_cm=float(np.percentile(errors,95)),
        mean_pwm_only_error_cm=float(shadow.mean()),max_pwm_only_error_cm=float(shadow.max()),
        p95_yaw_error_deg=float(np.percentile(np.abs([r['yaw_error_rad'] for r in records]),95)*180/np.pi),
        reasons=dict(Counter(r['control_reason'] for r in records)),path_reasons=dict(Counter(r['path_reason'] for r in records)),
        observation_status=dict(Counter(r['estimate'].get('status') for r in records)),
        saturation_fraction=float(np.mean([max(r['pwm'])>=.6-1e-6 for r in records])),
        map_hash=track.id,code_hash=manifest['sha256'],code_unchanged=manifest==code_manifest(),wall_s=time.perf_counter()-start,
        limits='Fixed corrected candidate map; synthetic raster camera. Historical turn ratios are proxies, other magnitudes are declared stress assumptions. Same-PWM shadow is NOT a second camera-free controlled trajectory. No new adaptation or controller tuning. Effective stalled-side model is not four-wheel contact physics. Laptop latency trace + assumed 50ms delivery, not measured Pi timing.')
    save(output/'summary.json',summary)
    with (output/'truth.jsonl').open('w',encoding='utf-8',newline='\n') as f:
        for r in plant.rows:f.write(json.dumps(r,default=json_default)+'\n')
    print(json.dumps(summary),flush=True);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--scenario',choices=SCENARIOS,required=True);p.add_argument('--direction',choices=['cw','ccw'],required=True)
    p.add_argument('--period',type=float,required=True);p.add_argument('--seed',type=int,default=17);p.add_argument('--seconds',type=float,default=240.)
    a=p.parse_args();run(a.output,a.scenario,a.direction,a.period,a.seed,a.seconds)
