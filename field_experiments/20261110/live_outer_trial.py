"""Bounded camera-feedback trial. GPIO only with --drive; no reverse or horn."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json,math,signal,sys,threading,time
from pathlib import Path
import cv2,numpy as np
from picamera2 import Picamera2
from drive_v3.runtime import Runtime,RecordingSink
from drive_v3.settings import load_settings
from drive_v3.map_model import TrackMap
from drive_v3.provenance import code_manifest
from drive_v3.start_region import right_red_prior


class GuardedMotors:
    def __init__(self,seconds,cap):
        import fcntl
        # Existing verified same-side wiring; do not import any old driving loop.
        sys.path.insert(0,'/home/pi/straight30')
        from probe import Motors,read_value,safety_reason
        self.read_value=read_value;self.safety_reason=safety_reason
        self.lockfile=open('/tmp/autodrive-power-probe.lock','a')
        fcntl.flock(self.lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.motor=Motors(1000);self.cap=cap;self.seconds=seconds
        self.lock=threading.RLock();self.done=threading.Event();self.ready=threading.Event()
        self.sample=None;self.error=None;self.active=False;self.deadline=float('inf')
        self.last=time.monotonic();self.applied=[0.,0.]
        self.sensor=threading.Thread(target=self._sensor,daemon=True);self.sensor.start()
        if not self.ready.wait(3):self.close();raise RuntimeError('No power telemetry')
        self.worker=threading.Thread(target=self._guard,daemon=True);self.worker.start()

    def _sensor(self):
        while not self.done.is_set():
            v=self.read_value(['pmic_read_adc','EXT5V_V'],r'=([0-9.]+)V');at=time.monotonic()
            t=self.read_value(['measure_temp'],r'=([0-9.]+)')
            f=self.read_value(['get_throttled'],r'=(0x[0-9a-fA-F]+)',16)
            self.sample={'voltage':v,'temp':t,'throttled':f,'voltage_at':at};self.ready.set()
            self.done.wait(.1)

    def _zero(self):self.motor.set(0.,0.);self.applied=[0.,0.]

    def _guard(self):
        while not self.done.wait(.02):
            with self.lock:
                now=time.monotonic()
                fault=self.safety_reason(self.sample,now,now,self.motor.pressed())
                if self.active and now>=self.deadline:fault='duration_complete'
                elif self.active and now-self.last>.5:fault='command_watchdog'
                if fault:self.error=fault;self.active=False;self._zero()

    def arm(self):
        with self.lock:
            if self.error:raise RuntimeError(self.error)
            self.last=time.monotonic();self.deadline=self.last+self.seconds;self.active=True

    def write(self,pwm):
        with self.lock:
            if self.error:raise RuntimeError(self.error)
            if not self.active:raise RuntimeError('Motor output not armed')
            if time.monotonic()>=self.deadline:
                self.error='duration_complete';self._zero();raise RuntimeError(self.error)
            values=np.asarray(pwm,float)
            if not np.isfinite(values).all() or values.min()<0 or values.max()>self.cap+1e-8:
                self._zero();raise RuntimeError('PWM outside forward trial limits')
            self.motor.set(*values);self.applied=values.tolist();self.last=time.monotonic()
            return self.applied.copy()

    def close(self):
        self.done.set()
        with self.lock:self.active=False;self._zero()
        for name in ('worker','sensor'):
            if hasattr(self,name):getattr(self,name).join(timeout=1.)
        self.motor.close();self.lockfile.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drive',action='store_true')
    parser.add_argument('--seconds',type=float,default=5.)
    parser.add_argument('--laps',type=int,default=0,choices=[0,1])
    parser.add_argument('--rolling-floor',type=float,choices=[.12,.30],default=.12)
    parser.add_argument('--pose',nargs=3,type=float,default=None)
    parser.add_argument('--direction',choices=['ccw','cw'],default='ccw')
    parser.add_argument('--mode',choices=['world','local'],default='world')
    args=parser.parse_args()
    if not 0<args.seconds<=(180 if args.laps else 5):parser.error('Trial duration exceeds limit')
    cv2.setNumThreads(1)
    def interrupt(*_):raise KeyboardInterrupt
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupt)
    output=Path(__file__).parent/'runs'/('live_' if args.drive else 'dry_')
    output=output.parent/(output.name+time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True,exist_ok=False)
    cfg,_=load_settings();cfg['planner']['acceptance_mode']='bounded_prefix'
    if args.mode=='local':cfg['planner'].update(mode='local_corridor',direction=args.direction)
    # Hand placement may start several cm off the map centre. These are candidate
    # offsets, not permissions to cross a boundary: the same footprint rules apply.
    cfg['planner']['offsets_cm']=[0.,-1.,1.,-2.,2.,-3.,3.,-4.,4.]
    cfg['planner']['path_valid_s']=.4
    cfg['planner']['replan_period_s']=.5  # include watchdog-held output in checked horizon
    # Field scoring permits partial line crossing if recovered. Provisional
    # trial limit; per-pose non-worsening and nominal recovery remain required.
    cfg['planner']['allowed_penetration_cm']=3.
    cfg['planner']['clearance_cm']=0.  # contact with painted boundary is not a failed mission
    cfg['planner']['uncertain_penetration_cm']=5.  # experimental support-point bound; means retain 3cm
    cfg['planner']['recovery_motion_tolerance_cm']=.02  # 0.2mm intermediate sweep, end must improve
    cfg['controller']['watchdog_s']=.5
    cfg['controller']['pwm_max']=.35
    if args.rolling_floor==.30:
        cfg['controller'].update(pwm_min=.30,pwm_max=.60,preferred_speed_cm_s=4.8)
        # Not a measured wheel-speed map: retain the existing model provisionally.
        # Plan for the maximum permitted body speed, even when cruising more slowly.
        cfg['planner'].update(normal_speed_cm_s=8.,degraded_speed_cm_s=8.)
    # No new unmeasured kick or direction gains; ordinary output slew is retained.
    cfg['controller']['trim_enabled']=False
    track=TrackMap();route=track.outer_route(args.direction);start_info=None
    if args.pose is None:
        pose,sigma,start_info=right_red_prior(track,cfg['vehicle'],args.direction)
        args.pose=pose.tolist();cfg['estimator']['initial_sigma']=sigma
    rt=Runtime(args.pose,route,cfg,start_s=0.,clock_id='boottime_relative',sink=RecordingSink())
    rt.track.clearance_fast(np.array([[0.,0.]]))  # all cold initialization before arming
    metadata={'mode':'drive' if args.drive else 'dry','seconds':args.seconds,'pose':args.pose,'direction':args.direction,'start_region':start_info,
        'settings':cfg,'code':code_manifest(),'motors':'existing /home/pi/straight30/probe.py Motors; paired left/right',
        'timestamp_note':'prediction uses last applied output until the actual new write time; PWM is not wheel odometry'}
    (output/'manifest.json').write_text(json.dumps(metadata,indent=2))
    camera=Picamera2();motor=None;started=False;reason='interrupted';count=0;applied=[0.,0.]
    try:
        camera.configure(camera.create_video_configuration(main={'size':(640,480),'format':'RGB888'},queue=False,buffer_count=4,controls={'FrameRate':30}))
        if args.drive:motor=GuardedMotors(args.seconds,cfg['controller']['pwm_max'])
        camera.start();started=True;time.sleep(2)
        print('PREPARING: motors zero; collecting initial observations',flush=True)
        origin=time.clock_gettime(time.CLOCK_BOOTTIME)-.1
        good=0;drive_start=None;lap_progress=0.;last_arc=None;held_since=None
        with (output/'steps.jsonl').open('w',buffering=1) as log:
            while True:
                if drive_start is not None and time.monotonic()-drive_start>=args.seconds:
                    reason='duration_complete';break
                if motor and motor.error:raise RuntimeError(motor.error)
                request=camera.capture_request()
                try:image=request.make_array('main');meta=request.get_metadata()
                finally:request.release()
                received=time.clock_gettime(time.CLOCK_BOOTTIME);capture=meta['SensorTimestamp']/1e9
                record,obs,debug,path=rt.step(image,capture-origin,received-origin,rotate180=True,shadow_pwm=applied)
                now=time.clock_gettime(time.CLOCK_BOOTTIME)
                if drive_start is not None:
                    arc=float(route.s[route.nearest(rt.estimator.primary.mean[:2])]);length=float(route.s[-1])
                    delta=(arc-last_arc+length/2)%length-length/2
                    # Estimated route progress, not independent lap ground truth.
                    # Ignore pose jumps and all changes while motors were stopped.
                    if max(applied)>0 and abs(delta)<10:lap_progress+=delta
                    last_arc=arc
                    if args.laps and args.mode!='local' and lap_progress>=length-1.:
                        reason='estimated_lap_complete';break
                    if path.speed_limit_cm_s<=0:
                        if held_since is None:held_since=time.monotonic()
                        elif time.monotonic()-held_since>8:
                            reason='held_for_8_seconds';break
                    else:held_since=None
                proposed=record['pwm'];phase='warmup' if drive_start is None else 'drive' if args.drive else 'dry'
                if drive_start is not None:
                    if now-capture>.4 or now-origin>path.valid_until_s:
                        log.write(json.dumps({'event':'stale_control_result','sensor_age_s':now-capture,'total_ms':record['total_ms']})+'\n')
                        proposed=[0.,0.]
                        rt.controller.stop('stale_control_result')
                        record['discarded_late_result']=True
                    applied=motor.write(proposed) if motor else [0.,0.]
                    write_at=time.clock_gettime(time.CLOCK_BOOTTIME)-origin
                    rt.timeline.step(write_at)
                    rt.timeline.command(write_at,*rt.model.forward_model(applied))
                    rt.last_pwm=np.array(applied);rt.last_v=rt.model.forward_model(applied)[0]
                else:
                    good=good+1 if record['estimate'].get('status') in ('observed','local_observed') and path.reason=='tracking' else 0
                record.update(phase=phase,actual_pwm=applied.copy(),sensor_to_result_ms=(now-capture)*1000,
                    metadata=meta,telemetry=motor.sample if motor else None,index=count,estimated_lap_progress_cm=lap_progress)
                log.write(json.dumps(record,default=lambda v:v.tolist())+'\n')
                cv2.imwrite(str(output/f'{count:04d}_raw.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,90])
                if count%10==0:print(json.dumps({'phase':phase,'frame':count,'path':path.reason,'applied':applied,'progress_cm':round(lap_progress,1)}),flush=True)
                count+=1
                if drive_start is None and count>=20:
                    if good<3:
                        if count>=60:raise RuntimeError('warmup_not_ready')
                        continue
                    # Reset controller output ramp: computed warmup commands were never applied.
                    rt.controller.stop('arm_reset');rt.last_pwm=np.zeros(2);rt.last_v=0.
                    if motor:motor.arm()
                    last_arc=float(route.s[route.nearest(rt.estimator.primary.mean[:2])])
                    drive_start=time.monotonic();print(f'DRIVE: maximum {args.seconds} seconds' if motor else 'DRY: GPIO disabled',flush=True)
    except (KeyboardInterrupt,Exception) as exc:
        reason=str(exc) or type(exc).__name__
    finally:
        if motor:motor.close()
        rt.close()
        if started:camera.stop()
        camera.close()
        summary={'output':str(output),'reason':reason,'frames':count,'drive':args.drive,'final_output':[0,0]}
        (output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
