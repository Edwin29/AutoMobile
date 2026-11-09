"""Pi-side bounded single motor experiment. Default: no GPIO, telemetry only."""
import argparse
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time


def validate(case, placement, duty, ramp, hold, frequency):
    if case not in ('idle','left','right','both','sequence','claude') or placement not in ('lifted','floor'):
        raise ValueError('Unknown experiment')
    if not all(math.isfinite(x) for x in (duty,ramp,hold,frequency)):
        raise ValueError('Nonfinite parameter')
    if not .05 <= duty <= .5 or not .3 <= ramp <= 2 or not .1 <= hold <= (5 if case in ('sequence','claude') else 2):
        raise ValueError('Limits: duty .05-.5, ramp .3-2 seconds, hold .1-2 seconds')
    if not 100 <= frequency <= 2000:
        raise ValueError('PWM frequency must be 100..2000 Hz')


def duty_at(elapsed, duty, ramp, hold):
    if elapsed < 0 or elapsed >= 2*ramp+hold:return 0.
    if elapsed < ramp:return duty*elapsed/ramp
    if elapsed < ramp+hold:return duty
    return max(0.,duty*(2*ramp+hold-elapsed)/ramp)


def safety_reason(sample, now, heartbeat, stop):
    if stop:return 'operator_or_connection_stop'
    if now-heartbeat>1.:return 'heartbeat_timeout'
    if sample is None or now-sample['voltage_at']>.75:return 'voltage_telemetry_stale'
    if sample['voltage'] is None:return 'voltage_read_failed'
    if sample['voltage']<4.80:return 'input_voltage_below_4.80V'
    if sample['temp'] is None or sample['throttled'] is None:return 'status_read_failed'
    if sample['temp']>=80:return 'temperature_at_least_80C'
    if sample['throttled'] & 1:return 'current_undervoltage_flag'
    return None


class Motors:
    def __init__(self, frequency):
        os.environ.setdefault('LG_WD','/tmp')
        from gpiozero import PWMOutputDevice,DigitalOutputDevice,Button
        self.devices=[];self.pwm=[];self.buttons=[];self.direction=[];self.signs=[1,1]
        try:
            for pin in (18,23):
                d=PWMOutputDevice(pin,frequency=frequency,initial_value=0)
                self.devices.append(d);self.pwm.append(d)
            for pin,value in ((22,0),(27,1),(25,0),(24,1)):
                d=DigitalOutputDevice(pin,initial_value=value)
                self.devices.append(d);self.direction.append(d)
            for pin in (5,6,13,19):
                d=Button(pin,pull_up=False);self.devices.append(d);self.buttons.append(d)
        except BaseException:
            self.close();raise

    def set(self,left,right):
        for i,v in enumerate((left,right)):
            sign=1 if v>=0 else -1
            if v and sign!=self.signs[i]:
                self.pwm[i].off()
                self.direction[2*i].value=int(sign<0)
                self.direction[2*i+1].value=int(sign>0)
                self.signs[i]=sign
            self.pwm[i].value=abs(v)

    def pressed(self):return any(b.is_pressed for b in self.buttons)

    def close(self):
        for d in self.pwm:
            try:d.off()
            except Exception:pass
        for d in reversed(self.devices):
            try:d.close()
            except Exception:pass


def read_value(command, pattern, base=None):
    try:
        r=subprocess.run(['vcgencmd',*command],capture_output=True,text=True,timeout=.3)
        m=re.search(pattern,r.stdout)
        return int(m.group(1),base) if base else float(m.group(1))
    except (OSError,subprocess.TimeoutExpired,ValueError,AttributeError):return None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--vision',action='store_true',help='Run Claude camera pipeline without its motor controller')
    p.add_argument('--hardware',action='store_true')
    p.add_argument('--simulate',action='store_true')
    p.add_argument('--case',choices=['idle','left','right','both','sequence','claude'],default='idle')
    p.add_argument('--placement',choices=['lifted','floor'],default='lifted')
    p.add_argument('--duty',type=float,default=.2)
    p.add_argument('--ramp',type=float,default=1)
    p.add_argument('--hold',type=float,default=.5)
    p.add_argument('--frequency',type=float,default=1000)
    args=p.parse_args();validate(args.case,args.placement,args.duty,args.ramp,args.hold,args.frequency)
    if args.hardware and args.simulate:p.error('hardware and simulate are exclusive')
    if args.case!='idle' and not (args.hardware or args.simulate):p.error('Motor case requires --hardware or --simulate')
    if args.case=='claude':
        args.vision=True
        if args.simulate:p.error('Claude profile uses the installed Pi code; use its offline validation instead')
    # Avoid two copies of THIS tool; gpiozero/lgpio reservations also protect pins.
    lock=None
    if args.hardware:
        import fcntl
        lock=open('/tmp/autodrive-power-probe.lock','w')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Load heavy modules before starting telemetry deadlines; no GPIO/camera opened here.
    if args.vision and not args.simulate:
        import cv2
        import numpy
        from picamera2 import Picamera2
        sys.path.insert(0,'/home/pi/TEST')
        import lane_pipeline
    controller=None
    if args.case=='claude':
        from claude_profile import ClaudeProfile
        controller=ClaudeProfile()
        args.frequency=controller.module.PWM_FREQ_HZ
    output=queue.Queue(maxsize=500)
    abort=threading.Event();quit_threads=threading.Event();start=threading.Event()
    state={'heartbeat':time.monotonic(),'sample':None,'phase':'waiting','left':0.,'right':0.}
    mutex=threading.Lock()
    seq=0

    def emit(kind,**data):
        nonlocal seq
        seq+=1
        row={'event':kind,'pi_monotonic':time.monotonic(),'seq':seq,**data}
        try:output.put_nowait(row)
        except queue.Full:abort.set()  # Logger/network congestion stops the trial.

    def writer():
        while True:
            row=output.get()
            try:print(json.dumps(row,allow_nan=False),flush=True)
            except (OSError,ValueError):abort.set();return
            finally:output.task_done()

    def reader():
        try:
            for line in sys.stdin:
                if len(line)>4096:abort.set();break
                try:cmd=json.loads(line).get('cmd')
                except (ValueError,AttributeError):abort.set();break
                if cmd=='ping':
                    with mutex:state['heartbeat']=time.monotonic()
                elif cmd=='start':start.set()
                elif cmd=='stop':abort.set()
                else:abort.set()
        finally:abort.set()

    def sensor():
        while not quit_threads.is_set():
            t=time.monotonic()
            if args.simulate:v,temp,th=5.05,45.,0;voltage_at=t
            else:
                v=read_value(['pmic_read_adc','EXT5V_V'],r'=([0-9.]+)V')
                voltage_at=time.monotonic()
                temp=read_value(['measure_temp'],r'=([0-9.]+)')
                th=read_value(['get_throttled'],r'=(0x[0-9a-fA-F]+)',16)
            sample={'voltage':v,'temp':temp,'throttled':th,'voltage_at':voltage_at,
                    'read_duration_s':time.monotonic()-t}
            with mutex:
                state['sample']=sample
                snap={k:state[k] for k in ('phase','left','right')}
            emit('sample',**sample,**snap)
            quit_threads.wait(max(0,.1-(time.monotonic()-t)))

    for f in (writer,reader,sensor):threading.Thread(target=f,daemon=True).start()
    for sig in (signal.SIGINT,signal.SIGTERM,getattr(signal,'SIGHUP',signal.SIGTERM)):
        signal.signal(sig,lambda *_:abort.set())
    boot=Path('/proc/sys/kernel/random/boot_id')
    reset=Path('/proc/device-tree/chosen/power/power_reset')
    emit('ready',config=vars(args),boot_id=boot.read_text().strip() if boot.exists() else None,
         power_reset_hex=reset.read_bytes().hex() if reset.exists() else None,
         gpio_opened=False)
    if controller:emit('controller_config',**controller.metadata())
    motors=None;reason='not_started'
    vision=None
    try:
        deadline=time.monotonic()+300
        while not start.is_set():
            with mutex:hb=state['heartbeat']
            if abort.is_set() or time.monotonic()-hb>1 or time.monotonic()>deadline:
                reason='waiting_cancelled_or_timeout';return
            time.sleep(.02)
        # Camera worker never owns GPIO; motor loop independently watches its freshness.
        if args.vision and not args.simulate:
            from vision_load import VisionLoad
            vision=VisionLoad();vision.start()
            deadline=time.monotonic()+20
            while not vision.last_frame:
                with mutex:sample,hb=state['sample'],state['heartbeat']
                reason=safety_reason(sample,time.monotonic(),hb,abort.is_set())
                if reason:return
                if vision.error:reason='vision_error:'+vision.error;return
                if time.monotonic()>deadline:reason='vision_start_timeout';return
                time.sleep(.02)
            emit('vision_ready')
        # Baseline always recorded before opening motor pins.
        with mutex:state['phase']='baseline'
        emit('baseline_start')
        end=time.monotonic()+3
        while time.monotonic()<end:
            with mutex:sample,hb=state['sample'],state['heartbeat']
            reason=safety_reason(sample,time.monotonic(),hb,abort.is_set())
            if reason:return
            time.sleep(.02)
        if args.hardware and args.case!='idle':motors=Motors(args.frequency)
        plan = [('forward',1,1),('left',.35,1),('right',1,.35),('reverse',-1,-1)] if args.case=='sequence' else [('active',int(args.case in ('left','both')),int(args.case in ('right','both')))]
        if controller:
            plan=[('claude_forward',0.,0),('claude_left',-.05,0),('claude_right',.05,0),('claude_forward_end',0.,0)]
        previous=time.monotonic()
        for name,lmul,rmul in plan:
            with mutex:state['phase']=name
            emit('motor_trial_start',case=name,physical=bool(motors))
            t0=time.monotonic()
            while time.monotonic()-t0<(args.hold if controller else 2*args.ramp+args.hold):
                now=time.monotonic()
                with mutex:sample,hb=state['sample'],state['heartbeat']
                reason=safety_reason(sample,now,hb,abort.is_set())
                if reason:return
                if vision and (vision.error or now-vision.last_frame>2):
                    reason='vision_failed_or_stale';return
                if motors and motors.pressed():reason='board_button';return
                duty=duty_at(now-t0,args.duty,args.ramp,args.hold)
                if controller:
                    left,right=controller.step(lmul,args.duty,now-previous,sample['voltage'])
                    previous=now
                else:left,right=duty*lmul,duty*rmul
                if motors:motors.set(left,right)
                with mutex:state.update(left=left,right=right)
                time.sleep(1/15 if controller else .02)
            if controller:
                emit('segment_transition',segment=name)
                continue
            if motors:motors.set(0,0)
            with mutex:state.update(phase='pause_'+name,left=0.,right=0.)
            emit('segment_stopped',segment=name)
            if args.case=='sequence':
                end_pause=time.monotonic()+2
                while time.monotonic()<end_pause:
                    with mutex:sample,hb=state['sample'],state['heartbeat']
                    reason=safety_reason(sample,time.monotonic(),hb,abort.is_set())
                    if reason:return
                    if motors and motors.pressed():reason='board_button';return
                    time.sleep(.02)
        if motors:motors.set(0,0)
        with mutex:state.update(phase='recovery',left=0.,right=0.)
        emit('motor_stopped')
        end=time.monotonic()+2
        while time.monotonic()<end and not abort.is_set():time.sleep(.02)
        reason='completed'
    except BaseException as e:
        reason=f'exception:{type(e).__name__}:{e}'
    finally:
        if motors:motors.close()
        if vision:
            vision.stop_event.set();vision.join(timeout=2)
            emit('vision_summary',frames=vision.frames,error=vision.error)
        with mutex:state.update(phase='ended',left=0.,right=0.)
        quit_threads.set();emit('end',reason=reason)
        # No unbounded join: stdout could be blocked after a network failure.
        end=time.monotonic()+.5
        while output.unfinished_tasks and time.monotonic()<end:time.sleep(.01)
        if lock:lock.close()


if __name__=='__main__':main()
