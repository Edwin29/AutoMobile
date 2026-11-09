"""Run on laptop: persist telemetry locally, provide a short remote control lease."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from probe import validate


def summarize(rows):
    remote=[r['remote'] for r in rows if 'remote' in r]
    samples=[r for r in remote if r.get('event')=='sample']
    phases={}
    for phase in dict.fromkeys(r.get('phase') for r in samples):
        vals=[r['voltage'] for r in samples if r.get('phase')==phase and r.get('voltage') is not None]
        phases[phase]={'samples':len(vals),'min_voltage':min(vals) if vals else None,
                       'mean_voltage':sum(vals)/len(vals) if vals else None}
    ending=next((r for r in reversed(remote) if r.get('event')=='end'),None)
    gaps=[b['voltage_at']-a['voltage_at'] for a,b in zip(samples,samples[1:])]
    return {'phases':phases,'end_reason':ending['reason'] if ending else 'no_end_event_transport_or_device_failure_unconfirmed',
            'max_observed_sample_gap_s':max(gaps) if gaps else None,
            'last_received_sample':samples[-1] if samples else None,
            'note':'Transport loss alone does not prove power loss. ADC polling cannot resolve millisecond transients. PWM duty is not measured motor current or speed.'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--host',default='pi@PI_HOST')
    p.add_argument('--case',choices=['idle','left','right','both','sequence','claude'],default='idle')
    p.add_argument('--placement',choices=['lifted','floor'],default='lifted')
    p.add_argument('--duty',type=float,default=.2)
    p.add_argument('--ramp',type=float,default=1.)
    p.add_argument('--hold',type=float,default=.5)
    p.add_argument('--frequency',type=float,default=1000)
    p.add_argument('--vision',action='store_true')
    p.add_argument('--hardware',action='store_true')
    p.add_argument('--simulate',action='store_true')
    p.add_argument('--auto',action='store_true',help='Idle/simulation only')
    p.add_argument('--note',default='',help='Battery/placement/observations for comparison')
    args=p.parse_args();validate(args.case,args.placement,args.duty,args.ramp,args.hold,args.frequency)
    if args.case=='claude':
        args.vision=True
        if args.simulate:p.error('Claude profile requires the installed Pi controller')
    if args.hardware and args.simulate:p.error('Choose hardware OR simulation')
    if args.case!='idle' and not(args.hardware or args.simulate):p.error('Use --hardware for physical motor operation')
    if args.auto and args.hardware and args.case!='idle':p.error('Physical motor trials always require typed confirmation')
    root=Path(__file__).parent
    run=root/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True)
    (run/'config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
    flags=['--case',args.case,'--placement',args.placement,'--duty',str(args.duty),
           '--ramp',str(args.ramp),'--hold',str(args.hold),'--frequency',str(args.frequency)]
    if args.vision:flags.append('--vision')
    if args.simulate:
        cmd=[sys.executable,'-u',str(root/'probe.py'),'--simulate',*flags]
    else:
        cmd=['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
             '-o','ConnectTimeout=6','-o','ServerAliveInterval=2','-o','ServerAliveCountMax=2',
             args.host,'/home/pi/AI_CAR/env_hw/bin/python','-u','/home/pi/power_diagnostics/probe.py',*flags]
        if args.hardware:cmd.append('--hardware')
    print('LOG:',run,flush=True)
    stop=threading.Event();ready=threading.Event();ended=threading.Event()
    rows=[];last_rx=[time.monotonic()];send_lock=threading.Lock()
    stderr=open(run/'ssh_stderr.txt','w',encoding='utf-8')
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,
                          text=True,encoding='utf-8',bufsize=1)

    def send(command):
        with send_lock:
            if proc.poll() is None:
                proc.stdin.write(json.dumps({'cmd':command})+'\n');proc.stdin.flush()

    def receive():
        try:
            with open(run/'telemetry.jsonl','w',encoding='utf-8') as f:
                sync=0;printed=0
                for line in proc.stdout:
                    try:remote=json.loads(line)
                    except ValueError:remote={'event':'unparsed','text':line.rstrip()}
                    last_rx[0]=time.monotonic()
                    row={'received_utc':datetime.now(timezone.utc).isoformat(),'local_monotonic':last_rx[0],'remote':remote}
                    rows.append(row);f.write(json.dumps(row)+'\n');f.flush()
                    if last_rx[0]-sync>=.5 or remote.get('event')!='sample':
                        os.fsync(f.fileno());sync=last_rx[0]
                    if remote.get('event')=='ready':ready.set()
                    if remote.get('event')=='end':ended.set()
                    if remote.get('event')!='sample' or last_rx[0]-printed>=1:
                        print(json.dumps(remote),flush=True);printed=last_rx[0]
                os.fsync(f.fileno())
        finally:stop.set()

    def heartbeat():
        while not stop.is_set():
            try:
                if time.monotonic()-last_rx[0] > (2 if ready.is_set() else 10):
                    send('stop');stop.set();return
                send('ping')
            except (OSError,ValueError):stop.set();return
            stop.wait(.2)

    rx=threading.Thread(target=receive,daemon=True);rx.start()
    hb=threading.Thread(target=heartbeat,daemon=True);hb.start()
    try:
        if not ready.wait(10):raise RuntimeError('Remote probe not ready; see ssh_stderr.txt')
        if args.case!='idle' and not args.auto:
            expected='START '+args.placement
            print('Close every other motor program. Check wheels/clear space. Board SW1-4 or Ctrl+C stops.',flush=True)
            if input(f'Type {expected!r} to run ONE {args.case} trial: ').strip()!=expected:
                send('stop');return
        if stop.is_set():raise RuntimeError('Telemetry disconnected before start')
        time.sleep(.3)  # Allow first sensor sample before baseline.
        send('start')
        deadline=time.monotonic()+40+(4*(2*args.ramp+args.hold+2) if args.case in ('sequence','claude') else 2*args.ramp+args.hold)
        while not stop.is_set() and not ended.is_set() and time.monotonic()<deadline:
            time.sleep(.1)
    except KeyboardInterrupt:print('Stopping trial...',flush=True)
    finally:
        try:send('stop')
        except (OSError,ValueError):pass
        stop.set()
        try:proc.stdin.close()
        except OSError:pass
        try:proc.wait(timeout=4)
        except subprocess.TimeoutExpired:
            proc.terminate();proc.wait(timeout=4)
        rx.join(timeout=2);stderr.close()
        summary=summarize(rows)
        summary['ssh_exit_code']=proc.returncode
        (run/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print('SUMMARY:',json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
