"""Three one-second equal-duty starts; default dry, physical output only --drive."""
import argparse,json,signal,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--drive',action='store_true');a=p.parse_args()
    def interrupt(*_):raise KeyboardInterrupt
    for name in ('SIGINT','SIGTERM','SIGHUP'):
        if hasattr(signal,name):signal.signal(getattr(signal,name),interrupt)
    output=Path(__file__).parent/'runs'/('start_threshold_'+time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True,exist_ok=False)
    motor=None;reason='completed'
    try:
        if a.drive:
            from live_outer_trial import GuardedMotors
            motor=GuardedMotors(6.,.4)
            motor.arm()
        with (output/'steps.jsonl').open('w',buffering=1) as log:
            for duty in (.30,.35,.40):
                print(f'START {duty*100:.0f}% BOTH SIDES, ONE SECOND',flush=True)
                for phase,value,duration in [('start',duty,1.),('stopped',0.,.4)]:
                    until=time.monotonic()+duration
                    while time.monotonic()<until:
                        actual=motor.write([value,value]) if motor else [0.,0.]
                        log.write(json.dumps(dict(at=time.monotonic(),phase=phase,trial=duty,
                            requested=[value,value],actual=actual,telemetry=motor.sample if motor else None))+'\n')
                        time.sleep(.02)
    except (KeyboardInterrupt,Exception) as exc:reason=str(exc) or type(exc).__name__
    finally:
        if motor:motor.close()
        summary=dict(reason=reason,drive=a.drive,final_output=[0,0],output=str(output))
        (output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
