"""One bounded, open-loop straight trial on Pi. No GPIO unless --drive is given."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import queue
import signal
import sys
import threading
import time
from contextlib import ExitStack

HERE = Path(__file__).resolve().parent
if not (HERE / "probe.py").exists():  # repository checkout; the zip bundles the same helper
    sys.path.insert(0, str(HERE.parent / "power_diagnostics"))
from probe import Motors, duty_at, read_value, safety_reason

RAMP, HOLD, DUTY, FREQUENCY = 0.5, 10.0, 0.30, 1000
TOTAL = 2*RAMP + HOLD

def telemetry(done, readings):
    while not done.is_set():
        volts = read_value(["pmic_read_adc", "EXT5V_V"], r"=([0-9.]+)V")
        stamp = time.monotonic()
        temp = read_value(["measure_temp"], r"=([0-9.]+)")
        flags = read_value(["get_throttled"], r"=(0x[0-9a-fA-F]+)", 16)
        readings.put(dict(voltage=volts, voltage_at=stamp, temp=temp, throttled=flags))
        done.wait(.08)

def trial(label, drive, *, duty_limit=DUTY, hold_s=HOLD, right_sign=1):
    total_s=2*RAMP+hold_s
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = HERE / "runs" / f"{label}_{stamp}"
    root.mkdir(parents=True)
    mode = "drive" if drive else "dry"
    report = dict(label=label,mode=mode,duty=duty_limit,frequency_hz=FREQUENCY,ramp_s=RAMP,hold_s=hold_s,right_sign=right_sign,
                  status="started", min_voltage=None, actual_hold_s=None)
    info = root / "trial.json"
    info.write_text(json.dumps(report,indent=2),encoding="utf8")
    print(f"{label}  {mode.upper()}  |  {root}",flush=True)
    reason = "interrupted"
    try:
        with ExitStack() as stack:
            log = stack.enter_context((root/"pwm.csv").open("x",newline="",encoding="utf8",buffering=1))
            writer = csv.writer(log)
            writer.writerow(("elapsed_s","phase","target_left","target_right","sent_left","sent_right","voltage"))
            motor = None
            sample = None
            readings = queue.SimpleQueue()
            if drive:
                import fcntl
                lock = stack.enter_context(open("/tmp/autodrive-power-probe.lock","a"))
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                done = threading.Event()
                stack.callback(done.set)
                threading.Thread(target=telemetry,args=(done,readings),daemon=True).start()
                sample = readings.get(timeout=3)
            def check():
                nonlocal sample
                if not drive:
                    return
                while True:
                    try: sample = readings.get_nowait()
                    except queue.Empty: break
                now = time.monotonic()
                fault = safety_reason(sample,now,now,False)
                if fault:
                    raise RuntimeError(fault)
                value = sample["voltage"]
                report["min_voltage"] = min(value,report["min_voltage"]) if report["min_voltage"] is not None else value
                if motor and motor.pressed():
                    raise RuntimeError("board button")
            check()
            if drive:
                motor = Motors(FREQUENCY)
                stack.callback(motor.close)
            print("READY - motion begins in 3 seconds" if drive else "READY - GPIO disabled",flush=True)
            for _ in range(60):
                check()
                time.sleep(.05)
            expired=threading.Event()
            shutdown=threading.Event()
            output_lock=threading.Lock()
            start = previous = time.monotonic()
            last_write=start
            if motor:
                def guard_output():
                    while not shutdown.wait(.02):
                        with output_lock:
                            at=time.monotonic()
                            if at-last_write>.25 or at>start+total_s+.12:
                                expired.set()
                                motor.set(0.,0.)
                                return
                guard=threading.Thread(target=guard_output,daemon=True)
                guard.start()
                def finish_guard():
                    shutdown.set()
                    guard.join()
                stack.callback(finish_guard)
            previous_phase = None
            hold_start = None
            while True:
                now = time.monotonic()
                elapsed = now-start
                if elapsed >= total_s:
                    break
                if now-previous > .18:
                    raise RuntimeError("control loop delayed")
                previous = now
                check()
                phase = "RAMP_UP" if elapsed<RAMP else "HOLD" if elapsed<RAMP+hold_s else "RAMP_DOWN"
                duty = duty_at(elapsed,duty_limit,RAMP,hold_s)
                if motor:
                    with output_lock:
                        if expired.is_set():
                            raise RuntimeError("output watchdog")
                        motor.set(duty,right_sign*duty)
                        last_write=now
                sent=duty if drive else 0.
                writer.writerow((round(elapsed,4),phase,duty,right_sign*duty,sent,right_sign*sent,sample["voltage"] if drive else ""))
                if phase != previous_phase:
                    if phase=="HOLD": hold_start=elapsed
                    if phase=="RAMP_DOWN": report["actual_hold_s"]=elapsed-hold_start
                    print(f"{phase}  {elapsed:.3f}s",flush=True)
                    previous_phase=phase
                time.sleep(.02)
            if motor:
                with output_lock:
                    motor.set(0.,0.)
                    if expired.is_set():
                        raise RuntimeError("output watchdog")
            reason = "completed"
    except (KeyboardInterrupt,Exception) as exc:
        reason = f"aborted: {exc or type(exc).__name__}"
    finally:
        if "start" in locals():
            with (root/"pwm.csv").open("a",newline="",encoding="utf8") as stream:
                csv.writer(stream).writerow((round(time.monotonic()-start,4),"END",0.,0.,0.,0.,""))
        report["status"] = reason
        info.write_text(json.dumps(report,indent=2),encoding="utf8")
    print(f"END  {reason}  |  {root}",flush=True)
    return reason=="completed"

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial",choices=["A1","B1","A2","B2","A3","B3"],required=True)
    parser.add_argument("--drive",action="store_true",help="energize the Pi motor outputs for one trial")
    args=parser.parse_args()
    def interrupt(*_):
        raise KeyboardInterrupt
    for sig in (signal.SIGINT,signal.SIGTERM,getattr(signal,"SIGHUP",signal.SIGTERM)):
        signal.signal(sig,interrupt)
    sys.exit(0 if trial(args.trial,args.drive) else 1)
