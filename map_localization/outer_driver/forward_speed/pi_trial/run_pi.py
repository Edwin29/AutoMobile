"""Pi 5 camera runner; shadow by default, explicit --drive to energize GPIO."""
import argparse
from contextlib import ExitStack
import csv
import math
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
import cv2
import numpy as np
from controller import Driver

def voltage():
    reply=subprocess.run(["vcgencmd","pmic_read_adc","EXT5V_V"],capture_output=True,
                         text=True,check=True,timeout=.22)
    m=re.search(r"=\s*([0-9.]+)V",reply.stdout)
    if not m or float(m[1])<4.85:
        raise RuntimeError("5V rail low or unreadable: "+reply.stdout.strip())
    return float(m[1])

class Wheels:
    def __init__(self,stack):
        from gpiozero import DigitalOutputDevice, PWMOutputDevice, Button
        self.io=[]
        for pwm,a,b in ((18,22,27),(23,25,24)):
            self.io.append((stack.enter_context(PWMOutputDevice(pwm,frequency=1000,initial_value=0)),
                            stack.enter_context(DigitalOutputDevice(a)),stack.enter_context(DigitalOutputDevice(b))))
        self.buttons=[stack.enter_context(Button(p,pull_up=False)) for p in (5,6,13,19)]
        self.lock=threading.Lock()
        self.deadline=time.monotonic()+.32
        self.fault=""
        self.done=threading.Event()
        self.thread=threading.Thread(target=self.watch,daemon=True)
        self.thread.start()
        stack.callback(self.close)
    def zero(self):
        for pwm,a,b in self.io:
            pwm.value=0
            a.off()
            b.off()
    def watch(self):
        while not self.done.wait(.015):
            with self.lock:
                if time.monotonic()>self.deadline or any(b.is_pressed for b in self.buttons):
                    self.fault="stale drive command or stop button"
                    self.zero()
                    return
    def set(self,cmd):
        with self.lock:
            if self.fault or time.monotonic()>self.deadline or any(b.is_pressed for b in self.buttons):
                self.zero()
                raise RuntimeError(self.fault or "expired command or stop button")
            for (pwm,a,b),value in zip(self.io,cmd):
                if not math.isfinite(value) or abs(value)>1:
                    self.zero()
                    raise ValueError("invalid PWM")
                forward=value>=0
                if (bool(b.value)!=forward) and pwm.value:
                    pwm.value=0
                a.value=not forward
                b.value=forward
                pwm.value=abs(float(value))
            self.deadline=time.monotonic()+.32
    def close(self):
        self.done.set()
        self.thread.join()
        with self.lock:
            self.zero()

def make_driver(args):
    if args.output_feedback:
        from feedback_driver import Feedback, FixedGain
        driver=Feedback(args.direction)
        driver.yaw_tracker=FixedGain()
        driver.right_mix=driver.yaw_tracker.right=0.6217314378064263
    else:
        driver=Driver(args.direction,adaptive_yaw=args.adaptive_yaw)
    driver.target_laps=args.laps
    if args.photo_footprint:
        rim=np.load(Path(__file__).with_name("photo_plus5mm_rim.npy"),allow_pickle=False)
        if rim.ndim!=2 or rim.shape[1]!=2 or len(rim)<3 or not np.isfinite(rim).all():
            raise ValueError("invalid photo footprint")
        driver.rim=rim
    return driver

def run(args):
    from picamera2 import Picamera2
    driver=make_driver(args)
    cv2.setNumThreads(1)
    with ExitStack() as stack:
        camera=Picamera2()
        stack.callback(camera.close)
        camera.configure(camera.create_video_configuration(main={"format":"RGB888","size":tuple(map(int,driver.size))},
                                                           controls={"FrameRate":15},queue=False))
        camera.start()
        stack.callback(camera.stop)
        time.sleep(1.2)
        destination=args.log or Path("map_run_"+time.strftime("%Y%m%d_%H%M%S")+".csv")
        stream=stack.enter_context(destination.open("x",newline="",encoding="utf8"))
        writer=csv.writer(stream)
        debug=None
        if args.edge_debug:
            from edge_debug import EdgeBuffer
            debug=EdgeBuffer(driver,destination)
            stack.callback(debug.save)  # Wheels are closed before flushing images.
        writer.writerow(("t","mode","x","y","yaw_deg","edge_residual_cm","margin_cm",
                         "requested_L","requested_R","applied_L","applied_R","reason",
                         "next_left_yaw_gain","next_right_yaw_gain","gain_mark","window_pwm_deg","window_visual_deg",
                         "next_left_output_gain","next_right_output_gain","direction","target_laps",
                         "target_distance_cm","estimated_laps","footprint","prediction_forward_gain"))
        wheels=None
        if args.drive:
            voltage()
            wheels=Wheels(stack)
        applied=(0.,0.)
        boot=lambda:time.clock_gettime(time.CLOCK_BOOTTIME)
        begin=boot()
        last_stamp=None
        next_voltage=begin+1
        while boot()-begin<args.duration:
            req=camera.capture_request()
            try:
                metadata=req.get_metadata()
                frame=cv2.flip(req.make_array("main"),-1)
            finally:
                req.release()
            stamp=metadata["SensorTimestamp"]*1e-9
            age=boot()-stamp
            dt=stamp-last_stamp if last_stamp is not None else 1/15
            last_stamp=stamp
            if debug:
                driver.edge_observation=None
            if age<0 or age>.22:
                cmd=driver.stop("stale camera frame")
            else:
                cmd=driver.step(frame,dt,applied)
            if wheels and boot()>=next_voltage:
                voltage()
                next_voltage=boot()+1
            if boot()-stamp>.27:
                cmd=driver.stop("stale processed frame")
            if not driver.reason and boot()-begin>=args.duration:
                cmd=driver.stop("duration reached")
            if wheels:
                wheels.set(cmd)
                applied=tuple(cmd)
            tracker=driver.yaw_tracker
            event=(tracker.event or {}) if tracker and not driver.reason else {}
            writer.writerow((boot()-begin,"drive" if wheels else "shadow",*driver.pose[:2],
                             math.degrees(driver.pose[2]),driver.residual,driver.margin,*cmd,*applied,driver.reason,
                             tracker.left if tracker else 1., tracker.right if tracker else 1.,
                             event.get("mark"),event.get("raw_deg"),event.get("visual_deg"),
                             1/driver.left_mix if args.output_feedback else "",
                             1/driver.right_mix if args.output_feedback else "",args.direction,args.laps,
                             args.laps*driver.arc[-1],driver.distance/driver.arc[-1],
                             "photo_plus5mm" if args.photo_footprint else "rectangle",
                             driver.last_forward_gain))
            stream.flush()
            if debug:
                debug.append(frame,stamp-begin,metadata,driver.edge_observation,driver.reason)
            if driver.reason:
                break
    print(driver.reason or "duration reached",destination)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--direction",choices=["cw","ccw"],required=True)
    ap.add_argument("--duration",type=float,default=8.)
    ap.add_argument("--laps",type=int,choices=(1,2),default=1)
    ap.add_argument("--drive",action="store_true")
    ap.add_argument("--edge-debug",action="store_true",help="retain final 24 raw frames and map residuals")
    ap.add_argument("--photo-footprint",action="store_true",help="use the sampled photo outline with a 5 mm buffer")
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument("--adaptive-yaw",action="store_true",help="learn a separate yaw gain per turn direction from floor paint")
    mode.add_argument("--output-feedback",action="store_true",help="use the calibrated outer-lap output feedback")
    ap.add_argument("--log",type=Path)
    args=ap.parse_args()
    if not math.isfinite(args.duration) or args.duration<=0:
        ap.error("duration must be finite and positive")
    def terminate(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,terminate)
    run(args)
