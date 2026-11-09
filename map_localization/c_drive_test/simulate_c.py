"""Replay the archived 2026-10-31 drive.main against an ideal desktop camera/plant."""
import argparse
import ast
import contextlib
import csv
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ARCHIVE = PROJECT / "junction_logic/review_video"
SIM = PROJECT / "junction_logic/visibility_review/source/sim_drive.py"
sys.path.insert(0, str(ARCHIVE))
import lane_pipeline as lane

def noop(*args, **kwargs):
    pass

class Recorder:
    def __init__(self, *args):
        self.state, self.samples = {}, []
        self.volt, self.power = 5., None
        self.events = []
    start = stop = close = noop
    def event(self, value):
        self.events.append(value)
    def log(self, *args):
        return {"steer_wheel_deg": 0}

def module(name, **entries):
    sys.modules[name] = types.SimpleNamespace(**entries)

module("power_rec", Recorder=Recorder, read_ext5v=lambda:5., vcgencmd=lambda *a:"0x0")
module("steer_rec", SteerRecorder=Recorder)
module("video_rec", ChunkedVideo=noop, draw_overlay=noop)
module("plot_power", plot=noop)
module("plot_steer", plot=noop)
import lane_drive as drive

# Load the map renderer alone; use the Oct 31 calibration and vision functions.
tree = ast.parse(SIM.read_text(encoding="utf8"))
world_code = [n for n in tree.body if isinstance(n, (ast.Assign,ast.ClassDef,ast.FunctionDef))
              and not (isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="HERE")]
end = next(i for i,n in enumerate(world_code) if isinstance(n,ast.FunctionDef) and n.name=="yaw_rate")
env = dict(cv2=cv2,np=np,L=lane,HERE=ARCHIVE)
exec(compile(ast.Module(body=world_code[:end],type_ignores=[]),str(SIM),"exec"),env)
DT = 1/15
FWD = 24.5/2-2.14
base = 13.
corner = np.array([[-8,-12.25],[8,-12.25],[8,12.25],[-8,12.25]])
rim = np.concatenate([np.linspace(a,b,math.ceil(np.linalg.norm(a-b)/.25),endpoint=False)
                      for a,b in zip(corner,np.roll(corner,-1,axis=0))])

class Plant:
    def __init__(self, direction, pivot_behind_center_cm=0., left_yaw_response=1., right_yaw_response=1.,
                 drift_deg_s_at_30=0., forward_scale=1.):
        self.world = env["World"]()
        self.pivot_behind_center_cm = pivot_behind_center_cm
        self.left_yaw_response = left_yaw_response
        self.right_yaw_response = right_yaw_response
        self.drift_deg_s_at_30 = drift_deg_s_at_30
        self.forward_scale = forward_scale
        self.sign = 1 if direction=="cw" else -1
        self.pos = np.array([21.5,-87.25]) # body centre at photo-left crosswalk
        self.yaw = -self.sign*np.pi/2
        self.angle, self.travel, self.tick = math.atan2(self.pos[1],self.pos[0]), 0., 0
        self.out = (0.,0.)
        self.rows = []
    def observe(self):
        t = self.yaw
        rot = np.array([[math.cos(t),-math.sin(t)],[math.sin(t),math.cos(t)]])
        xy = rim@rot+self.pos
        cam = self.pos + FWD*rot[1]
        margins = self.world.sd_at(xy[:,0],xy[:,1])
        self.rows.append((self.tick*DT,*self.pos,math.degrees(t),float(margins.min()),*self.out))
        return self.world.render((*cam,t))
    def capture_array(self):
        l,r = np.array(self.out)*16. # cm/s for unit wheel command
        delta = (l-r)/base*DT
        delta *= self.left_yaw_response if delta < 0 else self.right_yaw_response
        delta += math.radians(self.drift_deg_s_at_30)*DT*(sum(self.out)/2)/.30
        mid = self.yaw+delta/2
        self.pos += self.forward_scale*(l+r)/2*DT*np.sinc(delta/(2*np.pi))*np.array([math.sin(mid),math.cos(mid)])
        # pos tracks the body centre. Its fixed offset from the pivot rotates too.
        self.pos += 2*self.pivot_behind_center_cm*math.sin(delta/2)*np.array([math.cos(mid),-math.sin(mid)])
        self.yaw += delta
        angle = math.atan2(self.pos[1],self.pos[0])
        self.travel -= self.sign*math.remainder(angle-self.angle,2*np.pi)
        self.angle = angle
        self.tick += 1
        return self.observe()
    create_video_configuration = align_configuration = configure = start = close = noop
    def set(self,l,r):
        self.out = l,r
    @property
    def is_pressed(self):
        return self.travel>=2*np.pi

def run(direction, profile):
    cv2.setNumThreads(1)
    plant = Plant(direction)
    module("picamera2",Picamera2=lambda:plant)
    module("gpiozero",Button=lambda *a,**kw:plant)
    drive.Motors=lambda:plant
    drive.time=types.SimpleNamespace(time=lambda:plant.tick*DT,sleep=noop)
    # Renderer already returns the upright, rectified image that analyze expects.
    drive.cv2=types.SimpleNamespace(**{k:getattr(cv2,k) for k in ("fisheye","CV_16SC2","INTER_LINEAR")},
                                    flip=lambda img,*a:img,remap=lambda img,*a:img)
    drive.os=types.SimpleNamespace(fsync=noop)
    log_capture=io.StringIO()
    options=["--corner-bias", "0" if profile=="no_bias" else "2.5",
             "--start-corner-delay","2" if profile in ("start2","extra") else "0"]
    if profile=="extra":
        options+=["--ccw"]
    sys.argv=["lane_drive.py","--duration","240","--corners","99","--save-every","0",*options]
    with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(log_capture):
        drive.HERE=Path(folder)
        for filename in ["ground.npz","calib_checkerboard.npz"]:
            shutil.copyfile(ARCHIVE/filename,drive.HERE/filename)
        drive.main()
        generated=next((drive.HERE/"runs").iterdir())
        settings=json.loads((generated/"params.json").read_text())
    arr=np.asarray(plant.rows)
    stages=[arr[(arr[:,1]*sx>50)&(arr[:,2]*sy>50),4] for sx,sy in [(1,-1),(1,1),(-1,1),(-1,-1)]]
    negative=np.flatnonzero(arr[:,4]<0)
    result=dict(direction=direction,profile=profile,parameters=settings,elapsed_s=arr[-1,0],
                full_revolution=plant.travel>=2*np.pi,turns=plant.travel/(2*np.pi),
                minimum_cm=arr[:,4].min(),negative_frame_percent=100*len(negative)/len(arr),
                first_negative_s=arr[negative[0],0] if len(negative) else None,
                corner_minimum_cm=dict(zip(["photo_bottom_left","photo_bottom_right","photo_top_right","photo_top_left"],
                                           [v.min() if len(v) else None for v in stages])),
                drive_end=next(s for s in log_capture.getvalue().splitlines() if s.startswith("[drive] 정지 (")))
    result["drive_end"]=result["drive_end"].split("기록:")[0].rstrip(" ,")
    (HERE/f"{direction}_{profile}.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf8")
    np.savetxt(HERE/f"{direction}_{profile}.csv",arr,delimiter=",",fmt="%.6f",header="t,x,y,yaw,body_clear_cm,previous_left,previous_right",comments="")
    print(direction,profile,result["full_revolution"],result["minimum_cm"],result["negative_frame_percent"],flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("direction",choices=["cw","ccw"])
    p.add_argument("profile",choices=["no_bias","base","start2","extra"])
    a=p.parse_args()
    run(a.direction,a.profile)
