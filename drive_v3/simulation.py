"""Closed-loop raster camera simulation; truth is isolated from Runtime after initialization."""
from pathlib import Path
import argparse
import json
import time
import cv2
import numpy as np

from .camera import CameraGeometry
from .map_model import TrackMap, HERE
from .geometry import to_world, advance, footprint, wrap
from .runtime import Runtime
from .settings import load_settings
from .provenance import code_manifest

PROCESS_CODE_MANIFEST=code_manifest()


def json_default(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)


class Renderer:
    def __init__(self,track,camera):
        self.track=track;self.camera=camera
        self.ppcm=4.; self.origin=-125.
        y,x=np.mgrid[0:1001,0:1001]
        world=np.stack((x/self.ppcm+self.origin,y/self.ppcm+self.origin),axis=-1)
        margin=track.clearance(world)
        texture=np.empty((*x.shape,3),np.uint8);texture[:]=(95,215,125)
        texture[margin>=0]=(47,60,50)
        texture[np.abs(margin)<.22]=(70,220,210)
        def pixel(p): return tuple(np.rint((np.array(p)-self.origin)*self.ppcm).astype(int))
        for m in track.landmarks:
            if m["kind"]=="blue_circle_candidate":
                cv2.circle(texture,pixel(m["xy"]),round(m["radius_cm"]*self.ppcm),(225,165,30),-1)
            elif m["kind"] in ("dashline","red_line"):
                a,b=np.array(m["endpoints"])
                if m["kind"]=="red_line":cv2.line(texture,pixel(a),pixel(b),(50,50,200),2)
                else:
                    n=max(1,int(np.linalg.norm(b-a)/1.5))
                    for k in range(n):cv2.line(texture,pixel(a+(b-a)*k/n),pixel(a+(b-a)*(k+.65)/n),(190,120,200),2)
            elif m["kind"]=="crosswalk":
                center=np.array(m["xy"])
                for offset in np.arange(-5.4,5.5,1.8):
                    q=np.array([[offset-.45,-3.],[offset+.45,3.]])
                    if m["approach"]=="ew":q=q[:,::-1]
                    cv2.rectangle(texture,pixel(center+q[0]),pixel(center+q[1]),(215,220,220),-1)
        self.texture=texture
        yy,xx=np.mgrid[:camera.size[1],:camera.size[0]]
        uv=np.c_[xx.ravel(),yy.ravel()].astype(float).reshape(-1,1,2)
        bev=cv2.perspectiveTransform(uv,camera.M).reshape(-1,2)
        self.ground=camera.pixels_to_body(bev)
        self.valid=(self.ground[:,1]>camera.lens_ahead)&(self.ground[:,1]<150)&(np.abs(self.ground[:,0])<150)

    def render(self,pose,scenario="baseline",time_s=0.):
        world=to_world(self.ground,pose)
        maps=((world-self.origin)*self.ppcm).astype(np.float32)
        shape=(self.camera.size[1],self.camera.size[0])
        image=cv2.remap(self.texture,maps[:,0].reshape(shape),maps[:,1].reshape(shape),cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,borderValue=(90,90,90))
        image.reshape(-1,3)[~self.valid]=(90,90,90)
        if scenario in ("short_dropout","long_dropout") and 8<time_s<8+(.3 if scenario=="short_dropout" else 1.5):image[:]=245
        if scenario=="glare" and 8<time_s<13:
            cv2.ellipse(image,(200,335),(95,70),20,0,360,(250,250,250),-1)
        if scenario=="occlusion" and 8<time_s<13:
            cv2.rectangle(image,(0,160),(250,450),(70,70,70),-1)
        return image


def run(output,direction="ccw",scenario="baseline",duration=200.,dt=.2,video=False,junction=None,controller=None,vo=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    provenance=PROCESS_CODE_MANIFEST
    (output/"code_manifest.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    cfg,config_id=load_settings();track=TrackMap()
    if controller:cfg["controller"]["mode"]=controller
    cfg["estimator"]["use_visual_odometry"]=vo
    route=track.outer_route(direction)
    if junction:
        node,entry,turn=junction.split(":")
        route=track.junction_routes(node,entry,cfg["vehicle"])[turn]
    # Start on a long straight, explicit initial pose; subsequent truth never enters estimator.
    index=route.nearest([-86.25,-20.25]) if route.closed else 0
    pose=np.r_[route.points[index],route.heading[index]]
    render=Renderer(track,CameraGeometry(HERE/"config/camera_legacy_track.json"))
    rim=footprint(cfg["vehicle"])
    records=[];progress=0.;last_s=route.s[index];minimum=float("inf");zero_since=None
    writer=None
    if video:writer=cv2.VideoWriter(str(output/"preview.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),1/dt,(960,480))
    start=time.perf_counter(); applied=np.zeros(2);queue=[]
    outcome="time_limit"
    with Runtime(pose,route,cfg,clock_id="sim") as rt, (output/"steps.jsonl").open("w",encoding="utf-8") as stream:
        for step in range(1,int(duration/dt)+1):
            now=step*dt
            # Independent plant parameters, including physical wheel deadband and left bias.
            wheel=np.where(np.abs(applied)>=.12,applied,0.)*16
            v=wheel.mean()*(1.27 if scenario=="speed_bias" else 1.)
            omega=(wheel[0]-wheel[1])/13
            if scenario=="yaw_loss":omega*=.645 if omega<0 else .632
            if scenario=="left_bias" and v>0:omega-=.035
            pose=advance(pose,v,omega,dt)
            image=render.render(pose,scenario,now)
            record,obs,debug,path=rt.step(image,now,now,rectified=True)
            queue.append(np.asarray(record["pwm"]))
            applied=queue.pop(0) if len(queue)> (1 if scenario=="delay" else 0) else np.zeros(2)
            margin=float(track.clearance(to_world(rim,pose)).min());minimum=min(minimum,margin)
            i=route.nearest(pose[:2]);s=route.s[i];delta=s-last_s
            if route.closed:delta=(delta+route.s[-1]/2)%route.s[-1]-route.s[-1]/2
            progress+=delta;last_s=s
            record.update(truth=pose.tolist(),true_margin_cm=margin,progress_cm=progress,
                          pose_error_cm=float(np.linalg.norm(pose[:2]-rt.estimator.primary.mean[:2])),
                          yaw_error_rad=float(wrap(pose[2]-rt.estimator.primary.mean[2])))
            stream.write(json.dumps(record,default=json_default)+"\n");records.append(record)
            if writer:
                frame=image.copy()
                if len(path.points_body_cm)>1:
                    pixels=rt.camera.body_to_rectified(path.points_body_cm)
                    cv2.polylines(frame,[np.clip(pixels,-10000,10000).astype(np.int32)],False,(255,0,255),2)
                canvas=np.zeros((480,960,3),np.uint8);canvas[:,:640]=frame
                for row,text in enumerate([f"{direction} / {scenario}",f"t {now:.1f}s  progress {progress:.1f}cm",record["path_reason"],
                                            f"margin {margin:.2f}cm",f"pose error {record['pose_error_cm']:.2f}cm",f"PWM {applied.round(3)}"]):
                    cv2.putText(canvas,text,(645,35+row*35),cv2.FONT_HERSHEY_SIMPLEX,.45,(220,220,220),1)
                writer.write(canvas)
            if margin<-.25:outcome="boundary_violation";break
            if route.closed and progress>=route.s[-1]-.5:outcome="lap_complete";break
            if not route.closed and rt.planner.completed_route:outcome="route_complete";break
            zero=np.max(np.abs(applied))<1e-6
            if zero and zero_since is None:zero_since=now
            if not zero:zero_since=None
            if zero_since is not None and now-zero_since>2.:outcome="held";break
    if writer:writer.release()
    timings=np.array([r["total_ms"] for r in records])
    summary=dict(direction=direction,scenario=scenario,junction=junction,controller=cfg["controller"]["mode"],vo=vo,outcome=outcome,frames=len(records),
                 simulated_s=records[-1]["now_s"],progress_cm=progress,route_length_cm=float(route.s[-1]),
                 minimum_margin_cm=minimum,last_reason=records[-1]["path_reason"],last_control=records[-1]["control_reason"],
                 max_pose_error_cm=max(r["pose_error_cm"] for r in records),
                 median_step_ms=float(np.median(timings)),p95_step_ms=float(np.percentile(timings,95)),
                 wall_s=time.perf_counter()-start,settings_hash=rt.config_id,map_hash=track.id,code_hash=provenance["sha256"],
                 source_changed_during_process=code_manifest()["sha256"]!=provenance["sha256"],
                 limits="Synthetic planar raster, no real shadows/3D objects; finite footprint samples; no physical guarantee")
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2));return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True);p.add_argument("--direction",choices=["cw","ccw"],default="ccw")
    p.add_argument("--scenario",choices=["baseline","speed_bias","yaw_loss","left_bias","delay","short_dropout","long_dropout","glare","occlusion"],default="baseline")
    p.add_argument("--seconds",type=float,default=200);p.add_argument("--dt",type=float,default=.2)
    p.add_argument("--video",action="store_true");p.add_argument("--junction");p.add_argument("--controller",choices=["feedback","pure_pursuit"]);p.add_argument("--vo",action="store_true")
    a=p.parse_args();run(a.output,a.direction,a.scenario,a.seconds,a.dt,a.video,a.junction,a.controller,a.vo)


if __name__=="__main__":main()
