"""Delayed image-map observations and actual signed-command feedback.

The map is for resolving direction instructions, not replacing neural steering.
No environment state, vehicle preset, route progress or truth is accepted.
"""
import copy
import json
from pathlib import Path
import sys
import numpy as np
from bench.controllers import VENDOR_V4
from bench.map_worker import snapshot,restore
from .goal_development import ROOT,sha
from .junction_stage02 import dump

REPORT=ROOT/'cnn_finetune/results/goal_mission_map_20261113'


class DelayedImageMap:
    EVERY=3
    LATENCY_S=.070

    def __init__(self,direction,start_hint):
        sys.path.insert(0,str(VENDOR_V4))
        import drive_v4.core as core
        if not Path(core.__file__).resolve().is_relative_to(VENDOR_V4.resolve()):
            raise RuntimeError('Image map requires an isolated process with the pinned drive_v4 core')
        self.core_path=Path(core.__file__).resolve()
        self.visible=core.Driver(core.load_config(),direction=direction,pose=np.asarray(start_hint,float),clock='bench')
        self.working=core.Driver(core.load_config(),direction=direction,pose=np.asarray(start_hint,float),clock='bench')
        self.pending=[];self.n=0;self.applied=[];self.observations=[]
        self.last_capture=-np.inf;self.last_now=-np.inf;self.last_applied=-np.inf

    def publish(self,now):
        while self.pending and self.pending[0]['ready']<=now+1e-10:
            p=self.pending.pop(0);restore(self.visible,p['snapshot'])
            self.observations.append(dict(capture=p['capture'],ready=p['ready'],published=float(now)))

    def observe(self,image,capture,now):
        if not np.isfinite([capture,now]).all() or capture<self.last_capture or now<max(capture,self.last_now):
            raise ValueError('Frame timestamps must be finite, ordered and available')
        self.last_capture=capture;self.last_now=now;self.publish(now)
        if self.n%self.EVERY==0:
            ready=capture+self.LATENCY_S
            # The working result can be computed immediately offline, but only
            # its delayed published copy is ever read by the direction resolver.
            self.working.step(image,capture,ready,rotate180=False,rectified=False)
            self.pending.append(dict(capture=float(capture),ready=float(ready),snapshot=copy.deepcopy(snapshot(self.working))))
        self.n+=1;self.publish(now)
        return self.estimate(now)

    def estimate(self,now):
        self.publish(now);d=self.visible
        return dict(pose=np.asarray(d.map.pose_at(now),float).copy(),sxy=float(d.map.weight()[1]),
            map_capture=d.map.t,latest_published=None if not self.observations else dict(self.observations[-1]))

    def on_applied(self,t,pwm):
        pwm=np.asarray(pwm,float)
        if pwm.shape!=(2,) or not np.isfinite(pwm).all() or np.max(abs(pwm))>1 or not np.isfinite(t) or t<self.last_applied:
            raise ValueError('Expected ordered actual signed PWM within[-1,1]')
        self.last_applied=t;self.publish(t);m=self.visible.motor
        # Same nominal signed-motion extension as HybridJunction. No true gains
        # or velocities; both estimator copies receive identical applied motion.
        speed=np.sign(pwm)*np.where(abs(pwm)>=m.start,abs(pwm),0.)*m.c['cm_s_per_duty']
        raw=float((speed[0]-speed[1])/m.c['track_cm']);v=float(speed.mean());w=float(raw*m.yaw_gain(raw))
        for d in [self.visible,self.working]:
            d.pwm=pwm.copy();d.map.command(t,v,w);d.raw_cmds.append((float(t),raw));d.body_cmds.append((float(t),v,w))
            if len(d.raw_cmds)>200:d.raw_cmds=d.raw_cmds[-100:]
            if len(d.body_cmds)>200:d.body_cmds=d.body_cmds[-100:]
        self.applied.append(dict(t=float(t),pwm=pwm.copy(),nominal_v=v,nominal_w=w))

    def set_route(self,points):
        from drive_v3.map_model import Route
        points=np.asarray(points,float)
        if points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or not np.isfinite(points).all():raise ValueError('Expected finite map route points')
        for d in [self.visible,self.working]:
            rt=Route('neural_mission_instruction_route',points.copy());d.route=d.map.route=rt
            d.map._curvature_at=lambda s,rt=rt:np.interp(np.clip(s,0,rt.s[-1]),rt.s,rt.curvature)


def check():
    # This order reproduces the package contamination that the v2 resolver fixed.
    from .goal_mission_direction import DirectionResolver
    from bench.world import Track,Camera,Renderer
    track=Track();hint=track.start_pose('cw',14.5,10.)
    observer=DelayedImageMap('cw',hint);renderer=Renderer(track,Camera());image=renderer.render_raw(hint)
    before=copy.deepcopy(snapshot(observer.visible));capture=1/30;done=capture+.036;ready=capture+.070
    first=observer.observe(image,capture,done)
    assert first['map_capture'] is None and observer.working.map.t==capture
    assert snapshot(observer.visible)['map']==before['map']
    observer.on_applied(done,[.4,-.4]);assert observer.applied[-1]['nominal_v']==0 and observer.applied[-1]['nominal_w']>0
    assert observer.estimate(ready-1e-6)['map_capture'] is None
    published=observer.estimate(ready);assert published['map_capture']==capture
    np.testing.assert_allclose(published['pose'],observer.working.map.pose_at(ready),atol=1e-12,rtol=0)
    np.testing.assert_array_equal(observer.visible.map.commands,observer.working.map.commands)
    observer.on_applied(ready+.01,[0.,0.]);assert observer.applied[-1]['nominal_v']==observer.applied[-1]['nominal_w']==0
    observer.on_applied(ready+.02,[-.4,-.4]);assert observer.applied[-1]['nominal_v']<0 and observer.applied[-1]['nominal_w']==0
    assert len(observer.applied)==3 and len(observer.visible.raw_cmds)==len(observer.working.raw_cmds)
    route=np.array([[0.,-50.],[0.,0.],[50.,0.]])
    observer.set_route(route)
    np.testing.assert_array_equal(observer.visible.route.points,observer.working.route.points)
    # A new observation must not insert the map follower's proposed output into
    # either command history. Only explicit on_applied calls may do that.
    commands=copy.deepcopy(observer.working.map.commands)
    observer.n=3;observer.observe(image,.2,.236)
    assert observer.working.map.commands==commands and len(observer.applied)==3
    REPORT.mkdir(exist_ok=False)
    dump(REPORT/'checks.json',dict(source_sha256=sha(__file__),core_path=str(observer.core_path),core_sha256=sha(observer.core_path),
        delayed_publication=True,prediction_includes_actual_commands=True,signed_pivot_reverse_and_stop=True,
        no_map_proposals_applied=True,route_copies_equal=True,actor_inputs_unchanged=True,
        scope='Actual pinned map core, two synthetic camera observations and commanded-motion checks; not localization accuracy or full-mission driving',
        limits='Fixed70ms modeled map latency, every3camera frames. Covariance remains that of latest published estimate; no hardware latency claim.'))
    (REPORT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print('delayed image map checks passed',flush=True)


if __name__=='__main__':check()
