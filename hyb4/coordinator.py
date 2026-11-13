"""Pure hyb4 command ownership; no GPIO, truth, legacy pivot or map worker.

The integration adapter must provide an IMAGE-derived pose/uncertainty and an
ordinary-road policy proposal, then acknowledge the final actual PWM exactly
once. This is a tested seam, not a complete bench/Pi runtime registration.
"""
import numpy as np
from .control import PathTracker
from cnn_finetune.goal_mission_direction import DirectionResolver,_policy


class Coordinator:
    def __init__(self,junctions,direction,predictor):
        self.resolver=DirectionResolver(junctions,direction)
        self.interrupt=_policy.Interrupt();self.predictor=predictor;self.tracker=PathTracker()
        self.previous=np.zeros(2);self.last_capture=-np.inf;self.last_done=None
        self.last_applied=-np.inf;self.pending=None

    def push_direction(self,t,move):self.resolver.push_direction(t,move)
    def stop(self,t,reason='orchestrator'):self.interrupt.stop(t,reason)
    def resume(self,t,reason='orchestrator'):self.interrupt.resume(t,reason)
    def resume_all(self,t):self.interrupt.resume_all(t)

    def propose(self,image,capture,done,estimated_pose,sxy,ordinary_pwm):
        if self.pending is not None:raise RuntimeError('Previous proposal has no actual PWM receipt')
        if not np.isfinite([capture,done]).all() or capture<=self.last_capture or done<capture:
            raise ValueError('Invalid frame timestamps')
        if self.last_done is not None and done<=self.last_done:raise ValueError('Control clock moved backward')
        ordinary=np.asarray(ordinary_pwm,float)
        if ordinary.shape!=(2,) or not np.isfinite(ordinary).all() or max(abs(ordinary))>1:
            raise ValueError('Invalid ordinary-road proposal')
        if not self.interrupt.held:self.resolver.observe(done,estimated_pose,sxy)
        active=self.resolver.active;move=self.resolver.instruction
        diag=dict(active=active,instruction=move,held=sorted(self.interrupt.holds))
        packet=None
        try:
            # Warm recurrent state even outside junctions and during mission holds.
            packet=self.predictor.observe(image,capture,self.previous.copy(),move,active)
        except ValueError as exc:
            diag['path_error']=str(exc)
        if self.interrupt.held:
            out=np.zeros(2);diag['reason']='mission_hold';self.tracker.drive.reset()
        elif done-capture>.25:
            out=np.zeros(2);diag['reason']='stale_frame';self.tracker.drive.reset()
        elif active:
            try:
                if packet is None:raise ValueError('No trajectory for active junction')
                desired,info=self.tracker.desired(packet,done)
                dt=.036 if self.last_done is None else min(.25,done-self.last_done)
                out=self.tracker.drive.allocate(desired,dt);diag.update(info,reason='cnn_path')
            except ValueError as exc:
                out=np.zeros(2);diag.update(reason='invalid_path',path_error=str(exc));self.tracker.drive.reset()
        else:
            # Preserve the ordinary-road policy exactly. Handoff smoothing is a
            # pending connected-driving validation, not silently assumed here.
            out=ordinary.copy();diag['reason']='ordinary_policy';self.tracker.drive.reset()
        self.last_capture=float(capture);self.last_done=float(done)
        self.pending=(float(done),out.copy())
        return out,diag

    def on_applied(self,t,pwm):
        if not np.isfinite(t) or t<=self.last_applied:raise ValueError('Duplicate or unordered PWM receipt')
        if self.pending is not None and t<self.pending[0]:raise ValueError('Receipt predates proposal')
        self.tracker.acknowledge(t,pwm)
        value=np.asarray(pwm,float).copy()
        if self.pending is not None and not np.array_equal(value,self.pending[1]):self.tracker.drive.reset()
        self.previous=value;self.last_applied=float(t);self.pending=None

