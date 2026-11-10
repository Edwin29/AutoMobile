"""Six-part coordinator. A sink is injected; the default records commands only."""
from dataclasses import asdict
import time
import hashlib
import json
import numpy as np

from .contracts import FrameStamp
from .camera import CameraGeometry
from .observation import Observer
from .settings import load_settings
from .map_model import TrackMap, HERE
from .estimation import Estimator, TimedEstimator
from .planning import LocalPlanner
from .control import ResponseModel, Controller
from .mission import MissionManager
from .features import DetectionTracker
from .visual_motion import FloorVO
from .context import TargetTracker


class RecordingSink:
    """No hardware. Four logical outputs, two side commands; no wheel speed claims."""
    def __init__(self): self.last={}; self.horn=False
    def write(self,left,right,horn=False):
        if not np.isfinite([left,right]).all() or max(abs(left),abs(right))>1:
            raise ValueError("Invalid motor command")
        self.last=dict(front_left=left,rear_left=left,front_right=right,rear_right=right)
        self.horn=bool(horn)
    def stop(self): self.write(0.,0.,False)


class Runtime:
    def __init__(self, initial_pose, route=None, settings=None, start_s=0.,clock_id="sim",sink=None):
        self.cfg,self.config_id=load_settings() if settings is None else (settings,hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest())
        self.camera=CameraGeometry(HERE/"config/camera_legacy_track.json")
        self.observer=Observer(self.camera,HERE/"config/observation.json")
        self.observer.mark_cfg=self.cfg["marks"]
        self.track=TrackMap(); self.route=route or self.track.outer_route()
        self.estimator=Estimator(self.track,initial_pose,self.cfg["estimator"])
        self.timeline=TimedEstimator(self.estimator,start_s,clock_id)
        self.planner=LocalPlanner(self.track,self.route,self.cfg["planner"],self.cfg["vehicle"])
        self.model=ResponseModel(self.cfg["controller"],self.cfg["vehicle"])
        self.controller=Controller(self.model,self.cfg["controller"],self.cfg["controller"].get("mode","feedback"))
        self.mission=MissionManager(self.cfg["mission"])
        self.tracker=DetectionTracker(self.cfg["mission"])
        self.vo=FloorVO(self.camera,self.cfg["estimator"])
        self.target_tracker=TargetTracker(self.cfg["vehicle"],self.cfg["mission"])
        self.sink=sink or RecordingSink();self.sink.stop()
        self.clock=clock_id;self.last=start_s;self.sequence=0;self.last_pwm=np.zeros(2);self.last_v=0.
        self.last_motion=None
        self.last_capture=-1e9
        self.prepared_routes={}
        if self.route.junction:self.prepare_junction(self.route.junction,self.route.entry)

    def prepare_junction(self,junction,entry):
        """Expensive route construction is performed while stopped, outside step()."""
        if np.max(np.abs(self.last_pwm))>0:raise RuntimeError("Stop before preparing routes")
        self.sink.stop()
        self.prepared_routes[(junction,entry)]=self.track.junction_routes(junction,entry,self.cfg["vehicle"])

    def step(self, image, capture_s, now, *, rectified=False, rotate180=False,detections=(),context=None,shadow_pwm=None):
        begin=time.perf_counter(); stages={}
        try:
            if now<=self.last or capture_s>now: raise ValueError("Invalid runtime timestamps")
            stamp=FrameStamp(self.sequence,capture_s,self.clock);self.sequence+=1
            t=time.perf_counter()
            observation,debug=self.observer.process(image,stamp,rectified=rectified,rotate180=rotate180)
            stages["observe_ms"]=(time.perf_counter()-t)*1000
            fresh_capture=capture_s>self.last_capture
            self.last_motion=self.vo.update(debug,capture_s) if fresh_capture else None
            if fresh_capture:self.last_capture=capture_s
            t=time.perf_counter(); self.timeline.step(now,observation,self.last_motion if self.cfg["estimator"].get("use_visual_odometry",False) else None)
            stages["estimate_ms"]=(time.perf_counter()-t)*1000
            quality=self.estimator.last_diagnostics.get("status")=="observed"
            if context is None:
                context=self.target_tracker.update(observation,self.estimator,self.last_motion,now)
            tracked=self.tracker.update(detections,capture_s) if fresh_capture else []
            self.mission.observe(tracked,context,now)
            # VO replaces PWM motion prediction, not a second absolute pose observation.
            # Noise floors remain provisional; correlated camera errors are not independent sensors.
            speed=self.last_v
            if self.last_motion is not None:
                dt=self.last_motion.end_s-self.last_motion.start_s
                speed=float(np.linalg.norm(self.last_motion.translation_body_cm)/max(dt,1e-6))
            stationary=self.last_motion is not None and self.last_motion.status=="stationary"
            decision=self.mission.update(now,speed,context,stationary=stationary,observation_age_s=now-capture_s)
            if context and decision.direction and decision.reason in ("turn_committed","central_right_released"):
                parts=context.target_id.split(":")
                if len(parts)==2 and self.planner.route.id!=f"committed:{context.visit_id}:{context.target_id}":
                    routes=self.prepared_routes.get((parts[0],parts[1]),{})
                    if decision.direction in routes:
                        route=routes[decision.direction];route.id=f"committed:{context.visit_id}:{context.target_id}"
                        self.planner.set_route(route)
                    else:
                        decision.hold=True;decision.reason="route_not_prepared"
            t=time.perf_counter()
            # State is replayed to now. Raw fragment coordinates are still in the
            # capture-time body frame: never apply them as fresh local constraints.
            action_observation=observation if now-capture_s<1e-9 else None
            path=self.planner.plan(self.estimator,action_observation,now,self.clock,decision.hold,decision.speed_cap)
            stages["plan_ms"]=(time.perf_counter()-t)*1000
            output=self.controller.step(path,now,now-self.last,self.clock,quality)
            # Online gains update slowly and only from the prior interval's applied command.
            self.model.adapt(self.last_pwm,self.last_motion,quality and self.planner.state=="CRUISE")
            self.sink.write(*output.pwm,decision.horn)
            if shadow_pwm is None:
                self.timeline.command(now,output.v_cm_s,output.omega_rad_s)
            else:
                self.timeline.command(now,*self.model.forward_model(shadow_pwm))
            self.last=now;self.last_pwm=output.pwm.copy() if shadow_pwm is None else np.asarray(shadow_pwm).copy();self.last_v=output.v_cm_s
            record={"sequence":stamp.sequence,"capture_s":capture_s,"now_s":now,"clock":self.clock,
                    "pose":self.estimator.primary.mean.tolist(),"covariance":self.estimator.primary.P.tolist(),
                    "hypotheses":len(self.estimator.hypotheses),"estimate":self.estimator.last_diagnostics,
                    "plan":self.planner.last_diagnostics,"path_reason":path.reason,"control_reason":output.reason,
                    "pwm":output.pwm.tolist(),"predicted_v":output.v_cm_s,"predicted_omega":output.omega_rad_s,
                    "trim":output.trim,"mission":asdict(decision),"motion":asdict(self.last_motion) if self.last_motion else None,
                    "gains":[self.model.forward,self.model.left,self.model.right],"timings":stages,
                    "beliefs":[{"pose":h.mean.tolist(),"P":h.P.tolist(),"score":h.score,"associations":h.associations} for h in self.estimator.hypotheses],
                    "landmarks":[{"kind":m.kind,"body_cm":m.center_cm,"id":m.id,"flags":m.flags} for m in observation.landmarks if not m.kind.endswith("_patch")],
                    "config_hash":self.config_id,"map_hash":self.track.id,"camera_hash":self.camera.id,
                    "current_frame_constraints_used":action_observation is not None,
                    "total_ms":(time.perf_counter()-begin)*1000}
            return record,observation,debug,path
        except BaseException:
            self.sink.stop()
            raise

    def close(self):
        self.sink.stop()
        if hasattr(self.sink,"close"):self.sink.close()

    def __enter__(self): return self
    def __exit__(self,*args): self.close()
