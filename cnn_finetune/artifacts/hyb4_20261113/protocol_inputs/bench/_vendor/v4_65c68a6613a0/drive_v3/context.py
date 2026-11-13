"""Bind observed marks to an explicitly reserved visit; corrections cannot pass a line."""
from dataclasses import dataclass
import numpy as np
from .geometry import axes,to_body,sigma_poses
from .mission import MissionContext


@dataclass(frozen=True)
class Target:
    visit_id: str
    target_id: str
    kind: str
    map_marker_id: str | None = None
    possible_directions: tuple = ()


class TargetTracker:
    def __init__(self, vehicle, config):
        self.front=vehicle["front_cm"];self.cfg=config;self.target=None
        self.center=None;self.last_seen=-1e9;self.first=None;self.hits=0
        self.last_motion_end=-1e9;self.confirmed=False;self.passed=False
        self.last_capture=-1e9

    def arm(self,target):
        if self.target==target:return
        self.target=target;self.center=None;self.first=None;self.hits=0
        self.last_seen=-1e9;self.confirmed=False;self.passed=False
        self.last_capture=-1e9

    def update(self,observation,estimator,motion,now):
        if self.target is None:return None
        target=self.target
        captured=observation.stamp.capture_s
        if captured is None:captured=now
        if self.center is not None and motion is not None and motion.end_s>self.last_motion_end:
            self.center=(self.center-motion.translation_body_cm)@axes(motion.yaw_rad).T
            self.last_motion_end=motion.end_s
            # Physical relative motion, never a jump in the absolute map estimate.
            if self.confirmed and self.center[1]-self.front < -self.cfg["margin_cm"]:
                self.passed=True
        if target.kind=="junction":
            node,entry=target.target_id.split(":")
            xy=estimator.track.data["junctions"][node]["xy"]
            position=to_body([xy],estimator.primary.mean)[0]
            diag=estimator.last_diagnostics
            # No transverse mark required. Geometric junction evidence must actually be present.
            fresh=now-estimator.primary.last_observed_s<.2
            supported=fresh and len(diag.get("boundary_ids",[]))>=2 and 0<position[1]<35 and abs(position[0])<10
            return MissionContext(target.visit_id,target.target_id,target.kind,
                                  possible_directions=target.possible_directions,approach_confirmed=supported)
        candidates=[]
        for mark in observation.landmarks:
            if "localization_only" in mark.flags:continue
            assignments=[h.associations.get(mark.id) for h in estimator.hypotheses]
            if assignments and all(a==target.map_marker_id for a in assignments):candidates.append(mark)
        if len(candidates)==1 and captured>self.last_capture:
            mark=candidates[0]
            if self.first is None or captured-self.last_seen>self.cfg["track_expire_s"]:self.first=captured;self.hits=0
            self.hits+=1;self.last_seen=captured;self.center=np.array(mark.center_cm);self.last_capture=captured
            if self.hits>=self.cfg["minimum_hits"] and captured-self.first>=self.cfg["confirm_s"]-1e-9:self.confirmed=True
        fresh=now-self.last_seen<=self.cfg["track_expire_s"]
        distance=float(self.center[1]-self.front) if self.center is not None else None
        landmark=next((m for m in estimator.track.landmarks if m["id"]==target.map_marker_id),None)
        lower=None
        if landmark:
            lower=min(float(to_body([landmark["xy"]],p)[0,1]-self.front)
                      for h in estimator.hypotheses for p in sigma_poses(h.mean,h.P,1.))
        return MissionContext(target.visit_id,target.target_id,target.kind,distance,
                              target.possible_directions,self.confirmed and fresh,self.passed,
                              lower_distance_bound_cm=lower)
