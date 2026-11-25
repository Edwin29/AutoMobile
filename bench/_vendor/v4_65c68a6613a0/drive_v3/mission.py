"""Scoped mission reservations. Relocalization and tracker IDs never count as visits."""
from dataclasses import dataclass, field


@dataclass
class MissionContext:
    visit_id: str
    target_id: str
    kind: str  # pink, junction, center_crosswalk, outer_crosswalk
    distance_to_front_cm: float | None = None
    possible_directions: tuple = ()
    approach_confirmed: bool = False
    passed_confirmed: bool = False  # externally verified physical crossing, NOT pose jump
    exit_aligned: bool = False
    lower_distance_bound_cm: float | None = None


@dataclass
class MissionDecision:
    hold: bool = False
    speed_cap: float | None = None
    direction: str | None = None
    horn: bool = False
    reason: str = "cruise"
    events: list = field(default_factory=list)


class MissionManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.samples = {}; self.pending = {}; self.completed = set()
        self.executing = {}; self.committed = {}; self.conflicts = {}
        self.state = "CRUISE"; self.last_time = None

    def observe(self, detections, context, now):
        if context is None: return
        # Stable target + visit, not a volatile image tracker, is the mission identity.
        scope = (context.visit_id, context.target_id)
        for d in detections:
            allowed = ({"left", "right", "straight"} if context.kind == "junction" else
                       {"signal_red", "signal_green"} if context.kind == "center_crosswalk" else
                       {"stop", "slow", "horn"} if context.kind == "pink" else set())
            if d.kind not in allowed or d.timestamp > now or now-d.timestamp > self.cfg["track_expire_s"]:
                continue
            key = (*scope, d.kind)
            first, last, hits = self.samples.get(key, (d.timestamp, -1e9, 0))
            if d.timestamp <= last: continue
            if d.timestamp-last > self.cfg["track_expire_s"]: first, hits = d.timestamp, 0
            self.samples[key] = (first, d.timestamp, hits+1)
            if hits+1 >= self.cfg["minimum_hits"] and d.timestamp-first >= self.cfg["confirm_s"]-1e-9:
                self.pending[key] = d.timestamp
        # Old classifications are retained as reservations for signs, but not as current signal colors.
        self.samples = {k:v for k,v in self.samples.items() if now-v[1] <= self.cfg["track_expire_s"]}

    def update(self, now, speed, context=None,stationary=None,observation_age_s=0.):
        if self.last_time is not None and now < self.last_time: raise ValueError("Mission time reversed")
        self.last_time = now
        if context is None: self.state="CRUISE"; return MissionDecision()
        scope = (context.visit_id, context.target_id)
        kinds = {k[2] for k in self.pending if k[:2] == scope and k not in self.completed}
        if context.kind == "outer_crosswalk": return MissionDecision(reason="outer_crosswalk_localization_only")
        if context.kind == "junction":
            if scope in self.committed:
                return MissionDecision(direction=self.committed[scope], reason="turn_committed")
            if len(kinds)>1:
                self.conflicts.setdefault(scope,now)
                reason="direction_conflict_timeout" if now-self.conflicts[scope]>=self.cfg["conflict_timeout_s"] else "direction_conflict"
                self.state="CONFLICT"; return MissionDecision(hold=True,reason=reason)
            if not kinds: return MissionDecision(hold=context.approach_confirmed,reason="await_direction")
            direction=next(iter(kinds))
            if direction not in context.possible_directions:
                return MissionDecision(hold=True,reason="direction_not_available")
            if context.approach_confirmed:
                self.committed[scope]=direction
                return MissionDecision(direction=direction,reason="turn_committed",events=[{"kind":"turn_commit","scope":scope}])
            return MissionDecision(direction=direction,reason="direction_reserved")
        distance=context.distance_to_front_cm
        deadline=max(0.,speed)*(self.cfg["latency_s"]+max(0.,observation_age_s))+max(0.,speed)**2/(2*self.cfg["brake_cm_s2"])+self.cfg["margin_cm"]
        due=context.approach_confirmed and distance is not None and distance<=deadline
        if (not context.approach_confirmed and context.lower_distance_bound_cm is not None
                and context.lower_distance_bound_cm<=deadline):
            return MissionDecision(hold=True,reason="trigger_unconfirmed_near_deadline")
        if context.kind == "center_crosswalk":
            colors={k for k in kinds if now-self.pending[(*scope,k)]<=self.cfg["track_expire_s"]}
            if not due and scope not in self.executing:
                return MissionDecision(direction="right",reason="central_approach")
            if not self.cfg["rule_values_verified"] or self.cfg["signal_release"] not in ("green", "timed"):
                return MissionDecision(hold=True,reason="rules_required")
            if (self.cfg["signal_release"]=="timed" and scope in self.executing
                    and self.cfg["signal_stop_hold_s"] is not None
                    and now-self.executing[scope]>=self.cfg["signal_stop_hold_s"]):
                return MissionDecision(direction="right",reason="central_right_released")
            if "signal_red" in colors or not colors or len(colors)>1:
                self.executing.setdefault(scope, now)
                return MissionDecision(hold=True,reason="signal_red_or_unknown")
            release=self.cfg["signal_release"]
            if release=="timed" and scope in self.executing:
                hold=self.cfg["signal_stop_hold_s"]
                if hold is None: return MissionDecision(hold=True,reason="rules_required")
                if now-self.executing[scope]<hold: return MissionDecision(hold=True,reason="signal_stop_hold")
            return MissionDecision(direction="right",reason="central_right_released")
        if context.kind != "pink": return MissionDecision()
        if not kinds:
            if any(k[:2]==scope for k in self.completed): return MissionDecision(reason="mission_complete")
            return MissionDecision(hold=due,reason="missing_sign" if due else "await_sign")
        if context.passed_confirmed and any((*scope,k) not in self.executing for k in kinds):
            return MissionDecision(hold=True,reason="missed_deadline")
        decision=MissionDecision(reason="reserved")
        for kind in sorted(kinds):
            key=(*scope,kind)
            if not due and key not in self.executing: continue
            rule={"stop":"stop_hold_s","horn":"horn_duration_s","slow":"slow_speed_cm_s"}[kind]
            if not self.cfg["rule_values_verified"] or self.cfg[rule] is None or (kind=="slow" and self.cfg["slow_release"]!="passed"):
                decision.hold=True; decision.reason="rules_required"; continue
            if key not in self.executing:
                self.executing[key]=None if kind=="stop" else now
                decision.events.append({"kind":"mission_start","action":kind,"scope":scope})
            start=self.executing[key]
            if kind=="stop":
                previous_hold=decision.hold
                decision.hold=True; decision.reason="stop_hold"
                # A duration starts only after the caller reports a stationary estimate.
                at_rest=abs(speed)<.2 if stationary is None else stationary
                if at_rest and start is None: self.executing[key]=start=now
                if start is not None and now-start>=self.cfg[rule]:
                    self.completed.add(key); decision.hold=previous_hold; decision.reason="stop_complete"
            elif kind=="horn":
                decision.horn=now-start<self.cfg[rule]
                if not decision.horn: self.completed.add(key)
            elif kind=="slow":
                decision.speed_cap=self.cfg[rule]
                if context.passed_confirmed: self.completed.add(key); decision.speed_cap=None
        self.state="HOLD" if decision.hold else "EXECUTE"
        return decision

    def complete_visit(self, context):
        if not context.exit_aligned or not context.passed_confirmed:
            return False
        scope=(context.visit_id,context.target_id)
        for key in list(self.pending):
            if key[:2]==scope: self.completed.add(key); del self.pending[key]
        self.committed.pop(scope,None)
        return True
