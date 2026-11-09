"""Single-writer, motor-free reference for one run. Times use monotonic seconds."""
from dataclasses import dataclass
from math import isfinite

TURNS = {"LEFT", "RIGHT", "STRAIGHT"}
MISSIONS = {"STOP", "SLOW", "HORN"}


@dataclass(frozen=True)
class Sign:
    visit: str          # target pink gate OR target junction; unique per passage
    kind: str           # confirmed right-side classification
    captured_at: float


@dataclass(frozen=True)
class Pink:
    visit: str
    trigger: bool = False   # rising edge OR level in the upstream trigger ROI
    brake_gate: bool = False  # last safe braking point, predicted even if vision misses
    passed: bool = False    # front bumper passed the real line, independently verified
    stopped_before: bool = False  # fresh, continuous physical stop confirmation


@dataclass(frozen=True)
class Junction:
    visit: str
    center: bool
    allowed: frozenset[str]  # fresh LaneTracker reply in vehicle-relative coordinates
    at_entry: bool = False
    light: str | None = None   # fresh right-side RED/GREEN for this approach; else None
    cross_trigger: bool = False  # this incoming crosswalk in its upstream trigger ROI
    cross_brake_gate: bool = False
    stopped_before: bool = False


@dataclass(frozen=True)
class Decision:
    motion: str                # HOLD / FOLLOW / TURN
    speed_cap: float           # relative to normal lane speed; zero when holding
    reason: str
    turn: tuple[str, str] | None = None   # (junction visit, direction)
    horn: str | None = None    # stable command ID; GPIO adapter deduplicates
    slow_ids: tuple[str, ...] = ()  # speed-limit command IDs for acknowledgment


class Coordinator:
    def __init__(self, *, stop_s, red_s, slow_cap, observation_age_s, max_tick_gap_s):
        settings = (stop_s, red_s, slow_cap, observation_age_s, max_tick_gap_s)
        if not all(isfinite(v) and v > 0 for v in settings) or slow_cap >= 1:
            raise ValueError("Provide calibrated positive timings and 0 < slow_cap < 1")
        self.stop_s, self.red_s, self.slow_cap = stop_s, red_s, slow_cap
        self.observation_age_s, self.max_tick_gap_s = observation_age_s, max_tick_gap_s
        self.directions, self.missions = {}, {}
        self.triggered, self.red_seen, self.cross_gates = set(), set(), set()
        self.pink_brakes, self.cross_brakes, self.issued = set(), set(), set()
        self.pink_done, self.red_done, self.turn_done = set(), set(), set()
        self.slow_zones = set()
        self.stop_since = {}
        self.active_turn = None
        self.last_time = None
        self.active_pink = self.active_center = None
        self.fault = None
        self.armed = False

    def _dwelled(self, key, confirmed, now, duration):
        if not confirmed:
            self.stop_since.pop(key, None)
            return False
        return now >= self.stop_since.setdefault(key, now) + duration

    def step(self, now, *, signs=(), pink=None, junction=None, healthy=True,
             completed=frozenset(), slow_zone_finished=frozenset(), start_ready=False):
        """One atomic input snapshot. completed contains actuator acknowledgments:
        '<visit>/HORN' (pulse finished before line), '<visit>/SLOW' (speed reached
        before line), '<visit>/TURN' (exit verified). Context supplies unique IDs.
        STOP uses actual observed standstill. Read README before wiring adapters.
        """
        if not isfinite(now) or (self.last_time is not None and now <= self.last_time):
            raise ValueError("Ticks must use strictly increasing monotonic timestamps")
        gap = self.last_time is not None and now > self.last_time + self.max_tick_gap_s
        self.last_time = now
        if not healthy or gap:
            self.stop_since.clear()
            return Decision("HOLD", 0, "unhealthy_or_late_tick")
        if self.fault:
            return Decision("HOLD", 0, self.fault)
        self.armed |= start_ready  # placement and initial map pose acknowledged once per run
        if not self.armed:
            return Decision("HOLD", 0, "await_start_placement")

        for sign in signs:
            if not sign.visit or not 0 <= now-sign.captured_at <= self.observation_age_s:
                continue
            if sign.kind in TURNS and sign.visit not in self.turn_done:
                bank = self.directions
            elif sign.kind in MISSIONS and sign.visit not in self.pink_done:
                bank = self.missions
            else:
                continue
            old = bank.setdefault(sign.visit, sign.kind)
            if old != sign.kind:
                bank[sign.visit] = "CONFLICT"

        hold = None
        horn = None
        if self.active_turn and self.active_turn[0]+"/TURN" in completed:
            self.turn_done.add(self.active_turn[0])
            self.active_turn = None

        if self.active_pink and (pink is None or pink.visit != self.active_pink):
            self.stop_since.clear()
            return Decision("HOLD", 0, "restore_pending_pink_context")
        if self.active_center and (junction is None or junction.visit != self.active_center or not junction.center):
            self.stop_since.clear()
            return Decision("HOLD", 0, "restore_pending_center_context")

        if pink and pink.visit not in self.pink_done:
            k = pink.visit
            if pink.trigger:
                self.triggered.add(k)
            if pink.brake_gate:
                self.pink_brakes.add(k)
            if pink.trigger or pink.brake_gate:
                self.active_pink = k
            kind = self.missions.get(k)
            if k in self.triggered and kind in MISSIONS:
                cmd = k+"/"+kind
                if kind == "SLOW":
                    self.slow_zones.add(k)
                done = (self._dwelled(cmd, pink.stopped_before and not pink.passed, now, self.stop_s)
                        if kind == "STOP" else cmd in completed and cmd in self.issued)
                if done:
                    self.pink_done.add(k)
                    self.active_pink = None
                elif kind == "HORN":
                    horn = cmd
                elif kind == "STOP":
                    hold = "pink_stop"
            if k not in self.pink_done:
                if pink.passed:
                    self.fault = "missed_pink_deadline:"+k
                    return Decision("HOLD", 0, self.fault)
                if k in self.pink_brakes or (k in self.triggered and kind not in MISSIONS):
                    hold = "pink_instruction_or_completion_required"
        self.slow_zones.difference_update(slow_zone_finished & self.pink_done)

        candidate = None
        if junction and junction.visit not in self.turn_done and self.active_turn is None:
            k = junction.visit
            if junction.center:
                self.active_center = k
                if junction.light == "RED":
                    self.red_seen.add(k)
                if junction.cross_trigger:
                    self.cross_gates.add(k)
                if junction.cross_brake_gate:
                    self.cross_brakes.add(k)
                if k in self.red_seen and k not in self.red_done:
                    if k in self.cross_gates:
                        if self._dwelled(k+"/RED", junction.stopped_before, now, self.red_s):
                            self.red_done.add(k)
                        else:
                            hold = "red_crosswalk_stop"
                    elif k in self.cross_brakes or junction.at_entry:
                        hold = "confirm_incoming_crosswalk"
                elif k not in self.red_done and junction.light != "GREEN" and (
                        k in self.cross_gates or k in self.cross_brakes or junction.at_entry):
                    hold = "confirm_right_signal"
                direction = "RIGHT"
            else:
                direction = self.directions.get(k)
            if junction.at_entry:
                if direction not in TURNS or direction not in junction.allowed:
                    hold = "confirm_instruction_and_legal_exit"
                else:
                    candidate = (k, direction)
        if horn:
            self.issued.add(horn)
        if pink and pink.visit in self.slow_zones:
            self.issued.add(pink.visit+"/SLOW")
        slow_ids = tuple(k+"/SLOW" for k in sorted(self.slow_zones))
        if hold:
            return Decision("HOLD", 0, hold, horn=horn, slow_ids=slow_ids)
        if candidate:
            self.active_turn = candidate
            self.active_center = None
        cap = self.slow_cap if self.slow_zones else 1.0
        if self.active_turn:
            return Decision("TURN", cap, "committed", self.active_turn, horn, slow_ids)
        return Decision("FOLLOW", cap, "lane_follow", horn=horn, slow_ids=slow_ids)
