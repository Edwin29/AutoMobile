"""Motor-free mission decisions. Inputs are confirmed events from the route tracker,
perception and action executor. A new instance is required for each physical approach.
No distance, right-side association or standstill is guessed from a bounding box.
"""
from dataclasses import dataclass
from math import isfinite


def braking_gate(front_clear_cm, speed_upper_cm_s, delay_s, decel_lower_cm_s2, margin_cm):
    """front_clear_cm is a conservative along-route distance from the front bumper
    to the near edge of the expected marking. None means localization is uncertain.
    Inputs for delay and deceleration must include the vehicle's actual stopping response.
    """
    if not all(isfinite(v) for v in (speed_upper_cm_s, delay_s, decel_lower_cm_s2, margin_cm)):
        raise ValueError("Braking parameters must be finite")
    if min(speed_upper_cm_s, delay_s, margin_cm) < 0 or decel_lower_cm_s2 <= 0:
        raise ValueError("Use nonnegative speed, delay, margin and positive deceleration")
    limit = speed_upper_cm_s * delay_s + speed_upper_cm_s**2/(2*decel_lower_cm_s2) + margin_cm
    return front_clear_cm is None or not isfinite(front_clear_cm) or front_clear_cm <= limit


@dataclass
class Approach:
    zone: str  # start, outer, signed, center or unknown; fixed for this encounter
    stop_seconds: float  # configured competition dwell
    sign: str | None = None
    completed: bool = False
    red_seen: bool = False
    stopped_since: float | None = None
    released: bool = False
    missed: bool = False

    def __post_init__(self):
        if not isfinite(self.stop_seconds) or self.stop_seconds <= 0:
            raise ValueError("Pass the required positive stop duration in seconds")

    def update(self, now, *, right_sign=None, right_light=None, at_braking_gate=False,
               passed_pink=False, stopped_before_crossing=False, action_done=None,
               crossing_ahead=False, at_turn_entry=False, placed_at_start=False):
        """now: monotonic seconds. right_*: stable, fresh and matched to this approach.
        at_braking_gate: last safe decision point BEFORE the expected mark; must also
        be provided by the map when vision misses the mark. crossing_ahead: the
        incoming crossing, confirmed from vision or a localized map. action_done:
        action acknowledged for this encounter, completed on the approach side.
        Returns (control request, one-shot event); the lane controller executes it.
        """
        if self.zone == "start":
            if not self.released and placed_at_start:
                self.released = True
                return "FOLLOW", "init_pose_at_start"
            return ("FOLLOW" if self.released else "WAIT_PLACEMENT"), None
        if self.zone == "outer":
            return "FOLLOW", None
        if self.zone == "signed":
            if self.sign and action_done == self.sign:
                self.completed = True
            if passed_pink and not self.completed:
                self.missed = True
            if self.missed:
                return "HOLD_MISSED_DEADLINE", None
            event = None
            if self.sign is None and right_sign in ("horn", "slow", "stop", "right", "straight", "left"):
                self.sign = right_sign
                event = self.sign
            if self.completed:
                return "FOLLOW_MISSION_SPEED", None
            if self.sign == "stop" or at_braking_gate:
                return "HOLD_BEFORE_PINK", event
            return ("APPLY_SPEED_LIMIT" if self.sign == "slow" else "FOLLOW"), event
        if self.zone != "center":
            return "REQUEST_LOCALIZATION", None
        if self.released:
            return "FOLLOW_RIGHT_PATH", None
        self.red_seen |= right_light == "red"
        if self.red_seen:
            if stopped_before_crossing and crossing_ahead:
                if self.stopped_since is None:
                    self.stopped_since = now
                if now - self.stopped_since >= self.stop_seconds:
                    self.released = True
                    return "FOLLOW_RIGHT_PATH", "select_right"
                return "HOLD_BEFORE_CROSSING", None
            self.stopped_since = None
            return ("HOLD_BEFORE_CROSSING" if at_braking_gate else "APPROACH_SLOW"), None
        if crossing_ahead and right_light == "green":
            if at_turn_entry:
                self.released = True
                return "FOLLOW_RIGHT_PATH", "select_right"
            return "APPROACH_SLOW", None
        return ("HOLD_BEFORE_CROSSING" if at_braking_gate else "APPROACH_SLOW"), None
