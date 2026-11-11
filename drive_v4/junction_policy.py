"""Junction decisions and orchestrator commands for the driving code (2026-11-11).

No motors, camera or GPIO here: pure logic, the same on the Pi and in the bench.

Division of work (user, 2026-11-11):
  * signs, lights and the pink-line / crosswalk triggers belong to the vision team and
    the orchestrator. The driving code only
      - keeps a queue of direction commands (one per sign; signs stand on the right),
      - notices by itself, from its map pose, that it reaches a junction (there is no
        trigger line before a junction), offers the movements that exist there, and
        takes the queued command if it fits, the default otherwise,
      - stops on an interrupt and resumes on the next (stop sign: the orchestrator
        sends stop, waits 3 s, sends resume; red light: stop, green: resume).
  * not every junction has a sign (one sign per kind on the course), so every
    junction needs a default:
      - arriving on the outer road at a T junction: straight (stay on the loop)
      - arriving from an inner road at a T junction on the outer road: turn the way
        the lap goes (cw -> right, ccw -> left), i.e. merge back into the loop
      - the central crossroad: always right (course rule); direction commands are
        not used there and stay queued
  * an unfinished run is the worst outcome: a missing command never stops the car.
"""
from dataclasses import dataclass, field
import math

ARMS = {"N": (0.0, 1.0), "E": (1.0, 0.0), "S": (0.0, -1.0), "W": (-1.0, 0.0)}
MOVES = ("straight", "left", "right")


def _arm_of(d):
    return min(ARMS, key=lambda k: math.hypot(ARMS[k][0] - d[0], ARMS[k][1] - d[1]))


def exit_arm(came_from, move):
    """Arm the car leaves through when it arrives on `came_from` and does `move`."""
    dx, dy = -ARMS[came_from][0], -ARMS[came_from][1]            # travel direction
    d = {"straight": (dx, dy), "left": (-dy, dx), "right": (dy, -dx)}[move]
    return _arm_of(d)


@dataclass
class Decision:
    junction: str
    came_from: str
    options: list
    move: str
    exit_arm: str
    source: str                 # "command" / "default" / "rule"
    note: str = ""
    command: object = None


@dataclass
class JunctionPolicy:
    junctions: dict                       # map_v4.json "junctions": name -> {xy, arms}
    lap_direction: str = "cw"             # cw / ccw
    centre: str = "center"
    max_command_age_s: float = 30.0       # a sign's command is for the next junction or two, not later
    queue: list = field(default_factory=list)
    log: list = field(default_factory=list)

    # --- commands from the orchestrator ------------------------------------------
    def push(self, t, move, source="sign"):
        if move not in MOVES:
            raise ValueError(f"direction command must be one of {MOVES}")
        self.queue.append(dict(t=float(t), move=move, source=source))
        self.log.append(dict(t=float(t), event="command_queued", move=move, source=source))

    # --- geometry ----------------------------------------------------------------
    def kind(self, junction, came_from):
        """'crossroad', 'outer_T' (arriving along the outer road) or 'inner_T' (arriving from an inner road)."""
        if junction == self.centre:
            return "crossroad"
        x, y = self.junctions[junction]["xy"]
        inner = _arm_of((-x, -y))                                  # the arm pointing to the centre
        return "inner_T" if came_from == inner else "outer_T"

    def options(self, junction, came_from):
        arms = self.junctions[junction]["arms"]
        return [m for m in MOVES if exit_arm(came_from, m) in arms and exit_arm(came_from, m) != came_from]

    def default(self, junction, came_from):
        k = self.kind(junction, came_from)
        if k == "crossroad":
            return "right", "rule"
        if k == "outer_T":
            return "straight", "default"
        return ("right" if self.lap_direction == "cw" else "left"), "default"

    # --- the decision, taken once per junction visit -----------------------------
    def decide(self, t, junction, came_from):
        opts = self.options(junction, came_from)
        move, source = self.default(junction, came_from)
        note, used = "", None
        for c in [c for c in self.queue if t - c["t"] > self.max_command_age_s]:
            self.log.append(dict(t=float(t), event="command_expired", move=c["move"], queued_t=c["t"]))
        self.queue = [c for c in self.queue if t - c["t"] <= self.max_command_age_s]
        if self.kind(junction, came_from) != "crossroad" and self.queue:
            c = self.queue.pop(0)
            if c["move"] in opts:
                move, source, used = c["move"], "command", c
            else:
                note, used = f"command {c['move']} not possible here (options {opts}); default used", c
        if move not in opts:                                       # map has no such exit: never stop
            note += f" default {move} impossible; took {opts[0]}"
            move = opts[0]
        d = Decision(junction, came_from, opts, move, exit_arm(came_from, move), source, note.strip(), used)
        self.log.append(dict(t=float(t), event="junction_decision", **d.__dict__))
        return d


@dataclass
class Interrupt:
    """Stop / resume from the orchestrator. While held the driver sends zero output and
    freezes its own phase timers; on resume it continues where it was."""
    held: bool = False
    since: float = None
    log: list = field(default_factory=list)

    def stop(self, t, reason="orchestrator"):
        if not self.held:
            self.held, self.since = True, float(t)
            self.log.append(dict(t=float(t), event="stop", reason=reason))

    def resume(self, t, reason="orchestrator"):
        if self.held:
            self.log.append(dict(t=float(t), event="resume", reason=reason, held_s=float(t) - self.since))
            self.held, self.since = False, None
