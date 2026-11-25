"""Sign / light / line rules for the vision process (2026-11-14). Pure logic, no camera,
no model, no sockets: the same code runs on the Pi, in replay and in the tests.

    rules = Rules()
    cmds  = rules.update(t, detections)   # every detector frame
    cmds += rules.tick(t)                 # also between frames (timers)

A detection is dict(name, score, box=(x, y, w, h)) in the 640x480 UPRIGHT frame.
A command is a dict; "cmd" commands go to bench.pi_run over UDP, "local" ones
(horn) are carried out by the vision process itself.

Rules (user, 2026-11-14):
  * slow / horn / stop sign: seen -> the action starts 2 s later, lasts 3 s, then ends.
      slow : {"cmd": "speed", "scale": SLOW_SCALE}   ...  {"cmd": "speed", "scale": 1.0}
      horn : {"local": "horn", "on": True}           ...  {"local": "horn", "on": False}
      stop : {"cmd": "stop", "reason": "stop_sign"}  ...  {"cmd": "resume", "reason": "stop_sign"}
  * a sign that fired is ignored for COOLDOWN_S (10 s), counted from the moment it fired,
    and after that until it has left the view for REARM_ABSENT_S (one sign = one action).
  * red + cross schedules stop and resume at +4 and +5 seconds. Green + cross
    allows passage. Only a confirmed arrow supplies a direction command.
  * left / right / straight sign: the model flips between arrow classes on ONE sign
    (field 2026-11-14: straight/right alternating on the start sign; the left sign read as
    "right" for its first two frames). So the arrow classes vote together while a sign is
    in view, and when the sign has left the view (ARROW_GAP_S) the winner is sent if it has
    ARROW_MAJORITY of the votes; an unclear sign sends nothing (the junction default is
    used, as for a missed sign). Deciding early was tried on the field log and picked
    "straight" on the start sign that later read "right" from close up.
    The driving code queues the command for the next junction it recognises.
A class must be detected in CONFIRM_FRAMES consecutive detector frames to count
(one false box must not stop the car).
"""
from collections import defaultdict
import math

FRAME_H = 480


class SignalMap:
    """Incoming roads to the four signal crosswalks at the central crossroads."""

    def __init__(self, track):
        cx, cy = track["junctions"]["center"]["xy"]
        self.approaches = []
        for outer, ax, ay, axis in (("north", 0, 1, "ns"), ("south", 0, -1, "ns"),
                                    ("east", 1, 0, "ew"), ("west", -1, 0, "ew")):
            crossings = [(x, y) for x, y, orientation in track["crosswalks"]
                         if orientation == axis and abs((x-cx)*ay - (y-cy)*ax) < 1
                         and (x-cx)*ax + (y-cy)*ay > 0]
            x, y = min(crossings, key=lambda xy: math.hypot(xy[0]-cx, xy[1]-cy))
            ox, oy = track["junctions"][outer]["xy"]
            self.approaches.append((x, y, ax, ay, (ox-x)*ax + (oy-y)*ay))

    def eligible(self, pose, sxy, observation_age):
        if pose is None or sxy is None or observation_age is None:
            return None
        if not all(math.isfinite(v) for v in (*pose, sxy, observation_age)) or sxy > 2 or observation_age > 0.5:
            return None
        px, py, yaw = pose
        for x, y, ax, ay, length in self.approaches:
            dx, dy = px-x, py-y
            if (-2 <= dx*ax + dy*ay <= length and abs(dx*ay - dy*ax) <= 10.5
                    and -math.sin(yaw)*ax - math.cos(yaw)*ay >= math.cos(math.radians(35))):
                return True
        return False

CONFIRM_FRAMES = 2
SIGN_DELAY_S = 2.0          # seen -> action starts
ACTION_S = 3.0              # action length
HORN_ACTION_S = 1.0     
COOLDOWN_S = 10.0           # same sign ignored this long after it fired
SLOW_SCALE = 0.8            # wheel duty factor while "slow" (bench.pi_run keeps the start duty)

LINE_CLASSES = ("line", "cross")   # pink stop line, crosswalk
LINE_MID_RATIO = 0.5               # box bottom at or below this fraction of the frame height
LINE_NEEDS_RED = True
RED_MEMORY_S = 4.0                 # red seen this recently counts as "red now"
LINE_COOLDOWN_S = 10.0             # after a red-light stop is released, the same line is ignored

TIMED_SIGNS = ("stop", "slow", "horn")
DIRECTION_SIGNS = ("left", "right", "straight")
# A sign fires again only after COOLDOWN_S AND after it has been OUT of view for
# REARM_ABSENT_S. Field 2026-11-14: a "straight" sign in view queued six commands, and a
# stop sign still in view 10 s later stopped the car a second time at the same sign.
REARM_SIGNS = TIMED_SIGNS
REARM_ABSENT_S = 2.0

ARROW_MIN_VOTES = 3         # detector frames (about 8 per second) before any decision
ARROW_MAJORITY = 0.65       # share of the votes the winner needs
ARROW_GAP_S = 1.0           # no arrow for this long = the sign has left the view
ARROW_EDGE_PX = 3           # an arrow box touching the left/right frame edge is cut off and
                            # does not vote (field 2026-11-14: the start "straight" sign read
                            # "right" in the 4 frames where it was sliding out on the right)
FRAME_W = 640


def _on_off(name):
    if name == "stop":
        return {"cmd": "stop", "reason": "stop_sign"}, {"cmd": "resume", "reason": "stop_sign"}
    if name == "slow":
        return ({"cmd": "speed", "scale": SLOW_SCALE, "reason": "slow_sign"},
                {"cmd": "speed", "scale": 1.0, "reason": "slow_sign"})
    if name == "horn":
        return {"local": "horn", "on": True}, {"local": "horn", "on": False}
    raise ValueError(name)


class Rules:
    def __init__(self, frame_h=FRAME_H):
        self.frame_h = frame_h
        self.streak = defaultdict(int)
        self.fired_at = {}          # name -> t when it last fired
        self.timers = []            # [(t_due, command)]
        self.red_seen_at = None
        self.red_hold = False
        self.line_free_at = -1e9    # line trigger ignored until this time
        self.blind_until = -1e9     # ignore all detections until this time
        self.log = []               # (t, text) for the run log
        self.last_seen = {}         # raw detections (no confirmation), for re-arming
        self.armed = {}             # REARM_SIGNS: False after firing until out of view
        self.arrow_votes = {}       # class -> frames, for the arrow sign now in view
        self.arrow_last = None      # last time an arrow was seen

    # ---- helpers -------------------------------------------------------------------
    def _note(self, t, text):
        self.log.append((round(t, 3), text))

    def _ready(self, name, t):
        last = self.fired_at.get(name)
        if last is not None and t - last < COOLDOWN_S:
            return False
        return self.armed.get(name, True)

    def _fire(self, name, t):
        self.fired_at[name] = t
        if name in REARM_SIGNS:
            self.armed[name] = False

    def _line_at_mid(self, dets):
        bottoms = [d["box"][1] + d["box"][3] for d in dets if d["name"] in LINE_CLASSES]
        return bool(bottoms) and max(bottoms) >= LINE_MID_RATIO * self.frame_h

    # ---- per detector frame ----------------------------------------------------------
    def update(self, t, dets, frame=None, signal_eligible=None):
        if t < self.blind_until:
            return self.tick(t)
        out = []
        names = {d["name"] for d in dets}
        for n in names:
            self.last_seen[n] = t
        for n in REARM_SIGNS:
            if not self.armed.get(n, True) and n not in names and t - self.last_seen.get(n, -1e9) >= REARM_ABSENT_S:
                self.armed[n] = True
        if self._line_at_mid(dets):
            names.add("_line_mid")
        if signal_eligible is not False and {"red", "cross"} <= names:
            names.add("_red_cross")
        for n in set(self.streak) | names:
            self.streak[n] = self.streak[n] + 1 if n in names else 0
        seen = {n for n in names if self.streak[n] >= CONFIRM_FRAMES}

        for n in TIMED_SIGNS:
            if n in ("stop", "horn") and "dashline" not in names:
                continue
            if n in seen and self._ready(n, t):
                self._fire(n, t)
                duration = HORN_ACTION_S if n == "horn" else ACTION_S
                on, off = _on_off(n)
                self.timers.append((t + SIGN_DELAY_S, on))
                self.timers.append((t + SIGN_DELAY_S + duration, off))
                self._note(t, f"{n} seen: action {t + SIGN_DELAY_S:.1f}-{t + SIGN_DELAY_S + duration:.1f}")

        out += self._arrows(t, dets, frame)

        if "red" in seen:
            self.red_seen_at = t
        red_now = self.red_seen_at is not None and t - self.red_seen_at <= RED_MEMORY_S

        if signal_eligible is not False and "green" in names and "red" not in names and "cross" in names and t >= self.line_free_at:
            self.red_seen_at = None
            self.line_free_at = t + LINE_COOLDOWN_S
            self._fire("traffic_light", t)
            self._note(t, "green + cross: continue")

        if "_red_cross" in seen and t >= self.line_free_at:
            self.red_seen_at = None
            self.blind_until = t + 5.0
            self.line_free_at = t + LINE_COOLDOWN_S
            self._fire("traffic_light", t)
            self.timers.append((t + 4.0, {"cmd": "stop", "reason": "red_light"}))
            self.timers.append((t + 5.0, {"cmd": "resume", "reason": "red_light"}))
            self._note(t, "red + cross: stop at +4.0s, resume at +5.0s")

        return out + self.tick(t)

    # ---- arrow signs: one vote per frame, one command per sign -------------------------
    def _arrows(self, t, dets, frame):
        arrows = [d for d in dets if d["name"] in DIRECTION_SIGNS]
        whole = [d for d in arrows if d["box"][0] > ARROW_EDGE_PX
                 and d["box"][0] + d["box"][2] < FRAME_W - ARROW_EDGE_PX]
        if frame is not None:
            # Live mission votes belong to the approach where this frame was captured.
            if not whole:
                return []
            best = max(whole, key=lambda d: d["score"])
            return [dict(cmd="arrow_vote", frame=frame, move=best["name"])]
        if arrows and not whole:              # still in view, but only a cut-off part: no vote
            self.arrow_last = t
            return []
        arrows = whole
        if arrows:
            best = max(arrows, key=lambda d: d["score"])["name"]
            self.arrow_votes[best] = self.arrow_votes.get(best, 0) + 1
            self.arrow_last = t
            return []
        if self.arrow_last is None or t - self.arrow_last < ARROW_GAP_S:
            return []
        out = []                                   # the sign has left the view: decide now
        if self.arrow_votes:
            win, n = max(self.arrow_votes.items(), key=lambda kv: kv[1])
            total = sum(self.arrow_votes.values())
            if total >= ARROW_MIN_VOTES and n / total >= ARROW_MAJORITY:
                out = self._arrow_send(t, win, n, total)
            else:
                self._note(t, f"arrow sign unclear {dict(self.arrow_votes)}: nothing sent")
        self.arrow_votes, self.arrow_last = {}, None
        return out

    def _arrow_send(self, t, move, n, total):
        last = self.fired_at.get("arrow:" + move)
        if last is not None and t - last < COOLDOWN_S:     # the same sign split by a gap in view
            self._note(t, f"{move} again within {COOLDOWN_S:.0f} s: not sent twice")
            return []
        self.fired_at["arrow:" + move] = t
        self._note(t, f"{move} sign ({n}/{total} votes): direction queued {dict(self.arrow_votes)}")
        return [{"cmd": "direction", "move": move}]

    # ---- timers (call often, also without frames) -----------------------------------
    def tick(self, t):
        due = [c for td, c in self.timers if td <= t]
        self.timers = [(td, c) for td, c in self.timers if td > t]
        for c in due:
            self._note(t, f"timer: {c}")
        return due

    def status(self, t):
        """Short state for the live view."""
        cool = {n: round(COOLDOWN_S - (t - ft), 1) for n, ft in self.fired_at.items() if t - ft < COOLDOWN_S}
        wait = [n for n in REARM_SIGNS if not self.armed.get(n, True)]
        return dict(pending=[(round(td - t, 1), c.get("cmd") or c.get("local"), c.get("scale", c.get("on", "")))
                             for td, c in sorted(self.timers, key=lambda x: x[0])],
                    cooldown={k: v for k, v in cool.items() if not k.startswith("arrow:")},
                    arrows=dict(self.arrow_votes), wait_out_of_view=wait, red_hold=self.red_hold,
                    red_now=self.red_seen_at is not None and t - self.red_seen_at <= RED_MEMORY_S)

    def pending(self):
        return [(round(td, 2), c) for td, c in sorted(self.timers, key=lambda x: x[0])]

    def shutdown(self):
        """Commands that undo whatever is in progress (vision process ending)."""
        out = [c for _, c in self.timers if c.get("cmd") in ("resume",) or c.get("scale") == 1.0
               or c.get("on") is False]
        self.timers = []
        return out
