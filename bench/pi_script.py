"""Scripted orchestrator for field tests: signs / lights decided in advance, fired from the
car's OWN map pose (no truth on the car). Stands in for the vision team + orchestrator
so nobody has to type commands while the car drives.

    ./run_mission.sh --drive --direction cw --script inner_loop
    ./run_mission.sh --drive --direction ccw --script "dir:south:W:left, red:S:4"
    ./run_mission.sh --list-scripts

Tokens (comma separated):
    dir:JUNCTION:ARM:MOVE[:AT_CM]      reserve MOVE when the car is on ARM of JUNCTION heading in,
                                       AT_CM (default 60) or less from its centre (a sign seen there)
    red:ARM[:HOLD_S[:TRIGGER_CM]]      crossroad red light on that arm: stop (reason red_light) when
                                       the front is TRIGGER_CM (default 4) before the crosswalk,
                                       resume after HOLD_S (default 4)
    stop:PINK[:HOLD_S[:TRIGGER_CM]]    stop sign: stop (reason stop_sign) when pink line PINK
                                       (index, see --list-scripts) is TRIGGER_CM (default 20) ahead
                                       of the front; resume after HOLD_S (default 3)
Each token fires once. If the map has lost the car (sxy above MAX_SXY) nothing fires and a
direction then falls back to the junction default, as it would with a missed sign.
"""
import math

import numpy as np

ARMS = {"N": (0.0, 1.0), "E": (1.0, 0.0), "S": (0.0, -1.0), "W": (-1.0, 0.0)}
MAX_SXY = 2.0

PRESETS = {
    "cw": {
        "lap": "",                                                        # no sign: outer lap, all straight
        "inner_loop": "dir:south:E:right, red:S:4",                       # south T -> crossroad (red) -> east T merge
        "inner_loop_west": "dir:west:S:right, red:W:4",                   # west T -> crossroad (red) -> south T merge
        "stop_sign": "stop:6:3",                                          # north road pink line near the NE corner
        "reverse": "dir:south:E:right, dir:east:W:left",                  # merge the other way: cw -> ccw
        "all": "stop:2:3, dir:south:E:right, red:S:4",                    # south road pink line comes first
    },
    "ccw": {
        "lap": "",
        "inner_loop": "dir:south:W:left, red:S:4",                        # south T -> crossroad (red) -> east T merge
        "inner_loop_east": "dir:east:S:left, red:E:4",                    # east T -> crossroad (red) -> north T merge
        "stop_sign": "stop:6:3",
        "reverse": "dir:south:W:left, dir:east:W:right",                  # merge the other way: ccw -> cw
        "all": "stop:1:3, dir:south:W:left, red:S:4",
    },
}


def parse(spec, direction):
    spec = (spec or "").strip()
    spec = PRESETS.get(direction, {}).get(spec, spec)
    events = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        p = tok.split(":")
        if p[0] == "dir":
            events.append(dict(kind="direction", junction=p[1], arm=p[2].upper(), move=p[3],
                               at_cm=float(p[4]) if len(p) > 4 else 60.0))
        elif p[0] == "red":
            events.append(dict(kind="red_light", arm=p[1].upper(), hold_s=float(p[2]) if len(p) > 2 else 4.0,
                               trigger_cm=float(p[3]) if len(p) > 3 else 4.0))
        elif p[0] == "stop":
            events.append(dict(kind="stop_sign", pink=int(p[1]), hold_s=float(p[2]) if len(p) > 2 else 3.0,
                               trigger_cm=float(p[3]) if len(p) > 3 else 20.0))
        else:
            raise ValueError(f"unknown script token {tok!r}")
    return events


class ScriptOrchestrator:
    def __init__(self, events, track_data, front_cm=14.5):
        self.events = [dict(e, done=False) for e in events]
        self.d, self.front = track_data, front_cm
        self.pending = []                       # (time, kind, arg, reason)
        self.log = []

    def tick(self, t, est, sxy):
        """est: the car's map pose (x, y, yaw). Returns the commands due now:
        [(kind, arg, reason)] with kind in direction / stop / resume."""
        if est is not None and sxy is not None and sxy <= MAX_SXY:
            self._watch(t, np.asarray(est, float))
        due = [p for p in self.pending if p[0] <= t]
        self.pending = [p for p in self.pending if p[0] > t]
        return [(k, a, r) for _, k, a, r in due]

    def _fire(self, t, e, *cmds):
        e["done"] = True
        self.log.append(dict(t=float(t), event=e["kind"], spec={k: v for k, v in e.items() if k != "done"}))
        self.pending.extend(cmds)

    def _watch(self, t, pose):
        h = np.array([math.sin(pose[2]), math.cos(pose[2])])
        front = pose[:2] + self.front * h
        for e in self.events:
            if e["done"]:
                continue
            if e["kind"] == "direction":
                c, a = np.asarray(self.d["junctions"][e["junction"]]["xy"], float), np.asarray(ARMS[e["arm"]])
                rel = pose[:2] - c
                if 0 < rel @ a <= e["at_cm"] and abs(rel @ np.array([a[1], -a[0]])) < 10 and h @ -a > 0.8:
                    self._fire(t, e, (t, "direction", e["move"], "sign"))
            elif e["kind"] == "stop_sign":
                a, b = (np.asarray(p, float) for p in self.d["pink_lines"][e["pink"]])
                mid = (a + b) / 2
                ahead = float((mid - front) @ h)
                across = abs(float((mid - front) @ np.array([h[1], -h[0]])))
                if 0 < ahead <= e["trigger_cm"] and across < 15:
                    self._fire(t, e, (t, "stop", None, "stop_sign"), (t + e["hold_s"], "resume", None, "stop_sign"))
            elif e["kind"] == "red_light":
                arm = np.asarray(ARMS[e["arm"]])
                for x0, y0, _ in self.d["crosswalks"]:
                    cw = np.array([x0, y0], float)
                    if abs(cw @ np.array([arm[1], -arm[0]])) < 1 and 0 < cw @ arm < 40:   # the crossroad's crosswalk on that arm
                        ahead = float((cw - front) @ h) - 3.0
                        across = abs(float((cw - front) @ np.array([h[1], -h[0]])))   # field-sim 2026-11-11:
                        if 0 < ahead <= e["trigger_cm"] and across < 12 and h @ -arm > 0.8:  # without it a car 87 cm
                                                                                             # away fired at the start
                            self._fire(t, e, (t, "stop", None, "red_light"),
                                       (t + e["hold_s"], "resume", None, "red_light"))


def describe(track_data):
    lines = ["presets (--script NAME, per --direction):"]
    for d, ps in PRESETS.items():
        for k, v in ps.items():
            lines.append(f"  {d:4s} {k:16s} {v or '(no sign)'}")
    lines.append("pink lines (stop:INDEX):")
    for i, (a, b) in enumerate(track_data["pink_lines"]):
        m = (np.asarray(a) + np.asarray(b)) / 2
        x, y = m
        road = ("east" if x > 0 else "west") if abs(x) > abs(y) else ("north" if y > 0 else "south")
        lines.append(f"  {i}: centre ({x:.1f}, {y:.1f}) on the {road} outer road")
    return "\n".join(lines)
