"""One scorer for every driver. Uses truth only; driver flags are never evidence.

Sampled every 20 ms of simulated time:
  margin     footprint-to-boundary distance (+ inside the road, - over the line)
  episodes   continuous stretches with margin < 0; 'recovered' if back inside for 1 s
  progress   arc along a reference centre line (signed, wraps)
  corners    for each of the four outer corners: the car's heading change between
             45 cm before and 45 cm after the corner centre (along the reference line).
             Proper = 60..120 deg in the right direction. This captures the whole turn
             wherever the car starts turning (a box-based count missed early turn-ins).
  outcome    lap_complete / severe_departure / held (zero output 5 s) / time_limit
             After the lap distance the run continues (at most 60 cm) until every corner
             is scored and any open line crossing has either recovered or not.
"""
import numpy as np

from .world import wrap

CORNER_BOX_CM = 60.0     # identifies corner regions only; the start zone (y -63..-33) is outside
CORNER_WINDOW_CM = 45.0  # heading measured this far before/after each corner centre
LAP_GRACE_CM = 60.0


def corner_box(x, y):
    return ("E" if x > 0 else "W") + ("N" if y > 0 else "S") if abs(x) > CORNER_BOX_CM and abs(y) > CORNER_BOX_CM else None


class Scorer:
    def __init__(self, track, rim, route, direction, start_pose):
        from .world import to_world
        self._to_world = to_world
        self.track, self.rim, self.direction = track, rim, direction
        r = route if direction == "cw" else route[::-1]
        self.route = np.vstack((r, r[:1]))
        self.arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.route, axis=0), axis=1))]
        self.length = float(self.arc[-1])
        self.start_arc = self.last_arc = self._arc(start_pose)
        self.progress = 0.0
        self.t = 0.0
        self.min_margin = np.inf
        self.outside_s = self.zero_s = self.zero_run_s = 0.0
        self.episode, self.episodes, self.inside_since = None, [], None
        self.corner_plan = self._corner_centres()            # [(name, centre progress)], driving order
        self.corner_in = {}                                   # name -> heading at window start
        self.corners = []
        self.end = None
        self.lap_reached = None
        self.first_corner_progress = self.corner_plan[0][1] - 35.0 if self.corner_plan else 45.0
        self.start_yaw = float(start_pose[2])

    def _arc(self, pose):
        return float(self.arc[int(np.argmin(np.sum((self.route[:-1] - pose[:2]) ** 2, axis=1)))])

    def _corner_centres(self):
        """Progress (from the start) of each corner centre, in driving order."""
        n = len(self.route) - 1
        prog = (self.arc[:n] - self.start_arc) % self.length
        order = np.argsort(prog)
        groups = []
        for i in order:
            box = corner_box(*self.route[i])
            if box is None:
                continue
            if groups and groups[-1][0] == box and prog[i] - groups[-1][2] < 5.0:
                groups[-1][1].append(prog[i]); groups[-1][2] = prog[i]
            else:
                groups.append([box, [prog[i]], prog[i]])
        return [(g[0], float(np.mean(g[1]))) for g in groups]

    def sample(self, t, dt, pose, pwm):
        self.t = t
        d = self.track.clearance(self._to_world(self.rim, pose))
        m = float(d.min())
        self.min_margin = min(self.min_margin, m)
        a = self._arc(pose)
        self.progress += (a - self.last_arc + self.length / 2) % self.length - self.length / 2
        self.last_arc = a
        if m < 0:
            self.outside_s += dt
            self.inside_since = None
            if self.episode is None:
                self.episode = dict(start_s=t, start_progress_cm=self.progress, min_cm=m)
            self.episode["min_cm"] = min(self.episode["min_cm"], m)
        elif self.episode is not None:
            self.inside_since = t if self.inside_since is None else self.inside_since
            if t - self.inside_since >= 1.0 - 1e-9:
                self.episode.update(end_s=t, recovered=True)
                self.episodes.append(self.episode)
                self.episode, self.inside_since = None, None
        if np.max(np.abs(pwm)) < 1e-8:
            self.zero_s += dt
            self.zero_run_s += dt
        else:
            self.zero_run_s = 0.0
        # corners: heading across a window around each centre
        expected = -1 if self.direction == "ccw" else 1
        done = {c["corner"] for c in self.corners}
        for name, centre in self.corner_plan:
            if name in done:
                continue
            if name not in self.corner_in and self.progress >= max(0.0, centre - CORNER_WINDOW_CM):
                self.corner_in[name] = (self.start_yaw if centre - CORNER_WINDOW_CM <= 0 else float(pose[2]), t)
            if name in self.corner_in and self.progress >= centre + CORNER_WINDOW_CM:
                yaw0, t0 = self.corner_in[name]
                deg = float(np.degrees(wrap(pose[2] - yaw0)))
                self.corners.append(dict(corner=name, start_s=t0, end_s=t, turn_deg=deg,
                                         proper=60 <= expected * deg <= 120))
        if d.max() < 0 or m < -12:
            self.end = "severe_departure"
        elif self.progress >= self.length - 0.5:
            self.lap_reached = self.progress if self.lap_reached is None else self.lap_reached
            all_scored = len(self.corners) >= len(self.corner_plan)
            if (all_scored and self.episode is None) or self.progress >= self.lap_reached + LAP_GRACE_CM:
                self.end = "lap_complete"
        elif self.zero_run_s >= 5.0:
            self.end = "held"

    def summary(self, outcome):
        if self.episode is not None:
            self.episode.update(end_s=self.t, recovered=False)
            self.episodes.append(self.episode)
            self.episode = None
        proper = sum(c["proper"] for c in self.corners)
        unrecovered = sum(not e["recovered"] for e in self.episodes)
        first = [e["min_cm"] for e in self.episodes if e["start_progress_cm"] < self.first_corner_progress + 80]
        later = [e["min_cm"] for e in self.episodes if e["start_progress_cm"] >= self.first_corner_progress + 80]
        return dict(outcome=outcome, verified_lap=outcome == "lap_complete" and proper == 4 and unrecovered == 0,
                    progress_cm=self.progress, lap_cm=self.length, simulated_s=self.t,
                    proper_corners=proper, corners=self.corners, corner_plan=self.corner_plan,
                    min_margin_cm=self.min_margin, outside_s=self.outside_s, zero_output_s=self.zero_s,
                    clean_lap=self.min_margin >= 0,
                    worst_first_corner_cm=min(first) if first else 0.0,
                    worst_after_first_corner_cm=min(later) if later else 0.0,
                    episodes=self.episodes, unrecovered_episodes=unrecovered)
