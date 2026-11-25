"""Start-zone finish detector using the live map and previously applied motor commands."""
import math


class ReturnStop:
    def __init__(self, zone):
        self.x, self.y, self.yaw = zone
        self.departed = False
        self.travel_cm = 0.0
        self.last_t = None
        self.inside_since = None

    def in_zone(self, pose):
        x, y, yaw = pose
        return (self.x[0] < x < self.x[1] and self.y[0] < y < self.y[1]
                and math.cos(yaw - self.yaw) > 0.85)

    def sample(self, t, info, previous_speed):
        dt = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t
        if 0 < dt <= 0.2:
            self.travel_cm += abs(previous_speed) * dt
        else:
            self.inside_since = None
        pose, sxy, age = info.get("est"), info.get("sxy"), info.get("obs_age")
        if (pose is None or sxy is None or age is None or not all(math.isfinite(v) for v in (*pose, sxy, age))
                or sxy > 2.0 or not 0 <= age <= 0.3):
            self.inside_since = None
            return False
        x, y, _ = pose
        if x < self.x[0]-5 or x > self.x[1]+5 or y < self.y[0]-5 or y > self.y[1]+5:
            self.departed = True
        if not self.departed or self.travel_cm <= 300 or not self.in_zone(pose):
            self.inside_since = None
            return False
        if self.inside_since is None:
            self.inside_since = t
        return t - self.inside_since >= 0.3
