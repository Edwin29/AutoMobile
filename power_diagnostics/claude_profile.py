"""Use the installed driving controller's functions with scripted curvature inputs."""
import hashlib
import importlib
from pathlib import Path
import sys


class ClaudeProfile:
    def __init__(self):
        sys.path.insert(0, '/home/pi/TEST')
        self.module = importlib.import_module('lane_drive')
        self.curvature = 0.
        self.output = (0., 0.)

    def metadata(self):
        m = self.module
        return dict(source=m.__file__, sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
                    accel=m.ACCEL_PER_S, smoothing=m.STEER_SMOOTH,
                    pwm=m.PWM_FREQ_HZ, wheel_track_cm=m.WHEEL_TRACK_CM,
                    corner_slow_cm=m.CORNER_SLOW_CM,
                    scripted_curvatures=[0., -.05, .05, 0.],
                    note='Scripted curvature, not measured lane following; no SD image writing.')

    def step(self, curvature, speed, dt, voltage):
        m = self.module
        self.curvature = m.STEER_SMOOTH*self.curvature + (1-m.STEER_SMOOTH)*curvature
        target = m.wheel_speeds(self.curvature, speed)
        if voltage < m.VOLT_WARN:
            target = tuple(x*.8 for x in self.output)
        self.output = tuple(float(x) for x in m.slew(self.output, target, m.ACCEL_PER_S*dt))
        return self.output
