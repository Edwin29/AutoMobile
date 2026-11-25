import math
import unittest

from .return_stop import ReturnStop


class ReturnStopTest(unittest.TestCase):
    def setUp(self):
        self.stop = ReturnStop(((-10, 10), (-15, 15), 0))
        self.t = 0

    def frame(self, x=0, yaw=0, speed=10, age=0.1, sxy=1, dt=0.1):
        self.t += dt
        info = dict(est=(x, 0, yaw), sxy=sxy, obs_age=age)
        return self.stop.sample(self.t, info, speed)

    def travel(self, **kwargs):
        for _ in range(320):
            self.assertFalse(self.frame(**kwargs))

    def test_staying_at_start_cannot_finish(self):
        self.travel()

    def test_departure_without_travel_cannot_finish(self):
        self.frame(x=25)
        self.travel(speed=0)
        self.frame(x=25, dt=40)
        self.assertFalse(self.frame())
        self.assertLess(self.stop.travel_cm, 3)

    def test_heading_and_continuous_recent_observations(self):
        self.travel(x=25)
        self.travel(yaw=math.pi)
        for invalid in (dict(age=0.5), dict(age=None), dict(sxy=3), dict(dt=1)):
            self.assertFalse(self.frame())
            self.assertFalse(self.frame())
            self.assertFalse(self.frame(**invalid))
        self.assertTrue(any(self.frame() for _ in range(6)))
