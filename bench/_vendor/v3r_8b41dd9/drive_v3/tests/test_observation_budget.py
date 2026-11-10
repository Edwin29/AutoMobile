import unittest
from types import SimpleNamespace
import numpy as np
from drive_v3.estimation import Belief
from drive_v3.map_model import TrackMap
from drive_v3.planning import LocalPlanner
from drive_v3.settings import load_settings


class CameraBudgetTests(unittest.TestCase):
    def setUp(self):
        cfg=load_settings()[0]
        track=TrackMap()
        self.planner=LocalPlanner(track,track.outer_route('ccw'),cfg['planner'],cfg['vehicle'])
        self.belief=Belief(np.array([-86.25,40.,np.pi]),np.diag([.01,.01,.00001]),last_observed_s=1.)
        self.est=SimpleNamespace(hypotheses=[self.belief],motion_distance=0.)

    def test_fresh_delayed_camera_resets_distance_budget(self):
        for stamp,distance in [(1.,0.),(1.2,6.),(1.4,12.)]:
            self.belief.last_observed_s=stamp;self.est.motion_distance=distance
            result=self.planner.plan(self.est,None,stamp+.02,'test')
            self.assertNotEqual(result.reason,'observation_budget_exhausted')

    def test_repeated_capture_does_not_reset_distance_budget(self):
        self.planner.plan(self.est,None,1.02,'test')
        self.est.motion_distance=4.1
        self.assertEqual(self.planner.plan(self.est,None,1.2,'test').reason,'observation_budget_exhausted')

    def test_old_observation_still_times_out(self):
        self.assertEqual(self.planner.plan(self.est,None,1.8,'test').reason,'observation_budget_exhausted')


if __name__=='__main__':unittest.main()
