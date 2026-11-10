import unittest
import numpy as np
from drive_v3.compare_recovery import settings
from drive_v3.control import ResponseModel
from drive_v3.recovery_control import RollingController,rolling_allocate
from drive_v3.estimation import Estimator
from drive_v3.map_model import TrackMap
from drive_v3.contracts import PathDecision
from drive_v3.recovery_planning import RecoveryPlanner


class RecoveryExperimentTests(unittest.TestCase):
    def test_allocator_is_bounded_and_symmetric(self):
        cfg=settings('recovery');model=ResponseModel(cfg['controller'],cfg['vehicle'])
        for curvature in np.linspace(-.3,.3,41):
            p=rolling_allocate(model,4.8,4.8*curvature,8.)
            q=rolling_allocate(model,4.8,-4.8*curvature,8.)
            np.testing.assert_allclose(p,q[::-1])
            self.assertGreaterEqual(p.min(),.30-1e-9);self.assertLessEqual(p.max(),.60+1e-9)
            self.assertLessEqual(model.forward_model(p)[0],8.+1e-9)

    def test_low_mission_speed_cap_cannot_be_silently_raised(self):
        cfg=settings('recovery');model=ResponseModel(cfg['controller'],cfg['vehicle'])
        self.assertIsNone(rolling_allocate(model,3.,0.,3.))

    def test_stop_and_expired_path_always_zero(self):
        cfg=settings('recovery');c=RollingController(ResponseModel(cfg['controller'],cfg['vehicle']),cfg['controller'])
        p=[[0,0],[0,10],[0,20]]
        out=c.step(PathDecision(p,8.,.1,'x','tracking'),1.,.2,'x',True)
        self.assertTrue(np.all(out.pwm==0))
        out=c.step(PathDecision(p,0.,2.,'x','mission_hold'),1.,.2,'x',True)
        self.assertEqual(out.reason,'mission_hold');self.assertTrue(np.all(out.pwm==0))

    def test_stationary_uncertainty_not_motion_noise(self):
        cfg=settings('recovery');e=Estimator(TrackMap(),[0,0,0],cfg['estimator']);p=e.primary.P.copy()
        for i in range(50):e.predict(0,0,.2)
        np.testing.assert_array_equal(e.primary.mean,[0,0,0])
        self.assertLess(float(np.trace(e.primary.P-p)),.02)
        e.predict(5,0,1.)
        self.assertGreater(float(np.trace(e.primary.P-p)),2.)

    def test_startup_both_sides_zero_until_rolling_floor(self):
        cfg=settings('recovery');c=RollingController(ResponseModel(cfg['controller'],cfg['vehicle']),cfg['controller'])
        path=PathDecision([[0,0],[2,10],[8,20]],8.,10.,'x','tracking')
        for i in range(8):
            out=c.step(path,(i+1)*.1,.1,'x',True)
            self.assertTrue(np.all(out.pwm==0) or np.all(out.pwm>=.30-1e-9))

    def test_deep_rescue_budget_and_no_overlap_remain_stops(self):
        cfg=settings('recovery_rescue');track=TrackMap()
        # Both guards precede route sampling; no route or vehicle hardware needed.
        c=RollingController(ResponseModel(cfg['controller'],cfg['vehicle']),cfg['controller'])
        planner=RecoveryPlanner(track,None,cfg['planner'],cfg['vehicle'],c)
        e=Estimator(track,[-92.,0.,0.],cfg['estimator']);e.motion_distance=50.;planner.last_motion_distance=0.
        self.assertEqual(planner.plan(e,None,1.,'x').reason,'recovery_travel_budget')
        e.primary.mean=np.array([-120.,0.,0.])
        self.assertEqual(planner.plan(e,None,2.,'x').reason,'estimated_no_road_overlap')
