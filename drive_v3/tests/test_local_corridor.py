import unittest
import numpy as np
from drive_v3.local_corridor import LocalCorridorPlanner


class CameraStub:
    ppcm=5.;lat_min=-40.;lens_ahead=12.;fwd_max=40.

    def body_to_pixels(self,points):
        p=np.asarray(points)
        return np.c_[(p[:,0]-self.lat_min)*self.ppcm,(self.fwd_max+self.lens_ahead-p[:,1])*self.ppcm]


class LocalCorridorTests(unittest.TestCase):
    def masks(self,curve=False,opening=False):
        x=np.arange(401)[None,:]/5.-40.
        y=52.-np.arange(201)[:,None]/5.
        center=-.02*(y-19.)**2 if curve else np.zeros_like(y)
        left=x<center-12.;right=x>center+12.
        if opening:left=left&(y<25.)
        return dict(colored=left|right,valid=np.ones((201,401),bool))

    def test_no_boundary_never_authorizes_motion(self):
        d=self.masks();d['colored'][:]=False
        path=LocalCorridorPlanner(CameraStub()).plan_local(d,1.,1.02,'test')
        self.assertEqual(path.speed_limit_cm_s,0.)

    def test_stale_image_never_authorizes_motion(self):
        path=LocalCorridorPlanner(CameraStub()).plan_local(self.masks(),1.,1.5,'test')
        self.assertEqual(path.reason,'local_frame_stale')

    def test_T_opening_uses_outside_boundary_without_turning_into_branch(self):
        planner=LocalCorridorPlanner(CameraStub())
        path=planner.plan_local(self.masks(opening=True),1.,1.02,'test')
        self.assertEqual(path.reason,'tracking')
        self.assertGreater(planner.last_diagnostics['width_inferred_rows'],0)
        self.assertLess(np.max(np.abs(np.array(path.points_body_cm)[:,0])),.3)

    def test_corner_lookahead_reaches_observed_curve(self):
        planner=LocalCorridorPlanner(CameraStub())
        path=planner.plan_local(self.masks(curve=True),1.,1.02,'test')
        self.assertEqual(path.reason,'tracking')
        self.assertGreater(path.lookahead_cm,np.linalg.norm(planner.last_diagnostics['first_observed_body_cm']))
        self.assertLess(path.points_body_cm[-1][0],-5.)


if __name__=='__main__':unittest.main()
