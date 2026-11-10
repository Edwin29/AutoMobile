import unittest
import numpy as np
from drive_v3.path_acceptance import judge_profiles,commitment_distance,evaluate_prefix
from drive_v3.settings import load_settings
from drive_v3.map_model import TrackMap
from drive_v3.geometry import footprint
from drive_v3.estimation import Belief


class AcceptanceTests(unittest.TestCase):
    def setUp(self):self.all=load_settings()[0];self.cfg=self.all['planner']

    def test_existing_small_intrusion_can_recover(self):
        ok,reason,_=judge_profiles([[-.3,-.2,.3],[.4,.4,.5]],[[-.3,.3,.6]],self.cfg)
        self.assertTrue(ok,reason)

    def test_each_candidate_is_checked_not_only_global_worst(self):
        ok,reason,_=judge_profiles([[-.3,-.3,-.2],[1.,.5,-.2]],[[.5,.5,.6]],self.cfg)
        self.assertFalse(ok);self.assertEqual(reason,'candidate_worsening')

    def test_tolerance_cannot_ratchet_under_target(self):
        ok,reason,_=judge_profiles([[.21,.19,.3]],[[.21,.3,.4]],self.cfg)
        self.assertFalse(ok);self.assertEqual(reason,'candidate_worsening')

    def test_field_sweep_tolerance_requires_each_endpoint_to_recover(self):
        self.cfg['recovery_motion_tolerance_cm']=.02
        ok,reason,_=judge_profiles([[-.2,-.21,.1],[.1,.09,.1]],[[-.2,.3,.4]],self.cfg)
        self.assertTrue(ok,reason)
        ok,reason,_=judge_profiles([[-.2,-.21,.1],[.1,.09,.09]],[[-.2,.3,.4]],self.cfg)
        self.assertFalse(ok);self.assertEqual(reason,'candidate_end_worsening')

    def test_large_intrusion_remains_rejected_even_if_improving(self):
        ok,reason,_=judge_profiles([[-.8,-.6,-.3]],[[-.8,-.3,.5]],self.cfg)
        self.assertFalse(ok);self.assertEqual(reason,'penetration_depth')

    def test_looser_support_bound_does_not_relax_nominal_bound(self):
        self.cfg.update(allowed_penetration_cm=3.,uncertain_penetration_cm=5.)
        ok,reason,_=judge_profiles([[-2.,-1.,0.],[-4.,-3.,-2.]],[[-2.,0.,1.]],self.cfg,short_means=[[-2.,-1.,0.]])
        self.assertTrue(ok,reason)
        ok,reason,_=judge_profiles([[-3.1,-2.,0.],[-4.,-3.,-2.]],[[-3.1,0.,1.]],self.cfg,short_means=[[-3.1,-2.,0.]])
        self.assertFalse(ok);self.assertEqual(reason,'nominal_penetration_depth')

    def test_nominal_intrusion_cannot_persist_without_improvement(self):
        ok,reason,_=judge_profiles([[-.2,-.2,-.2]],[[-.2,-.2,.3]],self.cfg)
        self.assertFalse(ok);self.assertEqual(reason,'nominal_short_no_recovery')

    def test_commitment_grows_with_speed_and_delay(self):
        self.assertGreater(commitment_distance(self.cfg,5,.2),commitment_distance(self.cfg,5,0))
        self.assertGreater(commitment_distance(self.cfg,5,0),commitment_distance(self.cfg,3,0))

    def test_path_shorter_than_stopping_commitment_rejected(self):
        path=np.array([[0.,0.],[0.,1.],[0.,2.]])
        belief=Belief(np.array([-86.25,40.,np.pi]),np.eye(3)*.001)
        ok,reason,_=evaluate_prefix(TrackMap(),footprint(self.all['vehicle']),path,[belief],self.cfg,4.)
        self.assertFalse(ok);self.assertEqual(reason,'insufficient_commitment_horizon')


if __name__=='__main__':unittest.main()
