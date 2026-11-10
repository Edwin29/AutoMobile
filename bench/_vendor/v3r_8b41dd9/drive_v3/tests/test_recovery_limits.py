import unittest
from drive_v3.recovery_limits import depth_rejection


class RecoveryLimitsTests(unittest.TestCase):
    def test_field_like_initial_depth_does_not_veto_escape(self):
        self.assertEqual(depth_rejection([-6.06,-6.06,-3.,1.7]),'deep_departure')
        self.assertIsNone(depth_rejection([-6.06,-6.06,-3.,1.7],allow_existing=True))

    def test_new_deep_departure_is_still_rejected(self):
        self.assertEqual(depth_rejection([1.,-6.06,2.],allow_existing=True),'deep_departure')

    def test_existing_departure_cannot_get_deeper_or_stay_outside(self):
        self.assertEqual(depth_rejection([-6.1,-6.3,2.],allow_existing=True),'existing_deep_worsens')
        self.assertEqual(depth_rejection([-6.1,-6.,-1.],allow_existing=True),'existing_deep_not_back_inside')

    def test_preexisting_non_deep_profiles_keep_same_decision(self):
        for profile in [[3.,2.,4.],[-1.,-2.,1.],[-5.,-5.8,2.],[-5.,-7.,2.]]:
            self.assertEqual(depth_rejection(profile),depth_rejection(profile,allow_existing=True))
