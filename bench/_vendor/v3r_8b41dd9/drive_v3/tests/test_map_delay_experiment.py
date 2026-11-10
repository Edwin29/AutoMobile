import unittest
import numpy as np
from drive_v3.map_delay_experiment import completion, Plant, experiment_settings, MAPS
from drive_v3.map_model import Route, TrackMap


class TimingAndMapTests(unittest.TestCase):
    def test_single_worker_overrun_does_not_queue_old_frames(self):
        done,next_capture,delay=completion(.5,.5,0,[.58,.40])
        self.assertAlmostEqual(done,1.08)
        self.assertEqual(next_capture,done)
        done2,next2,_=completion(next_capture,.5,1,[.58,.40])
        self.assertAlmostEqual(done2,1.48)
        self.assertAlmostEqual(next2,1.58)

    def test_no_motor_command_before_completion(self):
        class WideRoad:
            def clearance(self,p):return np.full(len(p),100.)
        route=Route('straight',np.array([[0.,0.],[0.,500.],[0.,1000.]]))
        plant=Plant(np.zeros(3),WideRoad(),route,np.array([[0.,0.]]),0)
        plant.advance_to(.5)
        np.testing.assert_allclose(plant.pose,[0,0,0])
        plant.pwm=np.array([.3,.3])
        plant.advance_to(1.)
        np.testing.assert_allclose(plant.pose,[0,2.4,0],atol=1e-9)

    def test_distinct_maps_change_clearance_without_mutating_source(self):
        old=TrackMap(MAPS['old']);new=TrackMap(MAPS['corrected'])
        self.assertNotEqual(old.id,new.id)
        p=np.array([[0.,-98.]])
        self.assertGreater(old.clearance(p)[0],0)
        self.assertLess(new.clearance(p)[0],0)

    def test_period_changes_timing_not_road_limits(self):
        a=experiment_settings(.2,0.); b=experiment_settings(.6,.585)
        for key in ('hard_road_depth_cm','normal_speed_cm_s','coast_s'):
            self.assertEqual(a['planner'][key],b['planner'][key])
        for key in ('pwm_min','pwm_max','preferred_speed_cm_s'):
            self.assertEqual(a['controller'][key],b['controller'][key])
        self.assertGreater(b['controller']['watchdog_s'],.6+.585)


if __name__=='__main__':unittest.main()
