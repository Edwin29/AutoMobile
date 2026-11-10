import unittest
from types import SimpleNamespace
import numpy as np
from drive_v3.field_stress import velocity,StressPlant,render_input
from drive_v3.map_model import TrackMap,Route
from drive_v3.geometry import footprint,wrap
from drive_v3.compare_recovery import settings
from drive_v3.control import ResponseModel


class FieldStressTests(unittest.TestCase):
    def test_adjacent_float_at_pwm_floor_is_not_a_stalled_wheel(self):
        cfg=settings('recovery');model=ResponseModel(cfg['controller'],cfg['vehicle'])
        low=np.nextafter(.3,0.)
        for pwm in ([.6,low],[low,.6]):
            v,w=model.forward_model(pwm)
            self.assertAlmostEqual(v,7.2);self.assertAlmostEqual(abs(w),4.8/13.)

    def test_numerical_tolerance_does_not_bypass_real_deadband(self):
        cfg=settings('recovery');model=ResponseModel(cfg['controller'],cfg['vehicle'])
        v,w=model.forward_model([.3-1e-6,.6])
        self.assertAlmostEqual(v,4.8);self.assertAlmostEqual(w,-9.6/13.)

    def test_zero_command_cannot_create_drift(self):
        v,w,_=velocity([0.,0.],{'drift':-.035,'turn_proxy':True},1.,.693,.632)
        self.assertEqual((v,w),(0.,0.))

    def test_directional_response_changes_yaw_not_forward_speed(self):
        for pwm,ratio in [([.3,.6],.693),([.6,.3],.632)]:
            v,w,_=velocity(pwm,{},0,.693,.632)
            v2,w2,_=velocity(pwm,{'turn_proxy':True},0,.693,.632)
            self.assertEqual(v,v2);self.assertAlmostEqual(w2/w,ratio)

    def test_common_power_loss_preserves_instantaneous_curvature(self):
        v,w,_=velocity([.6,.3],{},1,.69,.63)
        vv,ww,_=velocity([.6,.3],{'sag':.65,'sag_s':12},1,.69,.63)
        self.assertAlmostEqual(vv/v,.65);self.assertAlmostEqual(ww/vv,w/v)
        v3,w3,_=velocity([.6,.3],{'sag':.65,'sag_s':12},13,.69,.63)
        self.assertEqual((v,w),(v3,w3))

    def test_stalled_left_side_and_release(self):
        v,w,_=velocity([.3,.3],{'stall_s':2.5},1,.69,.63)
        self.assertAlmostEqual(v,3.6);self.assertLess(w,0)
        v,w,_=velocity([.3,.3],{'stall_s':2.5},3,.69,.63)
        self.assertAlmostEqual(v,4.8);self.assertEqual(w,0)

    def test_slip_is_not_leaked_into_command_only_predictor(self):
        initial=np.array([-87.,-20.,np.pi]);route=Route('test',np.array([[-87.,-20.],[-87.,-40.],[-87.,-60.]]))
        plant=StressPlant(initial,TrackMap(),route,footprint(settings('recovery')['vehicle']),0,
            {'slip_cm':6.,'slip_deg':8.},'ccw',17,(.693,.632))
        plant.event_time=-2.;plant.advance_to(.02)
        np.testing.assert_allclose(plant.shadow[:2],initial[:2])
        self.assertAlmostEqual(wrap(plant.shadow[2]-initial[2]),0.)
        self.assertAlmostEqual(np.linalg.norm(plant.pose[:2]-initial[:2]),6.)

    def test_camera_dropout_clears_by_time_not_by_frame_count(self):
        renderer=SimpleNamespace(render=lambda p:np.full((480,640,3),80,np.uint8))
        plant=SimpleNamespace(pose=np.zeros(3),age=lambda t:t,direction='ccw')
        image,flags=render_input(renderer,plant,.9,{'dropout_s':1.2},17)
        self.assertTrue(np.all(image==245));self.assertIn('blank_camera',flags)
        image,flags=render_input(renderer,plant,1.3,{'dropout_s':1.2},17)
        self.assertTrue(np.all(image==80));self.assertNotIn('blank_camera',flags)

if __name__=='__main__':unittest.main()
