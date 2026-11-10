import unittest
import numpy as np
from drive_v3.control import Controller,ResponseModel
from drive_v3.settings import load_settings
from drive_v3.contracts import PathDecision


class RollingFloorTests(unittest.TestCase):
    def run_curve(self,radius):
        cfg=load_settings()[0];c=cfg['controller']
        c.update(pwm_min=.30,pwm_max=.60,preferred_speed_cm_s=4.8,trim_enabled=False)
        model=ResponseModel(c,cfg['vehicle']);control=Controller(model,c,'pure_pursuit')
        s=np.linspace(0,28,57)
        p=np.c_[-radius*(1-np.cos(s/radius)),radius*np.sin(s/radius)]
        results=[]
        for i in range(30):
            now=i*.1
            out=control.step(PathDecision(p.tolist(),8.,now+.25,'test','tracking'),now,.1,'test',True)
            self.assertTrue(np.all(out.pwm<=.60+1e-9))
            self.assertTrue(np.all((out.pwm<1e-9)|(out.pwm>=.30-1e-9)))
            self.assertLessEqual(out.v_cm_s,8.+1e-9)
            results.append(out)
        return results

    def test_rolls_on_feasible_curve_within_planned_speed(self):
        out=self.run_curve(30)[-1]
        self.assertTrue(np.all(out.pwm>=.30))
        self.assertLess(out.pwm[0],out.pwm[1])

    def test_impossible_ratio_stops_instead_of_clipping_inside_wheel(self):
        out=self.run_curve(10)[-1]
        self.assertEqual(out.reason,'deadband_path_infeasible')
        self.assertTrue(np.all(out.pwm==0))


if __name__=='__main__':unittest.main()
