"""Meaningful geometry/speed bounds and common-loop regression checks."""
import unittest
import numpy as np

from cnn_finetune import junction_cornering as c


class CorneringChecks(unittest.TestCase):
    def test_tangent_curvature_and_finite_difference(self):
        p,h,k,s=c.path_for(('south','N','right'),[16,16,5,5,0,0],True)
        self.assertLess(abs(k[0]),1e-12);self.assertLess(abs(k[-1]),1e-12)
        numeric=np.gradient(np.unwrap(h),s)
        self.assertLess(np.max(abs(numeric[5:-5]-k[5:-5])),.002)
        self.assertAlmostEqual(abs(float(c.wrap(h[-1]-h[0]))),np.pi/2)

    def test_speed_slows_before_curve_and_respects_bounds(self):
        s=np.arange(0,40,.25);k=np.where((s>18)&(s<25),.2,0.)
        v=c.speed_profile(k,s,6)
        self.assertLess(v[np.searchsorted(s,15)],6)
        self.assertTrue(np.all(v*abs(k)<=.45+1e-9))
        self.assertTrue(np.all(abs(np.diff(v*v))<=2*np.diff(s)+1e-9))

    def test_original_simulation_exact_regression(self):
        for key,preset,level,seed in [(('center','N','left'),'field_estimate','fixed',1303),
                                      (('south','N','right'),'sluggish','mild',1307)]:
            result=c.run(key,'old','legacy',preset,level,seed,'regression')
            old=c.OLD/'dynamic'/('_'.join(key)+f'_{preset}_{level}_{seed}_v{c.old_speed(key)}_l0.0')
            a=np.load(result['artifact']+'/trace.npz');b=np.load(old/'trace.npz')
            np.testing.assert_array_equal(a['samples'],b['samples'])
            np.testing.assert_array_equal(a['commands'],b['commands'])


if __name__=='__main__':unittest.main()
