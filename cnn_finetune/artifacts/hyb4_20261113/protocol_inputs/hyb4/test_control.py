"""Control contracts, causal command receipt and geometric invariances."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from .control import LocalPath, SignedDrive, AppliedMotion, PathTracker


class ControlTests(unittest.TestCase):
    def test_signed_inverse_and_saturation(self):
        d=SignedDrive()
        for v,w in [(4.,0.),(2.,1.),(2.,-1.)]:
            command=d.inverse(v,w)
            np.testing.assert_allclose(d.intended_motion(command),[v,w],atol=1e-12)
        self.assertLess(d.inverse(2.,1.)[1],0.)
        v,w=d.intended_motion(d.inverse(12.,3.))
        self.assertAlmostEqual(w/v,3./12.)
        self.assertAlmostEqual(max(abs(d.inverse(12.,3.))),1.)

    def test_pulse_density_integrated_duty(self):
        d=SignedDrive();rng=np.random.default_rng(51)
        intervals=rng.uniform(.02,.06,1000);desired=np.array([.12,-.21])
        applied=np.array([d.allocate(desired,dt) for dt in intervals])
        error=((applied-desired)*intervals[:,None]).sum(axis=0)
        self.assertLess(max(abs(error)),.36*intervals.max())
        self.assertTrue(np.all((applied==0)|(abs(applied)==.36)))
        d.reset();np.testing.assert_array_equal(d.allocate([0.,0.],.036),[0.,0.])

    def test_receipts_and_delay(self):
        m=AppliedMotion()
        np.testing.assert_array_equal(m.at(.2),[0.,0.])
        m.acknowledge(.2,[.5,-.5])
        np.testing.assert_array_equal(m.at(.249),[0.,0.])
        v,w=m.at(.5);self.assertAlmostEqual(v,0.);self.assertGreater(w,0.)
        m.acknowledge(.5,[0.,0.]);self.assertGreater(m.at(.52)[1],0.)
        self.assertLess(m.at(1.)[1],w*.1)
        for t in [.1,float('inf'),float('nan')]:
            with self.assertRaises(ValueError):m.at(t)

    def test_path_ownership_age_and_steering(self):
        pts=np.c_[np.zeros(48),np.arange(48)*.5-2];speed=np.full(48,4.)
        p=LocalPath(pts,speed,0.);pts[:]=5;speed[:]=0
        self.assertEqual(p.points_cm[0,0],0.);self.assertEqual(p.speeds_cm_s[0],4.)
        straight,_=PathTracker().desired(p,.036)
        left=LocalPath(p.points_cm+[-1,0],p.speeds_cm_s,0.)
        right=LocalPath(p.points_cm+[1,0],p.speeds_cm_s,0.)
        l,_=PathTracker().desired(left,.036);r,_=PathTracker().desired(right,.036)
        np.testing.assert_allclose(l,r[::-1]);self.assertLess(l[0],l[1])
        self.assertEqual(straight[0],straight[1])
        for t in [-.01,.251]:
            with self.assertRaises(ValueError):PathTracker().desired(p,t)
        with self.assertRaises(ValueError):LocalPath(np.zeros((5,2)),np.ones(5),0.)

    def test_temporal_network_and_checkpoint_contract(self):
        import torch
        from .model import TrajectoryPolicy,Predictor,AUX_SCHEMA
        torch.set_num_threads(2);torch.manual_seed(17)
        model=TrajectoryPolicy().eval()
        with torch.no_grad():
            images=torch.rand(1,3,3,66,200);aux=torch.zeros(1,3,len(AUX_SCHEMA))
            path,speed,h=model(images,aux)
            hs=None;out=[]
            for i in range(3):
                p,_,hs=model(images[:,i:i+1],aux[:,i:i+1],hs);out.append(p)
            torch.testing.assert_close(path,torch.cat(out,dim=1))
        self.assertEqual(tuple(path.shape),(1,3,48,2));self.assertEqual(tuple(h.shape),(1,1,64))
        self.assertTrue(torch.all((speed>0)&(speed<8)))
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'policy.pt';torch.save({'format':'signed-pwm-v1'},file)
            with self.assertRaises(ValueError):Predictor(file)
            torch.save({'format':'hyb4-path-v1','aux_schema':AUX_SCHEMA,'trained':False},file)
            with self.assertRaises(ValueError):Predictor(file)


if __name__=='__main__':unittest.main()
