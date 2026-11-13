import unittest
import numpy as np
from bench.world import Track
from .control import LocalPath
from .coordinator import Coordinator


class FakePredictor:
    def __init__(self):self.calls=[]
    def observe(self,image,capture,previous,direction,active):
        self.calls.append((previous.copy(),direction,active))
        if active:return LocalPath(np.c_[np.zeros(48),np.arange(48)*.5-2],np.full(48,4.),capture)


class CoordinatorTests(unittest.TestCase):
    def make(self):
        p=FakePredictor();return Coordinator(Track().data['junctions'],'cw',p),p

    def test_actual_receipt_not_proposal_and_warm_state(self):
        c,p=self.make();ordinary=np.array([.4,.42])
        proposed,info=c.propose(None,.1,.136,[80,50,0],.5,ordinary)
        np.testing.assert_array_equal(proposed,ordinary);self.assertEqual(info['reason'],'ordinary_policy')
        self.assertEqual(len(c.tracker.motion.commands),1)
        with self.assertRaises(RuntimeError):c.propose(None,.2,.236,[80,50,0],.5,ordinary)
        c.on_applied(.136,[0,0])  # external hold overrides the nonzero proposal
        with self.assertRaises(ValueError):c.on_applied(.136,[0,0])
        c.propose(None,.2,.236,[80,50,0],.5,ordinary)
        np.testing.assert_array_equal(p.calls[-1][0],[0,0]);self.assertEqual(len(p.calls),2)

    def test_junction_ownership_and_independent_holds(self):
        c,p=self.make();pose=[0,30,np.pi]
        c.push_direction(0,'left')
        _,info=c.propose(None,.1,.136,pose,.5,[.8,.8]);c.on_applied(.136,[.36,.36])
        self.assertTrue(info['active']);self.assertEqual(info['instruction'],'right')
        self.assertEqual(info['reason'],'cnn_path');self.assertEqual(len(c.resolver.policy.queue),1)
        c.stop(.14,'red');c.stop(.15,'stop_sign');c.resume(.16,'stop_sign')
        out,info=c.propose(None,.2,.236,pose,.5,[.8,.8]);c.on_applied(.236,out)
        np.testing.assert_array_equal(out,[0,0]);self.assertEqual(info['held'],['red'])
        self.assertEqual(p.calls[-1][1],'right');self.assertEqual(len(c.resolver.policy.queue),1)
        c.resume(.24,'red')
        _,info=c.propose(None,.3,.336,pose,.5,[.8,.8])
        self.assertEqual(info['reason'],'cnn_path');self.assertEqual(len(c.resolver.policy.log),2)


if __name__=='__main__':unittest.main()
