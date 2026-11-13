import unittest
import numpy as np
from .goal_development import LocalScene, terminal_at_gate
from .junction_learning import Scene, Encoder, setup, model_at, OUT


class GoalChecks(unittest.TestCase):
    def test_exit_gate_does_not_require_arbitrary_one_cm(self):
        self.assertIsNone(terminal_at_gate(54.9,2,3,1,2))
        self.assertEqual(terminal_at_gate(55,2,3,1,2),'stable_exit')
        self.assertEqual(terminal_at_gate(55,2,3,-.01,2),'unstable_exit')
        self.assertEqual(terminal_at_gate(55,.1,3,1,.5),'unstable_exit')
        self.assertEqual(terminal_at_gate(55,.1,11,1,2),'unstable_exit')
        self.assertEqual(terminal_at_gate(55,20,0,1,2),'missed_exit')

    def test_changed_scorer_preserves_prefix_physics_and_inputs(self):
        setup();encoder=Encoder().cuda().eval();policy=model_at(OUT/'models/outcome/epoch_8.pt')
        a=Scene(('east','W','right'),'mild',4101,'sluggish',encoder)
        b=LocalScene(('east','W','right'),'mild',4101,'sluggish',encoder)
        a.finish(policy,.8);b.finish(policy,.8)
        np.testing.assert_array_equal(a.physics,b.physics)
        for k in ['x','f','aux','action','applied']:
            np.testing.assert_array_equal(a.records[-1][k],b.records[-1][k])
        c,d=b.clone(),b.clone();d.teacher=lambda *unused:np.array([-1.,1.])
        c.step(policy);d.step(policy)
        np.testing.assert_array_equal(c.records[-1]['action'],d.records[-1]['action'])


if __name__=='__main__':unittest.main()
