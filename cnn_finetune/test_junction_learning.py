import unittest
import numpy as np
import torch

from .junction_learning import Scene,Encoder,TemporalPolicy,setup,outcome_cost


class LearningChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):setup()

    def test_streaming_memory_equals_full_sequence(self):
        torch.manual_seed(17);m=TemporalPolicy().cuda().eval()
        f=torch.randn(1,80,1152,device='cuda');aux=torch.randn(1,80,4,device='cuda')
        with torch.no_grad():
            full,_=m(f,aux);h=None;stream=[]
            for i in range(80):y,h=m(f[:,i:i+1],aux[:,i:i+1],h);stream.append(y)
        self.assertLess(float((full-torch.cat(stream,dim=1)).abs().max()),1e-5)

    def test_camera_vehicle_allocator_hidden_fork(self):
        torch.manual_seed(17);m=TemporalPolicy().cuda().eval();encoder=Encoder().cuda().eval()
        env=Scene(('east','W','right'),'mild_light',2001,encoder=encoder).finish(m,.4)
        a,b=env.clone(),env.clone()
        for _ in range(8):self.assertTrue(a.step(m));self.assertTrue(b.step(m))
        np.testing.assert_array_equal(np.asarray(a.physics),np.asarray(b.physics))
        for k in ['x','f','aux','action','applied']:
            np.testing.assert_array_equal(a.records[-1][k],b.records[-1][k])
        # Oracle labels must not influence student's input or applied commands.
        a,b=env.clone(),env.clone();b.teacher=lambda *unused:np.array([1.,-1.])
        a.step(m);b.step(m);np.testing.assert_array_equal(a.records[-1]['action'],b.records[-1]['action'])
        np.testing.assert_array_equal(a.records[-1]['applied'],b.records[-1]['applied'])

    def test_failure_cost_and_wrong_exit_penalty(self):
        encoder=Encoder().cuda().eval();env=Scene(('east','W','right'),'normal',2001,encoder=encoder).finish(until=.4)
        base=outcome_cost(env,0)
        env.outcome='severe_departure';failed=outcome_cost(env,0)
        env.outcome='wrong_exit';wrong=outcome_cost(env,0)
        self.assertGreater(failed['total'],base['total']+90)
        self.assertAlmostEqual(wrong['total']-failed['total'],50)


if __name__=='__main__':unittest.main()
