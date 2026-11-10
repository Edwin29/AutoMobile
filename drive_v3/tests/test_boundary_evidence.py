import unittest
import copy
import numpy as np
from drive_v3.boundary_evidence import sample_evidence, weighted_huber
from drive_v3.contracts import BoundaryFragment
from drive_v3.settings import load_settings


def line(name,start,end,x=0.):
    y=np.linspace(start,end,round((end-start)/.2)+1)
    return BoundaryFragment(name,np.c_[np.full(len(y),x),y].tolist(),['end','end'],'left',{'length_cm':end-start})


class EvidenceTests(unittest.TestCase):
    def setUp(self):self.cfg=load_settings()[0]['estimator']
    def mass(self,fragments):return float(np.sum(sample_evidence(fragments,self.cfg)[1]**2))

    def test_fragmenting_same_line_does_not_multiply_information(self):
        whole=self.mass([line('a',0,12)])
        split=self.mass([line(str(i),i*2,(i+1)*2) for i in range(6)])
        self.assertLessEqual(split,whole*1.05)
        self.assertGreater(split,whole*.8)

    def test_duplicate_long_and_short_fragment_do_not_add_evidence(self):
        for length in [1.4,10.]:
            a=line('a',0,length);b=copy.deepcopy(a);b.id='b'
            self.assertAlmostEqual(self.mass([a,b]),self.mass([a]),places=9)

    def test_isolated_short_fragment_has_less_influence_than_long_boundary(self):
        self.assertLess(self.mass([line('short',0,1.4)]),self.mass([line('long',0,10)])*.05)

    def test_join_gap_is_not_counted_as_observed_length(self):
        a=line('a',0,6);b=line('b',7,13)
        self.assertAlmostEqual(self.mass([a,b]),self.mass([line('whole',0,12)]),places=9)

    def test_huber_transition_is_independent_of_fragment_weight(self):
        z=np.array([.25,4.]);weights=np.array([.1,1.])
        a=weighted_huber(weights)(z)/(weights*weights)[None,:]
        b=weighted_huber(np.ones(2))(z)
        np.testing.assert_allclose(a,b)
        self.assertAlmostEqual(a[1,1],.5)


if __name__=='__main__':unittest.main()
