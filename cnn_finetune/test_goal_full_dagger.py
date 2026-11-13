import unittest
import numpy as np
from .goal_full_dagger import Labeler,QueryScene,supervision_mask
from .goal_teacher_probe import FastTeacher,BASE
from .goal_exit_reference_probe import centered_reference
from .goal_teacher_validation import SELECTED,legal_start
from .junction_learning import Encoder,setup


class FullDaggerChecks(unittest.TestCase):
    def test_rendered_labeler_and_fast_teacher_agree(self):
        setup();enc=Encoder().cuda().eval()
        a=QueryScene(('east','W','right'),'mild',6101,'sluggish',enc)
        b=FastTeacher(('east','W','right'),'mild',6101,'sluggish');legal_start(b)
        b.points,b.h,b.s,b.k,_=centered_reference(b,SELECTED['end_cm']);b.controls={k:SELECTED[k] for k in BASE}
        a.finish(until=2);b.finish(until=2);np.testing.assert_array_equal(a.physics,b.physics)
        physics=np.asarray(a.physics);labeler=Labeler(a.key,a.preset);pred=[]
        captures=np.array([r['capture'] for r in a.records]);ids=np.searchsorted(physics[:,0],captures)
        for i in ids:pred.append(labeler.teacher(physics[i,1:4],physics[i,4],physics[i,5]))
        np.testing.assert_allclose(pred,[r['label'] for r in a.records],atol=3e-8,rtol=0)
        mask=supervision_mask(a.key,a.preset,physics,captures);self.assertEqual(len(mask),len(a.records));self.assertTrue(mask.all())


if __name__=='__main__':unittest.main()
