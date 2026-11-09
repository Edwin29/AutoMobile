"""Exercise association, direction separation and freeze behavior on synthetic printed crosswalks."""
import unittest
import cv2
import numpy as np
from controller import HERE
from yaw_gain import GainTracker, PaintDetector

class GainTests(unittest.TestCase):
    def setUp(self):
        with np.load(HERE/"track.npz") as assets:
            self.det=PaintDetector(assets)
        yy,xx=np.indices((180,300),dtype=np.float32)
        self.det.maps=[np.dstack([xx,yy])]*2

    def crossing(self,yaw,glare=False):
        im=np.full((180,300,3),55,np.uint8)
        rad=np.radians(yaw)
        R=np.array([[np.cos(rad),-np.sin(rad)],[np.sin(rad),np.cos(rad)]])
        for x in np.arange(6)*3.15-7.875:
            rect=np.array([[x-1,-6],[x+1,-6],[x+1,6],[x-1,6]])+[0,36.36]
            xy=(rect@R.T-[0,12.36])*[5,-5]+[150,210]
            cv2.fillConvexPoly(im,np.rint(xy).astype(np.int32),(225,225,225))
        if glare:im[:]=255
        return self.det.detect(im)

    def test_turns(self):
        for sign in [-1,1]:
            tracker=GainTracker()
            tracker.update(self.crossing(0),0,.1,0.)
            for step in range(1,7):
                marks=self.crossing(sign*2.4*step)
                self.assertEqual([m.kind for m in marks],["cross"])
                tracker.update(marks,sign*4.,.2,0.)
            self.assertLess(tracker.gain(sign),13/18 if sign<0 else 1.)
            self.assertEqual(tracker.gain(-sign),1. if sign<0 else 13/18)

    def test_abstains(self):
        tracker=GainTracker()
        for _ in range(24):
            tracker.update(self.crossing(0),0,.1,0.)
        self.assertEqual(tracker.left,13/18)
        self.assertEqual(tracker.right,1.)
        tracker.update(self.crossing(0,True),-3,.1,0.)
        self.assertIsNone(tracker.previous)
        for _ in range(20):
            tracker.update(self.crossing(0),-4,.2,0.)
        self.assertEqual(tracker.left,13/18)

if __name__=="__main__":unittest.main()

