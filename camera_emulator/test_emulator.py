import unittest
import cv2
import numpy as np
from emulator import Emulator


class Tests(unittest.TestCase):
    def test_inverse_distortion_mapping(self):
        e=Emulator();f=e.virtual_K[0,0]
        uv=np.stack([e.mx[::40,::40],e.my[::40,::40]],-1).astype(float)
        rays=(uv-e.virtual_K[:2,2])/f
        back=cv2.fisheye.distortPoints(rays.reshape(-1,1,2),e.K,e.D).reshape(uv.shape)
        y,x=np.mgrid[0:480:40,0:640:40]
        self.assertLess(np.max(np.abs(back-np.stack([x,y],-1))),.001)

    def test_crop_and_passthrough(self):
        e=Emulator();frame=np.full((720,1280,3),137,np.uint8)
        base,out,crop=e.apply(frame,blur=0,noise=0,quality=100,distort=False)
        self.assertEqual(crop,[160,0,960,720])
        self.assertEqual(out.shape,(480,640,3))
        np.testing.assert_array_equal(base,out)

    def test_repeatable_noise(self):
        frame=np.full((480,640,3),128,np.uint8)
        a=Emulator(seed=2);b=Emulator(seed=2)
        np.testing.assert_array_equal(a.apply(frame)[1],b.apply(frame)[1])
        self.assertFalse(np.array_equal(a.apply(frame)[1],b.apply(frame,noise=0)[1]))


if __name__=='__main__':unittest.main()
