import unittest
import numpy as np
from drive_v3.map_model import TrackMap
from drive_v3.settings import load_settings
from drive_v3.start_region import right_red_prior


class StartRegionTests(unittest.TestCase):
    def test_heading_changes_axle_offset_not_body_region(self):
        t=TrackMap();v=load_settings()[0]['vehicle']
        north,sigma,bounds=right_red_prior(t,v,'ccw')
        south,_,other=right_red_prior(t,v,'cw')
        np.testing.assert_allclose(north,[85.25,-50.5,0])
        np.testing.assert_allclose(south,[85.25,-46.,np.pi])
        self.assertEqual(bounds['body_midpoint_y_cm'],[-50.75,-45.75])
        self.assertEqual(bounds['body_midpoint_y_cm'],other['body_midpoint_y_cm'])
        self.assertGreater(sigma[1],1.)

    def test_supplied_marker_layout(self):
        t=TrackMap();pink=np.asarray(t.data['pink_lines']);red=np.asarray(t.data['red_lines'])
        self.assertFalse(np.any(np.all(np.isclose(pink.mean(axis=1),[85.25,-65]),axis=1)))
        self.assertTrue(np.any(np.all(np.isclose(pink.mean(axis=1),[66.25,-87.25]),axis=1)))
        np.testing.assert_allclose(65-red.mean(axis=1)[:,1],[98.5,128.])


if __name__=='__main__':unittest.main()
