import math
import unittest
import numpy as np
from guard import TianGraph,PoseFilter,pose_from_landmarks,ground_points,Vehicle,Sample,RoadMask,Supervisor,demo_road


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.graph=TianGraph()
        self.supervisor=Supervisor(demo_road(),Vehicle(.16,.22,.001))
        self.stationary=[Sample(0,.9,.6,math.pi/2,0),Sample(.5,.9,.6,math.pi/2,0)]
        self.kw=dict(covariance=np.zeros((3,3)),now=1,last_camera=1,turn='straight',
                     hypotheses=[('00','10')],min_deceleration=.3,latency=.1,
                     tracking_error_bound=0,simulation=True)

    def check_path(self,path=None,**overrides):
        return self.supervisor.evaluate(path or self.stationary,**(self.kw|overrides))

    def test_no_right_at_bottom_t_junction_approaching_east(self):
        self.assertEqual(self.graph.exits('00','10'),{'straight':'20','left':'11'})
        self.assertEqual(self.check_path(turn='right').reason,'forbidden_turn')

    def test_turns_depend_on_arrival_direction(self):
        self.assertEqual(set(self.graph.exits('20','10')),{'straight','right'})
        self.assertEqual(set(self.graph.exits('11','10')),{'left','right'})

    def test_graph_has_twelve_edges(self):
        self.assertEqual(sum(map(len,self.graph.adj.values()))//2,12)

    def test_center_has_three_exits_no_uturn(self):
        self.assertEqual(set(self.graph.exits('10','11')),{'straight','left','right'})

    def test_intersection_of_hypotheses(self):
        self.assertEqual(self.graph.common_turns([('00','10'),('20','10')]),{'straight'})
        self.assertFalse(self.check_path(hypotheses=[('00','10'),('20','10')]).allow)

    def test_unknown_map_stops_live_use(self):
        self.assertEqual(self.check_path(simulation=False).reason,'unmeasured_map')

    def test_centered_narrow_road_can_fit_under_ideal_bounds(self):
        self.assertTrue(self.check_path().allow)

    def test_one_centimeter_lateral_shift_is_rejected(self):
        shifted=[Sample(p.t,p.x+.01,p.y,p.yaw,p.speed) for p in self.stationary]
        self.assertFalse(self.check_path(shifted).allow)

    def test_uncertainty_consumes_clearance(self):
        self.assertFalse(self.check_path(covariance=np.diag([.004**2,.004**2,0])).allow)

    def test_yaw_consumes_clearance(self):
        rotated=[Sample(p.t,p.x,p.y,p.yaw+.15,p.speed) for p in self.stationary]
        self.assertFalse(self.check_path(rotated).allow)

    def test_stale_or_out_of_order_camera_stops(self):
        self.assertFalse(self.check_path(last_camera=.1).allow)
        self.assertFalse(self.check_path(last_camera=2).allow)
        self.assertFalse(self.check_path(last_camera=None).allow)

    def test_stopping_horizon_required(self):
        path=[Sample(0,.9,.6,math.pi/2,.3),Sample(.2,.9,.63,math.pi/2,0)]
        self.assertEqual(self.check_path(path).reason,'insufficient_stopping_horizon')

    def test_swept_path_crossing_island_rejected(self):
        path=[Sample(0,.13,.6,0,.1),Sample(1,.9,.6,0,0)]
        self.assertFalse(self.check_path(path).allow)

    def test_entire_footprint_not_just_corners(self):
        mask=np.ones((200,200),np.uint8);mask[100,100]=0
        road=RoadMask(mask,.005)
        self.assertFalse(road.footprint_fits(Sample(0,.5,.5,0,0),Vehicle(.16,.22,0),0))

    def test_camera_rigid_pose_recovers_known_transform(self):
        a=np.array([[0,0],[.2,0],[0,.2],[.2,.2]])
        angle=.7;c,s=math.cos(angle),math.sin(angle)
        b=a@np.array([[c,s],[-s,c]])+[.8,.6]
        pose,error=pose_from_landmarks(a,b)
        np.testing.assert_allclose(pose,[.8,.6,.7],atol=1e-9)
        self.assertLess(error,1e-9)

    def test_degenerate_landmarks_rejected(self):
        with self.assertRaises(ValueError):
            pose_from_landmarks([[0,0],[1,0],[2,0]],[[0,0],[1,0],[2,0]])

    def test_homography_horizon_rejected(self):
        with self.assertRaises(ValueError):
            ground_points([[1,1]],np.zeros((3,3)))

    def test_filter_camera_gate_and_freshness(self):
        f=PoseFilter([.8,.6,0],np.diag([.02**2,.02**2,.1**2]))
        f.predict(.1,0,.1,1)
        R=np.diag([.005**2,.005**2,.03**2])
        self.assertFalse(f.update_camera([5,5,0],R,1,1))
        self.assertIsNone(f.last_camera_time)
        self.assertTrue(f.update_camera([.811,.6,0],R,1,1))
        self.assertFalse(f.update_camera([.811,.6,0],R,1,1.1))

    def test_nonfinite_inputs_fail_closed(self):
        self.assertFalse(self.check_path(latency=float('nan')).allow)
        self.assertFalse(self.check_path(now=float('nan')).allow)

    def test_camera_candidate_requires_measured_projection(self):
        from vision import yellow_boundary_candidates
        image=np.zeros((50,50,3),np.uint8)
        with self.assertRaises(ValueError):
            yellow_boundary_candidates(image,None,None)

    def test_yellow_candidate_projection(self):
        from vision import yellow_boundary_candidates
        image=np.zeros((50,50,3),np.uint8);image[10:40,20:25]=(0,255,255)
        points,mask=yellow_boundary_candidates(image,np.ones((50,50)),np.diag([.01,.01,1]))
        self.assertGreater(len(points),0)
        self.assertTrue(np.all((points[:,0]>=.2)&(points[:,0]<=.24)))

    def test_wrong_landmark_correspondences_rejected(self):
        with self.assertRaises(ValueError):
            pose_from_landmarks([[0,0],[.2,0],[0,.2]],[[0,0],[.2,0],[1,1]])


if __name__=='__main__':unittest.main()
