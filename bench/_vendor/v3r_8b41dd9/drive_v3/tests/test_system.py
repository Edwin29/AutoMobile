"""Behavioral contracts across geometry, timing, estimation, missions and control."""
import copy
import unittest
from unittest.mock import patch
from dataclasses import replace
import cv2
import numpy as np

from drive_v3.settings import load_settings
from drive_v3.geometry import advance,to_body,to_world,footprint
from drive_v3.map_model import TrackMap,Route
from drive_v3.estimation import Estimator,TimedEstimator
from drive_v3.contracts import ObservationFrame,FrameStamp,BoundaryFragment,LandmarkCandidate,PathDecision
from drive_v3.control import ResponseModel,Controller
from drive_v3.features import Detection,DetectionTracker,group_marks
from drive_v3.mission import MissionManager,MissionContext
from drive_v3.runtime import Runtime,RecordingSink
from drive_v3.visual_motion import MotionObservation,FloorVO
from drive_v3.actuation import WatchdogSink
from drive_v3.context import Target,TargetTracker


def observation(pose,t=0.,sequence=0):
    # Independent straight stretch with only longitudinally invariant walls.
    fragments=[]
    for i,x in enumerate((-99.25,-73.25)):
        world=np.c_[np.full(20,x),np.linspace(-45,-25,20)]
        fragments.append(BoundaryFragment(str(i),to_body(world,pose).tolist(),["end","end"],"unknown",{"length_cm":20.}))
    return ObservationFrame(FrameStamp(sequence,t,"test"),"test","test",fragments,[],[],"observed",{})


class GeometryEstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.track=TrackMap();cls.cfg=load_settings()[0]

    def test_transform_roundtrip_and_right_turn(self):
        p=np.array([[1.,3.],[-5.,8.]])
        np.testing.assert_allclose(to_body(to_world(p,[4,8,.7]),[4,8,.7]),p)
        q=advance([0,0,0],5,.5,1)
        self.assertGreater(q[0],0);self.assertGreater(q[1],0)

    def test_fast_clearance_never_overstates_exact(self):
        p=np.random.default_rng(19).uniform(-110,110,(5000,2))
        exact=self.track.clearance(p);approx=self.track.clearance_fast(p)
        self.assertTrue(np.all(approx<=exact+1e-8))
        self.assertLess(np.max(exact-approx),.5)

    def test_true_footprint_not_point_only(self):
        center=[-90,-35,0]
        self.assertGreater(self.track.clearance(np.array(center[:2])),0)
        self.assertLess(self.track.clearance(to_world(footprint(self.cfg["vehicle"]),center)).min(),2)

    def test_straight_does_not_invent_longitudinal_information(self):
        pose=np.array([-86.25,-60,0.])
        est=Estimator(self.track,pose,self.cfg["estimator"])
        before=est.primary.P[1,1]
        self.assertTrue(est.observe(observation(pose)))
        self.assertAlmostEqual(before,est.primary.P[1,1],places=6)
        self.assertLess(est.primary.P[0,0],1.)

    def test_map_predictions_cannot_be_measurements(self):
        est=Estimator(self.track,[-86.25,-60,0.],self.cfg["estimator"])
        obs=observation(est.primary.mean)
        for f in obs.fragments:f.source="map_predicted"
        self.assertFalse(est.observe(obs))

    def test_uncertainty_grows_without_observation(self):
        est=Estimator(self.track,[-86.25,-60,0.],self.cfg["estimator"])
        before=est.primary.P.copy();est.predict(5,0,1.)
        self.assertTrue(np.all(np.diag(est.primary.P)>np.diag(before)))

    def test_delayed_observation_matches_chronological_replay(self):
        pose=[-86.25,-60,0.]
        a=TimedEstimator(Estimator(self.track,pose,self.cfg["estimator"]),0.,"test")
        b=TimedEstimator(Estimator(self.track,pose,self.cfg["estimator"]),0.,"test")
        for timeline in (a,b):timeline.command(0.,5.,0.)
        obs=observation(pose,.2,7)
        a.step(.2,obs);a.step(.4)
        b.step(.4,obs)
        np.testing.assert_allclose(a.estimator.primary.mean,b.estimator.primary.mean,atol=1e-10)
        np.testing.assert_allclose(a.estimator.primary.P,b.estimator.primary.P,atol=1e-10)

    def test_duplicate_frame_not_fused_twice(self):
        t=TimedEstimator(Estimator(self.track,[-86.25,-60,0.],self.cfg["estimator"]),0.,"test")
        obs=observation(t.estimator.primary.mean,.2,4)
        t.step(.2,obs);p=t.estimator.primary.P.copy();t.step(.2,obs)
        np.testing.assert_allclose(t.estimator.primary.P,p)
        t.step(.2,replace(obs,stamp=FrameStamp(99,.2,"test")))
        np.testing.assert_allclose(t.estimator.primary.P,p)
        self.assertRaises(ValueError,t.step,.3,replace(obs,stamp=FrameStamp(5,.3,"wrong")))

    def test_visual_delta_replaces_wrong_pwm_distance(self):
        est=Estimator(self.track,[0,0,0],self.cfg["estimator"])
        t=TimedEstimator(est,0.,"test");t.command(0.,10.,0.)
        motion=MotionObservation(0,.2,np.array([0.,1.]),0.,20,.1,"moving")
        t.step(.2,motion=motion)
        np.testing.assert_allclose(est.primary.mean,[0.,1.,0.])
        t.step(.3)
        np.testing.assert_allclose(est.primary.mean,[0.,2.,0.])

    def test_junction_entry_exit_and_asymmetry(self):
        a=self.track.junction_routes("center","N",self.cfg["vehicle"])
        b=self.track.junction_routes("center","E",self.cfg["vehicle"])
        self.assertEqual(set(a),{"left","right","straight"})
        self.assertEqual(a["right"].exit,"W")
        self.assertNotAlmostEqual(a["right"].clearance,b["right"].clearance,places=2)

    def test_blue_landmark_can_update_when_edges_missing(self):
        pose=np.array([-63.25,-80.,0.]);est=Estimator(self.track,pose,self.cfg["estimator"])
        obs=observation(pose);obs.fragments=[]
        obs.landmarks=[LandmarkCandidate("blue","blue_circle_candidate",[0,15.75],[],{})]
        self.assertTrue(est.observe(obs))
        self.assertEqual(est.primary.associations["blue"],"blue_00")

    def test_partial_transverse_line_does_not_fake_lateral_center(self):
        pose=np.array([-86.25,-90.,0.]);est=Estimator(self.track,pose,self.cfg["estimator"])
        obs=observation(pose)
        obs.landmarks=[LandmarkCandidate("pink","pink_line_candidate",[5,25],[],{},["localization_only"])]
        self.assertTrue(est.observe(obs))
        self.assertLess(abs(est.primary.mean[0]-pose[0]),.05)


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.cfg=load_settings()[0];self.model=ResponseModel(self.cfg["controller"],self.cfg["vehicle"])
        self.controller=Controller(self.model,self.cfg["controller"])

    def path(self,x=0):
        return PathDecision([[x,0],[x,5],[x,10],[x,20]],5.,1.,"test","tracking")

    def test_response_inverse_is_consistent(self):
        for v,w in ((4,0),(4,.15),(4,-.15)):
            np.testing.assert_allclose(self.model.forward_model(self.model.inverse(v,w)),[v,w])

    def test_path_to_right_requests_more_left_output(self):
        self.controller.speed=4;self.controller.pwm[:]=.25
        result=self.controller.step(self.path(1),.1,.1,"test",True)
        self.assertGreater(result.pwm[0],result.pwm[1])

    def test_expired_path_and_bad_clock_zero_output(self):
        for now,clock in ((2.,"test"),(.1,"wrong")):
            result=self.controller.step(self.path(),now,.1,clock,True)
            self.assertEqual(list(result.pwm),[0,0])

    def test_trim_frozen_when_observation_bad(self):
        self.controller.speed=4;self.controller.pwm[:]=.25
        for i in range(3):self.controller.step(self.path(1),.1*(i+1),.1,"test",False)
        self.assertEqual(self.controller.trim,0)

    def test_trim_uses_reference_error_even_when_local_path_starts_at_origin(self):
        self.controller.speed=5.;self.controller.pwm[:]=.3125
        path=self.path();path.reference_lateral_error_cm=1.
        path.reference_heading_error_rad=0.;path.reference_curvature_cm_inv=0.
        self.controller.step(path,.1,.1,"test",True)
        self.assertLess(self.controller.trim,0.)

    def test_nonfinite_path_limit_never_reaches_actuation(self):
        path=self.path();path.speed_limit_cm_s=float("nan")
        self.assertEqual(self.controller.step(path,.1,.1,"test",True).reason,"invalid_path_limits")

    def test_adaptation_requires_trust_and_is_bounded(self):
        self.cfg["controller"]["adaptive"]=True
        motion=MotionObservation(0,.2,np.array([0,.8]),0,20,.1,"moving")
        self.assertFalse(self.model.adapt([.3,.3],motion,False))
        self.assertTrue(self.model.adapt([.3,.3],motion,True))
        self.assertGreaterEqual(self.model.forward,.5);self.assertLessEqual(self.model.forward,1.5)
        motion.translation_body_cm[1]=100
        self.assertFalse(self.model.adapt([.3,.3],motion,True))


class MissionTests(unittest.TestCase):
    def setUp(self):
        self.cfg=copy.deepcopy(load_settings()[0]["mission"])
        self.cfg.update(rule_values_verified=True,stop_hold_s=1.,horn_duration_s=.5,slow_speed_cm_s=2.,slow_release="passed",signal_release="green")
        self.m=MissionManager(self.cfg)

    def confirm(self,kind,ctx):
        for i in range(3):self.m.observe([Detection(kind,[20,20,40,40],.9,i*.1,100+i)],ctx,i*.1)

    def test_tracker_reidentification_does_not_repeat_horn(self):
        c=MissionContext("visit1","pink0","pink",.5,approach_confirmed=True)
        self.confirm("horn",c)
        self.assertTrue(self.m.update(.2,4,c).horn)
        self.assertFalse(self.m.update(.8,4,c).horn)
        self.m.observe([Detection("horn",[20,20,40,40],.9,1.,999)],c,1.)
        self.assertFalse(self.m.update(1.,4,c).horn)
        self.assertFalse(self.m.update(1.1,4,c).hold)

    def test_stop_time_starts_after_stationary_evidence(self):
        c=MissionContext("v","p","pink",.5,approach_confirmed=True)
        self.confirm("stop",c)
        self.assertTrue(self.m.update(.2,4,c).hold)
        self.assertTrue(self.m.update(1.,0,c).hold)
        self.assertTrue(self.m.update(1.9,0,c).hold)
        self.assertFalse(self.m.update(2.1,0,c).hold)

    def test_unverified_rule_does_not_invent_duration(self):
        self.cfg["rule_values_verified"]=False
        c=MissionContext("v","p","pink",.5,approach_confirmed=True);self.confirm("stop",c)
        self.assertEqual(self.m.update(.2,0,c).reason,"rules_required")

    def test_missing_trigger_near_deadline_does_not_allow_blind_pass(self):
        c=MissionContext("v","p","pink",lower_distance_bound_cm=.1)
        self.assertEqual(self.m.update(.2,4,c).reason,"trigger_unconfirmed_near_deadline")

    def test_zero_command_alone_is_not_stationary_confirmation(self):
        c=MissionContext("v","p","pink",.5,approach_confirmed=True);self.confirm("stop",c)
        self.m.update(.2,0,c,stationary=False)
        self.assertTrue(self.m.update(5.,0,c,stationary=False).hold)

    def test_capture_delay_advances_mission_deadline(self):
        c=MissionContext("v","p","pink",3.,approach_confirmed=True);self.confirm("horn",c)
        self.assertFalse(self.m.update(.3,4,c).horn)
        self.assertTrue(self.m.update(.4,4,c,observation_age_s=.3).horn)

    def test_red_central_holds_outer_does_not(self):
        c=MissionContext("v","cross0","center_crosswalk",.5,approach_confirmed=True)
        self.confirm("signal_red",c)
        self.assertTrue(self.m.update(.2,4,c).hold)
        self.assertFalse(self.m.update(.3,4,replace(c,kind="outer_crosswalk")).hold)

    def test_direction_conflict_and_illegal_exit_hold(self):
        c=MissionContext("v","west:S","junction",possible_directions=("straight","left"),approach_confirmed=True)
        self.confirm("right",c)
        self.assertEqual(self.m.update(.2,4,c).reason,"direction_not_available")
        self.confirm("left",c)
        self.assertEqual(self.m.update(.3,4,c).reason,"direction_conflict")

    def test_pose_correction_is_not_visit_completion(self):
        c=MissionContext("v","center:N","junction",possible_directions=("right",),approach_confirmed=True)
        self.confirm("right",c);self.m.update(.2,4,c)
        self.assertFalse(self.m.complete_visit(c))
        self.assertTrue(self.m.complete_visit(replace(c,passed_confirmed=True,exit_aligned=True)))

    def test_single_detection_never_confirms(self):
        c=MissionContext("v","p","pink",.5,approach_confirmed=True)
        self.m.observe([Detection("stop",[0,0,10,10],.99,0)],c,0)
        self.assertEqual(self.m.update(.1,4,c).reason,"missing_sign")


class FeatureRuntimeTests(unittest.TestCase):
    def test_one_pixel_pink_dashes_survive_zero_polygon_area(self):
        from drive_v3.observation import patch_candidates,load_config
        from drive_v3.map_model import HERE
        class Geometry:
            ppcm=5.
            def pixels_to_body(self,p):return np.asarray(p)/5
        mask=np.zeros((100,100),bool)
        for x in range(15,75,10):mask[50,x:x+5]=True
        masks={name:np.zeros_like(mask) for name in ("blue","pink","red","white")};masks["pink"]=mask
        cfg,_=load_config(HERE/"config/observation.json")
        patches=patch_candidates(masks,np.ones_like(mask),Geometry(),cfg)
        grouped=group_marks(patches,load_settings()[0]["marks"])
        self.assertTrue(any(m.kind=="dashline" for m in grouped))
    def test_delayed_pixels_not_used_as_current_body_constraints(self):
        with Runtime([-86.25,-60,0.]) as rt:
            with patch.object(rt.planner,"plan",wraps=rt.planner.plan) as planner:
                record,*_=rt.step(np.zeros((480,640,3),np.uint8),.1,.2)
                self.assertIsNone(planner.call_args.args[1])
                self.assertFalse(record["current_frame_constraints_used"])
    def test_watchdog_expiry_zeros_all_wheels(self):
        state={};now=[0.]
        wheels={k:(lambda duty,k=k:state.__setitem__(k,duty)) for k in ("front_left","rear_left","front_right","rear_right")}
        sink=WatchdogSink(wheels,timeout_s=.3,clock=lambda:now[0])
        try:
            sink.write(.3,.4);self.assertEqual(state["rear_right"],.4)
            now[0]=.4;self.assertTrue(sink.check_expiry());self.assertTrue(all(v==0 for v in state.values()))
        finally:sink.close()

    def test_floor_vo_translation_sign(self):
        class Geometry:
            def pixels_to_body(self,p):return np.c_[np.asarray(p)[:,0]/5,20-np.asarray(p)[:,1]/5]
        cfg=load_settings()[0]["estimator"];vo=FloorVO(Geometry(),cfg)
        rng=np.random.default_rng(8);im=np.zeros((100,100,3),np.uint8)
        for x,y in rng.integers(10,85,(80,2)):cv2.circle(im,(int(x),int(y)),2,(200,200,200),-1)
        valid=np.ones((100,100),bool);empty=np.zeros_like(valid)
        a=dict(bev=im,valid=valid,road=valid,glare_suspect=empty,white=empty,pink=empty,red=empty)
        b=dict(a,bev=cv2.warpAffine(im,np.array([[1.,0,0],[0,1.,5]]),(100,100)))
        vo.update(a,0.);motion=vo.update(b,.2)
        self.assertIsNotNone(motion)
        np.testing.assert_allclose(motion.translation_body_cm,[0,1],atol=.06)

    def test_marker_visit_cannot_advance_on_map_pose_jump(self):
        cfg=load_settings()[0];tracker=TargetTracker(cfg["vehicle"],cfg["mission"])
        tracker.arm(Target("v1","pink0","pink","dashline_00"))
        estimator=Estimator(TrackMap(),[0,0,0],cfg["estimator"])
        estimator.primary.associations={"m":"dashline_00"}
        obs=observation([0,0,0]);obs.landmarks=[LandmarkCandidate("m","dashline",[0,20],[],{})]
        for i,t in enumerate((0.,.1,.2)):
            obs.stamp=FrameStamp(i,t,"test");ctx=tracker.update(obs,estimator,None,t)
        self.assertTrue(ctx.approach_confirmed)
        estimator.primary.mean[1]+=100
        obs.landmarks=[];ctx=tracker.update(obs,estimator,None,.3)
        self.assertFalse(ctx.passed_confirmed)
        motion=MotionObservation(.3,.4,np.array([0.,8.]),0,20,.1,"moving")
        self.assertTrue(tracker.update(obs,estimator,motion,.4).passed_confirmed)

    def test_stripes_group_but_one_white_patch_does_not(self):
        cfg=load_settings()[0]["marks"]
        patches=[]
        for i in range(3):
            p=np.array([[-3,-.3],[3,-.3],[3,.3],[-3,.3]])+[0,i*2]
            patches.append(LandmarkCandidate(str(i),"white_patch",p.mean(0).tolist(),p.tolist(),{}))
        self.assertEqual(group_marks(patches,cfg)[0].kind,"crosswalk")
        self.assertEqual(group_marks(patches[:1],cfg),[])

    def test_exception_stops_injected_sink(self):
        sink=RecordingSink();r=Runtime([-86.25,-60,0.],sink=sink)
        sink.write(.3,.3)
        with self.assertRaises(ValueError):r.step(np.zeros((10,10,3),np.uint8),.1,.1)
        self.assertTrue(all(v==0 for v in sink.last.values()))

    def test_tracker_preserves_track_when_class_changes(self):
        tracker=DetectionTracker(load_settings()[0]["mission"])
        a=tracker.update([Detection("left",[0,0,20,20],.9,0)],0)[0]
        b=tracker.update([Detection("right",[1,1,21,21],.9,.1)],.1)[0]
        self.assertEqual(a.track_id,b.track_id)


if __name__=="__main__":unittest.main()
