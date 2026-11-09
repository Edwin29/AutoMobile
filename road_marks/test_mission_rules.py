import unittest
from mission_rules import Approach, braking_gate
from observations import Observations


class Rules(unittest.TestCase):
    def test_conservative_braking_gate(self):
        # Illustrative values only: 2 cm latency + 2.5 cm braking + 1 cm margin.
        self.assertTrue(braking_gate(5.5,10,.2,20,1))
        self.assertFalse(braking_gate(6,10,.2,20,1))
        self.assertTrue(braking_gate(None,10,.2,20,1))

    def test_outer_crossing_and_red_are_ignored(self):
        for light in ("red","green",None):
            self.assertEqual(Approach("outer",1).update(0,right_light=light,crossing_ahead=True,
                             at_turn_entry=True,at_braking_gate=True),("FOLLOW",None))

    def test_center_waits_for_actual_stop_and_then_turns_once(self):
        a=Approach("center",1)
        self.assertEqual(a.update(0,right_light="red",crossing_ahead=True)[0],"APPROACH_SLOW")
        self.assertEqual(a.update(10,at_braking_gate=True)[0],"HOLD_BEFORE_CROSSING")
        stop=dict(stopped_before_crossing=True,crossing_ahead=True)
        self.assertIsNone(a.update(11,**stop)[1])
        self.assertIsNone(a.update(11.9,**stop)[1])
        self.assertEqual(a.update(12,**stop)[1],"select_right")
        self.assertIsNone(a.update(13,**stop)[1])

    def test_green_far_away_does_not_release_red_at_entry(self):
        a=Approach("center",1)
        self.assertIsNone(a.update(0,crossing_ahead=True,right_light="green")[1])
        self.assertEqual(a.update(2,crossing_ahead=True,right_light="red",at_braking_gate=True)[0],
                         "HOLD_BEFORE_CROSSING")
        b=Approach("center",1)
        self.assertEqual(b.update(2,crossing_ahead=True,right_light="green",at_turn_entry=True)[1],"select_right")
        self.assertEqual(Approach("center",1).update(3,crossing_ahead=True,at_braking_gate=True)[0],
                         "HOLD_BEFORE_CROSSING")

    def test_sign_deadline_and_acknowledgment(self):
        for sign in ("horn","stop","slow"):
            a=Approach("signed",1)
            self.assertEqual(a.update(0,right_sign=sign)[1],sign)
            self.assertEqual(a.update(1,right_sign=sign,at_braking_gate=True),("HOLD_BEFORE_PINK",None))
            self.assertEqual(a.update(2,action_done=sign)[0],"FOLLOW_MISSION_SPEED")
            self.assertEqual(a.update(3,passed_pink=True)[0],"FOLLOW_MISSION_SPEED")
        a=Approach("signed",1)
        self.assertEqual(a.update(0,at_braking_gate=True)[0],"HOLD_BEFORE_PINK")
        self.assertEqual(a.update(1,passed_pink=True,right_sign="horn"),("HOLD_MISSED_DEADLINE",None))

    def test_start_is_one_time(self):
        a=Approach("start",1)
        self.assertEqual(a.update(0)[0],"WAIT_PLACEMENT")
        self.assertEqual(a.update(1,placed_at_start=True)[1],"init_pose_at_start")
        self.assertIsNone(a.update(2,placed_at_start=True)[1])

    def test_right_side_and_temporal_gate(self):
        o=Observations()
        # Boxes from the real 04 segment photo image_20261014_203342_562230.
        # Repeated coordinates here test the logic only, not accuracy over time.
        detections=[dict(kind="stop",box=[165,234,191,278]),dict(kind="slow",box=[586,184,640,279]),
                    dict(kind="dashline",box=[189,428,477,462])]
        args=(detections,(480,640),[(100,380),(500,380),(550,479),(75,479)],
              [(540,100),(639,100),(639,330),(540,330)])
        self.assertEqual(o.update(0,*args),{})
        self.assertEqual(o.update(.1,*args),dict(right_sign="slow",pink="dashline"))
        self.assertEqual(o.update(.8,*args),{})


if __name__=="__main__":
    unittest.main()
