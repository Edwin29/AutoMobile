import unittest
from orchestrator import Coordinator, Pink, Junction, Sign


def new():
    c = Coordinator(stop_s=1, red_s=2, slow_cap=.5, observation_age_s=.5, max_tick_gap_s=1)
    c.step(0, start_ready=True)
    return c


def tee(**kwargs):
    return Junction("lap1/T", False, frozenset({"LEFT", "STRAIGHT"}), **kwargs)


def center(**kwargs):
    return Junction("lap1/C", True, frozenset({"LEFT","RIGHT","STRAIGHT"}), **kwargs)


class IntegrationCases(unittest.TestCase):
    def test_interleaved_inbox_and_two_distinct_executors(self):
        c = new()
        left = Sign("lap1/T","LEFT",.1)
        horn = Sign("lap1/P","HORN",.1)
        self.assertIsNone(c.step(.1,signs=[left,horn,left]).horn)
        d = c.step(.2,pink=Pink("lap1/P",trigger=True))
        self.assertEqual((d.horn,d.turn),("lap1/P/HORN",None))
        d = c.step(.3,pink=Pink("lap1/P",brake_gate=True),junction=tee(at_entry=True))
        self.assertEqual((d.motion,d.horn),("HOLD","lap1/P/HORN"))
        d = c.step(.4,pink=Pink("lap1/P"),junction=tee(at_entry=True),completed={"lap1/P/HORN"})
        self.assertEqual((d.turn,d.horn),(("lap1/T","LEFT"),None))
        self.assertEqual(c.step(.5).turn,d.turn)  # camera loses junction while turning
        self.assertEqual(c.step(.6,completed={"lap1/T/TURN"},junction=tee(at_entry=True)).motion,"FOLLOW")

    def test_center_stop_overrides_other_direction_and_green_not_required_after_dwell(self):
        c = new()
        c.step(.1,signs=[Sign("lap1/C","LEFT",.1)],junction=center(light="RED"))
        self.assertEqual(c.step(.2,junction=center()).motion,"FOLLOW")
        self.assertEqual(c.step(.3,junction=center(cross_trigger=True,at_entry=True)).motion,"HOLD")
        for t in (.4,1.4):
            self.assertEqual(c.step(t,junction=center(stopped_before=True,at_entry=True)).motion,"HOLD")
        self.assertEqual(c.step(2.4,junction=center(stopped_before=True,at_entry=True)).turn,("lap1/C","RIGHT"))

    def test_center_and_pink_must_both_clear(self):
        c = new()
        c.step(.1,signs=[Sign("P","STOP",.1)],pink=Pink("P",trigger=True),
               junction=center(cross_trigger=True,light="RED"))
        for t in (.2,1.2):
            d = c.step(t,pink=Pink("P",stopped_before=True),junction=center(stopped_before=True,at_entry=True))
            self.assertEqual(d.motion,"HOLD")
        self.assertEqual(c.step(2.2,junction=center(stopped_before=True,at_entry=True)).turn,("lap1/C","RIGHT"))

    def test_unknown_signal_and_missing_crosswalk(self):
        c = new()
        self.assertEqual(c.step(.1,junction=center(cross_brake_gate=True)).motion,"HOLD")
        self.assertEqual(c.step(.2,junction=center(light="RED",at_entry=True)).motion,"HOLD")
        self.assertEqual(c.step(.3).motion,"HOLD")  # unresolved center cannot disappear
        c = new()
        self.assertEqual(c.step(.1,junction=center(light="GREEN",at_entry=True)).turn,("lap1/C","RIGHT"))

    def test_outer_crosswalk_is_observation_only(self):
        c = new()
        self.assertEqual(c.step(.1,junction=tee(light="RED",cross_trigger=True)).motion,"FOLLOW")
        self.assertEqual(c.step(.2,junction=tee(at_entry=True)).motion,"HOLD")

    def test_unavailable_direction_stale_and_conflict(self):
        c = new()
        self.assertEqual(c.step(.1,signs=[Sign("lap1/T","RIGHT",.1)],junction=tee(at_entry=True)).motion,"HOLD")
        c = new()
        self.assertEqual(c.step(.8,signs=[Sign("lap1/T","LEFT",.1)],junction=tee(at_entry=True)).motion,"HOLD")
        self.assertEqual(c.step(.9,signs=[Sign("lap1/T","LEFT",.9),Sign("lap1/T","STRAIGHT",.9)],
                                junction=tee(at_entry=True)).motion,"HOLD")

    def test_slow_precedes_line_and_persists_to_explicit_end(self):
        c = new()
        c.step(.1,signs=[Sign("P","SLOW",.1)],pink=Pink("P"))
        self.assertEqual(c.step(.2,pink=Pink("P",trigger=True)).speed_cap,.5)
        self.assertEqual(c.step(.3,pink=Pink("P",passed=True),completed={"P/SLOW"}).speed_cap,.5)
        self.assertEqual(c.step(.4,slow_zone_finished={"P"}).speed_cap,1)

    def test_missing_marker_holds_at_last_safe_point_and_miss_latches(self):
        c = new()
        self.assertEqual(c.step(.1,pink=Pink("P",brake_gate=True)).motion,"HOLD")
        self.assertEqual(c.step(.2,pink=Pink("P")).motion,"HOLD")
        self.assertEqual(c.step(.3,pink=Pink("P",passed=True)).reason,"missed_pink_deadline:P")
        self.assertEqual(c.step(.4).motion,"HOLD")

    def test_unhealthy_and_stale_tick_cut_output_and_reset_stop_timer(self):
        c = new()
        c.step(.1,signs=[Sign("P","STOP",.1)],pink=Pink("P",trigger=True,stopped_before=True))
        self.assertEqual(c.step(.2,healthy=False).motion,"HOLD")
        self.assertEqual(c.step(2,pink=Pink("P",stopped_before=True)).motion,"HOLD")
        self.assertEqual(c.step(2.1,pink=Pink("P",stopped_before=True)).motion,"HOLD")
        self.assertEqual(c.step(3.1,pink=Pink("P",stopped_before=True)).motion,"FOLLOW")

    def test_start_and_future_signs(self):
        c = Coordinator(stop_s=1, red_s=2, slow_cap=.5, observation_age_s=.5, max_tick_gap_s=1)
        self.assertEqual(c.step(0).motion,"HOLD")
        self.assertEqual(c.step(.1,start_ready=True,signs=[Sign("lap1/T","LEFT",1)],
                                junction=tee(at_entry=True)).motion,"HOLD")


if __name__ == "__main__":
    unittest.main()
