import json
import unittest
from collections import deque
from math import pi

from .controllers import HybridMission
from .controllers_mission import _load_policy
from .pi_run import parse_command
from .vision_rules import Rules
from .world import Track


class ApproachArrowTests(unittest.TestCase):
    def setUp(self):
        self.ctrl = HybridMission.__new__(HybridMission)
        jp = _load_policy()
        self.ctrl.junctions, self.ctrl.ARMS = Track().data["junctions"], jp.ARMS
        self.ctrl.policy = jp.JunctionPolicy(self.ctrl.junctions, "ccw")
        self.ctrl.active, self.ctrl.visit = False, None
        self.ctrl.arrow_frames, self.ctrl.arrow_votes = {}, deque(maxlen=128)
        self.rules = Rules()

    def vote(self, index, move="left"):
        t = 0.12 * index
        self.ctrl.remember_arrow_frame(index, t, [-87, 48, pi], 0.5)
        cmd, = self.rules.update(t, [dict(name=move, score=.9, box=[60, 20, 50, 50])], frame=index)
        parsed = parse_command(json.dumps(cmd).encode())
        return self.ctrl.accept_arrow_vote(t + .05, *parsed["arg"])

    def decide(self, target):
        arrow = self.ctrl.take_arrow(2, target)
        return self.ctrl.policy.decide(2, *target, arrow=arrow)

    def test_turn_is_taken_at_this_approach_and_never_carried_forward(self):
        for i in range(5):
            self.assertEqual(self.vote(i), ("west", "N"))
        self.ctrl.remember_arrow_frame(8, 1, [-87, 36, pi], 0.5)
        self.assertEqual(self.decide(("west", "N")).move, "left")
        self.assertIsNone(self.ctrl.accept_arrow_vote(1.4, 8, "left"))
        self.assertEqual(self.decide(("south", "W")).move, "straight")

    def test_wrong_target_illegal_exit_and_ambiguous_votes(self):
        for i in range(4):
            self.vote(i)
        self.assertEqual(self.decide(("north", "E")).move, "straight")
        for i in range(4):
            self.vote(i, "right")
        self.assertEqual(self.decide(("west", "N")).move, "straight")
        for i, move in enumerate(["right", "right", "left", "left"]):
            self.vote(i, move)
        self.assertEqual(self.decide(("west", "N")).move, "straight")

    def test_late_frame_and_frame_after_commit(self):
        self.ctrl.remember_arrow_frame(1, 1, [-87, 48, pi], 0.5)
        self.assertIsNone(self.ctrl.accept_arrow_vote(3, 1, "left"))
        self.ctrl.visit = ("west", "N")
        self.assertIsNone(self.vote(2))
