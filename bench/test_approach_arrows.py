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

    def test_only_branch_resolves_turn_votes_but_not_turn_vs_straight(self):
        for i in range(4):
            self.vote(i)
        self.assertEqual(self.decide(("north", "E")).move, "straight")
        for i in range(4):
            self.vote(i, "right")
        self.assertEqual(self.decide(("west", "N")).move, "left")
        for i, move in enumerate(["right", "right", "left", "left"]):
            self.vote(i, move)
        self.assertEqual(self.decide(("west", "N")).move, "left")
        for i, move in enumerate(["right", "straight", "left", "straight"]):
            self.vote(i, move)
        self.assertEqual(self.decide(("west", "N")).move, "straight")

    def test_both_directions_and_two_turn_merge(self):
        for hit, expected in [(("north", "W"), "right"), (("north", "S"), None)]:
            self.ctrl.arrow_votes.extend((1, hit, m) for m in ["left", "right", "left", "right"])
            arrow = self.ctrl.take_arrow(2, hit)
            self.assertEqual(arrow["move"] if arrow else None, expected)

    def test_late_frame_and_frame_after_commit(self):
        self.ctrl.remember_arrow_frame(1, 1, [-87, 48, pi], 0.5)
        self.assertIsNone(self.ctrl.accept_arrow_vote(3, 1, "left"))
        self.ctrl.visit = ("west", "N")
        self.assertIsNone(self.vote(2))

    def test_policy_rejects_vote_from_another_arm_or_junction(self):
        arrow = dict(move="left", source="vision_approach", junction="north", arm="E")
        for junction, arm in (("west", "N"), ("north", "W")):
            decision = self.ctrl.policy.decide(5, junction, arm, arrow=arrow)
            self.assertEqual((decision.move, decision.source), ("straight", "default"))
        self.assertEqual(self.ctrl.policy.decide(5, "north", "E", arrow=arrow).move, "left")
