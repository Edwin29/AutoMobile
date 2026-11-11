"""Junction policy: options, defaults, command queue, interrupts. No hardware."""
import json
import unittest
from pathlib import Path

from drive_v4.junction_policy import Interrupt, JunctionPolicy, exit_arm

MAP = json.loads((Path(__file__).resolve().parents[1] / "config" / "map_v4.json").read_text(encoding="utf-8"))


class Policy(unittest.TestCase):
    def setUp(self):
        self.cw = JunctionPolicy(MAP["junctions"], "cw")
        self.ccw = JunctionPolicy(MAP["junctions"], "ccw")

    def test_exit_arm(self):
        self.assertEqual(exit_arm("S", "straight"), "N")   # arriving from the south = heading north
        self.assertEqual(exit_arm("S", "left"), "W")
        self.assertEqual(exit_arm("S", "right"), "E")
        self.assertEqual(exit_arm("N", "right"), "W")

    def test_options_follow_the_map(self):
        self.assertEqual(self.cw.options("east", "N"), ["straight", "right"])      # cw on the east road: inner is right
        self.assertEqual(self.ccw.options("east", "S"), ["straight", "left"])
        self.assertEqual(self.cw.options("east", "W"), ["left", "right"])          # from the inner road: merge
        self.assertEqual(self.cw.options("center", "S"), ["straight", "left", "right"])

    def test_defaults(self):
        self.assertEqual(self.cw.decide(0, "east", "N").move, "straight")          # stay on the loop
        self.assertEqual(self.cw.decide(0, "north", "S").move, "right")            # merge with the lap (cw)
        self.assertEqual(self.ccw.decide(0, "north", "S").move, "left")
        d = self.cw.decide(0, "center", "E")
        self.assertEqual((d.move, d.source), ("right", "rule"))

    def test_command_used_once_and_only_if_possible(self):
        self.cw.push(1, "right")
        d = self.cw.decide(2, "east", "N")
        self.assertEqual((d.move, d.source), ("right", "command"))
        self.assertEqual(self.cw.decide(3, "south", "E").move, "straight")         # queue now empty: default
        self.cw.push(4, "left")                                                    # no left arm on the outer T (cw)
        d = self.cw.decide(5, "west", "S")
        self.assertEqual((d.move, d.source), ("straight", "default"))
        self.assertIn("not possible", d.note)

    def test_crossroad_keeps_the_queue(self):
        self.cw.push(1, "left")
        self.assertEqual(self.cw.decide(2, "center", "E").move, "right")           # rule; command not used
        d = self.cw.decide(3, "north", "S")                                        # still there for the next T
        self.assertEqual((d.move, d.source), ("left", "command"))
        self.cw.push(4, "straight")                                                # impossible from an inner road
        d = self.cw.decide(5, "north", "S")
        self.assertEqual((d.move, d.source), ("right", "default"))

    def test_old_commands_expire(self):
        self.cw.push(0, "right")
        d = self.cw.decide(100, "east", "N")
        self.assertEqual((d.move, d.source), ("straight", "default"))

    def test_interrupt(self):
        i = Interrupt()
        i.stop(1.0)
        i.stop(1.5)                     # repeated stop keeps the first time
        self.assertTrue(i.held)
        i.resume(4.0)
        self.assertFalse(i.held)
        self.assertAlmostEqual(i.log[-1]["held_s"], 3.0)


if __name__ == "__main__":
    unittest.main()
