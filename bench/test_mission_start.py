import subprocess
import sys
import unittest
from types import SimpleNamespace

from .pi_run import describe_start, start_pose
from .vehicle import PRESETS
from .vision_rules import Rules
from .world import Track


class MissionStartTests(unittest.TestCase):
    def describe(self, direction, pose=None, start=None):
        args = SimpleNamespace(direction=direction, pose=pose, start=start)
        track = Track()
        return describe_start(args, track, start_pose(args, track, PRESETS["field_estimate"]))

    def test_red_start_both_directions(self):
        self.assertIn("south/E exits=straight,right", self.describe("cw"))
        self.assertIn("east/S exits=straight,left", self.describe("ccw"))

    def test_outer_heading_and_map_arms(self):
        with self.assertRaisesRegex(ValueError, "implies ccw"):
            self.describe("cw", pose=[87, -48, 0])
        with self.assertRaisesRegex(ValueError, "implies cw"):
            self.describe("ccw", start="west:S:24")
        with self.assertRaisesRegex(ValueError, "existing junction arm"):
            self.describe("cw", start="west:W:24")
        self.assertIn("west/N exits=straight,left", self.describe("ccw", start="west:N:24"))
        self.assertIn("center/S", self.describe("cw", start="center:S:30"))

    def test_real_drive_requires_explicit_direction(self):
        result = subprocess.run([sys.executable, "-m", "bench.pi_run", "--drive"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--drive requires --direction", result.stderr)

    def test_lights_and_arrows(self):
        def detection(name):
            return dict(name=name, score=0.99, box=[50, 80, 40, 60])

        for color in ("green", "red"):
            rules = Rules()
            commands = rules.update(0, [detection(color), detection("cross")])
            for time in (4, 5, 8, 9, 30):
                commands.extend(rules.tick(time))
            expected = ["stop", "resume"] if color == "red" else []
            self.assertEqual([cmd["cmd"] for cmd in commands], expected)
        for move in ("left", "straight", "right"):
            rules = Rules()
            for time in (0, 0.2, 0.4):
                self.assertEqual(rules.update(time, [detection(move)]), [])
            self.assertEqual(rules.update(2, []), [dict(cmd="direction", move=move)])
