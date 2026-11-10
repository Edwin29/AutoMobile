"""Behavioral tests for gaps, branches, ambiguous markers and camera conventions."""
from pathlib import Path
import json
import tempfile
import unittest

import cv2
import numpy as np

from drive_v3.camera import CameraGeometry
from drive_v3.contracts import FrameStamp, EvidenceSource
from drive_v3.fragments import thin, trace_skeleton, make_fragments, connection_candidates
from drive_v3.observation import Observer, fit_circle, load_config, patch_candidates
from drive_v3.replay import main

HERE = Path(__file__).resolve().parents[1]


class FlatGeometry:
    ppcm = 5.
    def pixels_to_body(self, points):
        p = np.asarray(points, float)
        return np.column_stack((p[:, 0]/5-10, 20-p[:, 1]/5))
    def body_to_pixels(self, points):
        p = np.asarray(points, float)
        return np.column_stack(((p[:, 0]+10)*5, (20-p[:, 1])*5))


class ObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, _ = load_config(HERE/"config/observation.json")
        cls.camera = CameraGeometry(HERE/"config/camera_legacy_track.json")

    def fragments(self, mask, road=None):
        if road is None:
            road = np.zeros_like(mask)
        return make_fragments(mask, road, np.ones_like(mask), FlatGeometry(), self.cfg)[0]

    def test_cross_has_four_arms_not_an_average_diagonal(self):
        mask = np.zeros((101, 101), np.uint8)
        cv2.line(mask, (10, 50), (90, 50), 1, 1)
        cv2.line(mask, (50, 10), (50, 90), 1, 1)
        chains = trace_skeleton(mask)
        self.assertEqual(len(chains), 4)
        for p, reasons, _ in chains:
            self.assertIn("branch", reasons)
            self.assertTrue(np.all(p[:, 0] == 50) or np.all(p[:, 1] == 50))

    def test_t_junction_has_three_separate_arms(self):
        mask = np.zeros((101, 101), np.uint8)
        mask[50, 10:91] = 1
        mask[10:51, 50] = 1
        chains = trace_skeleton(mask)
        self.assertEqual(len(chains), 3)
        self.assertTrue(all("branch" in reasons for _, reasons, _ in chains))

    def test_gap_stays_unobserved_even_with_connection(self):
        mask = np.zeros((101, 101), np.uint8)
        mask[10:40, 40] = 1
        mask[50:90, 40] = 1
        fragments = self.fragments(mask)
        self.assertEqual(len(fragments), 2)
        connections, _ = connection_candidates(fragments, self.cfg)
        self.assertEqual(len(connections), 1)
        for f in fragments:
            pixels = FlatGeometry().body_to_pixels(f.points_cm)
            self.assertFalse(np.any((pixels[:, 1] >= 40) & (pixels[:, 1] < 50)))
            self.assertEqual(f.source, EvidenceSource.OBSERVED)
        self.assertEqual(connections[0].status, "unobserved_connection_candidate")

    def test_parallel_edges_are_not_joined_sideways(self):
        mask = np.zeros((101, 101), np.uint8)
        mask[10:90, 40] = mask[10:90, 50] = 1
        fragments = self.fragments(mask)
        self.assertEqual(len(fragments), 2)
        self.assertEqual(connection_candidates(fragments, self.cfg)[0], [])

    def test_curve_points_remain_on_observed_mask(self):
        mask = np.zeros((101, 101), np.uint8)
        cv2.ellipse(mask, (50, 50), (30, 30), 0, 0, 150, 1, 3)
        fragments = self.fragments(mask)
        self.assertGreater(len(fragments), 0)
        for f in fragments:
            p = np.rint(FlatGeometry().body_to_pixels(f.points_cm)).astype(int)
            self.assertTrue(mask[p[:, 1], p[:, 0]].all())

    def test_closed_loop_is_retained(self):
        mask = np.zeros((81, 81), np.uint8)
        cv2.rectangle(mask, (20, 20), (60, 60), 1, 1)
        paths = trace_skeleton(mask)
        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0][2])

    def test_side_requires_road_evidence(self):
        mask = np.zeros((101, 101), np.uint8)
        mask[10:90, 40] = 1
        self.assertEqual(self.fragments(mask)[0].side_hint, "unknown")
        road = np.zeros_like(mask)
        road[:, 41:] = 1
        self.assertEqual(self.fragments(mask, road)[0].side_hint, "left")

    def test_short_noise_is_counted_not_promoted(self):
        mask = np.zeros((101, 101), np.uint8)
        mask[40:43, 40] = 1
        fragments, _, dropped = make_fragments(mask, mask*0, np.ones_like(mask), FlatGeometry(), self.cfg)
        self.assertEqual(fragments, [])
        self.assertGreater(dropped, 0)

    def test_circle_arc_fit_not_visible_pixel_centroid(self):
        angles = np.linspace(-1.1, 1.1, 80)
        points = np.c_[3+2*np.cos(angles), 5+2*np.sin(angles)]
        center, radius, residual, coverage = fit_circle(points)
        np.testing.assert_allclose(center, [3, 5], atol=1e-8)
        self.assertAlmostEqual(radius, 2.)
        self.assertLess(residual, 1e-8)
        self.assertLess(coverage, self.cfg["circle_arc_min_deg"])

    def test_blue_disk_detected_as_candidate_without_map_id(self):
        masks = {k: np.zeros((101, 101), np.uint8) for k in ("blue", "pink", "red", "white")}
        cv2.circle(masks["blue"], (50, 50), 15, 1, -1)
        marks = patch_candidates(masks, np.ones((101, 101), bool), FlatGeometry(), self.cfg)
        self.assertEqual(len(marks), 1)
        self.assertEqual(marks[0].kind, "blue_circle_candidate")
        self.assertIsNone(marks[0].map_id)

    def test_clipped_circle_not_promoted(self):
        masks = {k: np.zeros((101, 101), np.uint8) for k in ("blue", "pink", "red", "white")}
        cv2.circle(masks["blue"], (50, 50), 15, 1, -1)
        valid = np.ones((101, 101), bool)
        valid[:, :50] = False
        marks = patch_candidates(masks, valid, FlatGeometry(), self.cfg)
        self.assertEqual(marks[0].kind, "blue_patch")
        self.assertIn("view_or_roi_clipped", marks[0].flags)

    def test_half_occluded_disk_not_a_confident_circle(self):
        masks = {k: np.zeros((101, 101), np.uint8) for k in ("blue", "pink", "red", "white")}
        cv2.circle(masks["blue"], (50, 50), 15, 1, -1)
        masks["blue"][:, :50] = 0
        marks = patch_candidates(masks, np.ones((101, 101), bool), FlatGeometry(), self.cfg)
        self.assertEqual(marks[0].kind, "blue_patch")

    def test_image_edge_circle_is_clipped_without_roi(self):
        masks = {k: np.zeros((101, 101), np.uint8) for k in ("blue", "pink", "red", "white")}
        cv2.circle(masks["blue"], (0, 50), 15, 1, -1)
        marks = patch_candidates(masks, np.ones((101, 101), bool), FlatGeometry(), self.cfg)
        self.assertIn("view_or_roi_clipped", marks[0].flags)

    def test_calibration_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            descriptor = json.loads((HERE/"config/camera_legacy_track.json").read_text(encoding="utf-8"))
            descriptor["npz"] = str(self.camera.asset)
            descriptor["npz_sha256"] = "0"*64
            path = Path(d)/"camera.json"
            path.write_text(json.dumps(descriptor), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                CameraGeometry(path)

    def test_pink_patch_is_not_a_dashline_mission(self):
        masks = {k: np.zeros((101, 101), np.uint8) for k in ("blue", "pink", "red", "white")}
        masks["pink"][40:45, 30:65] = 1
        marks = patch_candidates(masks, np.ones((101, 101), bool), FlatGeometry(), self.cfg)
        self.assertEqual(marks[0].kind, "pink_patch")
        self.assertIn("region_only_not_mission_marker", marks[0].flags)

    def test_body_reference_offset_added_exactly_once(self):
        pixels = np.array([[200., 100.], [50., 160.]])
        body = self.camera.pixels_to_body(pixels)
        self.assertAlmostEqual(body[0, 1], 30+12.36)
        np.testing.assert_allclose(self.camera.body_to_pixels(body), pixels)

    def test_blank_frame_has_no_boundary_and_no_fake_timestamp(self):
        observer = Observer(self.camera, HERE/"config/observation.json")
        frame, _ = observer.process(np.zeros((480, 640, 3), np.uint8), FrameStamp(0, None, "unavailable"),
                                    rectified=False, rotate180=False)
        self.assertEqual(frame.status, "insufficient_boundary_observation")
        self.assertEqual(frame.fragments, [])
        self.assertIsNone(frame.stamp.capture_s)
        json.dumps(frame.to_dict(), allow_nan=False)

    def test_wrong_resolution_is_rejected(self):
        with self.assertRaises(ValueError):
            self.camera.prepare(np.zeros((240, 320, 3), np.uint8), rectified=False, rotate180=False)

    def test_double_preprocessing_is_rejected(self):
        with self.assertRaises(ValueError):
            self.camera.prepare(np.zeros((480, 640, 3), np.uint8), rectified=True, rotate180=True)

    def test_replay_provenance_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root/"blank.png"
            cv2.imwrite(str(path), np.zeros((480, 640, 3), np.uint8))
            output = root/"run"
            main(["--input", str(path), "--output", str(output)])
            summary = json.loads((output/"summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["frame_count"], 1)
            self.assertEqual(summary["mode"], "offline_observation_no_control")
            self.assertIn("observation.py", summary["code_sha256"])
            with self.assertRaises(FileExistsError):
                main(["--input", str(path), "--output", str(output)])


if __name__ == "__main__":
    unittest.main()
