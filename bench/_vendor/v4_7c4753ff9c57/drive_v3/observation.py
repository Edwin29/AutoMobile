"""HSV observation only. No pose matching, road completion, mission or motor output."""
from pathlib import Path
import json
import math
import time

import cv2
import numpy as np

from .camera import sha256_file
from .contracts import LandmarkCandidate, ObservationFrame
from .fragments import make_fragments, connection_candidates


def load_config(path):
    path = Path(path)
    content = json.loads(path.read_text(encoding="utf-8"))
    if content.get("schema") != 1:
        raise ValueError("Unsupported observation config")
    cfg = content["values"]
    for name, value in cfg.items():
        if not np.isfinite(np.asarray(value, dtype=float)).all():
            raise ValueError(f"Non-finite setting: {name}")
    if not 0 <= cfg["forward_min_cm"] < cfg["forward_max_cm"]:
        raise ValueError("Invalid forward ROI")
    if cfg["min_fragment_cm"] <= 0 or cfg["max_connection_candidates"] < 0:
        raise ValueError("Invalid fragment/connection limits")
    return cfg, sha256_file(path)


def fit_circle(points):
    """Algebraic center estimate and geometric residual; not an occlusion solver."""
    p = np.asarray(points, float)
    if len(p) < 5:
        return None
    origin = p.mean(0)
    q = p-origin
    A = np.column_stack((2*q, np.ones(len(q))))
    if np.linalg.matrix_rank(A) < 3:
        return None
    solution, _, _, _ = np.linalg.lstsq(A, (q*q).sum(1), rcond=None)
    center = solution[:2]+origin
    r2 = solution[2]+solution[:2]@solution[:2]
    if r2 <= 0:
        return None
    radius = math.sqrt(r2)
    residual = float(np.sqrt(np.mean((np.linalg.norm(p-center, axis=1)-radius)**2)))
    angles = np.sort(np.mod(np.arctan2(p[:, 1]-center[1], p[:, 0]-center[0]), 2*np.pi))
    coverage = float(np.degrees(2*np.pi-np.diff(np.r_[angles, angles[0]+2*np.pi]).max()))
    return center, radius, residual, coverage


def patch_candidates(masks, valid, geometry, cfg):
    result = []
    valid_inside = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8),
                            borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    for name in ("blue", "pink", "red", "white"):
        contours, _ = cv2.findContours((masks[name] & valid).astype(np.uint8),
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            area = float(cv2.contourArea(contour)/geometry.ppcm**2)
            pixels = contour[:, 0, :]
            # Rasterized thin pink strokes have zero polygon area, despite
            # measurable length. Do not erase them with a filled-patch area gate.
            thin_pink=(name=="pink" and len(pixels)>=3 and np.max(np.ptp(pixels,axis=0))>=2)
            if area < cfg["min_patch_area_cm2"] and not thin_pink:
                continue
            points = geometry.pixels_to_body(pixels)
            clipped = bool((~valid_inside[pixels[:, 1], pixels[:, 0]]).any())
            kind = name+"_patch"
            center = points.mean(0)
            quality = {"area_cm2": area,"thin_stroke_retained":bool(thin_pink and area<cfg["min_patch_area_cm2"])}
            flags = ["view_or_roi_clipped"] if clipped else []
            if name == "blue":
                circle = fit_circle(points)
                if circle is not None:
                    circle_center, radius, residual, arc = circle
                    fill = area/(np.pi*radius**2)
                    quality.update(radius_cm=radius, circle_rms_cm=residual,
                                   arc_coverage_deg=arc, disk_fill_fraction=float(fill))
                    if (not clipped and cfg["circle_radius_min_cm"] <= radius <= cfg["circle_radius_max_cm"]
                            and residual <= cfg["circle_residual_max_cm"]
                            and arc >= cfg["circle_arc_min_deg"] and fill >= cfg["circle_min_fill"]):
                        center, kind = circle_center, "blue_circle_candidate"
                    else:
                        flags.append("circle_center_not_accepted")
                else:
                    flags.append("circle_fit_degenerate")
            # Patch centroids are image regions, NOT completed marker/landmark centers.
            if kind.endswith("_patch"):
                flags.append("region_only_not_mission_marker")
            stride = max(1, len(points)//96)
            result.append(LandmarkCandidate(
                f"patch_{len(result):03d}", kind, center.round(5).tolist(),
                points[::stride].round(5).tolist(), quality, flags))
    return result


class Observer:
    def __init__(self, geometry, config_path):
        self.geometry = geometry
        self.cfg, self.config_id = load_config(config_path)
        from .settings import load_settings
        self.mark_cfg = load_settings()[0]["marks"]

    def process(self, image, stamp, *, rectified, rotate180):
        begin = time.perf_counter()
        if stamp.capture_s is not None and (not math.isfinite(stamp.capture_s) or stamp.capture_s < 0):
            raise ValueError("Invalid capture timestamp")
        g, c = self.geometry, self.cfg
        und, bev = g.prepare(image, rectified=rectified, rotate180=rotate180)
        hsv = cv2.cvtColor(und, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        colored = cv2.inRange(hsv, tuple(c["colored_low"]), tuple(c["colored_high"])) > 0
        raw = {
            "colored": colored,
            "road": (v < c["road_v_max"]) & ~colored,
            "blue": cv2.inRange(hsv, tuple(c["blue_low"]), tuple(c["blue_high"])) > 0,
            "pink": cv2.inRange(hsv, tuple(c["pink_low"]), tuple(c["pink_high"])) > 0,
            "red": ((h <= c["red_h_low_max"]) | (h >= c["red_h_high_min"])) & (s >= c["red_s_min"]) & (v >= c["red_v_min"]),
            "white": (s <= c["white_s_max"]) & (v >= c["white_v_min"]),
            "glare_suspect": (s <= c["glare_s_max"]) & (v >= c["glare_v_min"]),
        }
        masks = {name: g.warp_mask(mask) for name, mask in raw.items()}
        yy = np.arange(g.bev_size[1])[:, None]
        forward = g.fwd_max-yy/g.ppcm
        valid = g.valid_bev & (forward >= c["forward_min_cm"]) & (forward <= c["forward_max_cm"])
        # Ambiguous colored patches are not treated as proof of dark drivable floor.
        road = masks["road"] & ~(masks["blue"] | masks["pink"] | masks["red"] | masks["white"])
        interior = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8),
                            borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        boundary = masks["colored"] & (cv2.dilate((road & valid).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & interior
        fragments, skeleton, dropped = make_fragments(boundary, road, valid, g, c)
        connections, truncated = connection_candidates(fragments, c)
        landmarks = patch_candidates(masks, valid, g, c)
        from .features import group_marks
        landmarks += group_marks(landmarks, self.mark_cfg)
        known = road | masks["colored"] | masks["blue"] | masks["pink"] | masks["red"] | masks["white"]
        masks.update(road=road, valid=valid, unknown=valid & ~known, boundary=boundary, skeleton=skeleton)
        elapsed_ms = (time.perf_counter()-begin)*1000
        diagnostics = {
            "processing_ms": elapsed_ms,
            "valid_pixels": int(valid.sum()),
            "unclassified_pixels": int(masks["unknown"].sum()),
            "glare_suspect_pixels": int((masks["glare_suspect"] & valid).sum()),
            "boundary_pixels": int(boundary.sum()),
            "discarded_short_chains": dropped,
            "connection_candidates_truncated": truncated,
            "rectified_input": bool(rectified), "rotate180": bool(rotate180),
            "marker_grouping_implemented": True,
            "yolo_implemented": False,
            "calibration_session_match_verified": False,
            "quality_is_probability": False,
        }
        status = "boundary_observations" if fragments else "insufficient_boundary_observation"
        frame = ObservationFrame(stamp, g.id, self.config_id, fragments, connections,
                                 landmarks, status, diagnostics)
        return frame, {"rectified": und, "bev": bev, **masks}
