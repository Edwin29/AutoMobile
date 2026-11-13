"""The FIXED world: track geometry, camera projection, image rendering, start zones.

Nothing here is tunable per experiment. The track is the surveyed/corrected map
(bench/world/track.json = drive_v4/config/map_v4.json: left 26, bottom 24, right 26
(user-confirmed), inner 18.5/21). The camera is the calibration both drivers use
(bench/world/camera_track.npz). This module imports no driver code.

Conventions (same as the drivers): x east, y north (cm); yaw clockwise from north;
pose origin = axle midpoint; body x right, y forward.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
WORLD = HERE / "world"

# Camera constants from drive_v3/config/camera_legacy_track.json (same npz hash).
CAMERA = dict(pixels_per_cm=5.0, lateral_min_cm=-40.0, forward_max_cm=50.0, lens_ahead_body_cm=12.36)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def axes(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def to_world(points, pose):
    return np.asarray(points) @ axes(pose[2]) + np.asarray(pose[:2])


def advance(pose, v, omega, dt):
    a = omega * dt
    p = np.asarray(pose, float).copy()
    p[:2] += v * dt * np.sinc(a / (2 * np.pi)) * np.array([np.sin(p[2] + a / 2), np.cos(p[2] + a / 2)])
    p[2] = wrap(p[2] + a)
    return p


def box_sdf(points, box, radius):
    p = np.asarray(points)
    x0, x1, y0, y1 = box
    q = np.abs(p - np.array([(x0 + x1) / 2, (y0 + y1) / 2])) - np.array([(x1 - x0) / 2 - radius, (y1 - y0) / 2 - radius])
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(np.max(q, axis=-1), 0) - radius


class Track:
    def __init__(self, path=WORLD / "track.json"):
        self.data = json.loads(Path(path).read_text(encoding="utf-8"))

    def clearance(self, points):
        """Signed distance to the nearest road boundary (+ inside the road)."""
        d = [-box_sdf(points, self.data["outer"], self.data["outer_radius_cm"])]
        d += [box_sdf(points, b, self.data["block_radius_cm"]) for b in self.data["blocks"]]
        return np.min(np.stack(d, axis=-1), axis=-1)

    def start_pose(self, direction, vehicle_front_cm=14.5, vehicle_rear_cm=10.0):
        """Centre of the right red-line start zone, axle-midpoint pose."""
        region = self.data["start_regions"]["right_red"]
        yaw = region["heading_by_direction"][direction]
        ys = sorted(float(np.mean(self.data["red_lines"][i], axis=0)[1]) for i in region["red_indices"])
        x = float(np.mean(region["road_x_cm"]))
        y = float(np.mean(ys))
        shift = (vehicle_front_cm - vehicle_rear_cm) / 2   # body midpoint is ahead of the axle
        return np.array([x - shift * math.sin(yaw), y - shift * math.cos(yaw), yaw])


class Camera:
    """Rectified camera view of a flat floor (the drivers' own calibration)."""

    def __init__(self, npz=WORLD / "camera_track.npz"):
        with np.load(npz, allow_pickle=False) as z:
            self.size = tuple(int(v) for v in z["size"])
            self.M = z["M"].copy()
            self.route = z["route"].copy()   # legacy centre line, used ONLY for progress scoring
            self.K, self.D, self.rect_K = z["K"].copy(), z["D"].copy(), z["rect_K"].copy()
        self._raw_maps = None
        c = CAMERA
        self.ppcm, self.lat_min, self.fwd_max, self.lens_ahead = (
            c["pixels_per_cm"], c["lateral_min_cm"], c["forward_max_cm"], c["lens_ahead_body_cm"])
        yy, xx = np.mgrid[:self.size[1], :self.size[0]]
        uv = np.c_[xx.ravel(), yy.ravel()].astype(float).reshape(-1, 1, 2)
        bev = cv2.perspectiveTransform(uv, self.M).reshape(-1, 2)
        self.ground = np.column_stack((bev[:, 0] / self.ppcm + self.lat_min,
                                       self.fwd_max - bev[:, 1] / self.ppcm + self.lens_ahead))
        self.valid = (self.ground[:, 1] > self.lens_ahead) & (self.ground[:, 1] < 150) & (np.abs(self.ground[:, 0]) < 150)

    def raw_geometry(self):
        """Floor point (body cm) seen by every pixel of the RAW fisheye frame (upright).

        Each raw pixel is undistorted with the camera's own fisheye model into the
        rectified plane, then mapped to the floor with the same homography as the
        rectified view. Unlike bending the rectified image back, this also fills the
        wide raw periphery that the rectified crop does not contain."""
        if self._raw_maps is None:
            w, h = self.size
            u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
            raw = np.stack([u.ravel(), v.ravel()], -1).reshape(-1, 1, 2)
            rect = cv2.fisheye.undistortPoints(raw, self.K, self.D, R=np.eye(3), P=self.rect_K).reshape(-1, 2)
            hom = np.c_[rect, np.ones(len(rect))] @ self.M.T
            in_front = hom[:, 2] > 1e-9
            bev = hom[:, :2] / np.where(in_front, hom[:, 2], 1.0)[:, None]
            ground = np.column_stack((bev[:, 0] / self.ppcm + self.lat_min,
                                      self.fwd_max - bev[:, 1] / self.ppcm + self.lens_ahead))
            valid = in_front & (ground[:, 1] > self.lens_ahead) & (ground[:, 1] < 150) & (np.abs(ground[:, 0]) < 150)
            self._raw_maps = (ground, valid)
        return self._raw_maps


class Renderer:
    """Paints the track once, then samples it through the camera for any pose."""
    ppcm, origin = 4.0, -125.0

    def __init__(self, track, camera):
        self.track, self.camera = track, camera
        y, x = np.mgrid[0:1001, 0:1001]
        world = np.stack((x / self.ppcm + self.origin, y / self.ppcm + self.origin), axis=-1)
        margin = track.clearance(world)
        tex = np.empty((*x.shape, 3), np.uint8)
        tex[:] = (95, 215, 125)
        tex[margin >= 0] = (47, 60, 50)
        tex[np.abs(margin) < .22] = (70, 220, 210)
        d = track.data
        px = lambda p: tuple(np.rint((np.array(p) - self.origin) * self.ppcm).astype(int))
        for xb in d["blue_x"]:
            for yb in d["blue_y"]:
                cv2.circle(tex, px([xb, yb]), round(d["blue_radius_cm"] * self.ppcm), (225, 165, 30), -1)
        for xy in d["blue_extra"]:
            cv2.circle(tex, px(xy), round(d["blue_radius_cm"] * self.ppcm), (225, 165, 30), -1)
        for a, b in d["red_lines"]:
            cv2.line(tex, px(a), px(b), (50, 50, 200), 2)
        for a, b in d["pink_lines"]:
            a, b = np.array(a), np.array(b)
            n = max(1, int(np.linalg.norm(b - a) / 1.5))
            for k in range(n):
                cv2.line(tex, px(a + (b - a) * k / n), px(a + (b - a) * (k + .65) / n), (190, 120, 200), 2)
        for x0, y0, approach in d["crosswalks"]:
            for off in np.arange(-5.4, 5.5, 1.8):
                q = np.array([[off - .45, -3.], [off + .45, 3.]])
                if approach == "ew":
                    q = q[:, ::-1]
                cv2.rectangle(tex, px(np.array([x0, y0]) + q[0]), px(np.array([x0, y0]) + q[1]), (215, 220, 220), -1)
        self.texture = tex
        self.base_texture = tex.copy()

    def add_floor_line(self, a, b, color=(180, 190, 190), width=3):
        px = lambda p: tuple(np.rint((np.array(p) - self.origin) * self.ppcm).astype(int))
        cv2.line(self.texture, px(a), px(b), color, width)

    def render(self, pose, ground=None):
        cam = self.camera
        g = cam.ground if ground is None else ground
        world = to_world(g, pose)
        maps = ((world - self.origin) * self.ppcm).astype(np.float32)
        shape = (cam.size[1], cam.size[0])
        image = cv2.remap(self.texture, maps[:, 0].reshape(shape), maps[:, 1].reshape(shape), cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(90, 90, 90))
        image.reshape(-1, 3)[~cam.valid] = (90, 90, 90)
        return image

    def render_raw(self, pose, car_front_px=28, car_front_bgr=(38, 38, 42)):
        """The camera's own raw fisheye frame (upright, i.e. after the 180-degree flip).
        Bottom rows are covered by the car's front, as on the real camera."""
        cam = self.camera
        ground, valid = cam.raw_geometry()
        world = to_world(ground, pose)
        maps = ((world - self.origin) * self.ppcm).astype(np.float32)
        shape = (cam.size[1], cam.size[0])
        image = cv2.remap(self.texture, maps[:, 0].reshape(shape), maps[:, 1].reshape(shape), cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(90, 90, 90))
        image.reshape(-1, 3)[~valid] = (90, 90, 90)
        if car_front_px:
            image[-car_front_px:] = car_front_bgr
        return image
