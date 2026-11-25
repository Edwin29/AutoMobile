"""Explicit legacy camera adapter, independent of any historical driver import."""
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class CameraGeometry:
    def __init__(self, descriptor):
        self.descriptor = Path(descriptor).resolve()
        self.metadata = json.loads(self.descriptor.read_text(encoding="utf-8"))
        if self.metadata["schema"] != 1:
            raise ValueError("Unsupported camera schema")
        self.asset = (self.descriptor.parent / self.metadata["npz"]).resolve()
        asset_hash = sha256_file(self.asset)
        if asset_hash != self.metadata["npz_sha256"]:
            raise ValueError("Camera asset hash mismatch; review calibration before reuse")
        self.id = hashlib.sha256((sha256_file(self.descriptor) + asset_hash).encode()).hexdigest()
        with np.load(self.asset, allow_pickle=False) as z:
            self.size = tuple(int(v) for v in z["size"])
            self.bev_size = tuple(int(v) for v in z["bev"])
            self.M = z["M"].copy()
            self.remap = cv2.fisheye.initUndistortRectifyMap(
                z["K"], z["D"], np.eye(3), z["rect_K"], self.size, cv2.CV_32FC1)
        if self.M.shape != (3, 3) or not np.isfinite(self.M).all():
            raise ValueError("Invalid homography")
        self.ppcm = float(self.metadata["pixels_per_cm"])
        if self.ppcm <= 0:
            raise ValueError("pixels_per_cm must be positive")
        self.lat_min = float(self.metadata["lateral_min_cm"])
        self.fwd_max = float(self.metadata["forward_max_cm"])
        self.lens_ahead = float(self.metadata["lens_ahead_body_cm"])
        raw_valid = np.full((self.size[1], self.size[0]), 255, np.uint8)
        # Rectified archived images use these same intrinsics; padded pixels stay invalid.
        self.valid_rect = cv2.remap(raw_valid, *self.remap, cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT)
        self.valid_bev = cv2.warpPerspective(self.valid_rect, self.M, self.bev_size,
                                            flags=cv2.INTER_NEAREST) > 0

    def prepare(self, image, *, rectified, rotate180):
        if image is None or image.dtype != np.uint8 or image.shape != (self.size[1], self.size[0], 3):
            raise ValueError(f"Expected uint8 BGR {self.size}, no silent resize")
        if rectified and rotate180:
            raise ValueError("Rotate raw camera image before rectification, not a rectified frame")
        oriented = cv2.rotate(image, cv2.ROTATE_180) if rotate180 else image
        und = oriented.copy() if rectified else cv2.remap(oriented, *self.remap, cv2.INTER_LINEAR)
        bev = cv2.warpPerspective(und, self.M, self.bev_size, flags=cv2.INTER_LINEAR)
        return und, bev

    def warp_mask(self, mask):
        return (cv2.warpPerspective(mask.astype(np.uint8), self.M, self.bev_size,
                                    flags=cv2.INTER_NEAREST) > 0) & self.valid_bev

    def pixels_to_body(self, points):
        p = np.asarray(points, dtype=float).reshape(-1, 2)
        return np.column_stack((p[:, 0] / self.ppcm + self.lat_min,
                                self.fwd_max - p[:, 1] / self.ppcm + self.lens_ahead))

    def body_to_pixels(self, points):
        p = np.asarray(points, dtype=float).reshape(-1, 2)
        return np.column_stack(((p[:, 0] - self.lat_min) * self.ppcm,
                                (self.fwd_max + self.lens_ahead - p[:, 1]) * self.ppcm))

    def body_to_rectified(self, points):
        p = self.body_to_pixels(points).astype(np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, np.linalg.inv(self.M)).reshape(-1, 2)
