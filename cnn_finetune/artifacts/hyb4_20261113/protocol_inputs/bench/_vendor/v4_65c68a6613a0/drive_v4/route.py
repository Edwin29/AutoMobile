"""Route helpers: where the lane centre is relative to the driving route.

The camera measures the car's offset from the LANE CENTRE; the map pose gives
the offset from the ROUTE. lane_offset[i] links the two so their difference
(a bias from camera calibration + map geometry) can be learned online.

Evaluated and NOT adopted (2026-11-10): a minimum-curvature route. With a
24.5x16cm body in a 24-26cm lane the footprint-feasible lateral range is only
about +-3cm on straights and +-1-2cm in corners, and a clipped elastic band
made the tightest local radius smaller (10.8 -> 6.9cm), not larger.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def lane_offsets(track, points, heading, span=20.0, step=0.25):
    """(centre offset, lane width) of the road at each route point.

    centre offset: signed (along body-right normal) offset of the lane centre from
    the route point. width: road width measured along the same normal (used when the
    camera sees only one edge, instead of a fixed 24 cm guess)."""
    n = np.c_[np.cos(heading), -np.sin(heading)]
    d = np.arange(-span, span + 1e-9, step)
    q = points[:, None, :] + d[None, :, None] * n[:, None, :]
    inside = track.clearance(q.reshape(-1, 2)).reshape(q.shape[:2]) > 0   # exact (offline, cached)
    mid = len(d) // 2
    off = np.zeros(len(points))
    width = np.full(len(points), np.nan)
    for i in range(len(points)):
        row = inside[i]
        if not row[mid]:
            continue
        L = mid
        while L > 0 and row[L - 1]:
            L -= 1
        R = mid
        while R < len(d) - 1 and row[R + 1]:
            R += 1
        off[i] = (d[L] + d[R]) / 2
        if L > 0 and R < len(d) - 1:            # both edges found inside the scan
            width[i] = d[R] - d[L] + step
    return off, width


def with_lane_offsets(track, route):
    key = hashlib.sha256(json.dumps([track.id, route.id, len(route.points), float(route.s[-1]), 3]).encode()).hexdigest()[:16]
    cache = HERE / "config" / f"lane_geometry_{route.id}_{key}.npz"
    if cache.exists():
        z = np.load(cache)
        route.lane_offset, route.lane_width = z["offset"], z["width"]
    else:
        route.lane_offset, route.lane_width = lane_offsets(track, route.points, route.heading)
        np.savez(cache, offset=route.lane_offset, width=route.lane_width)
    return route
