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
    """Signed offset (along body-right normal) of the lane centre from each route point."""
    n = np.c_[np.cos(heading), -np.sin(heading)]
    d = np.arange(-span, span + 1e-9, step)
    q = points[:, None, :] + d[None, :, None] * n[:, None, :]
    inside = track.clearance_fast(q) > 0
    mid = len(d) // 2
    out = np.zeros(len(points))
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
        out[i] = (d[L] + d[R]) / 2
    return out


def with_lane_offsets(track, route):
    key = hashlib.sha256(json.dumps([track.id, route.id, len(route.points), float(route.s[-1])]).encode()).hexdigest()[:16]
    cache = HERE / "config" / f"lane_offset_{route.id}_{key}.npy"
    if cache.exists():
        route.lane_offset = np.load(cache)
    else:
        route.lane_offset = lane_offsets(track, route.points, route.heading)
        np.save(cache, route.lane_offset)
    return route
