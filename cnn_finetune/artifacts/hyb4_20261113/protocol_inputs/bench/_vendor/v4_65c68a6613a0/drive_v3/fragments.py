"""Trace observed skeleton edges, splitting at junctions and gaps.

No gap filling or map import. Connection suggestions never become observed points.
"""
import math
import cv2
import numpy as np

from .contracts import BoundaryFragment, ConnectionCandidate


def thin(mask):
    """Zhang-Suen thinning, using only numpy (no opencv-contrib dependency)."""
    a = np.pad(mask.astype(np.uint8), 1)
    while True:
        changed = False
        for phase in (0, 1):
            p = [a[:-2, 1:-1], a[:-2, 2:], a[1:-1, 2:], a[2:, 2:],
                 a[2:, 1:-1], a[2:, :-2], a[1:-1, :-2], a[:-2, :-2]]
            count = sum(p)
            transitions = sum(((p[i] == 0) & (p[(i+1) % 8] == 1)).astype(np.uint8)
                              for i in range(8))
            if phase == 0:
                clear = (p[0]*p[2]*p[4] == 0) & (p[2]*p[4]*p[6] == 0)
            else:
                clear = (p[0]*p[2]*p[6] == 0) & (p[0]*p[4]*p[6] == 0)
            remove = (a[1:-1, 1:-1] == 1) & (count >= 2) & (count <= 6) & (transitions == 1) & clear
            if remove.any():
                a[1:-1, 1:-1][remove] = 0
                changed = True
        if not changed:
            return a[1:-1, 1:-1].astype(bool)


def trace_skeleton(skeleton):
    """Return (pixel chains, endpoint reasons, closed). Trace each graph edge once."""
    nodes = set(zip(*np.nonzero(skeleton)))  # (row, col)
    neighbors = {}
    for y, x in sorted(nodes):
        linked = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (y+dy, x+dx)
                if (dx == dy == 0) or q not in nodes:
                    continue
                # Avoid triangle shortcuts around an orthogonal bend/junction.
                if dx and dy and ((y, x+dx) in nodes or (y+dy, x) in nodes):
                    continue
                linked.append(q)
        neighbors[(y, x)] = sorted(linked)
    used = set()
    paths = []

    def edge(a, b):
        return tuple(sorted((a, b)))

    def walk(a, b):
        chain = [a, b]
        used.add(edge(a, b))
        while len(neighbors[b]) == 2:
            choices = [c for c in neighbors[b] if edge(b, c) not in used]
            if not choices:
                break
            c = choices[0]
            used.add(edge(b, c))
            chain.append(c)
            a, b = b, c
            if b == chain[0]:
                break
        closed = chain[0] == chain[-1]
        reasons = ["loop" if closed else ("branch" if len(neighbors[p]) > 2 else "observed_end")
                   for p in (chain[0], chain[-1])]
        paths.append((np.array([(x, y) for y, x in chain], dtype=float), reasons, closed))

    for a in sorted(nodes):
        if len(neighbors[a]) != 2:
            for b in neighbors[a]:
                if edge(a, b) not in used:
                    walk(a, b)
    for a in sorted(nodes):  # remaining closed cycles
        for b in neighbors[a]:
            if edge(a, b) not in used:
                walk(a, b)
    return paths


def make_fragments(boundary, road, valid, geometry, cfg):
    skeleton = thin(boundary)
    support_distance = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 3)
    fragments = []
    dropped = 0
    for pixels, reasons, closed in trace_skeleton(skeleton):
        points = geometry.pixels_to_body(pixels)
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        if length < cfg["min_fragment_cm"]:
            dropped += 1
            continue
        if not closed and np.linalg.norm(points[0]) > np.linalg.norm(points[-1]):
            points, pixels, reasons = points[::-1], pixels[::-1], reasons[::-1]
        for i, endpoint in enumerate((pixels[0], pixels[-1])):
            x, y = endpoint.astype(int)
            if support_distance[y, x] <= 2:
                reasons[i] = "view_or_roi_limit"
        delta = np.gradient(points, axis=0)
        tangent = delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-9)
        left = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        scores = []
        valid_scores = []
        for sign in (1, -1):
            q = np.rint(geometry.body_to_pixels(points + sign*left*cfg["side_probe_cm"])).astype(int)
            inside = (q[:, 0] >= 0) & (q[:, 0] < road.shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < road.shape[0])
            q = q[inside]
            vv = valid[q[:, 1], q[:, 0]] if len(q) else np.array([], bool)
            rr = road[q[:, 1], q[:, 0]][vv] if len(q) else np.array([], bool)
            scores.append(float(rr.mean()) if len(rr) else 0.)
            valid_scores.append(float(vv.sum()/len(points)))
        left_road, right_road = scores
        side = "unknown"
        if not closed and min(valid_scores) >= cfg["side_probe_valid_min"]:
            if right_road >= cfg["side_support_min"] and right_road-left_road >= cfg["side_support_delta"]:
                side = "left"
            elif left_road >= cfg["side_support_min"] and left_road-right_road >= cfg["side_support_delta"]:
                side = "right"
        fragments.append(BoundaryFragment(
            f"edge_{len(fragments):03d}", points.round(5).tolist(), reasons, side,
            {"length_cm": length, "support_points": len(points),
             "road_left_fraction": left_road, "road_right_fraction": right_road,
             "probe_valid_fraction": min(valid_scores)}, closed))
    return fragments, skeleton, dropped


def connection_candidates(fragments, cfg):
    ends = []
    for f in fragments:
        if f.closed:
            continue
        p = np.asarray(f.points_cm)
        for end in (0, 1):
            if f.endpoint_reasons[end] != "observed_end":
                continue
            q, interior = (p[0], p[min(4, len(p)-1)]) if end == 0 else (p[-1], p[max(0, len(p)-5)])
            d = q-interior
            if np.linalg.norm(d) > 1e-9:
                ends.append((f.id, end, q, d/np.linalg.norm(d)))
    candidates = []
    for i, (fa, ea, a, da) in enumerate(ends):
        for fb, eb, b, db in ends[i+1:]:
            if fa == fb:
                continue
            distance = float(np.linalg.norm(b-a))
            if not 0 < distance <= cfg["connection_max_gap_cm"]:
                continue
            along = (b-a)/distance
            angle = math.degrees(math.acos(float(np.clip(min(da@along, db@(-along)), -1, 1))))
            if angle <= cfg["connection_max_angle_deg"]:
                candidates.append(ConnectionCandidate(fa, ea, fb, eb, distance, angle))
    candidates.sort(key=lambda c: (c.gap_cm, c.heading_mismatch_deg, c.fragment_a, c.fragment_b))
    limit = cfg["max_connection_candidates"]
    return candidates[:limit], max(0, len(candidates)-limit)
