"""Build drive_v4/config/map_v4.json from drive_v3's map with the road geometry fixed.

Finding (2026-11-10): drive_v3's road boxes are rotated 90 degrees relative to the
photo/plan frame in which the landmarks were verified. Photo measurements
(track_guard/measurements.photo.json): left 26, bottom 24, inner vertical 18,
inner horizontal 21 (user re-confirmed: the first T junction met driving north
from the red lines is 21cm wide). v3 had left 26, bottom 26, inner vertical 21,
inner horizontal 18.5. Rotating the road boxes 90 degrees clockwise,
(x, y) -> (y, -x), reproduces all four measurements.

Landmarks stay in the verified frame; items placed on an outer road keep their
along-road position and move with that road's centreline / outer edge.
Consequence: right road 26cm (user-confirmed 2026-11-10), top 26cm (unmeasured).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V3_MAP = HERE.parent / "drive_v3/config/map.json"
OUT = HERE / "config/map_v4.json"


def rotate_box(b):
    x0, x1, y0, y1 = b
    return [y0, y1, -x1, -x0]


def main():
    m = json.loads(V3_MAP.read_text(encoding="utf-8"))
    old_outer, old_blocks = m["outer"], m["blocks"]
    outer = rotate_box(old_outer)
    blocks = [rotate_box(b) for b in old_blocks]

    def centre(o, b):
        # outer-road centrelines: left/right in x, bottom/top in y
        bx0 = min(k[0] for k in b); bx1 = max(k[1] for k in b)
        by0 = min(k[2] for k in b); by1 = max(k[3] for k in b)
        return dict(left=(o[0] + bx0) / 2, right=(o[1] + bx1) / 2, bottom=(o[2] + by0) / 2, top=(o[3] + by1) / 2)

    c0, c1 = centre(old_outer, old_blocks), centre(outer, blocks)
    dx = dict(left=c1["left"] - c0["left"], right=c1["right"] - c0["right"])
    dy = dict(bottom=c1["bottom"] - c0["bottom"], top=c1["top"] - c0["top"])
    edge_dx = dict(left=outer[0] - old_outer[0], right=outer[1] - old_outer[1])
    edge_dy = dict(bottom=outer[2] - old_outer[2], top=outer[3] - old_outer[3])

    def road_of(x, y):
        """Which outer road a point lies on (old frame), else None."""
        if x <= -73.25: return "left"
        if x >= 73.25: return "right"
        if y <= -74.25: return "bottom"
        if y >= 74.25: return "top"
        return None

    def shift(p, edge=False):
        x, y = p
        r = road_of(x, y)
        if r in ("left", "right"):
            return [x + (edge_dx if edge else dx)[r], y]
        if r in ("bottom", "top"):
            return [x, y + (edge_dy if edge else dy)[r]]
        return [x, y]

    m["outer"], m["blocks"] = outer, blocks
    m["crosswalks"] = [shift([x, y]) + [d] for x, y, d in m["crosswalks"]]
    m["pink_lines"] = [[shift(a), shift(b)] for a, b in
                       [(l[0], l[1]) for l in m["pink_lines"]]]
    # a line spanning the road: shift both ends with the road it belongs to (use its midpoint)
    fixed = []
    for a, b in [(l[0], l[1]) for l in json.loads(V3_MAP.read_text(encoding="utf-8"))["pink_lines"]]:
        mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]
        s = shift(mid)
        d = [s[0] - mid[0], s[1] - mid[1]]
        fixed.append([[a[0] + d[0], a[1] + d[1]], [b[0] + d[0], b[1] + d[1]]])
    m["pink_lines"] = fixed
    reds = []
    for a, b in m["red_lines"]:
        d = dx["right"]
        reds.append([[a[0] + d, a[1]], [b[0] + d, b[1]]])
    m["red_lines"] = reds
    rr = m["start_regions"]["right_red"]
    rr["road_x_cm"] = [max(k[1] for k in blocks), outer[1]]
    m["blue_x"], m["blue_y"] = [-v for v in m["blue_y"][::-1]], list(m["blue_x"])
    m["blue_extra"] = [shift(p, edge=True) for p in m["blue_extra"]]
    for name, j in m["junctions"].items():
        j["xy"] = shift(j["xy"])
    m["provenance"] = ("drive_v3 map with road boxes rotated 90deg cw to match photo-measured widths "
                       "(left 26, bottom 24, inner vertical 18, inner horizontal 21); landmarks kept in the "
                       "verified frame and moved with their outer road. Right road 26 confirmed by the user (2026-11-10); top (26) unmeasured.")
    xs0 = sorted({b[0] for b in blocks}); xs1 = sorted({b[1] for b in blocks})
    ys0 = sorted({b[2] for b in blocks}); ys1 = sorted({b[3] for b in blocks})
    m["v4_fix"] = {"date": "2026-11-10", "source": "drive_v4/build_map.py", "road_widths_cm": {
        "left": xs0[0] - outer[0], "right": outer[1] - xs1[-1],
        "bottom": ys0[0] - outer[2], "top": outer[3] - ys1[-1],
        "inner_vertical": xs0[-1] - xs1[0], "inner_horizontal": ys0[-1] - ys1[0]}}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(m["v4_fix"], indent=1))
    print("outer", outer, "blocks", blocks)


if __name__ == "__main__":
    main()
