#!/usr/bin/env python3
"""카메라 이미지 -> 어안 보정 -> cm 단위 버드아이 -> 왼쪽/중앙/오른쪽 선 검출 (직선/코너).

사용법:
    python3 lane_pipeline.py                        # raw_640.jpg, 외곽 도로(26cm)
    python3 lane_pipeline.py --input corner_01.jpg  # 다른 이미지
    python3 lane_pipeline.py --capture --input x.jpg  # 카메라로 새로 촬영 후 처리
    python3 lane_pipeline.py --road cross           # 십자 교차로 도로(21cm)
결과 이미지는 <입력파일이름>_*.jpg 로 저장된다.

캘리브레이션 값은 calibrate_checkerboard.py 가 만든 calib_checkerboard.npz 를 쓴다.
캘리브레이션 이미지와 같은 조건(Picamera2 640x480 video config + 180도 flip)으로 촬영해야 한다.

지면 평면 추정 (--setup, 직선 도로에서 1회):
  두 차선의 소실점 -> 도로 진행 방향 d, 카메라 roll=0 가정 -> 지면 법선 n,
  두 차선 안쪽 경계 간격 = 도로폭 -> 카메라 높이 H.  결과는 ground.npz 에 저장.

선 검출 (버드아이, cm 단위):
  0) 검은 노면과 초록/노랑 영역의 경계 = 노란 선 안쪽 경계.
     (먼 곳의 노란 선은 얇고 흐려 초록과 색이 섞이므로 노란색 대신 이 경계를 쓴다)
  1) 로봇에서 가까운 경계점부터 진행 방향으로 창을 옮겨가며 추적 (코너에서도 동작)
  2) 진행 방향 기준 노면이 왼편이면 오른쪽 선, 오른편이면 왼쪽 선
  3) 안쪽 경계를 도로 쪽 법선으로 도로폭/2 옮겨 중앙선, 중앙선을 ±도로폭/2 옮겨 좌/우 선
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

YELLOW_LO = (20, 120, 140)
YELLOW_HI = (50, 255, 255)
COLORED_LO = (20, 100, 130)   # 초록 + 노랑 (도로 밖)
COLORED_HI = (87, 255, 255)   # 88 이상은 창문 빛이 노면에 비친 하늘색 반사라서 제외
ROAD_V_MAX = 130              # 이보다 어두우면 노면
ROI_TOP = 0.3  # 이 비율보다 위쪽(배경)은 무시

# 실측 맵 (2026-10-30, 노란 선 안쪽 경계끼리)
ROAD_WIDTH_CM = {'outer': 26.0,         # 외곽 도로 (왼쪽)
                 'outer_bottom': 24.0,  # 외곽 도로 (아래)
                 'cross_h': 21.0,       # 십자 도로 가로 방향
                 'cross_v': 18.0}       # 십자 도로 세로 방향
DEFAULT_ROAD = 'outer'      # 도로폭을 아직 모를 때(한쪽 선만 보일 때)의 기본값
WIDTH_SNAP_CM = 2.0         # 실측 간격이 맵 도로폭과 이 이내로 맞으면 그 도로로 판단
WIDTH_MEASURE_FWD_CM = 35   # 도로폭은 이 거리 이내에서만 측정 (코너/교차로에서는 도로가 넓어짐)
PX_PER_CM = 5
BEV_LAT_CM = (-40, 40)  # 버드아이 좌우 범위 (카메라 중심 기준, +가 오른쪽)
BEV_FWD_MAX_CM = 50     # 버드아이 전방 최대 거리 (그 이상은 해상도가 낮아 신뢰도 떨어짐)

TRACE_STEP_CM = 2.0     # 선 추적 한 걸음
TRACE_RADIUS_CM = 2.5   # 추적 창 반경
TRACE_MAX_GAP = 4       # 선이 끊겨도 이 걸음 수만큼은 예측 방향으로 계속 진행
TRACE_MAX_TURN = 40     # 한 걸음에 허용하는 최대 방향 변화(도)
TRACE_MINPIX = 8
LOOKAHEAD_CM = 18       # 주행 목표점까지 거리 (pure pursuit). 20 -> 18: 코너를 2cm 늦게 꺾기 시작 (2026-10-31 조정)
ROBOT_HALF_W_CM = 8.0   # 바퀴 포함 차폭 16cm / 2
WHEELBASE_CM = 15.0     # 앞뒤 바퀴 축 사이 거리 (트랙 사진 기준 추정) - 코너 내륜차 계산용
CLEAR_FWD_MAX_CM = 16   # 선 침범 여유는 로봇 바로 앞(보이는 가장 가까운 경계, 이 거리 이내)에서 계산
BRANCH_DEG = 15         # 선이 처음 방향보다 도로 바깥쪽으로 이만큼 꺾이면 그쪽 길이 열린 것(분기)
BRANCH_START_DEG = 5    # 분기 시작점 = 꺾임이 이 각도 미만이던 마지막 지점
PAIR_TOL_CM = 5         # 좌/우 경계 간격이 도로폭 + 이 값 이내일 때만 마주 보는 짝으로 사용
                        # (코너에서 넓어지는 폭은 최대 약 2cm, T자 분기로 열린 곳은 훨씬 넓어짐)
KEEP_MARGIN_CM = 1.5    # 좌우 선 직접 보정: 로봇 바로 앞에서 차체 옆면과 선 사이에 남길 여유
KEEP_OFFTRACK_FRAC = 0.5  # 코너 안쪽 선은 차체 뒤 내륜차(축간거리²/2R)의 이 비율만큼 여유를 더 둠

COLORS = {'left': (255, 0, 255), 'center': (0, 255, 0), 'right': (0, 0, 255)}


def capture(path):
    from picamera2 import Picamera2
    cam = Picamera2()
    cfg = cam.create_video_configuration(main={"format": "RGB888", "size": (640, 480)})
    cam.align_configuration(cfg)
    cam.configure(cfg)
    cam.start()
    time.sleep(2)  # AE/AWB 안정화
    img = cam.capture_array()
    cam.close()
    cv2.imwrite(path, cv2.flip(img, -1))  # 카메라가 뒤집혀 장착됨


def yellow_mask(img, roi_top=ROI_TOP):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m[:int(img.shape[0] * roi_top)] = 0
    return m


def edge_masks(und, M, size):
    """버드아이에서 (노면-초록/노랑 경계 마스크, 노면 마스크)."""
    hsv = cv2.cvtColor(und, cv2.COLOR_BGR2HSV)
    colored = cv2.inRange(hsv, COLORED_LO, COLORED_HI)
    road = ((hsv[..., 2] < ROAD_V_MAX) & (colored == 0)).astype(np.uint8) * 255
    # 영역 마스크를 변환한 뒤 경계를 구함 (가는 경계를 직접 변환하면 가까운 쪽에서 사라짐)
    bc = cv2.warpPerspective(colored, M, size) >= 128
    br = (cv2.warpPerspective(road, M, size) >= 128).astype(np.uint8)
    edge = bc & cv2.dilate(br, np.ones((3, 3), np.uint8)).astype(bool)
    return edge.astype(np.uint8), br


def line_points(mask):
    """마스크에서 가장 큰 두 연결요소를 좌/우 차선 점들로 반환."""
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    idx = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:2]
    idx.sort(key=lambda i: cent[i][0])
    pts = []
    for i in idx:
        ys, xs = np.nonzero(lab == i)
        pts.append(np.stack([xs, ys], 1).astype(np.float64))
    return pts  # [left, right]


def row_extent(pts):
    """행(y)마다 x 의 최소/최대."""
    ys = pts[:, 1].astype(int)
    rows = np.unique(ys)
    lo = np.full(ys.max() + 1, np.inf)
    hi = np.full(ys.max() + 1, -np.inf)
    np.minimum.at(lo, ys, pts[:, 0])
    np.maximum.at(hi, ys, pts[:, 0])
    return rows, lo[rows], hi[rows]


def inner_edge(pts, side):
    """행마다 도로 쪽 경계만 남김: 왼쪽 선은 오른쪽 끝, 오른쪽 선은 왼쪽 끝."""
    rows, lo, hi = row_extent(pts)
    return np.stack([hi if side == 'left' else lo, rows], 1).astype(np.float64)


def undistort(img, K, D, balance=0.0):
    h, w = img.shape[:2]
    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=balance)
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), newK, (w, h), cv2.CV_16SC2)
    return cv2.remap(img, m1, m2, cv2.INTER_LINEAR), newK


def homog_line(pts):
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return np.cross([x0, y0, 1.0], [x0 + vx, y0 + vy, 1.0])


def estimate_ground(lp, rp, Kn, road_w):
    """두 평행 차선(보정 영상 좌표)으로 지면 평면과 카메라 높이 추정."""
    lp, rp = inner_edge(lp, 'left'), inner_edge(rp, 'right')
    Ki = np.linalg.inv(Kn)
    v = np.cross(homog_line(lp), homog_line(rp))           # 소실점
    d = Ki @ v
    d /= np.linalg.norm(d) * np.sign(d[2])                 # 전방(z>0)
    n = np.cross(d, [1.0, 0, 0])                           # roll=0 가정
    n /= np.linalg.norm(n)
    if n[1] > 0:                                           # 카메라 y 는 아래 -> 위쪽 법선은 y<0
        n = -n
    e = np.cross(d, n)                                     # 오른쪽
    if e[0] < 0:
        e = -e

    def lateral_per_H(pts):
        r = Ki @ np.c_[pts, np.ones(len(pts))].T
        return np.median(-(e @ r) / (n @ r))

    H = road_w / (lateral_per_H(rp) - lateral_per_H(lp))
    pitch = np.degrees(np.arcsin(-n[2]))  # 광축이 지면 쪽으로 숙여진 각도
    return dict(d=d, n=n, e=e, H=H, pitch=pitch)


def ground_to_image(g, Kn):
    """지면 (lateral cm, forward cm, 1) -> 보정 영상 픽셀 호모그래피."""
    return Kn @ np.c_[g['e'], g['d'], -g['H'] * g['n']]


def bev_transform(g, Kn, img_h):
    """보정 영상 -> cm 버드아이 변환 행렬과 크기, 전방 범위."""
    Hg = ground_to_image(g, Kn)
    # 영상 하단 중앙이 닿는 지면 거리 = 버드아이 최소 전방 거리
    p = np.linalg.inv(Hg) @ [Kn[0, 2], img_h - 1, 1]
    fwd_min = np.floor(p[1] / p[2])
    lat0, lat1 = BEV_LAT_CM
    s = PX_PER_CM
    # 지면 (a, b) -> 버드아이 픽셀 (u, v): 오른쪽 +u, 전방은 위쪽
    A = np.array([[s, 0, -lat0 * s], [0, -s, BEV_FWD_MAX_CM * s], [0, 0, 1]])
    size = (int((lat1 - lat0) * s), int((BEV_FWD_MAX_CM - fwd_min) * s))
    return A @ np.linalg.inv(Hg), size, fwd_min


def px_to_lat(u):
    return u / PX_PER_CM + BEV_LAT_CM[0]


def lat_to_px(a):
    return (a - BEV_LAT_CM[0]) * PX_PER_CM


def fwd_to_px(b):
    return int(round((BEV_FWD_MAX_CM - b) * PX_PER_CM))


def px_to_fwd(v):
    return BEV_FWD_MAX_CM - v / PX_PER_CM


def to_cm(p):
    """버드아이 픽셀 (N,2) -> 지면 cm (lat, fwd)."""
    return np.stack([px_to_lat(p[:, 0]), px_to_fwd(p[:, 1])], 1)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def left_normal(t):
    """버드아이 픽셀 좌표(y 아래)에서 진행방향 t 의 왼쪽 법선."""
    return np.stack([t[..., 1], -t[..., 0]], -1)


def tangents(pts):
    return unit(np.gradient(pts, axis=0))


def smooth(pts, k=3):
    if len(pts) <= k:
        return pts
    pad = np.r_[pts[:1].repeat(k // 2, 0), pts, pts[-1:].repeat(k // 2, 0)]
    return np.stack([np.convolve(pad[:, i], np.ones(k) / k, 'valid') for i in range(2)], 1)


def dist_to_polyline(q, poly):
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.clip(((q[:, None] - a) * ab).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-9), 0, 1)
    proj = a + t[..., None] * ab
    return np.linalg.norm(q[:, None] - proj, axis=-1).min(1)


def near(P, c, r):
    return P[np.hypot(*(P - c).T) < r]


def trace_from(P, seed, d, size):
    """seed 에서 d 방향으로 선을 따라가며 창 중심점 목록 반환."""
    step, radius = TRACE_STEP_CM * PX_PER_CM, TRACE_RADIUS_CM * PX_PER_CM
    cos_max = np.cos(np.radians(TRACE_MAX_TURN))
    w, h = size
    pts, p, last, gap = [], seed, seed, 0
    while len(pts) < 300:
        q = p + d * step
        if not (0 <= q[0] < w and 0 <= q[1] < h):
            break
        nb = near(P, q, radius)
        c = nb.mean(0) if len(nb) >= TRACE_MINPIX else None
        mv = None if c is None else c - last
        # 픽셀이 없거나, 앞으로 나아가지 못하거나(선 끝), 급격히 꺾이면 끊김으로 처리
        if c is None or mv @ d < 0.4 * step or unit(mv) @ d < cos_max:
            gap += 1
            if gap > TRACE_MAX_GAP:
                break
            p = q
            continue
        gap = 0
        d = unit(d + unit(mv))  # 방향을 부드럽게 갱신
        p = last = c
        pts.append(c)
    return pts


def trace_lines(m, max_lines=4):
    """버드아이 경계 마스크의 선들을 로봇에서 가까운 것부터 추적.
    각 선은 로봇 쪽 끝에서 먼 쪽으로 정렬된 창 중심점 배열."""
    h, w = m.shape
    ys, xs = np.nonzero(m)
    P = np.stack([xs, ys], 1).astype(np.float64)
    radius = TRACE_RADIUS_CM * PX_PER_CM
    k = int(2 * radius) | 1
    dense = cv2.boxFilter(m.astype(np.float32), -1, (k, k), normalize=False)[ys, xs] >= TRACE_MINPIX
    used = ~dense                     # 주변 픽셀이 적은 점(잡음)은 시작점으로 쓰지 않음
    O = np.array([lat_to_px(0), fwd_to_px(0)])
    lines = []
    while len(lines) < max_lines:
        cand = np.nonzero(~used)[0]
        if len(cand) == 0:
            break
        i = cand[np.argmin(np.hypot(*(P[cand] - O).T))]
        seed = near(P, P[i], radius).mean(0)
        _, _, vt = np.linalg.svd(near(P, seed, 2 * radius) - seed)
        d0 = vt[0]
        pts = trace_from(P, seed, -d0, (w, h))[::-1] + [seed] + trace_from(P, seed, d0, (w, h))
        pts = np.array(pts)
        for p in pts:
            used |= np.hypot(*(P - p).T) < radius * 1.5
        if len(pts) < 4:
            continue
        if np.hypot(*(pts[-1] - O)) < np.hypot(*(pts[0] - O)):
            pts = pts[::-1]
        lines.append(smooth(pts))
    return lines


def classify_line(pts, road):
    """진행 방향 기준 노면이 어느 편인지로 좌/우 선 판별. (side, (왼편, 오른편 노면 비율)).
    nsign = +1 이면 노면이 진행 방향 왼편 (도로 쪽 법선 = left_normal * nsign)."""
    nl = left_normal(tangents(pts))
    off = 1.5 * PX_PER_CM
    h, w = road.shape

    def road_ratio(q):
        q = np.round(q).astype(int)
        ok = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
        return float(road[q[ok, 1], q[ok, 0]].mean()) if ok.any() else 0.0

    rl, rr = road_ratio(pts + nl * off), road_ratio(pts - nl * off)
    return ('right' if rl > rr else 'left'), (rl, rr)


def bend_profile(pts, nsign):
    """선이 처음 방향보다 꺾인 각도(도) 배열: + 는 노면 반대쪽(도로 바깥), - 는 노면 쪽(도로 안쪽).
    nsign: 노면이 진행 방향 왼편이면 +1 (선 이름이 아니라 실제 노면 위치로 판단 -> 이름이 틀려도 맞음)."""
    t = unit(np.gradient(to_cm(pts), axis=0))
    th = np.degrees(np.unwrap(np.arctan2(t[:, 0], t[:, 1])))
    return (th - np.median(th[:3])) * nsign


def branch_index(pts, nsign):
    """선이 도로 바깥쪽으로 BRANCH_DEG 이상 꺾이면 그 꺾임이 시작된 인덱스, 아니면 None."""
    out = bend_profile(pts, nsign)
    over = np.nonzero(out >= BRANCH_DEG)[0]
    if not len(over):
        return None
    before = np.nonzero(out[:over[0]] < BRANCH_START_DEG)[0]
    return int(before[-1]) + 1 if len(before) else 0


def project_to_polyline(q, poly):
    """각 점 q 의 polyline 위 최근접점, 거리, 그리고 최근접점이 polyline 양 끝이 아닌지(마주 보는지)."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.clip(((q[:, None] - a) * ab).sum(-1) / np.maximum((ab ** 2).sum(-1), 1e-9), 0, 1)
    proj = a + t[..., None] * ab
    d = np.linalg.norm(q[:, None] - proj, axis=-1)
    j = d.argmin(1)
    k = np.arange(len(q))
    interior = ~(((j == 0) & (t[k, j] <= 0)) | ((j == len(ab) - 1) & (t[k, j] >= 1)))
    return proj[k, j], d[k, j], interior


def build_center(det, half_w_px, prefer=None):
    """중앙선(버드아이 px). prefer 쪽(없으면 더 길게 보이는 쪽) 안쪽 경계를 기준으로,
    반대쪽 경계와 마주 보는 구간은 두 경계의 가운데(실측 폭), 한쪽만 보이는 구간은 가장 가까운 곳의
    실측 반폭만큼(실측이 전혀 없으면 도로폭/2) 도로 쪽으로 옮긴다. 코너에서 도로가 넓어져도 가운데를 유지."""
    length = {s: np.linalg.norm(np.diff(d['edge'], axis=0), axis=1).sum() for s, d in det.items()}
    sides = sorted(det, key=lambda s: (s != prefer, -length[s]))
    base = det[sides[0]]['edge']
    nroad = left_normal(tangents(base)) * det[sides[0]]['nsign']  # 실제 노면 쪽
    C = base + nroad * half_w_px
    if len(sides) > 1:
        foot, dist, facing = project_to_polyline(base, det[sides[1]]['edge'])
        max_pair = 2 * half_w_px + PAIR_TOL_CM * PX_PER_CM
        ok = facing & (dist < max_pair) & (((foot - base) * nroad).sum(1) > 0)
        if ok.any():
            C[ok] = (base[ok] + foot[ok]) / 2
            paired = np.nonzero(ok)[0]
            for i in np.nonzero(~ok)[0]:  # 짝 없는 점: 가장 가까운 짝의 실측 반폭으로 이어 붙임
                j = paired[np.abs(paired - i).argmin()]
                C[i] = base[i] + nroad[i] * dist[j] / 2
    return smooth(C, 5)


def extend_to_robot(C, bh):
    """중앙선 시작점에서 접선 반대 방향으로 버드아이 하단까지 직선 연장."""
    t0 = unit(C[min(3, len(C) - 1)] - C[0])
    step = TRACE_STEP_CM * PX_PER_CM
    ext = []
    p = C[0]
    for _ in range(30):
        if p[1] >= bh - 1:
            break
        p = p - t0 * step
        ext.append(p)
    return np.r_[np.array(ext[::-1]).reshape(-1, 2), C], len(ext)


def fit_circle(pts):
    """Kasa 원 피팅 -> (중심, 반지름)."""
    x, y = pts.T
    A = np.c_[x, y, np.ones_like(x)]
    D, E, F = np.linalg.lstsq(A, -(x ** 2 + y ** 2), rcond=None)[0]
    c = np.array([-D / 2, -E / 2])
    return c, np.sqrt(max(c @ c - F, 0))


def measure_width(det):
    """좌/우 안쪽 경계가 함께 보이는 가까운 구간(WIDTH_MEASURE_FWD_CM 이내)의 실측 간격(cm)."""
    if len(det) < 2:
        return None
    r, l = det['right']['edge'], det['left']['edge']
    fr, fl = px_to_fwd(r[:, 1]), px_to_fwd(l[:, 1])
    r = r[(fr <= WIDTH_MEASURE_FWD_CM) & (fr >= fl.min()) & (fr <= fl.max())]
    if len(r) < 5:
        return None
    return float(np.median(dist_to_polyline(r, l))) / PX_PER_CM


def snap_width(gap):
    """실측 간격 -> 가장 가까운 맵 도로 (이름, 폭). WIDTH_SNAP_CM 넘게 차이나면 None."""
    name = min(ROAD_WIDTH_CM, key=lambda k: abs(ROAD_WIDTH_CM[k] - gap))
    return (name, ROAD_WIDTH_CM[name]) if abs(ROAD_WIDTH_CM[name] - gap) <= WIDTH_SNAP_CM else None


def clearances(det):
    """로봇 바로 앞(CLEAR_FWD_MAX_CM 이내)의 경계점 중 로봇 왼편/오른편에서 중심에 가장 가까운 점과
    차체 옆면 사이의 여유(cm). 음수면 침범. 선 이름이 아니라 점이 로봇의 어느 편에 있는지로 계산."""
    out = {'left': None, 'right': None}
    if not det:
        return out
    e = to_cm(np.vstack([d['edge'] for d in det.values()]))
    e = e[e[:, 1] <= CLEAR_FWD_MAX_CM]
    if (e[:, 0] < 0).any():
        out['left'] = -ROBOT_HALF_W_CM - float(e[e[:, 0] < 0, 0].max())
    if (e[:, 0] >= 0).any():
        out['right'] = float(e[e[:, 0] >= 0, 0].min()) - ROBOT_HALF_W_CM
    return out


def analyze(und, M, size, road=None, turn_cmd='straight', road_hint=DEFAULT_ROAD,
            lookahead=LOOKAHEAD_CM, pivot_back=0.0, corner_bias=0.0, corner_delay=0.0,
            left_bias_extra=0.0, left_delay=None, keep_gain=0.0):
    """보정 영상 -> 선 검출, 분기 처리, 중앙선/좌우 선, 주행 수치.
    road: 맵 도로 이름(고정) 또는 None(자동: 좌우 선 실측 간격을 맵 도로폭에 맞춤,
          한쪽 선만 보이면 road_hint 유지).
    lookahead: pure pursuit 목표점 거리(cm). pivot_back: 회전 중심이 카메라 바로 아래보다 뒤에 있는
          거리(cm) -> 목표점까지 거리를 그만큼 길게 보고 곡률 계산.
    corner_bias: 코너에서 목표 경로를 바깥쪽으로 옮기는 최대 거리(cm, 0=사용 안 함).
    corner_delay: 앞에 코너가 보이면 목표점을 이만큼(cm) 가깝게 -> 그만큼 더 직진한 뒤 꺾기 시작.
    left_bias_extra / left_delay: 좌회전 코너에서만 쓰는 추가 바깥쪽 보정(cm, 반경 50cm 이하에서 전부)과
          회전 지연(cm, None 이면 corner_delay 그대로). 우회전(시계 방향 주행)은 영향 없음.
    keep_gain: 좌우 선 직접 보정 세기(1/cm², 0=사용 안 함). 중앙선만 따르지 않고, 로봇 바로 앞에서 실제 좌/우 선과의
          여유가 KEEP_MARGIN_CM(코너 안쪽은 내륜차만큼 더)보다 모자라면 그 선 반대쪽으로 곡률을 더한다.
    반환: (결과 dict 또는 실패 시 None, 로그 메시지 목록)."""
    bw, bh = size
    notes = []

    # 노면 경계(= 선 안쪽 경계) 추적 -> 좌/우 판별
    edge, road_mask = edge_masks(und, M, size)
    det = {}
    for pts in trace_lines(edge):
        side, rratio = classify_line(pts, road_mask)
        d = dict(edge=pts, road=rratio, nsign=1 if rratio[0] > rratio[1] else -1)
        if side in det:
            # 급한 코너에서는 앞을 가로지르는 바깥 경계의 방향이 뒤집혀 같은 이름이 나올 수 있음
            # -> 반대쪽 이름이 비어 있으면 더 왼쪽에 있는 선을 왼쪽으로 (가까운 두 선까지 사용)
            other = 'left' if side == 'right' else 'right'
            if other in det or len(det) >= 2:
                continue
            a, b = det[side], d
            if to_cm(a['edge'])[:, 0].mean() > to_cm(b['edge'])[:, 0].mean():
                a, b = b, a
            det = {'left': a, 'right': b}
        else:
            det[side] = d
    if not det:
        notes.append('[lanes] 차선 검출 실패')
        return None, notes
    clear = clearances(det)

    # 분기(교차로): 선이 도로 바깥쪽으로 꺾이면 그쪽 길이 열린 것.
    # 단, 반대쪽 선이 같은 방향(그 선 기준 안쪽)으로 꺾이면 갈림길이 아니라 도로 자체가 도는 코너 -> 그대로 사용.
    # 갈 방향(turn_cmd) 쪽 선만 꺾인 부분까지 따라가고, 나머지 선은 분기 시작 전까지만 사용
    inward = {s: bend_profile(d['edge'], d['nsign']).min() <= -BRANCH_DEG for s, d in det.items()}
    branch_pts, corner = {}, False
    for side, d in det.items():
        b = branch_index(d['edge'], d['nsign'])
        if b is None:
            continue
        if inward.get('left' if side == 'right' else 'right'):
            corner = True
            continue
        branch_pts[side] = d['edge'][b]
        d['branch_fwd'] = px_to_fwd(d['edge'][b][1])
    if corner:
        notes.append('[branch] 양쪽 선이 같은 방향으로 꺾임 -> 갈림길이 아닌 코너 (양쪽 선 모두 사용)')
    turn = turn_cmd
    if turn != 'straight' and turn not in branch_pts:
        notes.append(f'[branch] {turn} 요청 -> 그쪽 분기가 아직 안 보여 직진 유지')
        turn = 'straight'
    opened = ', '.join(f'{"왼쪽" if s == "left" else "오른쪽"} 길(전방 {det[s]["branch_fwd"]:.0f}cm부터)'
                       for s in branch_pts) or '없음'
    notes.append(f'[branch] 열린 분기: {opened} -> 주행 방향 {turn}')
    for side in list(det):
        if side in branch_pts and side != turn:
            b = branch_index(det[side]['edge'], det[side]['nsign'])
            if b < 3:
                del det[side]
            else:
                det[side]['edge'] = det[side]['edge'][:b]
    if not det:
        notes.append('[lanes] 분기 전 구간이 너무 짧음')
        return None, notes

    # 도로폭: 고정 지정이 없으면 실측 간격을 맵 도로폭에 맞춤
    gap = measure_width(det)
    if road is None:
        snapped = snap_width(gap) if gap is not None else None
        if snapped:
            road = snapped[0]
            notes.append(f'[width] 실측 간격 {gap:.1f}cm -> {road} ({ROAD_WIDTH_CM[road]:g}cm)')
        else:
            road = road_hint
            why = '한쪽 선만 보임' if gap is None else f'실측 간격 {gap:.1f}cm 가 맵과 안 맞음'
            notes.append(f'[width] {why} -> {road} ({ROAD_WIDTH_CM[road]:g}cm) 유지')
    road_w = ROAD_WIDTH_CM[road]
    hw = road_w / 2 * PX_PER_CM

    # 중앙선 = 마주 보는 좌/우 안쪽 경계의 가운데 (한쪽만 보이면 실측 반폭/도로폭/2 만큼 옮김),
    # 좌/우 선 = 중앙선 ±도로폭/2 (곡선 평행 이동)
    C_obs = build_center(det, hw, prefer=None if turn == 'straight' else turn)
    C, n_ext = extend_to_robot(C_obs, bh)
    nl = left_normal(tangents(C))
    lines = {'left': C + nl * hw, 'center': C, 'right': C - nl * hw}

    # 주행용 수치 (cm, 로봇=카메라 바로 아래 지면 원점, +lat 오른쪽, +fwd 전방)
    Ccm = to_cm(C)
    tcm = unit(np.gradient(Ccm, axis=0))
    Cobs_cm = to_cm(C_obs)
    R, bend = None, None
    if len(Cobs_cm) >= 6:
        cc, R = fit_circle(Cobs_cm)
        t_mid = tcm[n_ext + len(C_obs) // 2]
        p_mid = Cobs_cm[len(C_obs) // 2]
        # (lat, fwd) 는 x 오른쪽/y 위쪽 좌표계 -> 외적이 양수면 원 중심이 진행방향 왼편
        bend = '왼쪽' if t_mid[0] * (cc[1] - p_mid[1]) - t_mid[1] * (cc[0] - p_mid[0]) > 0 else '오른쪽'
        path_desc = '직선' if R > 100 else f'{bend} 코너, 중앙선 반경 약 {R:.0f}cm'
    else:
        path_desc = '판단 불가 (검출 구간 짧음)'

    # 코너 바깥쪽 보정: 앞(카메라)이 중앙선을 따라가면 차체 뒤가 내륜차(약 축간거리²/2R)만큼 안쪽 선을 밟으므로
    # 목표 경로를 그 절반만큼 코너 바깥쪽으로 옮겨 차체 전체가 도로 가운데를 지나게 함 (= 조금 늦게 꺾음)
    bias = 0.0
    is_corner = R is not None and R < 150
    if corner_bias > 0 and is_corner:
        bias = min(corner_bias, WHEELBASE_CM ** 2 / (4 * R))
    if bend == '왼쪽' and is_corner:
        # 좌회전 전용: 좌회전에서만 차체가 안쪽 선을 밟는 비대칭 보정 (급한 코너일수록 크게, 우회전 값은 그대로)
        bias += left_bias_extra * min(1.0, 50.0 / R)
        if left_delay is not None:
            corner_delay = left_delay
    Ct = C + nl * (bias * PX_PER_CM * (1 if bend == '오른쪽' else -1)) if bias else C
    Ctcm = to_cm(Ct)

    dist = np.hypot(*Ccm.T)
    i0 = int(dist.argmin())
    heading = np.degrees(np.arctan2(tcm[i0, 0], tcm[i0, 1]))
    dist_t = np.hypot(*Ctcm.T)
    look = lookahead - corner_delay if corner_delay and is_corner else lookahead
    ahead = np.nonzero((np.arange(len(Ct)) >= i0) & (dist_t >= look))[0]
    ia = int(ahead[0]) if len(ahead) else len(Ct) - 1
    la = Ctcm[ia]
    lp = la + [0.0, pivot_back]                # 회전 중심 기준 목표점
    curv_pp = 2 * lp[0] / max(lp @ lp, 1e-6)  # pure pursuit 곡률 (+ 오른쪽)

    # 좌우 선 직접 보정: 실제 선과의 여유가 모자란 쪽에서 밀어냄 (왼쪽이 모자라면 오른쪽(+)으로)
    keep = 0.0
    if keep_gain > 0:
        need = {'left': KEEP_MARGIN_CM, 'right': KEEP_MARGIN_CM}
        if is_corner and bend is not None:
            need['right' if bend == '오른쪽' else 'left'] += KEEP_OFFTRACK_FRAC * WHEELBASE_CM ** 2 / (2 * R)
        short = {s: 0.0 if clear[s] is None else max(0.0, need[s] - clear[s]) for s in need}
        keep = keep_gain * (short['left'] - short['right'])
        curv_pp += keep

    return dict(det=det, lines=lines, C=C, Ct=Ct, bias=bias, keep=keep, Ccm=Ccm, i0=i0, ia=ia, heading=heading,
                lookahead=la,
                curv_pp=curv_pp, path_desc=path_desc, path_R=R, path_bend=bend, branch_pts=branch_pts,
                turn=turn, road=road, road_w=road_w, gap=gap, clear=clear), notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', action='store_true')
    ap.add_argument('--input', default='raw_640.jpg')
    ap.add_argument('--calib', default='calib_checkerboard.npz')
    ap.add_argument('--road', choices=('auto', *ROAD_WIDTH_CM), default='auto',
                    help='auto: 좌우 선 실측 간격을 맵 도로폭에 맞춤')
    ap.add_argument('--turn', choices=('straight', 'left', 'right'), default='straight',
                    help='교차로(분기)에서 갈 방향. 그쪽 분기가 안 보이면 직진')
    ap.add_argument('--setup', action='store_true',
                    help='현재 영상(직선 도로, --road 로 도로 지정)으로 지면 평면을 다시 추정해 ground.npz 저장')
    args = ap.parse_args()
    if args.capture:
        capture(args.input)
    out_prefix = Path(args.input).stem + '_'

    img = cv2.imread(args.input)
    h, w = img.shape[:2]

    # 1) 체커보드 캘리브레이션 값으로 어안 왜곡 보정
    c = np.load(args.calib)
    assert tuple(c['size']) == (w, h), f'캘리브레이션 해상도 {tuple(c["size"])} != 입력 {(w, h)}'
    und, Kn = undistort(img, c['K'], c['D'])

    # 2) 지면 평면 (카메라 높이/각도) -> cm 버드아이
    if args.setup:
        lp, rp = line_points(yellow_mask(und))
        g = estimate_ground(lp, rp, Kn, ROAD_WIDTH_CM[DEFAULT_ROAD if args.road == 'auto' else args.road])
        np.savez('ground.npz', **g)
    g = dict(np.load('ground.npz'))
    print(f'[ground] 카메라 높이 H={float(g["H"]):.1f}cm, 숙인 각도(pitch)={float(g["pitch"]):.1f}°')
    M, size, fwd_min = bev_transform(g, Kn, h)
    bev = cv2.warpPerspective(und, M, size)
    bw, bh = size

    # 3~5) 선 검출, 분기 처리, 중앙선/좌우 선, 주행 수치
    res, notes = analyze(und, M, size, None if args.road == 'auto' else args.road, args.turn)
    for n in notes:
        print(n)
    if res is None:
        return
    det, lines, C, Ccm = res['det'], res['lines'], res['C'], res['Ccm']
    i0, ia, heading, la, curv_pp = res['i0'], res['ia'], res['heading'], res['lookahead'], res['curv_pp']
    branch_pts, turn, path_desc = res['branch_pts'], res['turn'], res['path_desc']
    road_w = res['road_w']
    half_w = road_w / 2

    print(f'[{res["road"]}] 도로폭(선 안쪽끼리) {road_w:g}cm -> 왼쪽-중앙-오른쪽 {half_w:g}cm 간격')
    print(f'[path  ] {path_desc}')
    print(f'[center] 로봇에서 가장 가까운 중앙선 점 ({Ccm[i0, 0]:+.1f}, {Ccm[i0, 1]:.1f})cm '
          f'-> 좌우 이탈 {Ccm[i0, 0]:+.1f}cm, 진행방향 대비 {heading:+.1f}°')
    print(f'[target] {LOOKAHEAD_CM}cm 앞 목표점 ({la[0]:+.1f}, {la[1]:.1f})cm, '
          f'pure pursuit 회전반경 {abs(1 / curv_pp) if abs(curv_pp) > 1e-4 else np.inf:.0f}cm '
          f'({"오른쪽" if curv_pp > 0 else "왼쪽"})')
    for side in ('left', 'right'):
        if side not in det:
            print(f'[{side:6s}] 미검출 -> 반대쪽 선에서 도로폭으로 추정')
            continue
        d = det[side]
        e_cm = to_cm(d['edge'])
        length = np.linalg.norm(np.diff(e_cm, axis=0), axis=1).sum()
        diff = dist_to_polyline(d['edge'], lines[side]) / PX_PER_CM
        print(f'[{side:6s}] 검출 길이 {length:.0f}cm ({e_cm[0, 0]:+.0f},{e_cm[0, 1]:.0f})->'
              f'({e_cm[-1, 0]:+.0f},{e_cm[-1, 1]:.0f})cm, '
              f'모델 선과 거리 중앙값 {np.median(diff):.1f}cm (최대 {diff.max():.1f}), '
              f'노면 비율 왼편/오른편 {d["road"][0]:.2f}/{d["road"][1]:.2f}')
    if len(det) == 2:
        gap = dist_to_polyline(det['right']['edge'], det['left']['edge']) / PX_PER_CM
        gap = gap[gap < road_w * 1.5]
        if len(gap):
            print(f'[check ] 실측 안쪽 경계 간격 {np.median(gap):.1f}cm '
                  f'({gap.min():.1f}~{gap.max():.1f}, 설정 {road_w:.0f}cm)')

    # 6) 그리기
    out = bev.copy()
    for a in range(BEV_LAT_CM[0], BEV_LAT_CM[1] + 1, 10):          # 10cm 격자
        cv2.line(out, (int(lat_to_px(a)), 0), (int(lat_to_px(a)), bh), (200, 200, 200), 1)
    for b in range(int(np.ceil(fwd_min / 10) * 10), BEV_FWD_MAX_CM + 1, 10):
        v = fwd_to_px(b)
        cv2.line(out, (0, v), (bw, v), (200, 200, 200), 1)
        cv2.putText(out, f'{b}', (2, v - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    c0 = int(lat_to_px(0))
    cv2.line(out, (c0, 0), (c0, bh), (255, 255, 0), 1)              # 로봇(카메라) 중심
    for name, ln in lines.items():
        cv2.polylines(out, [ln.astype(np.int32)], False, COLORS[name], 3)
    for d in det.values():                                          # 실제 검출된 안쪽 경계점
        for p in d['edge']:
            cv2.circle(out, tuple(np.int32(p)), 2, (255, 255, 255), -1)
    cv2.circle(out, tuple(np.int32(C[ia])), 6, (0, 255, 255), 2)    # 목표점
    for p in branch_pts.values():                                   # 분기 시작점
        cv2.drawMarker(out, tuple(np.int32(p)), (0, 165, 255), cv2.MARKER_TILTED_CROSS, 14, 3)
    cv2.imwrite(out_prefix + 'birdeye.jpg', out)

    ov = und.copy()                                                 # 보정 영상에 역투영
    Mi = np.linalg.inv(M)
    for name, ln in lines.items():
        back = cv2.perspectiveTransform(ln.reshape(-1, 1, 2).astype(np.float32), Mi)
        cv2.polylines(ov, [back.reshape(-1, 2).astype(np.int32)], False, COLORS[name], 3)

    def lab(im, t):
        im = cv2.resize(im, (w, h)) if im.shape[:2] != (h, w) else im.copy()
        cv2.putText(im, t, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return im
    bev_pad = np.zeros((h, w, 3), np.uint8)
    sc = min(w / bw, h / bh)
    r = cv2.resize(out, (int(bw * sc), int(bh * sc)))
    bev_pad[:r.shape[0], (w - r.shape[1]) // 2:(w - r.shape[1]) // 2 + r.shape[1]] = r
    grid = np.vstack([np.hstack([lab(img, '1 input'), lab(und, '2 undistorted')]),
                      np.hstack([lab(ov, '3 left / center / right'),
                                 lab(bev_pad, f'4 bird-eye (turn={turn})')])])
    cv2.imwrite(out_prefix + 'summary.jpg', grid)


if __name__ == '__main__':
    main()
