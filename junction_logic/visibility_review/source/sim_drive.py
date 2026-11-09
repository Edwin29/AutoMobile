#!/usr/bin/env python3
"""모의 주행: 실측 맵 위에서 로봇 카메라 영상을 만들어 실제 주행 코드(lane_pipeline.analyze + lane_drive 조향 로직)를
그대로 돌리고, 로봇 움직임은 주행 기록으로 추정한 모델(sysid.py)로 계산. 로봇은 움직이지 않는다.

  맵    : 팀원 디지털 트윈 도면 track_layout.json (github gawbi/track-digital-twin b4e2433, 줄자 실측 기반).
          블록 62.75x65cm(코너 반경 10.5), 가운데 도로 남북 21 / 동서 18.5cm, 바깥 순환로 26cm(동쪽만 24cm),
          바깥 코너 반경 32.5cm, 횡단보도·분홍 점선·빨간 선·파란 원 포함. 선 = 노면 바로 바깥 노란 선(1cm).
          사용자 실측 사진의 129/133/37cm 는 분홍 점선 사이 거리로 이 도면과 맞음.
  카메라: 캘리브레이션/지면 추정(calib_checkerboard.npz, ground.npz)의 역변환으로 보정 영상(und)을 그림
  움직임: 명령 지연 3프레임, 회전 = 좌/우 따로 맞춘 데드밴드 선형 모델 + 직진 쏠림, 전진 = KV x 평균 출력
  침범  : 차체(카메라 기준 좌우 ±8cm, 앞 4cm ~ 뒤 19cm) 옆면과 선 안쪽 경계 사이 실제 거리 (음수 = 침범)

사용법:
    python3 sim_drive.py --ccw                       # 지금 lane_drive 기본값으로 반시계 한 바퀴
    python3 sim_drive.py --ccw --left-bias-extra 0.5 --plot out.png
    python3 sim_drive.py                             # 시계 방향 (기존 확정 설정 확인용)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import lane_drive as D
import lane_pipeline as L

HERE = Path(__file__).resolve().parent

# ---- 맵 (cm, 원점 = 매트 중앙, x 동쪽 / y 북쪽, track_layout.json 과 같은 좌표) ----
OUTER = (-99.25, 97.25, -100.25, 100.25)   # 바깥 순환로 바깥 경계 (x0, x1, y0, y1): 블록 끝 ±73.25/±74.25 + 26 (동쪽 24)
OUTER_R = 32.5
BLOCK_R = 10.5
BLOCKS = [(-73.25, -10.5, 9.25, 74.25), (10.5, 73.25, 9.25, 74.25),     # (x0, x1, y0, y1)
          (-73.25, -10.5, -74.25, -9.25), (10.5, 73.25, -74.25, -9.25)]
MAT = (-111.75, 111.75, -114.0, 114.0)     # 초록 인쇄 영역 (서 12.5 / 동 14.5 / 남북 13.75cm 띠)
MAT_R, BORDER = 37.0, 6.0
# 횡단보도: (중심 x, y, 도로 방향) - 흰 줄 6개(폭 2, 간격 1.15, 길이 12cm)가 도로를 가로질러 늘어섬
CROSSWALKS = [(0.0, 25.75, 'ns'), (0.0, -25.75, 'ns'), (-27.0, 0.0, 'ew'), (27.0, 0.0, 'ew'),
              (-21.5, 87.25, 'ew'), (85.25, 20.25, 'ns'), (21.5, -87.25, 'ew'), (-86.25, -20.25, 'ns')]
ROAD_MARKS = [('pink', (-66.25, 97.25), (-66.25, 77.25)), ('pink', (-66.25, -77.25), (-66.25, -97.25)),
              ('pink', (66.25, -77.25), (66.25, -97.25)), ('pink', (-96.25, 65.0), (-76.25, 65.0)),
              ('pink', (-96.25, -65.0), (-76.25, -65.0)), ('pink', (75.25, 65.0), (95.25, 65.0)),
              ('pink', (75.25, -65.0), (95.25, -65.0)),
              ('red', (33.0, 95.25), (33.0, 79.25)), ('red', (62.0, 95.25), (62.0, 79.25))]
PROPS = [(x, y) for x in (-63.25, -20.5, 20.5, 63.25) for y in (-64.25, -19.25, 19.25, 64.25)] + \
        [(20.5, 107.1), (104.5, -19.25), (-20.5, -107.1), (-105.5, 19.25)]
PROP_D = 10.5
YELLOW_W = 1.0
EXT, PPC = 125.0, 5        # 그림 범위 ±EXT cm, cm 당 픽셀
COL = {'road': (61, 77, 47), 'green': (125, 215, 95), 'yellow': (82, 218, 165), 'stripe': (215, 220, 220),
       'pink': (190, 120, 200), 'red': (50, 50, 200), 'blue': (225, 165, 30),
       'white': (215, 220, 220), 'floor': (150, 155, 160), 'sky': (170, 175, 180)}

# ---- 로봇 ----
BODY_LAT = L.ROBOT_HALF_W_CM
BODY_FWD = (-19.0, 4.0)    # lane_eval 과 같은 차체 앞뒤 범위
D_CAM = 6.2                # 회전 중심 -> 카메라 거리 (sysid)
FPS = 14.4                 # 실제 주행 평균
KV = 16.0                  # cm/s per 평균 출력 (코너 사이 시간으로 추정, VO 전진은 직선에서 과소 추정)
LAG = 3
YAW = dict(kr=86.4, kl=90.8, dead=0.010, c=0.52)  # °/s, d=(L-R)/2 (sysid 데드밴드 모델)


def rrect_sdf(X, Y, x0, x1, y0, y1, r):
    """둥근 사각형 부호 거리 (안쪽 음수)."""
    cx, cy, hx, hy = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    qx, qy = np.abs(X - cx) - (hx - r), np.abs(Y - cy) - (hy - r)
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r


class World:
    def __init__(self):
        n = int(2 * EXT * PPC)
        c = (np.arange(n) + 0.5) / PPC - EXT
        X, Y = np.meshgrid(c, -c)  # 행은 위쪽(y 큰 쪽)부터
        sd_out = rrect_sdf(X, Y, *OUTER, OUTER_R)
        sd_blk = np.min([rrect_sdf(X, Y, *b, BLOCK_R) for b in BLOCKS], axis=0)
        self.sd = np.minimum(-sd_out, sd_blk).astype(np.float32)  # 노면 안쪽 양수 = 선까지 거리
        img = np.empty((n, n, 3), np.uint8)
        img[:] = COL['floor']
        x0, x1, y0, y1 = MAT
        img[rrect_sdf(X, Y, x0 - BORDER, x1 + BORDER, y0 - BORDER, y1 + BORDER, MAT_R + BORDER) < 0] = COL['white']
        img[rrect_sdf(X, Y, *MAT, MAT_R) < 0] = COL['green']
        for px, py in PROPS:
            img[np.hypot(X - px, Y - py) < PROP_D / 2] = COL['blue']
        img[(self.sd < 0) & (self.sd > -YELLOW_W)] = COL['yellow']
        road = self.sd >= 0
        img[road] = COL['road']
        for cx, cy, d in CROSSWALKS:  # 흰 줄은 도로를 가로질러 6개, 각 줄은 도로 방향으로 12cm
            along, across = (Y - cy, X - cx) if d == 'ns' else (X - cx, Y - cy)
            k = (across + 17.75 / 2) / 3.15
            stripe = (np.abs(along) <= 6) & (k >= 0) & (k < 6) & ((k % 1) * 3.15 < 2.0)
            img[stripe & road] = COL['stripe']
        for kind, (ax, ay), (bx, by) in ROAD_MARKS:
            p = cv2.line(np.zeros((n, n), np.uint8), self.px_int(ax, ay), self.px_int(bx, by), 1,
                         int(0.6 * PPC) if kind == 'pink' else PPC)
            if kind == 'pink':  # 점선 (2cm 칠, 2cm 비움)
                t = np.hypot(X - ax, Y - ay)
                p &= ((t % 4) < 2).astype(np.uint8)
            img[(p > 0) & road] = COL[kind]
        self.img = cv2.GaussianBlur(img, (0, 0), 0.6 * PPC / 2)
        # 카메라 보정 영상 픽셀 -> 지면 (lat, fwd)
        c = np.load(HERE / 'calib_checkerboard.npz')
        g = dict(np.load(HERE / 'ground.npz'))
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(c['K'], c['D'], (640, 480), np.eye(3),
                                                                      balance=0.0)
        self.M, self.size, _ = L.bev_transform(g, newK, 480)
        Hi = np.linalg.inv(L.ground_to_image(g, newK))
        u, v = np.meshgrid(np.arange(640, dtype=np.float64), np.arange(480, dtype=np.float64))
        p = np.tensordot(Hi, np.stack([u, v, np.ones_like(u)]), 1)
        ok = (p[2] > 1e-9)
        lat, fwd = p[0] / np.where(ok, p[2], 1), p[1] / np.where(ok, p[2], 1)
        self.sky = ~ok | (fwd > 250) | (fwd < 0)
        self.lat, self.fwd = lat.astype(np.float32), fwd.astype(np.float32)

    def to_px(self, X, Y):
        return ((X + EXT) * PPC - 0.5).astype(np.float32), ((EXT - Y) * PPC - 0.5).astype(np.float32)

    @staticmethod
    def px_int(x, y):
        return int(round((x + EXT) * PPC - 0.5)), int(round((EXT - y) * PPC - 0.5))

    def render(self, pose, rng=None):
        x, y, yaw = pose
        s, c = np.sin(yaw), np.cos(yaw)
        X = x + self.lat * c + self.fwd * s       # 오른쪽 (cos, -sin), 전방 (sin, cos)
        Y = y - self.lat * s + self.fwd * c
        mx, my = self.to_px(X, Y)
        und = cv2.remap(self.img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=COL['floor'])
        und[self.sky] = COL['sky']
        und = cv2.GaussianBlur(und, (0, 0), 1.0)
        if rng is not None:
            und = np.clip(und + rng.normal(0, 4, und.shape), 0, 255).astype(np.uint8)
        return und

    def sd_at(self, X, Y):
        mx, my = self.to_px(np.asarray(X, np.float64), np.asarray(Y, np.float64))
        return cv2.remap(self.sd, mx.reshape(1, -1), my.reshape(1, -1), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=-50).ravel()

    def body_clear(self, pose):
        """차체 왼쪽/오른쪽 옆면의 선까지 최소 여유(cm, 음수=침범)."""
        x, y, yaw = pose
        s, c = np.sin(yaw), np.cos(yaw)
        f = np.linspace(*BODY_FWD, 24)
        out = []
        for lat in (-BODY_LAT, BODY_LAT):
            out.append(float(self.sd_at(x + lat * c + f * s, y - lat * s + f * c).min()))
        return out


def yaw_rate(d):
    """d=(L-R)/2 -> °/s (+오른쪽)."""
    k = YAW
    return k['kr'] * max(d - k['dead'], 0) + k['kl'] * min(d + k['dead'], 0) + k['c']


# 코너 접근: 이름 -> (코너 블록 곡선 시작점 x, y, 진행 방향 yaw도). yaw 0 = 북, 90 = 동 (+ 오른쪽 회전)
APPROACH = {
    'ccw-E': (85.25, 63.75, 0.0), 'ccw-N': (-62.75, 87.25, -90.0),
    'ccw-W': (-86.25, -63.75, 180.0), 'ccw-S': (62.75, -87.25, 90.0),
    'cw-E': (85.25, -63.75, 180.0), 'cw-S': (-62.75, -87.25, -90.0),
    'cw-W': (-86.25, 63.75, 0.0), 'cw-N': (62.75, 87.25, 90.0),
}


def approach_pose(name, before, off=0.0, dyaw=0.0):
    """코너 곡선 시작 before cm 앞, 도로 중심에서 오른쪽으로 off cm, 진행 방향에서 dyaw도 돌린 자세 (x, y, yaw도)."""
    x, y, yaw = APPROACH[name]
    r = np.radians(yaw)
    f, rt = np.array([np.sin(r), np.cos(r)]), np.array([np.cos(r), -np.sin(r)])
    p = np.array([x, y]) - before * f + off * rt
    return float(p[0]), float(p[1]), yaw + dyaw


START = {  # 실제 출발 위치 (첫 프레임 정합, match_start.py): (x, y, yaw도)
    'ccw': approach_pose('ccw-E', 22.0),    # 동쪽 24cm 도로를 북쪽으로, 북동 코너(좌회전) 직전
    'cw': approach_pose('cw-E', 22.0),
}


def simulate(args, world, start=None, seed=None, model=None, max_time=200.0, record_frames=False):
    """lane_drive.main 의 조향 루프를 그대로 따라 한 바퀴. 반환: 프레임별 기록 list(dict)."""
    rng = np.random.default_rng(seed) if seed is not None else None
    model = model or {}
    kv = KV * model.get('kv_scale', 1.0)
    yaw_scale = {'r': model.get('kr_scale', 1.0), 'l': model.get('kl_scale', 1.0)}
    yaw_noise = model.get('yaw_noise', 0.0)
    x, y, yaw_d = start or START['ccw' if args.ccw else 'cw']
    pose = np.array([x, y, np.radians(yaw_d)])
    dt = 1.0 / FPS
    road_hint = args.start_road
    lost, curv_f = 0, 0.0
    prev_la, jump_cnt = None, 0
    target, out, good_target = (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)
    in_corner, corner_cnt, corners_done, seg_label = False, 0, 0, 'S0'
    corner_bend, corner_t0 = None, 0.0
    bend_votes, lap_bend = [], None
    start_phase = args.start_corner_delay > 0
    queue = [(0.0, 0.0)] * (LAG - 1)
    noise_w = 0.0
    rows = []
    reason = '시간 종료'
    for frame in range(int(max_time * FPS)):
        t = frame * dt
        und = world.render(pose, rng)
        if start_phase and not in_corner and t > D.START_WINDOW_S:
            start_phase = False
        start_d = args.start_corner_delay if start_phase else 0.0
        res, notes = L.analyze(und, world.M, world.size, None, 'straight', road_hint, args.lookahead,
                               args.pivot_back, args.corner_bias, max(args.corner_delay, start_d),
                               args.left_bias_extra, max(args.left_corner_delay, start_d),
                               args.line_keep_gain, args.robust)  # 방향 통합: 항상 적용
        hold = False
        if args.robust and res is not None:
            la0 = res['lookahead'][0]
            straight = not in_corner and (res['path_R'] is None or res['path_R'] >= D.CORNER_EXIT_R_CM)
            if straight and prev_la is not None and abs(la0 - prev_la) > D.JUMP_CM and jump_cnt < D.JUMP_HOLD_FRAMES:
                hold = True
                jump_cnt += 1
            else:
                prev_la, jump_cnt = la0, 0
        fixed_now = args.corner_mode == 'fixed' and in_corner
        if hold:
            pass
        elif res is None:
            lost += 1
            quality_lost = any(n.startswith('[quality] 인식') for n in notes)
            if not fixed_now and lost >= (D.QUALITY_STOP_FRAMES if quality_lost else D.LOST_STOP_FRAMES):
                reason = f'차선을 {lost}프레임 연속 놓침'
                break
            if quality_lost:
                target = (good_target[0] * D.QUALITY_COAST, good_target[1] * D.QUALITY_COAST)
            elif lost > D.LOST_HOLD_FRAMES:
                target = (0.0, 0.0)
        else:
            lost = 0
            road_hint = res['road']
            curv_f = args.smooth * curv_f + (1 - args.smooth) * res['curv_pp']
            target = good_target = D.wheel_speeds(curv_f, args.speed, args.track, args.corner_slow)
        if fixed_now:
            target = D.fixed_corner_target(corner_bend, args.corner_outer or args.speed, args.corner_ratio)
        out = D.slew(out, target, D.ACCEL_PER_S * dt)

        # 코너 세기 (lane_drive 와 같음)
        R = None if res is None else res['path_R']
        if not in_corner:
            if R is not None and R < D.CORNER_ENTER_R_CM:
                corner_cnt += 1
                bend_votes.append(res['path_bend'])
            else:
                corner_cnt, bend_votes = 0, []
            if corner_cnt >= D.CORNER_ENTER_FRAMES:
                bend = max(set(bend_votes), key=bend_votes.count)
                if not (lap_bend is not None and bend != lap_bend):
                    lap_bend = lap_bend or bend
                    in_corner = True
                    corner_bend, corner_t0 = bend, t
                    seg_label = f'C{corners_done + 1}'
                corner_cnt, bend_votes = 0, []
        elif args.corner_mode == 'fixed':
            el = t - corner_t0
            straight = R is not None and R >= D.CORNER_EXIT_R_CM and el >= args.corner_min_s
            corner_cnt = corner_cnt + 1 if straight else 0
            if el >= args.corner_max_s and corner_cnt < D.CORNER_FIXED_EXIT_FRAMES:
                corner_cnt = D.CORNER_FIXED_EXIT_FRAMES
        else:
            corner_cnt = corner_cnt + 1 if R is None or R >= D.CORNER_EXIT_R_CM else 0
        done = False
        exit_frames = D.CORNER_FIXED_EXIT_FRAMES if args.corner_mode == 'fixed' else D.CORNER_EXIT_FRAMES
        if in_corner and corner_cnt >= exit_frames:
            in_corner, corner_cnt = False, 0
            corners_done += 1
            seg_label = f'S{corners_done}'
            start_phase = False
            if args.corners and corners_done >= args.corners:
                reason = f'코너 {corners_done}개 통과'
                done = True

        cl, cr = world.body_clear(pose)
        rows.append(dict(t=t, frame=frame, x=pose[0], y=pose[1], yaw=np.degrees(pose[2]), seg=seg_label,
                         left=out[0], right=out[1], curv=curv_f, ok=res is not None, clear_l=cl, clear_r=cr,
                         front_l=None if res is None or res['clear']['left'] is None else res['clear']['left'],
                         front_r=None if res is None or res['clear']['right'] is None else res['clear']['right'],
                         bias=0.0 if res is None else res['bias'], keep=0.0 if res is None else res['keep'],
                         R=R, und=und if record_frames else None))
        if done:
            break
        if world.sd_at([pose[0]], [pose[1]])[0] < -12:
            reason = '트랙 이탈'
            break

        # 로봇 움직임 (명령 지연 LAG 프레임)
        queue.append(out)
        lw, rw = queue.pop(0)
        s, d = (lw + rw) / 2, (lw - rw) / 2
        w = yaw_rate(d)
        w *= yaw_scale['r'] if w > 0 else yaw_scale['l']
        if yaw_noise:
            noise_w = 0.8 * noise_w + rng.normal(0, yaw_noise * 0.6)
            w += noise_w
        v = kv * s
        f = np.array([np.sin(pose[2]), np.cos(pose[2])])
        cen = pose[:2] - D_CAM * f
        pose[2] += np.radians(w) * dt
        f2 = np.array([np.sin(pose[2]), np.cos(pose[2])])
        cen = cen + v * dt * (f + f2) / 2
        pose[:2] = cen + D_CAM * f2
    return rows, reason


def summarize(rows, ccw):
    """구간별 (프레임 수, 안쪽 침범 %, 안쪽 최소, 바깥 침범 %, 바깥 최소). 반시계: 안쪽 = 왼쪽."""
    segs = {}
    for r in rows:
        segs.setdefault(r['seg'], []).append(r)
    segs['ALL'] = rows
    out = {}
    for k, v in segs.items():
        cl = np.array([r['clear_l'] for r in v])
        cr = np.array([r['clear_r'] for r in v])
        inner, outer = (cl, cr) if ccw else (cr, cl)
        out[k] = dict(n=len(v), in_pct=float((inner < 0).mean() * 100), in_min=float(inner.min()),
                      out_pct=float((outer < 0).mean() * 100), out_min=float(outer.min()))
    return out


def print_summary(summ, title=''):
    if title:
        print(title)
    print('  구간  프레임 |  안쪽 침범%/최소  |  바깥 침범%/최소')
    for k, s in summ.items():
        print(f'  {k:4s} {s["n"]:6d} |  {s["in_pct"]:5.0f}%  {s["in_min"]:+5.1f}cm |  {s["out_pct"]:5.0f}%  '
              f'{s["out_min"]:+5.1f}cm')


def plot(rows, world, path, title=''):
    """맵 위 차체 궤적 + 시간별 여유 그래프."""
    sc = 3
    n = int(2 * EXT * sc)
    m = cv2.resize(world.img, (n, n), interpolation=cv2.INTER_AREA)
    for k, r in enumerate(rows):
        if k % 3:
            continue
        yaw = np.radians(r['yaw'])
        s, c = np.sin(yaw), np.cos(yaw)
        pts = []
        for lat, f in ((-BODY_LAT, BODY_FWD[1]), (BODY_LAT, BODY_FWD[1]), (BODY_LAT, BODY_FWD[0]),
                       (-BODY_LAT, BODY_FWD[0])):
            X, Y = r['x'] + lat * c + f * s, r['y'] - lat * s + f * c
            pts.append(((X + EXT) * sc, (EXT - Y) * sc))
        bad = min(r['clear_l'], r['clear_r']) < 0
        cv2.polylines(m, [np.int32(pts)], True, (0, 0, 255) if bad else (255, 200, 0), 1, cv2.LINE_AA)
    for r in rows:
        cv2.circle(m, (int((r['x'] + EXT) * sc), int((EXT - r['y']) * sc)), 1, (255, 255, 255), -1)
    cv2.putText(m, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(m, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    # 여유 그래프
    gw, gh = 640, n
    g = np.full((gh, gw, 3), 255, np.uint8)
    T = max(r['t'] for r in rows) or 1
    def px(t, v):
        return int(40 + (t / T) * (gw - 60)), int(gh / 2 - v * (gh / 2 - 30) / 10)
    cv2.line(g, px(0, 0), px(T, 0), (0, 0, 200), 1)
    for v in (-5, 5, 10):
        cv2.line(g, px(0, v), px(T, v), (220, 220, 220), 1)
        cv2.putText(g, f'{v:+d}', (4, px(0, v)[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
    prev = None
    for r in rows:
        if r['seg'] != prev:
            cv2.putText(g, r['seg'], (px(r['t'], 0)[0], gh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
            if r['seg'].startswith('C'):
                cv2.line(g, (px(r['t'], 0)[0], 20), (px(r['t'], 0)[0], gh - 20), (200, 200, 240), 1)
            prev = r['seg']
    for key, col in (('clear_l', (200, 100, 30)), ('clear_r', (0, 140, 255))):
        pts = np.int32([px(r['t'], max(-10, min(10, r[key]))) for r in rows])
        cv2.polylines(g, [pts], False, col, 2, cv2.LINE_AA)
    cv2.putText(g, 'body clearance [cm]  blue=left  orange=right', (40, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.hstack([m, g]))


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start-road', choices=L.ROAD_WIDTH_CM, default=L.DEFAULT_ROAD)
    ap.add_argument('--speed', type=float, default=0.3)
    ap.add_argument('--corners', type=int, default=4)
    ap.add_argument('--lookahead', type=float, default=D.LOOKAHEAD_CM)
    ap.add_argument('--track', type=float, default=D.WHEEL_TRACK_CM)
    ap.add_argument('--smooth', type=float, default=D.STEER_SMOOTH)
    ap.add_argument('--pivot-back', type=float, default=D.PIVOT_BACK_CM)
    ap.add_argument('--corner-slow', type=float, default=D.CORNER_SLOW_CM)
    ap.add_argument('--corner-delay', type=float, default=D.CORNER_DELAY_CM)
    ap.add_argument('--start-corner-delay', type=float, default=D.START_CORNER_DELAY_CM)
    ap.add_argument('--ccw', action='store_true', help='반시계 출발 위치와 안쪽/바깥 표시에만 사용 (조향은 방향 통합)')
    ap.add_argument('--left-bias-extra', type=float, default=D.LEFT_BIAS_EXTRA_CM)
    ap.add_argument('--left-corner-delay', type=float, default=D.LEFT_CORNER_DELAY_CM)
    ap.add_argument('--line-keep-gain', type=float, default=D.LINE_KEEP_GAIN)
    ap.add_argument('--corner-bias', type=float, default=D.CORNER_BIAS_CM)
    ap.add_argument('--robust', action='store_true')
    # 코너 고정 조향 (2026-11-05): 코너 진입 판정 후에는 차선 대신 정해진 좌/우 출력으로 돎
    ap.add_argument('--corner-mode', choices=('line', 'fixed'), default=D.CORNER_MODE_DEFAULT,
                    help='코너에서 line=차선 추종, fixed=고정 조향 (C 모델 코너 비율)')
    ap.add_argument('--corner-ratio', type=float, default=D.CORNER_FIXED_RATIO,
                    help='고정 조향의 안쪽/바깥 바퀴 출력 비율 (C 모델 코너 중앙값 0.63, 작을수록 세게 돎)')
    ap.add_argument('--corner-outer', type=float, default=None, help='고정 조향의 바깥 바퀴 출력 (기본 --speed)')
    ap.add_argument('--corner-min-s', type=float, default=D.CORNER_FIXED_MIN_S,
                    help='고정 조향을 최소 이 시간(초) 유지한 뒤에야 직선 판정으로 빠져나옴')
    ap.add_argument('--corner-max-s', type=float, default=D.CORNER_FIXED_MAX_S,
                    help='고정 조향 최대 시간(초), 넘으면 차선 추종으로 복귀')
    # 모의 전용
    ap.add_argument('--start', type=float, nargs=3, metavar=('X', 'Y', 'YAW'), help='출발 위치 (cm, cm, 도)')
    ap.add_argument('--seed', type=int, default=None, help='영상 잡음/회전 잡음 난수 (없으면 잡음 없음)')
    ap.add_argument('--yaw-noise', type=float, default=0.0, help='회전 잡음 표준편차 °/s')
    ap.add_argument('--plot', default=None)
    ap.add_argument('--json', default=None, help='요약과 프레임별 위치/여유를 저장')
    return ap


def main():
    args = parser().parse_args()
    world = World()
    model = dict(yaw_noise=args.yaw_noise)
    rows, reason = simulate(args, world, tuple(args.start) if args.start else None, args.seed, model)
    print(f'종료: {reason}, {rows[-1]["t"]:.1f}초, 놓친 프레임 {sum(not r["ok"] for r in rows)}')
    print_summary(summarize(rows, args.ccw), '반시계' if args.ccw else '시계')
    if args.plot:
        plot(rows, world, args.plot, ('CCW' if args.ccw else 'CW'))
        print('그림:', args.plot)
    if args.json:
        keep = ('t', 'x', 'y', 'yaw', 'seg', 'left', 'right', 'curv', 'ok', 'clear_l', 'clear_r', 'bias', 'keep', 'R')
        json.dump(dict(args=vars(args), reason=reason, summary=summarize(rows, args.ccw),
                       rows=[{k: r[k] for k in keep} for r in rows]), open(args.json, 'w'), ensure_ascii=False)


if __name__ == '__main__':
    main()
