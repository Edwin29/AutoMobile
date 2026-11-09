#!/usr/bin/env python3
"""주행 기록의 선 침범 분석 (저장된 frame_*.jpg, 없으면 녹화한 video_raw.mp4 를 다시 분석).

  앞쪽 여유: 로봇 바로 앞(보이는 가장 가까운 경계)에서 차체 옆면(중심 ±8cm)과 좌/우 선 안쪽 경계 사이(cm)
  차체 여유: 앞에서 본 선을 영상 기반 이동 추정(위에서 본 영상끼리 ECC 정합)으로 로봇 위치까지 옮겨,
            차체 영역(좌우 ±8cm, 앞뒤 BODY_REAR_CM~BODY_FRONT_CM) 옆의 선과의 최소 여유(cm)
  음수 = 그 선을 침범. 코너에서 안쪽 선을 밟는 건 차체 중간/뒤라서 '차체 여유'로만 보인다.

사용법:
    python3 lane_eval.py runs/<주행>              # 구간별 요약 + runs/<주행>/clearance.png
    python3 lane_eval.py runs/<A> runs/<B> ...    # 여러 주행(조향 설정) 비교 표
구간(S0, C1, S1, ...)은 lane_drive.py 의 직선/코너 구분 (코너 판정은 앞을 보고 하므로 실제 회전보다 이르다).
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import lane_pipeline as L

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'CUR'))
from plot_power import C_CUT, C_LEFT, C_RIGHT, C_TEXT, FONT, Panel, save_png  # noqa: E402

BODY_FRONT_CM = 4.0     # 차체 앞 끝 (카메라 바로 아래 지면 기준, +전방) - 트랙 사진으로 추정
BODY_REAR_CM = -19.0    # 차체 뒤 끝 (차 길이 약 23cm 추정)
LINE_MEMORY_S = 6.0     # 앞에서 본 선을 기억하는 시간
ECC_CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4)


def num(v):
    return float(v) if v not in (None, '', 'None') else np.nan


class Replayer:
    def __init__(self):
        c = np.load(HERE / 'calib_checkerboard.npz')
        g = dict(np.load(HERE / 'ground.npz'))
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(c['K'], c['D'], (640, 480), np.eye(3),
                                                                      balance=0.0)
        self.m1, self.m2 = cv2.fisheye.initUndistortRectifyMap(c['K'], c['D'], np.eye(3), newK, (640, 480),
                                                               cv2.CV_16SC2)
        self.M, self.size, _ = L.bev_transform(g, newK, 480)
        valid = cv2.warpPerspective(np.full((480, 640), 255, np.uint8), self.M, self.size)
        self.valid = cv2.erode(valid, np.ones((9, 9), np.uint8))

    def frame(self, img):
        und = cv2.remap(img, self.m1, self.m2, cv2.INTER_LINEAR)
        gray = cv2.cvtColor(cv2.warpPerspective(und, self.M, self.size), cv2.COLOR_BGR2GRAY)
        return und, cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)

    def motion(self, prev, cur):
        """prev 버드아이 px 좌표 -> cur 좌표로 옮기는 2x3 행렬 (실패 시 None)."""
        W = np.eye(2, 3, dtype=np.float32)
        try:
            _, W = cv2.findTransformECC(prev, cur, W, cv2.MOTION_EUCLIDEAN, ECC_CRIT, self.valid, 5)
            return W
        except cv2.error:
            return None


def body_clearance(store):
    """기억 중인 선 점(현재 로봇 기준 px)으로 차체 옆 좌/우 최소 여유(cm).
    선 이름이 아니라 점이 로봇 중심선의 어느 편에 있는지로 좌/우를 나눈다."""
    out = {'left': np.nan, 'right': np.nan}
    if not store:
        return out
    cm = L.to_cm(np.vstack([p for _, p, _ in store]))
    cm = cm[(cm[:, 1] >= BODY_REAR_CM) & (cm[:, 1] <= BODY_FRONT_CM) & (np.abs(cm[:, 0]) < 25)]
    if (cm[:, 0] < 0).any():
        out['left'] = -L.ROBOT_HALF_W_CM - float(cm[cm[:, 0] < 0, 0].max())
    if (cm[:, 0] >= 0).any():
        out['right'] = float(cm[cm[:, 0] >= 0, 0].min()) - L.ROBOT_HALF_W_CM
    return out


def frames_of(run, video_step=2):
    """(프레임 번호, 원본 영상) 순서대로. 저장된 frame_*.jpg 가 있으면 그것, 없으면 녹화한 video_raw.mp4 를
    video_step 프레임마다 (영상의 n 번째 장 = 주행 프레임 n)."""
    jpgs = sorted(run.glob('frame_*.jpg'))
    if jpgs:
        for f in jpgs:
            yield int(f.stem.split('_')[1]), cv2.imread(str(f))
        return
    cap = cv2.VideoCapture(str(run / 'video_raw.mp4'))
    fr = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if fr % video_step == 0:
            yield fr, img
        fr += 1
    cap.release()


def analyze_run(run):
    """저장 프레임마다 (t, frame, segment, 앞쪽 좌, 앞쪽 우, 차체 좌, 차체 우, 누적 회전도)."""
    by_frame = {int(r['frame']): r for r in csv.DictReader(open(run / 'log.csv'))}
    rp = Replayer()
    data, store, prev, hint, yaw = [], [], None, L.DEFAULT_ROAD, 0.0
    for fr, img in frames_of(run):
        r = by_frame.get(fr)
        if r is None:
            continue
        t = num(r['t'])
        und, gray = rp.frame(img)
        if prev is not None:
            W = rp.motion(prev, gray)
            if W is None:
                store = []  # 이동을 모르면 기억한 선을 버림
            else:
                yaw += np.degrees(np.arctan2(W[1, 0], W[0, 0]))
                store = [(s, p @ W[:, :2].T + W[:, 2], ts) for s, p, ts in store if t - ts <= LINE_MEMORY_S]
        prev = gray
        res, _ = L.analyze(und, rp.M, rp.size, None, 'straight', hint)
        front = {'left': np.nan, 'right': np.nan}
        if res:
            hint = res['road']
            front = {s: np.nan if v is None else v for s, v in res['clear'].items()}
        body = body_clearance(store)  # 이번 프레임에서 새로 본 선은 차체 앞쪽이라 넣기 전에 계산
        if res:
            store += [(s, d['edge'].astype(np.float32), t) for s, d in res['det'].items()]
        data.append((t, fr, r.get('segment', ''), front['left'], front['right'], body['left'], body['right'], yaw))
    return data


def summarize(data):
    """구간별 [프레임 수, (앞쪽좌, 앞쪽우, 차체좌, 차체우) 각각 침범 %와 최소 여유]."""
    segs = {}
    for row in data:
        segs.setdefault(row[2], []).append(row[3:7])
    segs['ALL'] = [row[3:7] for row in data]
    summary = {}
    for seg, v in segs.items():
        a = np.array(v, float)
        cols = []
        for k in range(4):
            x = a[:, k][np.isfinite(a[:, k])]
            cols.append((100 * (x < 0).mean() if len(x) else np.nan, x.min() if len(x) else np.nan))
        summary[seg] = (len(a), cols)
    return summary


def plot_clearance(run, data):
    a = np.array([(row[0], *row[3:8]) for row in data], float)
    img = np.full((470, 1100, 3), 255, np.uint8)
    cv2.putText(img, f'Line clearance [cm]  (<0 = crossing the line)  {run.name}', (20, 28), FONT, 0.6, C_TEXT, 2,
                cv2.LINE_AA)
    cv2.putText(img, 'blue=left line  orange=right line   thin dots=front of robot   thick=robot body (odometry)',
                (20, 52), FONT, 0.45, C_TEXT, 1, cv2.LINE_AA)
    p = Panel(img, 80, 90, 980, 320, (0, max(a[:, 0].max(), 1)), (-8, 10), 'clearance', xlabel='time [s]',
              yfmt='{:.1f}')
    starts = [(row[0], row[2]) for k, row in enumerate(data) if k == 0 or row[2] != data[k - 1][2]]
    for k, (ts, seg) in enumerate(starts):
        te = starts[k + 1][0] if k + 1 < len(starts) else a[-1, 0]
        if seg.startswith('C'):
            p.band(ts, te, (235, 235, 250))
        p.text(ts, seg, C_TEXT)
    p.hline(0, C_CUT, 'line')
    for col, color in ((1, C_LEFT), (2, C_RIGHT)):
        for x, y in zip(a[:, 0], a[:, col]):
            if np.isfinite(y):
                cv2.circle(img, (p.px(x), p.py(y)), 1, color, -1)
    p.line(a[:, 0], a[:, 3], C_LEFT, 2)
    p.line(a[:, 0], a[:, 4], C_RIGHT, 2)
    out = run / 'clearance.png'
    save_png(out, img)
    return out


def params_of(run):
    f = run / 'params.json'
    return json.load(open(f)) if f.exists() else {}


def main():
    runs = [Path(a) for a in sys.argv[1:]] or [max((HERE / 'runs').glob('2*/'), key=lambda p: p.stat().st_mtime)]
    table = []
    for run in runs:
        data = analyze_run(run)
        s = summarize(data)
        print(f'== {run.name}  {params_of(run) or ""}  (프레임 {len(data)}, 추정 누적 회전 {data[-1][7]:+.0f}°)')
        print('  구간  프레임 | 앞쪽 좌 침범%/최소 | 앞쪽 우 침범%/최소 | 차체 좌 침범%/최소 | 차체 우 침범%/최소')
        for seg, (n, cols) in s.items():
            cells = ' | '.join(f'{pc:5.0f}% {mn:+5.1f}cm' for pc, mn in cols)
            print(f'  {seg:4s} {n:5d}  | {cells}')
        print(f'  그래프: {plot_clearance(run, data)}')
        table.append((run.name, params_of(run), s['ALL'][1]))
    if len(runs) > 1:
        print('\n== 비교 (전체 구간 침범 프레임 % / 최소 여유)')
        for name, prm, cols in table:
            cells = ' | '.join(f'{k} {pc:3.0f}% {mn:+5.1f}'
                               for k, (pc, mn) in zip(('앞좌', '앞우', '차좌', '차우'), cols))
            print(f'  {name} {prm}\n      {cells}')


if __name__ == '__main__':
    main()
