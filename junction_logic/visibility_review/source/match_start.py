#!/usr/bin/env python3
"""실제 주행 첫 프레임과 모의 카메라 영상의 차선 경계가 가장 잘 맞는 출발 위치 찾기 (sim_drive --start 용).
네 코너 접근(sim_drive.APPROACH) 각각에서 코너까지 거리·좌우 위치·방향을 바꿔 가며 경계 거리(챔퍼)를 비교한다.
사용법: python3 match_start.py runs/<주행> {ccw|cw} [코너까지 거리 시작 끝] [방향 범위(도)]"""
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

import lane_pipeline as L
import sim_drive as S
from lane_eval import Replayer

_W = None


def edge_pts(und, M, size):
    e, _ = L.edge_masks(und, M, size)
    p = np.argwhere(e > 0)[:, ::-1]
    return p[L.px_to_fwd(p[:, 1]) <= 35]


def search(job):
    global _W
    if _W is None:
        _W = S.World()
    name, real_path, dr, yr = job
    real, _ = Replayer().frame(cv2.imread(real_path))
    rpts = edge_pts(real, _W.M, _W.size)
    rmask = np.zeros(_W.size[::-1], np.uint8)
    rmask[rpts[:, 1], rpts[:, 0]] = 1
    rdt = cv2.distanceTransform(1 - rmask, cv2.DIST_L2, 3)
    out = []
    for before in np.arange(dr[0], dr[1], 2.0):
        for off in np.arange(-4, 4.1, 1.0):
            for dyaw in np.arange(-yr, yr + 0.1, 3.0):
                pose = S.approach_pose(name, before, off, dyaw)
                p = edge_pts(_W.render((pose[0], pose[1], np.radians(pose[2]))), _W.M, _W.size)
                if len(p) < 20:
                    continue
                m = np.zeros_like(rmask)
                m[p[:, 1], p[:, 0]] = 1
                sdt = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
                score = (np.minimum(rdt[p[:, 1], p[:, 0]], 25).mean()
                         + np.minimum(sdt[rpts[:, 1], rpts[:, 0]], 25).mean()) / 2 / L.PX_PER_CM
                out.append((score, name, before, off, dyaw, pose))
    return sorted(out)[:3]


def main():
    run, mode = sys.argv[1], sys.argv[2]
    dr = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (4, 50)
    yr = float(sys.argv[5]) if len(sys.argv) > 5 else 15
    names = [n for n in S.APPROACH if n.startswith(mode + '-')]
    with ProcessPoolExecutor(4) as ex:
        res = sorted(sum(ex.map(search, [(n, f'{run}/frame_00000.jpg', dr, yr) for n in names]), []))
    for score, name, before, off, dyaw, pose in res[:6]:
        print(f'평균 어긋남 {score:.2f}cm  {name} 코너 {before:.0f}cm 앞, 오른쪽 {off:+.0f}cm, 방향 {dyaw:+.0f}°'
              f'  ->  --start {pose[0]:.1f} {pose[1]:.1f} {pose[2]:.0f}')


if __name__ == '__main__':
    main()
