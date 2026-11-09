#!/usr/bin/env python3
"""실제 주행 첫 프레임과 모의 카메라 영상의 차선 경계가 가장 잘 맞는 출발 위치 찾기 (sim_drive START 용).
사용법: python3 match_start.py runs/<주행> {ccw|cw} [코너까지 x 범위 시작 끝]"""
import sys

import cv2
import numpy as np

import lane_pipeline as L
import sim_drive as S
from lane_eval import Replayer


def edge_pts(und, M, size):
    e, _ = L.edge_masks(und, M, size)
    p = np.argwhere(e > 0)[:, ::-1].astype(np.float32)
    return p[L.px_to_fwd(p[:, 1]) <= 35]


def main():
    run, mode = sys.argv[1], sys.argv[2]
    xr = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (20, 60)
    w = S.World()
    rp = Replayer()
    real, _ = rp.frame(cv2.imread(f'{run}/frame_00000.jpg'))
    rpts = edge_pts(real, w.M, w.size)
    rmask = np.zeros(w.size[::-1], np.uint8)
    rmask[rpts[:, 1].astype(int), rpts[:, 0].astype(int)] = 1
    rdt = cv2.distanceTransform(1 - rmask, cv2.DIST_L2, 3)
    x0, y0, yaw0 = S.START[mode]
    sgn = 1 if x0 > 0 else -1          # 진행 방향 (+x 동쪽 / -x 서쪽)
    best = []
    for xs in np.arange(xr[0], xr[1], 2.0):
        for off in np.arange(-4, 4.1, 1.0):
            for dyaw in np.arange(-15, 36, 3.0):
                pose = (sgn * xs, y0 + off, np.radians(yaw0 + dyaw))
                p = edge_pts(w.render(pose), w.M, w.size)
                if len(p) < 20:
                    continue
                m = np.zeros_like(rmask)
                m[p[:, 1].astype(int), p[:, 0].astype(int)] = 1
                sdt = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3)
                score = (np.minimum(rdt[p[:, 1].astype(int), p[:, 0].astype(int)], 25).mean()
                         + np.minimum(sdt[rpts[:, 1].astype(int), rpts[:, 0].astype(int)], 25).mean()) / 2
                best.append((score / L.PX_PER_CM, pose[0], pose[1], yaw0 + dyaw))
    best.sort()
    for b in best[:5]:
        print(f'평균 어긋남 {b[0]:.2f}cm  ->  x {b[1]:.1f}, y {b[2]:.1f}, yaw {b[3]:.0f}')


if __name__ == '__main__':
    main()
