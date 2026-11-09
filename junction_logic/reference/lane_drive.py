#!/usr/bin/env python3
"""카메라 차선 검출(lane_pipeline.analyze) + pure pursuit 로 라인트레이서 주행.

사용법:
    python3 lane_drive.py --dry-run                   # 모터는 끄고 인식/모터 명령만 출력
    python3 lane_drive.py --speed 0.3 --duration 15   # 실제 주행
    python3 lane_drive.py --corners 4 --duration 300  # 코너 4개를 돌면(트랙 한 바퀴) 정지
    python3 lane_drive.py --road cross_h --turn right # 21cm 도로, 교차로에서 우회전

안전: 보드 버튼(SW1~4) 중 아무거나 누르거나 Ctrl+C -> 즉시 정지.
      차선을 LOST_STOP_FRAMES 프레임 연속 놓치거나 --duration 초가 지나면 정지.

전원(모터 시작 시 Pi 리셋) 대책:
  - 소프트 스타트: 모터 출력 변화량을 ACCEL_PER_S 로 제한 (0 -> 0.3 에 0.5초)
  - PWM 주파수 PWM_FREQ_HZ (gpiozero 기본 100Hz 는 펄스마다 큰 전류가 흐름)
  - 카메라 FPS 를 낮춰 Pi 부하/전류를 줄임
  - Pi 입력 전압(EXT5V)/소비전력을 power.csv 에 기록 (CUR/power_rec.py), VOLT_CUT 미만이면 모터 정지
  - 로그는 줄마다 fsync -> 갑자기 전원이 나가도 직전까지 남음
기록: runs/<시각>/log.csv(프레임별), power.csv(전원, 구간별), power.png(부하 그래프),
      --save-every 프레임마다 원본 프레임(frame_XXXXX.jpg).
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import lane_pipeline as L

os.environ.setdefault('LG_WD', '/tmp')  # lgpio 알림 파이프(.lgd-nfy*)를 작업 폴더 대신 /tmp 에 생성
# 주의: lgpio 는 GPIO 를 처음 열 때 작업 폴더를 LG_WD 로 바꾼다 -> 파일 경로는 모두 HERE 기준 절대 경로로
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'CUR'))
from power_rec import Recorder, read_ext5v, vcgencmd  # noqa: E402

# 모터 드라이버 배선 (AI_CAR/practice 코드 기준): A = 왼쪽 바퀴, B = 오른쪽 바퀴, IN1=0/IN2=1 이 전진
PIN_PWM = {'L': 18, 'R': 23}
PIN_IN1 = {'L': 22, 'R': 25}
PIN_IN2 = {'L': 27, 'R': 24}
PIN_BUTTONS = (5, 6, 13, 19)

WHEEL_TRACK_CM = 13.0    # 좌우 바퀴 중심 간격 (차폭 16cm 기준 추정)
CORNER_SLOW_CM = 10.0    # 곡률이 클수록 감속: v / (1 + |곡률| * 이 값)
STEER_SMOOTH = 0.5       # 곡률 저역통과 (0 이면 필터 없음)
LOST_HOLD_FRAMES = 3     # 차선을 놓쳐도 이 프레임까지는 직전 명령 유지
LOST_STOP_FRAMES = 10    # 연속으로 이만큼 놓치면 정지 후 종료

PWM_FREQ_HZ = 1000       # 모터 PWM 주파수
ACCEL_PER_S = 0.6        # 바퀴 출력 최대 변화 속도 (1초당 PWM 0.6)
VOLT_WARN = 4.85         # 이 미만이면 가속 금지 + 감속
VOLT_CUT = 4.75          # 이 미만이면 모터 정지 후 종료

# 코너 세기 (이력 적용): 중앙선 반경이 ENTER_R 미만인 프레임이 ENTER_FRAMES 만큼 이어지면 코너 진입,
# 반경이 EXIT_R 이상(또는 직선/판단 불가)인 프레임이 EXIT_FRAMES 만큼 이어지면 코너 통과.
# 한 바퀴(--corners) 셀 때는 첫 코너와 같은 방향만 인정 (외곽 한 바퀴는 코너 방향이 모두 같음)
CORNER_ENTER_R_CM = 50
CORNER_EXIT_R_CM = 100
CORNER_ENTER_FRAMES = 20
CORNER_EXIT_FRAMES = 15

THROTTLE_BITS = {0: '저전압(현재)', 1: '주파수제한(현재)', 2: '스로틀(현재)', 3: '온도제한(현재)',
                 16: '저전압 발생', 17: '주파수제한 발생', 18: '스로틀 발생', 19: '온도제한 발생'}


def read_throttled():
    try:
        return int(vcgencmd('get_throttled').split('=')[1], 16)
    except (IndexError, ValueError):
        return None


def describe_throttled(v):
    if v is None:
        return '읽기 실패'
    return ', '.join(name for bit, name in THROTTLE_BITS.items() if v >> bit & 1) or '정상'


class Motors:
    def __init__(self, freq=PWM_FREQ_HZ):
        from gpiozero import DigitalOutputDevice, PWMOutputDevice
        self.pwm = {k: PWMOutputDevice(p, frequency=freq) for k, p in PIN_PWM.items()}
        self.in1 = {k: DigitalOutputDevice(p) for k, p in PIN_IN1.items()}
        self.in2 = {k: DigitalOutputDevice(p) for k, p in PIN_IN2.items()}

    def set(self, left, right):
        """-1.0 ~ 1.0, 양수가 전진."""
        for k, v in (('L', left), ('R', right)):
            v = float(np.clip(v, -1.0, 1.0))
            self.in1[k].value = 0 if v >= 0 else 1
            self.in2[k].value = 1 if v >= 0 else 0
            self.pwm[k].value = abs(v)

    def stop(self):
        self.set(0.0, 0.0)

    def close(self):
        self.stop()
        for dev in (*self.pwm.values(), *self.in1.values(), *self.in2.values()):
            dev.close()


def wheel_speeds(curv, speed):
    """곡률(1/cm, +오른쪽) -> 좌/우 바퀴 명령."""
    v = speed / (1 + abs(curv) * CORNER_SLOW_CM)
    left = v * (1 + curv * WHEEL_TRACK_CM / 2)
    right = v * (1 - curv * WHEEL_TRACK_CM / 2)
    m = max(abs(left), abs(right), 1.0)  # 1 을 넘으면 비율 유지하며 줄임
    return left / m, right / m


def slew(cur, target, max_step):
    """현재 출력에서 목표까지 한 번에 max_step 이내로만 이동 (소프트 스타트/급조향 방지)."""
    return tuple(c + float(np.clip(t - c, -max_step, max_step)) for c, t in zip(cur, target))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--road', choices=('auto', *L.ROAD_WIDTH_CM), default='auto',
                    help='auto: 좌우 선 실측 간격을 맵 도로폭에 맞춤 (한쪽만 보이면 직전 도로 유지)')
    ap.add_argument('--start-road', choices=L.ROAD_WIDTH_CM, default=L.DEFAULT_ROAD,
                    help='auto 일 때 출발 도로 (첫 프레임에 한쪽 선만 보일 경우 사용)')
    ap.add_argument('--turn', choices=('straight', 'left', 'right'), default='straight')
    ap.add_argument('--speed', type=float, default=0.3, help='기본 PWM (0~1)')
    ap.add_argument('--duration', type=float, default=15.0, help='최대 주행 시간(초)')
    ap.add_argument('--corners', type=int, default=0, help='코너를 이 개수만큼 통과하면 정지 (0=사용 안 함)')
    ap.add_argument('--fps', type=float, default=15, help='카메라 프레임레이트')
    ap.add_argument('--dry-run', action='store_true', help='모터는 돌리지 않음')
    ap.add_argument('--save-every', type=int, default=5, help='N 프레임마다 원본 저장 (0=저장 안 함)')
    ap.add_argument('--calib', default=str(HERE / 'calib_checkerboard.npz'))
    args = ap.parse_args()
    road_fixed = None if args.road == 'auto' else args.road
    road_hint = args.start_road  # 자동 모드에서 직전에 판단한 도로 (한쪽 선만 보일 때 유지)

    # 보정/버드아이 변환은 미리 한 번만 계산
    c = np.load(args.calib)
    size_img = tuple(int(v) for v in c['size'])
    newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        c['K'], c['D'], size_img, np.eye(3), balance=0.0)
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(c['K'], c['D'], np.eye(3), newK, size_img, cv2.CV_16SC2)
    g = dict(np.load(HERE / 'ground.npz'))
    M, size, _ = L.bev_transform(g, newK, size_img[1])

    run_dir = HERE / 'runs' / datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    log = open(run_dir / 'log.csv', 'w', newline='')
    wr = csv.writer(log)

    def log_row(row):
        wr.writerow(row)
        log.flush()
        os.fsync(log.fileno())

    log_row(['t', 'frame', 'ok', 'segment', 'offset_cm', 'heading_deg', 'target_lat', 'target_fwd',
             'curv', 'left', 'right', 'ext5v', 'pi_power_w', 'road', 'gap_cm', 'path_R', 'turn', 'path', 'proc_ms'])

    print(f'[power] 시작 전 EXT5V {read_ext5v()}V, {vcgencmd("measure_temp")}, '
          f'상태: {describe_throttled(read_throttled())}')
    # 구간(직선 S0, 코너 C1, 직선 S1, ...)별로 전원 부하를 기록 -> CUR/plot_power.py 로 그래프
    rec = Recorder(run_dir / 'power.csv', ('step', 'label', 'left', 'right', 'phase'))
    rec.state.update(step=0, label='S0', left=0.0, right=0.0, phase='idle')
    rec.start()

    from gpiozero import Button
    from picamera2 import Picamera2
    from plot_power import plot
    buttons = [Button(p, pull_up=False) for p in PIN_BUTTONS]
    motors = None if args.dry_run else Motors()
    cam = Picamera2()
    cfg = cam.create_video_configuration(main={"format": "RGB888", "size": size_img},
                                         controls={"FrameRate": args.fps})
    cam.align_configuration(cfg)
    cam.configure(cfg)
    cam.start()
    time.sleep(1.5)  # AE/AWB 안정화

    print(f'[drive] {"DRY-RUN (모터 꺼짐)" if args.dry_run else "주행"} '
          f'road={args.road}{f" (출발 {road_hint})" if road_fixed is None else ""} '
          f'turn={args.turn} speed={args.speed} duration={args.duration}s '
          f'{f"corners={args.corners} " if args.corners else ""}fps={args.fps} PWM {PWM_FREQ_HZ}Hz -> {run_dir}')
    print('[drive] 정지: 보드 버튼 아무거나 / Ctrl+C')
    lost, curv_f = 0, 0.0
    target, out = (0.0, 0.0), (0.0, 0.0)
    in_corner, corner_cnt, corners_done, seg = False, 0, 0, 0
    bend_votes, lap_bend = [], None
    t_start = t_prev = time.time()
    frame = 0
    reason = '시간 종료'
    rec.state['phase'] = 'hold'
    rec.event('step S0')

    def new_segment(label):
        nonlocal seg
        seg += 1
        rec.state.update(step=seg, label=label)
        rec.event(f'step {label}')

    try:
        while time.time() - t_start < args.duration:
            if any(b.is_pressed for b in buttons):
                reason = '버튼 정지'
                break
            if rec.volt is not None and rec.volt < VOLT_CUT:
                reason = f'전압 강하 {rec.volt:.2f}V < {VOLT_CUT}V'
                break
            img = cv2.flip(cam.capture_array(), -1)  # 카메라가 뒤집혀 장착됨
            t0 = time.time()
            und = cv2.remap(img, m1, m2, cv2.INTER_LINEAR)
            res, notes = L.analyze(und, M, size, road_fixed, args.turn, road_hint)
            proc_ms = (time.time() - t0) * 1000

            if res is None:
                lost += 1
                if lost >= LOST_STOP_FRAMES:
                    reason = f'차선을 {lost}프레임 연속 놓침'
                    break
                if lost > LOST_HOLD_FRAMES:
                    target = (0.0, 0.0)
            else:
                lost = 0
                road_hint = res['road']
                curv_f = STEER_SMOOTH * curv_f + (1 - STEER_SMOOTH) * res['curv_pp']
                target = wheel_speeds(curv_f, args.speed)
            if rec.volt is not None and rec.volt < VOLT_WARN:  # 전압이 낮으면 가속 금지, 서서히 감속
                target = (out[0] * 0.8, out[1] * 0.8)
            now = time.time()
            out = slew(out, target, ACCEL_PER_S * (now - t_prev))
            t_prev = now
            if motors:
                motors.set(*out)
            rec.state.update(left=out[0], right=out[1])

            # 코너 세기 (중앙선 반경 기준, 진입/통과 모두 여러 프레임 연속일 때만 인정)
            R = None if res is None else res['path_R']
            t = now - t_start
            if not in_corner:
                if R is not None and R < CORNER_ENTER_R_CM:
                    corner_cnt += 1
                    bend_votes.append(res['path_bend'])
                else:
                    corner_cnt, bend_votes = 0, []
                if corner_cnt >= CORNER_ENTER_FRAMES:
                    bend = max(set(bend_votes), key=bend_votes.count)
                    if args.corners and lap_bend is not None and bend != lap_bend:
                        print(f'{t:5.1f}s [corner] {bend} 코너 신호 무시 (이 바퀴는 {lap_bend} 코너)')
                    else:
                        lap_bend = lap_bend or bend
                        in_corner = True
                        new_segment(f'C{corners_done + 1}')
                        print(f'{t:5.1f}s [corner] {corners_done + 1}번째 코너 진입 ({bend})')
                    corner_cnt, bend_votes = 0, []
            else:
                corner_cnt = corner_cnt + 1 if R is None or R >= CORNER_EXIT_R_CM else 0
            if in_corner and corner_cnt >= CORNER_EXIT_FRAMES:
                in_corner, corner_cnt = False, 0
                corners_done += 1
                new_segment(f'S{corners_done}')
                print(f'{t:5.1f}s [corner] {corners_done}번째 코너 통과')
                if args.corners and corners_done >= args.corners:
                    reason = f'코너 {corners_done}개 통과 (목표 {args.corners}개)'
                    break
                plot(run_dir / 'power.csv', snapshot=True, title='Lane drive load')  # 중간 저장

            p_now = '' if rec.power is None else f'{rec.power:.2f}'
            if res is not None:
                off = res['Ccm'][res['i0'], 0]
                la = res['lookahead']
                log_row([f'{t:.3f}', frame, 1, rec.state['label'], f'{off:.2f}', f'{res["heading"]:.1f}',
                         f'{la[0]:.2f}', f'{la[1]:.2f}', f'{curv_f:.4f}', f'{out[0]:.3f}', f'{out[1]:.3f}',
                         rec.volt, p_now, res['road'], '' if res['gap'] is None else f'{res["gap"]:.1f}',
                         '' if res['path_R'] is None else f'{res["path_R"]:.0f}',
                         res['turn'], res['path_desc'], f'{proc_ms:.1f}'])
                if frame % 15 == 0:
                    print(f'{t:5.1f}s #{frame:4d} [{rec.state["label"]}] 이탈 {off:+5.1f}cm 각도 {res["heading"]:+5.1f}° '
                          f'-> L {out[0]:+.2f} R {out[1]:+.2f} | {res["road"]} | {rec.volt}V {p_now}W '
                          f'| {res["path_desc"]} | {proc_ms:.0f}ms')
            else:
                log_row([f'{t:.3f}', frame, 0, rec.state['label'], '', '', '', '', f'{curv_f:.4f}',
                         f'{out[0]:.3f}', f'{out[1]:.3f}', rec.volt, p_now, '', '', '', '', ' '.join(notes),
                         f'{proc_ms:.1f}'])
                print(f'{t:5.1f}s #{frame:4d} 차선 놓침 ({lost}) -> L {out[0]:+.2f} R {out[1]:+.2f} | {rec.volt}V')
            if args.save_every and frame % args.save_every == 0:
                cv2.imwrite(str(run_dir / f'frame_{frame:05d}.jpg'), img)
            frame += 1
    except KeyboardInterrupt:
        reason = 'Ctrl+C'
    finally:
        if motors:
            motors.close()
        cam.close()
        rec.event(f'stopped: {reason}')
        rec.stop()
        log.close()
    dt = time.time() - t_start
    print(f'[drive] 정지 ({reason}). {frame} 프레임 / {dt:.1f}s = {frame / max(dt, 1e-6):.1f} fps, '
          f'코너 {corners_done}개 통과, 기록: {run_dir}')

    # 구간별 전원 부하 요약
    drive = [s for s in rec.samples if s[0]['phase'] == 'hold' and s[1] is not None]
    if drive:
        print(f'[power] 주행 중 EXT5V 최저 {min(s[1] for s in drive):.3f}V, '
              f'상태: {describe_throttled(read_throttled())}')
        for label in dict.fromkeys(s[0]['label'] for s in drive):
            ss = [s for s in drive if s[0]['label'] == label]
            v = [s[1] for s in ss]
            p = [s[2] for s in ss if s[2] is not None]
            lw = np.mean([s[0]['left'] for s in ss])
            rw = np.mean([s[0]['right'] for s in ss])
            print(f'  {label:3s}: 바퀴 평균 L {lw:.2f} R {rw:.2f} | EXT5V 최저 {min(v):.3f}V 평균 {np.mean(v):.3f}V '
                  f'| Pi 소비전력 {np.mean(p) if p else float("nan"):.2f}W')
    print(f'[power] 그래프: {plot(run_dir / "power.csv", title="Lane drive load")}')


if __name__ == '__main__':
    main()
