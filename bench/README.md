# bench — 두 주행 코드가 함께 쓰는 시뮬레이터

주행 코드(drive_v3, drive_v4)와 시뮬레이터의 역할을 나눈다. 시뮬레이터는 주행 코드 내부를 모르고, 주행 코드도 시뮬레이터를 import하지 않는다.

| 구분 | 담당 | 바꿀 수 있나 |
|---|---|---|
| **세계** `world.py`, `world/` | 트랙 지도(실측 반영 수정 지도), 카메라 보정·투영, 영상 생성, 출발 구역 | **고정** |
| **차량** `vehicle.py` | 출발 최소 출력(좌/우), 최대 유효 출력, 듀티당 속도(좌/우), 좌회전·우회전 효율, 주행 중 쏠림, 명령 지연, 바퀴 반응 시간 | **실험마다 지정** |
| **방해 요소** `disturb.py` | 반사, 가림, 화면 끊김, 흐림·황색조, 바닥 이음새, 출력 저하 | 실험마다 지정 |
| **타이밍** `run.py` | 카메라 30fps, 주행 코드의 처리 시간(각자 측정값 또는 강제 지정) | 실험마다 지정 |
| **채점** `score.py` | 20ms마다 실제 차체와 선 사이 여유, 선 밖 시간, 복귀, 코너 통과, 완주/멈춤/이탈 | 하나로 통일 |
| **연결부** `controllers.py` | 각 주행 코드를 `step(영상, 촬영 시각, 적용 시각) → PWM`으로 감싼다 | — |

주행 코드가 받는 것은 보정된 카메라 영상, 시각, 출발 구역 중앙 위치(모든 코드 동일)뿐이다. 실제 위치나 차량 값은 받지 않는다.

## 차량 값

`vehicle.py`의 `VehicleParams` 기본값은 "이상적인 차"다. 실제 값을 모르기 때문에 프리셋과 무작위 범위로 여러 환경을 만든다.

| 프리셋 | 내용 |
|---|---|
| ideal | 명령만큼 정확히 움직임, 지연 없음 |
| field_estimate | 실주행 로그 추정: 회전 효율 0.6/0.6, 속도 16.8, 지연 50ms, 반응 0.1초 |
| weak_left_turn | 좌회전 0.5, 우회전 0.7 |
| sluggish | 지연 150ms, 반응 0.25초 |
| high_start_duty | 출발 최소 출력 0.34(주행 코드는 0.30을 하한으로 씀) |
| low_saturation | 출력 0.5 이상은 더 빨라지지 않음 |
| unequal_sides | 왼쪽 바퀴가 10% 느림(오른쪽으로 쏠림) |

무작위(`random:<seed>`) 범위: 출발 최소 출력 0.26~0.34(좌우 따로), 최대 유효 출력 0.5~1.0, 속도 14~18(좌우 차이 ±6%), 회전 효율 0.45~1.0(좌우 따로), 쏠림 ±0.03rad/s, 지연 0~150ms, 반응 0~0.25초.

## 실행

```bash
python -m bench.run --controller v4 --vehicle field_estimate --direction ccw --out runs/x
python -m bench.run --controller v3r --vehicle random:7 --effects glare,low_power --out runs/y
python -m bench.run --controller v4 --vehicle ideal --vehicle-json '{"turn_eff_left":0.5,"actuation_delay_s":0.1}' --out runs/z
python -m bench.matrix --out runs/m --workers 8          # 정해진 전체 비교
python -m bench.matrix --out runs/m --report             # 표 만들기
```

- `--timing own`: 각 코드의 처리 시간. v4는 Pi 실측 70ms, 수정 v3는 노트북 측정 약 0.46초(Pi 미측정).
- `--timing fast` / `slow`: 두 코드에 같은 처리 시간을 강제한다. 알고리즘만 비교할 때 쓴다.

## 실차 기반 카메라 (`--effects realcam`, `bench/realcam.py`)

실차 Pi 사진 약 2,500장(10/30~11/10)에서 본 현상을 원본 어안 영상에 넣는다. v4/v3r은 훼손된 원본을 보정(undistort)한 영상을 받는다. 차에서 일어나는 순서와 같다.

- 색: 매트 도로/초록 색을 실측값으로 바꿨다(초록 V 202, 도로 V 74).
- 광택 매트의 조명 반사: 멀리 있는 조명이라 차의 위치가 아니라 **방향**에 따라 움직인다. 매트 주름 하이라이트도 넣었다.
- 렌즈: 초점 흐림과 주변부 보라색 감광.
- 자동노출·화이트밸런스: 장면이 바뀌면 늦게 따라간다. 시작 직후 어두운 보라색 프레임도 재현한다.
- 지평선 위 배경: 실제 사진(사람, 의자, 다른 차)을 쓰고, 차 방향에 따라 돌아간다(`world/realcam/bg_*.jpg`).
- `realcam_harsh`는 여기에 다음을 더한다: 사람·가구 그림자(고정된 것과 걸어다니는 것), 차 자체 그림자, 긴 노출(모션블러), 노출 헌팅, 뿌연 안개, 어두울 때 노이즈, 렌즈 플레어.
- `realcam_light / shadow / exposure / blur / optics`: harsh 요인을 하나씩만 typical 위에 켠다.

torch로 계산하며, GPU가 있으면 GPU를 쓴다. RTX 4060에서 프레임당 typical 5.4ms, harsh 10.5ms이고, CPU에서는 300ms와 500ms다. 미리보기 명령:

```bash
python -m bench.realcam_preview --out preview.jpg --speed --device cuda
```

## 주행 코드 버전

- v4: 이 저장소의 `drive_v4` 작업본.
- 수정 v3: `codex/v3-recovery-comparison` 브랜치의 커밋 `8b41dd9`를 `_vendor/v3r_8b41dd9`에 고정해 두었다. 그쪽 작업이 바뀌어도 결과가 흔들리지 않는다.

## 한계

평평한 합성 바닥, 그림자·실제 반사 없음, 두 바퀴 등가 모델(네 바퀴 접촉 역학 아님). 차량 값은 범위 시험용 가정이고, `field_estimate`도 로그에서 추정한 값이지 실측 정답이 아니다.
