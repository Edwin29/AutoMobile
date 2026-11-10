# 세 번째 주행 프로그램 — v0.2 실험판

[합의한 여섯 파트](../docs/DRIVE_V3_DESIGN_20261110.md)를 별도 코드로 연결했다. **노트북에서 사진·영상 재생, 합성 카메라 폐루프, 미션 로직 시험을 실행하는 통합 실험판**이다. 기존 C안·실차 배포본은 변경하지 않는다. 실차 완주가 검증된 배포판이라는 뜻은 아니다.

## 구현 범위

| 파트 | 구현 | 주요 파일 |
|---|---|---|
| 1 관측 | HSV/보정/BEV, 분기·공백 보존 조각 추적, 좌우 후보, 파란 원, 점선/실선/횡단보도 후보, 선택적 YOLO, SORT 계열 추적 | `observation.py`, `fragments.py`, `features.py` |
| 2 지도·경로 | 경계/표식 ID, 비대칭 지도, 외곽 방향별 여유 개선, 진입·출구별 Bézier 경로/제한 최적화, 차체 쓸림 여유표 | `map_model.py`, `build_map.py` |
| 3 상태 | PWM 예측, 강건 정합 후 판정, 관측 가능한 방향의 EKF, 조건부 소수 가설, 혁신 검사, 국소 재탐색, 지연 프레임 재처리, 선택적 바닥 VO | `estimation.py`, `visual_motion.py` |
| 4 경로 결정 | 합류 스플라인 후보, 히스테리시스, 동일 경로를 자세 후보들에서 검사, 짧은 예측 구간, 재관측, 제한적인 전진 복귀 | `planning.py` |
| 5 제어 | Pure Pursuit/곡률 FF+오차 FB 비교, 동적 trim, 공유 PWM 응답모델/역변환, 최소 출력·포화·변화율, 선택적 적응 | `control.py`, `actuation.py` |
| 6 미션 | 방문·대상별 예약, 중복 방지, 충돌 보류, 방향 확정, 핑크선 마지노선, STOP/SLOW/HORN, 중앙 신호등/우회전, 통과와 위치 보정 분리 | `mission.py`, `context.py` |
| 연결·검증 | 기본 기록용 출력, 예외 시 0 출력, 실제 영상 shadow, 합성 카메라 폐루프, JSONL/소스 해시 | `runtime.py`, `shadow.py`, `simulation.py` |

## 실행

저장소 루트 `github_publish/AutoMobile`에서 Python 3.11을 사용한다. 설치와 실행에 **같은 인터프리터**를 쓴다.

```powershell
py -3.11 -m venv .venv-v3
.\.venv-v3\Scripts\python.exe -m pip install -r drive_v3/requirements.txt
.\.venv-v3\Scripts\python.exe -m unittest discover -s drive_v3/tests -v
```

아래 `python`은 위 가상환경 실행 파일로 바꿔도 된다. 기본 명령은 GPIO를 불러오지 않는다. 출력 폴더는 덮어쓰지 않으므로 새 이름을 지정한다.

```powershell
# 양방향 폐루프와 영상. seconds는 상한이다.
python -m drive_v3 simulate --direction ccw --seconds 220 --video --output drive_v3/runs/my_ccw
python -m drive_v3 simulate --direction cw --seconds 220 --video --output drive_v3/runs/my_cw

# 전진 배율 1.27 / 회전 반응 저하
python -m drive_v3 simulate --scenario speed_bias --output drive_v3/runs/my_speed
python -m drive_v3 simulate --scenario yaw_loss --output drive_v3/runs/my_turn

# 제어기 / VO 비교
python -m drive_v3 simulate --controller feedback --output drive_v3/runs/my_feedback
python -m drive_v3 simulate --vo --output drive_v3/runs/my_vo

# 중앙 교차로 북쪽 진입 → 우회전
python -m drive_v3 simulate --junction center:N:right --output drive_v3/runs/my_junction
python -m drive_v3 build-map --output drive_v3/runs/my_map
python -m drive_v3 matrix --seconds 25 --output drive_v3/runs/my_matrix
```

`held`는 정지, `time_limit`는 설정 시간 종료다. **어느 것도 완주가 아니다.** `lap_complete`/`route_complete`만 완료다. 25초 매트릭스는 짧은 기능 비교이고 한 바퀴 시험이 아니다.

## 실제 자료 재생

```powershell
python -m drive_v3 observe --input "../../map_localization/outer_driver/field_runs/ccw_v135_20261106_174607_edges" --glob "[0-9][0-9].png" --output drive_v3/runs/my_observation
python -m drive_v3.review_saved --workspace "../.." --output drive_v3/runs/my_real_review
python -m drive_v3 shadow "YOUR_VIDEO.mp4" --pose -86.25 -20.25 3.14159 --direction ccw --seconds 10 --output drive_v3/runs/my_shadow
```

기본 입력은 **이미 180도 회전된, 렌즈 미보정 BGR 640×480**이다. 회전하지 않은 raw에만 `--rotate180`, 같은 `rect_K`로 보정한 영상에만 `--rectified`를 쓴다. 둘을 함께 지정하지 않는다. 크기 자동 확대나 외부 파라미터 자동 변경은 없다.

`shadow`는 기록 영상의 차가 새 명령대로 움직였다고 계산하지 않는다. `--commands commands.csv`가 있으면 `time_s,left_pwm,right_pwm`를 영상 시간에 맞춰 읽어 기존 명령으로 예측한다. 없으면 운동 예측은 0이며 불확실성이 커진다. 이는 진단 한계이지 실제 차가 정지했다는 주장도, 새 코드의 폐루프 성능도 아니다. shadow 시간은 프레임 번호/공칭 FPS이며 VFR·로그 시간 오프셋은 별도 정렬이 필요하다. pose는 사용자 초기 가정으로 정답이 아니다.

## 프로그램 연결

`Runtime.step(image, capture_s, now, ...)`에 같은 단조 시계의 촬영/처리 시각을 전달한다. 반환은 `record, observation, debug_images, path`다. record에는 자세·공분산·가설·잔차·정지 사유·제한 후 PWM·미션 이벤트·계산 시간·설정/지도/보정 해시가 있다.

지연 프레임은 상태 추정에 과거 시각으로 반영한다. 그 원시 경계는 현재 차체 좌표의 제약으로 재사용하지 않으며 미션 마지노선에는 지연 시간을 더한다. 정밀한 국소 영상 운동 보상은 아직 없으므로, 지연 프레임의 국소 관측 경로는 제한된다.

```python
from drive_v3.runtime import Runtime
from drive_v3.context import Target

with Runtime(initial_pose, route, start_s=start, clock_id="sensor_monotonic") as car:
    # 경로 생성은 정지 중 사전 수행. 제어 루프에서는 최적화하지 않는다.
    car.prepare_junction("west", "S")
    car.target_tracker.arm(Target("lap1-west", "west:S", "junction",
                                  possible_directions=("straight", "right")))
    record, obs, debug, path = car.step(frame, capture_s, now,
                                       detections=right_roi_detections)
```

방문 ID는 상위 진행 관리에서 부여한다. 이미지 추적 ID나 절대 자세 보정으로 방문을 만들지 않는다. 바닥 미션에는 `map_marker_id`도 지정한다. 대응이 가설마다 다르면 트리거를 확정하지 않는다. 확인된 표식 이후 VO 상대 이동으로 통과를 판정할 수 있지만 절대 자세가 이동한 것만으로 통과시키지 않는다. 다음 방문 예약·교차로 완료 확인 인터페이스는 제공하며, 랜덤 표지의 전체 과제 순회는 아직 종단 검증하지 않았다.

`YoloAdapter(model_path)`는 명시적으로 생성할 때만 Ultralytics를 불러온다. 오른쪽 ROI에서 `Detection`을 반환한다. 모델에 없는 직진·좌회전 클래스를 만들어내지 않는다. 실제 가중치 추론은 이번 단위 시험에 포함하지 않았다.

`RecordingSink`가 기본이다. GPIO 연결은 기존 하드웨어 드라이버의 네 바퀴 콜백을 `WatchdogSink`에 주입한다. 별도 스레드가 명령 만료 시 0 출력을 시도한다. 현재 좌/우 명령을 앞뒤에 같이 전달하며 바퀴별 실측 속도를 안다는 뜻은 아니다. 핀 배치·Picamera2 스트리밍·부저 배선은 임의로 정하거나 Pi에 배포하지 않았다. 프로세스 강제 종료·OS 정지·전원 상실까지 소프트웨어 스레드가 보장하지는 않는다.

## 기본값과 기능 제한

- 설정은 `config/observation.json`, `config/runtime.json`, `config/map.json`에 모았다. 초기 공학값이며 통계적으로 검증된 확률·안전 한계가 아니다.
- 기본 추종은 Pure Pursuit다. FF+FB도 같은 경로로 비교 가능하다. 합류 경로가 원점에서 시작해 오차가 항상 0이 되는 실수를 피하도록 별도의 기준 경로 오차를 전달한다. 두 제어기의 회전량을 무조건 더하지 않는다.
- 동적 trim은 0에서 시작한다. 관측이 좋고 직선이며 포화·출력 제한이 없는 구간에서 천천히 갱신한다. 고정 보상값을 무조건 불러오지 않는다.
- 응답모델 적응과 VO 예측은 기본 **꺼짐**이다. 반복 무늬에서 VO 잔차가 작아도 이동량이 틀리는 사례가 있어 2차원 특징 분포 검사도 추가했다. VO는 진단/정지 확인에 사용하며, 상태 예측 사용 여부는 별도다. 켜면 해당 구간 PWM 예측을 대체하며 독립 절대 센서로 중복 융합하지 않는다. 공유 카메라 오차 상관과 VO 잡음은 아직 보정되지 않았다.
- STOP 시간, HORN 길이, SLOW 속도/해제, 신호 해제 규칙은 아직 `null`이다. 시험에서는 명시한 가상 규칙을 주입한다. 실제 기본값은 `rules_required`로 보류한다.
- 합쳐진 핑크선은 `pink_line_candidate`로 위치 보정에만 사용한다. 점선 패턴이 확인됐다고 하거나 미션을 실행하지 않는다.
- 자동 후진, 제자리 회전, 시동 킥, CNN 분할, 무제한 예측 주행은 활성화하지 않았다.

## 좌표와 검증 한계

차체 x=오른쪽, y=전방, cm. yaw는 월드 북쪽에서 시계방향 양수다. 앞끝에서 14.5cm를 기준으로 전장 24.5cm·폭 16cm 직사각형을 쓴다. 실제 순간 회전 중심의 실측값이라는 뜻은 아니다. 렌즈 전방 오프셋 12.36cm는 한 번만 적용한다.

내부 21/18cm 도로의 지도 축 대응과 파란 원 좌표는 기존 자료를 계승했으며 독립 재실측값이 아니다. 거리장 격자 조회는 보간 오차를 감안해 셀 대각선 0.354cm를 차감한다. 유한 sigma 자세 표본·차체 표본 검사는 모든 연속 공간 위험의 증명이 아니다. 후보 경로 생성 실패도 물리적으로 가능한 경로가 전혀 없다는 증명이 아니다.

원 관측부 기록은 [v0.1 검증](VALIDATION_20261110.md), 통합 결과·실패 원인은 [v0.2 검증 보고서](VALIDATION_V02.md)를 읽는다. **새 코드의 양방향 완주나 실차 운용 가능을 전제로 하지 않는다.**
