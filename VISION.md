# 표지판 탐지 연결 (2026-11-14)

## 구조

```
pi_run (기존 주행 프로세스)                 vision (새 프로세스)
  카메라 → upright 프레임 ──공유메모리──→  YOLO11n (ncnn, 2스레드, 최대 8fps)
                                            vision_rules: 표지판 판단·타이머·쿨다운
  UDP 8765 명령 수신  ←──── stop / resume / direction / speed
  감속(speed) 적용                           경적(horn)은 여기서 GPIO로 직접
```

- 카메라는 pi_run 하나만 연다. vision 프로세스는 자기가 쉬고 있을 때만 프레임을 한 장 받는다(복사 약 0.4ms). 주행 루프는 YOLO를 기다리지 않는다.
- vision은 `./send.sh`와 똑같은 UDP 명령을 보낸다. 정지·재개·방향 처리는 기존 코드를 그대로 쓴다.

## 파일

| 파일 | 내용 |
|---|---|
| `bench/vision_rules.py` (새 파일) | 규칙만 있음(카메라·모델 없음). 숫자는 파일 위쪽 상수에서 바꾼다 |
| `bench/vision.py` (새 파일) | 검출기(letterbox + 후처리), 프레임 전달, UDP 전송, 경적, 재생 모드 |
| `bench/pi_run.py` (수정) | `--vision` 옵션, 프레임 전달, `speed` 명령(감속) |
| `bench/pi_cmd.py` (수정) | `./send.sh slow`, `./send.sh normal` 추가 |

## 규칙

| 대상 | 동작 |
|---|---|
| stop | 확정 후 2초 뒤 정지(사유 stop_sign), 3초 뒤 재개 |
| slow | 확정 후 2초 뒤 차선 주행 duty ×0.8, 3초 뒤 ×1.0. 좌우 비율(곡률)은 그대로 두고, 느린 바퀴가 0.31 밑으로 내려가지 않는 만큼만 줄임. 급코너·교차로 회전은 줄이지 않음 |
| horn | 확정 후 2초 뒤 경적 켬, 3초 뒤 끔 (`--horn-pin`이 없으면 기록만) |
| left / right / straight | 화면 좌우 끝에 잘리지 않은 화살표를 프레임당 한 표로 집계한다. 촬영 시 지도 추정상 접근 중인 삼거리와 진입로에만 집계한다. 기존 34 cm 결정 시점에 최근 8초간 3표 이상이고 한 방향이 65% 이상이며 해당 출구가 있으면 그쪽을 택한다. 근거가 부족하면 기존 기본 경로를 따른다. 결정 이후 도착한 표는 다음 삼거리로 넘어가지 않는다. |
| 쿨다운 | stop·slow·horn은 발동 시각부터 10초 동안 무시하고, 그 뒤에도 화면에서 2초 이상 사라졌다가 다시 보여야 다시 발동한다. |
| 신호등과 횡단보도 | 같은 프레임에서 빨강과 횡단보도를 검출하면 4초 뒤 정지하고 5초 뒤 재개한다. 초록과 횡단보도를 검출하면 통과한다. 두 경우 모두 방향 명령은 생성하지 않는다. |
| 확정 | 같은 클래스가 연속 2프레임 검출돼야 인정 |

## 설치 (Pi)

```bash
cd ~/my_env/hybrid_mission
.venv/bin/pip install ncnn
mkdir -p models/signs
cp ~/yolo_real/new_model/model.ncnn.param ~/yolo_real/new_model/model.ncnn.bin models/signs/
```

## 실행

```bash
# 1) 저장 사진으로 먼저 확인 (모터 없음, 보낼 명령만 출력)
.venv/bin/python -m bench.vision --replay pi_runs/drive_hyb2m_20261111_175226 --fps 10

# 2) 드라이런: 카메라 + 판단 + 탐지, 모터 꺼짐. ms가 10 이하인지 본다
./run_mission.sh --vision --max-seconds 60

# 3) 탐지 화면 보기: 노트북 브라우저에서 http://<Pi 주소>:5000 (주소는 시작할 때 출력됨)
./run_mission.sh --vision --vision-view

# 4) 오른쪽 빨간 출발 구역에서 남쪽을 보는 시계 방향 출발. 북쪽을 보면 --direction ccw 사용.
./run_mission.sh --drive --direction cw --vision
./run_mission.sh --drive --direction cw --vision --horn-pin 18     # 부저가 GPIO18(BCM)에 있을 때
```

- `--script`와 함께 쓰지 않는다(같은 표지판이 두 번 동작한다).
- 탐지 기록: `pi_runs/<실행>/vision.jsonl` (프레임마다 검출 결과, 보낸 명령)
- 손으로 시험: `./send.sh slow`, `./send.sh normal`, `./send.sh resume --reason red_light`
- `--vision-model none`: 모델 없이 프레임 전달만 시험한다.
