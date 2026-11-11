# 인계: A5b 미세조정 (데스크탑 에이전트용) — 2026-11-11 밤

노트북에서 작업하던 Claude가 데스크탑의 로컬 에이전트(Codex 또는 Claude)에게 넘기는 문서다. 계획 전체는 [README.md](README.md)에 있다. **이 문서가 우선한다.**

## 0. 반드시 지킬 것

- **언어:** 사용자에게 하는 모든 설명과 보고는 **한국어**로 한다.
- **차량 금지:** 실차 모터, Pi의 카메라·GPIO를 쓰지 않는다. Pi는 6단계에서 사진을 **읽어 복사**할 때만 접속한다(주소와 접속 방법은 사용자에게 받는다). `~/hybrid_mission`은 수정하지 않는다(7단계는 사용자 승인 뒤).
- **다른 프로그램 보호:** 데스크탑의 다른 프로그램(OCR 서버, Robotomy_AI, LM Studio, Codex·Claude 앱 등)을 멈추거나 건드리지 않는다.
  - 자원 기준(2026-11-11 확인): 남은 메모리 약 6.5GB, GPU 8GB 중 약 2GB 사용 중.
  - **동시 작업자는 6~8개로 제한**한다. 시작 전에 남은 메모리를 다시 확인한다.
- **저장 위치:** 데이터·체크포인트·로그는 **`D:\autodrive_cnn\`** 아래에 둔다. C:는 20GB만 남아 있다. 저장소에는 코드, 결과 요약, 최종 모델(약 1MB)만 넣는다.
- **읽기만:** 팀 코드(`track-digital-twin`), `cnn_map_hybrid_lab`, `v3_recovery_lab`, 원본 A5b 파일(`bench/_vendor/tdt_models_2026-11-01`)은 수정하지 않는다.
- **커밋:** 비밀번호·토큰·개인 호스트·환경 폴더는 커밋하지 않는다. 커밋 메시지 끝에는 다음 줄을 넣는다.
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  ```
  (Codex라면 자기 규칙을 따른다.)
- **보고:** 실제로 실행한 검사만 보고한다. 시뮬레이터 결과와 실차 결과를 섞지 않는다.
- **멈출 지점:** 단계마다 "확인"을 통과하지 못하면 다음 단계로 가지 않고 사용자에게 알린다. **3단계가 끝나면 결과를 보고하고 멈춘다.** 4단계 DAgger는 사용자 승인 뒤에 한다.

## 1. 저장소와 환경

- **저장소:** `https://github.com/Edwin29/AutoMobile`. `main`과 `drive-v4-bench`는 같은 내용이다(2026-11-11 병합).
  ```bat
  git clone https://github.com/Edwin29/AutoMobile.git D:\autodrive_cnn\AutoMobile
  cd /d D:\autodrive_cnn\AutoMobile
  git checkout -b cnn-finetune
  ```
  작업은 `cnn-finetune` 브랜치에서 한다.
- **파이썬:** 데스크탑 기본 Python 3.11.8(pyenv). torch 2.5.1+cu121(CUDA 사용 가능), OpenCV 4.10, numpy, scipy가 있다.
  - `onnx` 패키지는 없다. ONNX 내보내기(7단계)가 필요할 때만 별도 가상환경을 만든다. **시스템 파이썬에는 설치하지 않는다.**
    ```bat
    python -m venv --system-site-packages D:\autodrive_cnn\venv
    D:\autodrive_cnn\venv\Scripts\pip install onnx
    ```
- **환경 변수:** `BENCH_CNN_SPEED=1.0` (CNN 속도. `bench.matrix`는 기본으로 넣는다).
- **한글 경로 주의:** 사용자 폴더 이름이 한글이라 `cv2.imread`가 실패한다. 파일을 바이트로 읽어 `cv2.imdecode`를 쓴다(`bench/realcam.py`의 `imread` 참고). 데이터는 D:의 ASCII 경로에 둔다.
- **GPU 카메라 효과:** `bench/realcam.py`가 torch(CUDA)로 실차 기록 기반 효과를 입힌다. 효과 이름은 `realcam`(실차 수준), `realcam_harsh`(가혹), `realcam_light/shadow/exposure/blur/optics`(요인별)이다.

## 2. 지금까지 한 것

- **0단계 완료** (노트북, 2026-11-11):
  - `cnn_finetune/model.py`:
    - `LanePolicy`: 팀 PilotNet과 같은 구조
    - `load_a5b()`: OpenCV로 ONNX 가중치를 읽어 옮김
    - `preprocess_bgr()`: bench `TeamCNN._preprocess`와 같은 전처리
    - `export_onnx()`
  - `cnn_finetune/convert_check.py`: 시뮬레이터 사진 300장에서 PyTorch 사본과 원본 ONNX의 출력 차이가 **최대 2.1e-7**. 다시 내보내기 검사는 onnx 패키지가 없어 건너뛰었다.
  - **데스크탑에서 다시 확인:** `python -m cnn_finetune.convert_check --n 300` → `"ok": true`

## 3. 핵심 배경 (반드시 이해할 것)

- **A5b 출력:** `[linear, angular] ∈ [−1, 1]`, angular + = 왼쪽.
  - 실행 시 팀 변환(`bench/controllers.py` `TeamCNN.step`, min_duty 0.175, max_duty 0.66, turn_mix 0.72, speed_scale 1.0)으로 바퀴 PWM을 만든다.
  - 그다음 `--cnn-mixer v4`이면 `HybridCNN._remap_cnn`이 곡률을 유지한 채 v4 할당기(`MotorModel.mix`)로 다시 만든다.
- **문제:** A5b 정답은 Isaac Sim의 이상적인 차량 기준이다. 실차는 출발 문턱 0.30, 회전 효율 약 0.6이다. 실차 코너 프레임의 54~59%에서 안쪽 바퀴 명령이 문턱 아래였다(`drive_v4/field_runs/hyb2m_20261111`).
- **이번 미세조정의 정답 기준:** 실행 경로인 **팀 변환 → v4 할당기**를 거쳤을 때 교사가 원한 곡률과 평균 듀티가 나오게 하는 `[linear, angular]`를 정답으로 쓴다.
  - 할당기의 회전 효율은 v4 설정값을 쓴다: `yaw_gain_left` 0.63, `yaw_gain_right` 0.62, `cm_s_per_duty` 16, `track_cm` 13. 코드는 `bench/_vendor/<V4_PIN>/drive_v4/core.py`의 `MotorModel`.
  - 실행할 때는 `--cnn-mixer v4`를 쓴다는 전제다.
- **좌표:** bench 월드는 x 오른쪽(동), y 위(북), yaw는 +y 기준 시계 방향 +. 차량 기준점은 앞끝에서 14.5cm 뒤(축 중간).

## 3-1. 정정 (2026-11-11 밤, 첫 역산 검사 결과를 보고)

데스크탑 첫 검사에서 7조건 중 4조건이 "실패"했다. 대부분 **이 문서의 확인 기준이 잘못된 것**이었다. 아래로 바꾼다.

1. **정답이 맞는지는 할당 "전"에서 판단한다.**
   - `HybridCNN._remap_cnn`이 팀 변환 PWM에서 계산하는 값이 목표와 같으면 정답이 맞다.
     - 곡률: `k = raw × yaw_gain(raw) / v` (`speed = pwm × cm_s_per_duty`, `v` = 평균, `raw = (왼쪽 − 오른쪽) / track_cm`)
     - 평균 듀티: 팀 PWM의 평균
   - v4 할당기는 그 곡률을 평균적으로 내도록 설계돼 있다. 안쪽 바퀴가 출발 문턱(0.30) 아래로 갈 상황이면 두 바퀴를 같은 비율로 올리므로, **할당 후 평균 듀티가 목표보다 높아지는 것은 정상**이다(예: 0.43 → 0.467). 값은 기록하되 실패로 보지 않는다.
2. **역산은 [linear, angular] 전체 범위(−1~1)에서 찾는다.** 팀 변환은 바퀴 크기가 0.175보다 작을 수 없어서, 안쪽 바퀴 역회전(음수)을 허용하지 않으면 큰 곡률을 표현하지 못한다(예: 0.06이 0.050에서 막힘).
3. **목표 곡률 상한:** 안쪽 바퀴 0, 바깥 최대일 때 낼 수 있는 곡률은 약 2·g/track ≈ 0.095/cm(반경 약 10.5cm)다. 교사의 목표는 **±0.085/cm로 자른다.** 외곽 코너는 약 0.05/cm다.
4. **할당 후 실제 곡률을 확인할 때는** 여러 번 할당한 결과에서 **(평균 회전 속도) ÷ (평균 속도)**로 계산한다. 회전·속도는 `MotorModel.velocity`로 계산하며 출발 문턱이 적용된다. PWM을 먼저 평균 낸 뒤 곡률을 계산하면 출발 문턱 때문에 틀린다.
5. **허용 오차:** 할당 전 곡률 1e-4/cm, 할당 전 평균 듀티 1e-3. 할당 후 곡률은 목표의 ±5% 안이면 통과로 기록한다(번갈아 주기의 주기 오차 포함).

## 3-2. 정정 2 (재검사 9조건 중 4조건 실패를 보고)

곡률 상한 0.085/cm는 틀렸다. **팀 변환으로 한 번에 표현할 수 있는 곡률에는 빈 구간이 있다.**

- 팀 변환의 바퀴 듀티는 0이거나, 크기가 0.175~0.66이다.
- 바깥 0.66, 안쪽 0.175일 때 곡률이 최대 **0.0954 × (0.66 − 0.175) / (0.66 + 0.175) ≈ 0.0554/cm**다. 여기서 0.0954 = 2·g/track, g ≈ 0.62.
- 안쪽을 0으로 하면 0.095/cm, 안쪽을 역회전(−0.175 이하)하면 0.16/cm 이상으로 건너뛴다.
- 따라서 **0.0554~0.095/cm는 평균 듀티를 어떻게 잡아도 정답으로 만들 수 없다.**

**수정 기준**

1. **목표 곡률은 ±0.055/cm로 자른다.**
   - 외곽 코너 중앙선은 약 0.051/cm이고, 실차 CNN이 코너에서 낸 명령도 약 0.044/cm였다.
   - 교차로 회전은 학습에서 뺀다.
   - 검사 조건 ±0.06, ±0.085는 "잘림(정상)"으로 기록한다.
2. **곡률은 정확히(1e-4), 평균 듀티는 목표 0.43에 가장 가깝게** 맞춘다(제약 최적화).
   - 큰 곡률에서는 평균 듀티가 내려가도 된다. 예: 0.055/cm에서 약 0.42.
   - 정답에 실제로 쓴 평균 듀티는 메타데이터에 기록한다.
3. 그 밖의 3-1 기준은 그대로다. 할당 전 판단, 할당 후 듀티 상승은 정상, 할당 후 곡률은 평균 회전 속도 ÷ 평균 속도로 ±5% 안.

**참고(나중):** 출력 정의를 "곡률·속도"로 바꾸고 팀 변환을 거치지 않게 하면 이 빈 구간이 없어진다. 실행 코드 변경이 필요하므로 이번 A안에서는 하지 않는다.

## 4. 1단계 — 교사

- **파일:** `cnn_finetune/teacher.py`
- **입력:** 실제 자세(시뮬레이터 정답)와 주행 방향.
- **경로:** 외곽은 `bench/world.py` `Camera.route`(점수용 중앙선)를 방향에 맞게 쓴다. 안쪽 도로는 직선 중앙선(x=0 또는 y=0).
- **조향:** 앞 15~20cm 순수 추종으로 목표 곡률 `k*`를 정한다. 진행 방향 오차를 줄이는 항을 더해도 된다.
- **속도:** A5b 평소 수준(평균 듀티 약 0.43)을 기본으로 하고, 곡률이 크면 줄여도 된다.
- **정답 역산:** `(k*, 평균 듀티)` → `[linear, angular]`. 팀 변환과 v4 할당기 기준으로 수치 탐색한다(격자 또는 뉴턴). 역산이 맞는지 단위 시험으로 확인한다.
- **교사 단독 주행 확인:** 교사가 직접 PWM을 내서 시뮬레이터를 돈다. `bench.run`과 같은 루프를 쓰고, 컨트롤러로 등록하거나 별도 스크립트로 만든다.
  - 차량: `field_estimate`, `weak_left_turn`, `sluggish`, `unequal_sides`, `random:<seed>` 몇 개
  - 카메라: `realcam`, `realcam_harsh`
  - 양방향
- **확인:** 선 안 밟은 완주가 거의 전부여야 한다. 아니면 교사를 고친다.

## 5. 2단계 — 데이터

- **파일:** `cnn_finetune/collect.py`
- **수집 방식:** 교사가 운전하면서, 매 프레임 CNN 입력(전처리 후 3×66×200, uint8로 저장)과 정답 `[linear, angular]`, 메타데이터(자세, 차량, 효과, 시드)를 기록한다.
- **DART 흔들기:** 교사 명령에 잠깐씩 틀어짐을 넣는다. 정답은 흔들기 전 교사 값이다.
- **다양화:** 방향, 차량 특성 무작위(`bench/vehicle.py` `random_params`), 카메라 효과 무작위, 시작 위치 무작위, 외곽과 안쪽 도로.
- **제외:** 교차로 중심 35cm 안은 뺀다. 지도 `junctions` 좌표 기준.
- **분량:** 약 5만 장.
- **저장:** `D:\autodrive_cnn\data\v1\` 샤드 npz. 시드별로 학습·검증·시험을 나눈다(같은 시드가 섞이지 않게).
- **확인:** 정답 분포(직선·코너·복귀 비율), 교차로 제외가 됐는지, 샤드 몇 개를 그림으로 확인.

## 6. 3단계 — 1차 학습과 짧은 비교

- **파일:** `cnn_finetune/train.py`
- **학습:** `load_a5b()`에서 출발. AdamW, 작은 학습률(1e-5~5e-5), 몇 에포크. 증강은 팀과 비슷하게 하되 좌우 반전 시 angular 부호를 뒤집는다.
- **검증:** 교사와의 MAE(전체, 코너, 복귀 장면 따로). 같은 검증 세트에서 원본 A5b와 비교한다.
- **짧은 시뮬레이터 비교:** 새 모델을 bench 컨트롤러로 등록한다(예: `TeamCNN` 하위 클래스에 torch 모델 또는 ONNX 경로). hyb2 / `--cnn-mixer v4` 기준으로 A5b와 비교한다.
  - `bench.matrix`의 기존 묶음(예: `J_xcheck`의 실차 수준·harsh 카메라, 4차량 × 양방향, 시드 2개)을 축소해 돌린다.
  - 지표: 선 안 밟은 완주, 선까지 여유, 코너 후 평행 회복 시간, 안쪽 바퀴 문턱 아래 비율.
- **보고 후 멈춤.** 결과는 `cnn_finetune/results/` 요약과 사용자 보고(한국어)로 남긴다.

## 7. 참고 명령

```bat
cd /d D:\autodrive_cnn\AutoMobile
set BENCH_CNN_SPEED=1.0
python -m cnn_finetune.convert_check --n 300
python -m bench.run --controller hyb2 --vehicle field_estimate --effects realcam --direction ccw --seed 41 --out D:\autodrive_cnn\runs\x
python -m bench.matrix --out D:\autodrive_cnn\runs\m --workers 6 --group J_xcheck --controller hyb2,cnnA5b --seeds 41
python -m bench.mission --scenario inner_loop --out D:\autodrive_cnn\runs\mission_x
```

## 8. 관련 문서

- [CNN 미세조정 계획](README.md)
- [작업 기록](../docs/DRIVE_V4_BENCH_20261111.md) 6~11절: 실차 기반 카메라, hyb2·hyb3, 교차로, 미션
- [실차 결과](../drive_v4/field_runs/hyb2m_20261111/README.md)
- [에이전트 공통 안내](../AGENTS.md)
