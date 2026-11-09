# 재현 방법

Python 3.11 환경에서 분석했다. requirements-desktop.txt는 게시 시 로컬 버전이다. requirements에는 현재 smoke 검사에 사용하는 numpy·OpenCV·torch를 기록했다. 과거 그래프 스크립트에는 matplotlib 등 추가 의존성이 있을 수 있다. 모든 과거 스크립트에 대한 통합 환경 검증을 뜻하지 않는다. 영상 출력에는 ffmpeg가 필요하다.

실행 예시는 저장소 루트 기준이다.

    python -m venv .venv
    .venv/Scripts/Activate.ps1
    python -m pip install -r requirements-desktop.txt
    python tools/check_archive.py
    python tools/smoke_sim.py

smoke_sim은 GPIO·카메라 없이 렌더러를 사용하는 두 관측기의 3스텝을 확인한다.

## 기본 한 바퀴

    python map_localization/outer_driver/validate_sim.py ccw
    python map_localization/outer_driver/validate_sim.py cw
    python road_ai_v1/sim_clean.py /path/to/road_results_clean.zip

clean ZIP은 Drive에서 받는다. sim_clean 결과는 closed_loop_clean.json에 쓰므로 기존 결과를 먼저 복사하거나 별도 체크아웃을 사용한다. 기본 이상 구동 모델이며 실차 비대칭·지연의 완전한 모델이 아니다.

## 촬영 사진 적용

    python road_ai_v1/replay_clean.py road_ai_v1/20261106_193448_316130 /path/to/road_results_clean.zip road_ai_v1/replay_new

사진 ZIP을 road_ai_v1 아래에 먼저 푼다. 사진은 이미 장착 방향으로 180도 회전한 BGR다. 다시 뒤집지 않고 해당 세션 remap과 320×240 축소를 적용한다. 카메라 장착이 달라지면 보정 검증이 필요하다.

## 학습과 실차 재생

Drive Colab의 ACTION=view는 기존 결과 표시이고 train은 새 학습이다. clean/mild/hard는 현재 코드에서 새 모델로 시작한다. mild가 clean 가중치를 이어 학습한다는 의미는 아니다. 확보한 clean 결과는 validation이며 final-test 결과는 없다.

real_video_review.py에는 원본 영상 6개와 field_runs PNG가 필요하다. investigate_edges.py와 oracle_test.py에도 실제 PNG가 필요하고 oracle_test에는 257장과 기존 마스크도 필요하다. Git에는 해당 수치 결과가 있으나 큰 원본은 별도 접근이 필요하다.

일부 옛 스크립트의 개인 경로는 일반 예시 경로로 바꿨다. source_manifest.json에서 수정 여부를 확인하고 실제 입력 경로를 설정한다.

## Pi 프로그램

run_pi.py, capture_track.py, capture_images.py, probe.py, straight_test.py는 옵션에 따라 카메라·GPIO·모터를 사용할 수 있다. OS 제공 picamera2와 GPIO 라이브러리를 확인해야 한다. 저장소 게시 중 Pi 실행은 수행하지 않았다. 과거 README 명령은 당시 시험 절차로 읽는다.
