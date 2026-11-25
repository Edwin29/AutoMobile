Pi 차선 주행 모델 시험 안내 (2026-11-01)

이 묶음에는 학습된 모델만 들어 있다. 실행 코드는 GitHub에 있다.
  코드: https://github.com/gawbi/track-digital-twin  (main 브랜치)
  모델: 이 zip (GitHub Release에서도 받을 수 있음: https://github.com/gawbi/track-digital-twin/releases/tag/models-2026-11-01)


1. 들어 있는 모델

  폴더                   이름  학습 데이터                              상태
  models/lane-real-C5b/  C5b   A5b -> 실제 주행으로 미세조정            실차 미시험 (1순위)
  models/lane-v5b/       A5b   반사·가혹 조건 Isaac Sim 데이터만        실차 미시험 (2순위)
  models/lane-real-C/    C     Isaac Sim -> 실제 미세조정 (기존)        큰길 커브 성공, 좁은 도로·반사 구간 이탈
  models/lane-real-B/    B     실제 주행 데이터만                        직선·큰길 커브 성공, 한 바퀴 실패
  models/lane-v4b/       A     Isaac Sim 데이터만 (기존)                실차 커브 4곳 모두 이탈 (비교용)
  configs/real_car.json        모터 보정값 (좌우 duty 계산)


2. Pi 준비

  [경우 1] 2026-10-31에 쓴 그 Pi (~/track-digital-twin 폴더와 .venv-pi가 이미 있음)
    노트북에서 이 폴더 안으로 들어가 모델만 복사하면 끝:
      scp -r models configs pi@<Pi주소>:track-digital-twin/

  [경우 2] 새 Pi (Raspberry Pi OS 64-bit)
      sudo apt install -y python3-picamera2 python3-opencv git
      git clone https://github.com/gawbi/track-digital-twin.git
      cd track-digital-twin
      python3 -m venv --system-site-packages .venv-pi
      .venv-pi/bin/pip install onnxruntime pillow
      .venv-pi/bin/python -c "import picamera2, gpiozero, onnxruntime, cv2; print('ok')"
    그다음 [경우 1]처럼 models, configs를 복사한다.
    인터넷이 안 되면 노트북의 pi_offline 폴더(install_offline.sh)를 쓴다.


3. 출발 전 확인 (오늘 Pi가 4번 꺼진 원인)

  - Pi에서 다른 무거운 프로그램(시뮬레이션, 학습, VS Code 원격 등)을 끈다.
  - 온도 70도 이하:          vcgencmd measure_temp
  - 충전 케이블을 빼고 배터리가 충분한지 확인한다.
  - 한 사람은 차 옆에서 대기, 도로를 벗어나면 들어 올리기만 한다 (다시 놓지 않기).


4. 실행 (Pi에서)

  cd ~/track-digital-twin
  .venv-pi/bin/python run_vehicle.py --policy models/lane-real-C5b/policy.onnx \
      --motors gpio --flip180 --calibration configs/real_car.json \
      --speed-scale 0.5 --duration 180 --warmup 1 --output runs/C5b-lap-1

  - 모델을 바꾸려면 --policy 경로와 --output 이름만 바꾼다.
  - 180초 뒤 자동 정지. 중간에 멈추려면 Ctrl+C.
  - 속도는 0.5(50%) 유지. 더 올리면 Pi 전원이 꺼질 수 있다.


5. 비교 방법 (중요)

  - 모든 모델을 같은 출발점, 같은 방향(시계방향), 같은 조명에서 돌린다.
    추천 출발점: 바깥 큰길 직선 가운데.
  - 순서: C5b -> A5b -> C (기존 기준)
  - 위에서 휴대폰으로 촬영하고, 몇 초에 어디서 벗어났는지 적는다.
  - 결과 폴더 runs/<이름>/ (telemetry.jsonl, summary.json)과 영상을 모아서 전달한다.


6. 알려진 한계

  - 차선만 따라간다. 표지판·신호등·교차로 방향 선택은 하지 않는다.
  - 왼쪽 위 모서리(조명 반사 심함)와 동쪽 좁은 도로(24cm)에서 이탈할 수 있다.
  - 카메라 입력: 640x480, 180도 회전(--flip180), 아래쪽 절반만 사용.
    카메라 각도(아래로 약 23도)가 바뀌면 모델이 잘 안 될 수 있다.
