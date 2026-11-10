# 팀원 CNN 차선 주행 모델 (고정 사본)

- 출처: https://github.com/gawbi/track-digital-twin (코드 커밋은 SOURCE_COMMIT.txt)
- 모델: GitHub Release `models-2026-11-01`의 `pi_models_2026-10-31.zip` (4.66MB, 2026-11-11 받음)
- 입력: 차량 카메라 원본 640x480(180° 회전 후, 어안 왜곡 그대로) → 아래 절반 → YUV → 3x3 흐림 → 200x66
- 출력: [전진, 회전] in [-1, 1] (`normalized_linear_angular_v1`, 회전 + = 왼쪽)
- 모터: configs/real_car.json의 DutyMixer (min_duty 0.175, max_duty 0.66, turn_mix 0.72), 실행 속도 배율 0.5
- 이 폴더는 bench에서 비교 시험용으로만 읽는다. 원저작자의 저장소 라이선스를 따른다.
