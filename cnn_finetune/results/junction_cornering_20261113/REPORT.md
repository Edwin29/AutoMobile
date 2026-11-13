# 교차로 코너링 경로 개선 실험

2026-11-13. 이전 단계 0–2의 매우 작은 반경 회전을 더 넓고 점진적인 코너링으로 개선하는 후속 실험이다. CNN 학습과 하드웨어 동작은 수행하지 않았다.

24개 회전 이동을 탐색했고, 개발 검증 기준을 통과한 교사 후보는 **16/24개**다. 이 중 현재 미션에서 허용되는 이동은 14개다. 나머지는 이전 후보와 함께 보존하며 새 교사로 채택하지 않는다. 중앙 좌회전 4개는 기하학 실험이며 미션에 적용하지 않는다.

## 변경과 분리 비교

- 이전 목적함수는 경계 여유를 크게 만드는 것이 우선이어서 최소 반경 약 1.5cm인 경로도 선택했다. 이번에는 약 0.5cm/1cm 여유 후보를 각각 탐색하고, 최대 곡률·곡률 변화·곡률 변화율을 줄였다. 그 결과를 촘촘한 차체 윤곽 검사로 다시 확인했다.
- 진입·회전·출구를 모두 5차 Bézier로 연결하고 접속점의 곡률을 0으로 맞췄다. 미분은 해석적으로 계산했다. 기존 경로도 그대로 보존했다. 탐색은 최적해/가능 최대 반경의 증명이 아니다.
- 곡률 기반 기존 속도 방식과 회전속도/횡가속 상한 및 전후방 가속 제약을 둔 미리 감속하는 속도 계획을 비교했다. 조향 저역통과 필터를 추가하지 않았다. 속도 계획이 더 나쁘면 기존 속도 방식을 선택했다.
- 24이동 × (기존 1 + 새 경로 2 × 속도 2) × 2차체 × 2조건 × 개발 seed1303 = 480회 선택 평가 후 이동별 조합을 고정했다. 선택 자료도 모두 보존했다.
- 검증은 24이동 × 기존/후보 2 × 4차체 × fixed/mild/harsh 3 × 3시드 = 1728회다. 이는 개발 시드 1301/1303/1307의 비교이며 선택에 사용한 1303도 포함한다. 새로운 최종 시험 일반화 성적은 아니다. 최종 시험 및 학습 시드는 미사용이다.
- 모든 쌍의 시작 자세와 외부 사건 일정이 같음을 검사했다. 36ms 처리, 최대 20ms 물리 적분, 실제 구동 지연·출력 문턱·포화·역회전 밀림을 유지했다. 원래 실행과 새 공통 루프의 PWM/물리 궤적 재현 검사도 수행했다.

## 전체 후보의 검증 성적

아래에는 보류 후보도 포함한다. 실패를 숨기거나 통과 사례만 골라 전체 개선으로 보고하지 않는다. 초기 경계를 밟은 배치도 무이탈 분모에서 제외하지 않았다.

| 조건 | 안정 출구: 이전 → 후보 | 무이탈 안정 출구: 이전 → 후보 |
|---|---:|---:|
| fixed | 288/288 → 288/288 | 228/288 → 240/288 |
| mild | 288/288 → 288/288 | 137/288 → 149/288 |
| harsh | 284/288 → 278/288 | 4/288 → 0/288 |

fixed의 주행별 중앙값: 교차로 중심 30cm 안에서의 각가속 RMS 0.425 → 0.243 rad/s². 통과 시간 19.02 → 17.85초.
밀림과 출력 증감은 이전 단계와 같은 고정 시드 합성 일정이다. 경로와 속도에 따라 사건 당시 차량 위치와 종료 전 노출량은 달라질 수 있다. 쌍별 노출 시간을 paired_changes.json에 함께 보존했다. 속도·회전속도 채널은 Vehicle의 구동 응답이며 외부에서 강제로 더한 자세 밀림은 포함하지 않으므로, 부드러움 개선 주장은 외부 밀림이 없는 fixed 비교에 한정한다. 광학 입력을 사용하지 않는 정답 자세 교사이므로 강한 조명에 대한 CNN 개선을 주장하지 않는다.

## 교사 후보의 채택 조건

fixed/mild 각각에서 안정 도달 및 무이탈 건수가 감소하지 않고, 중앙 정렬 시작에서 경계 침범·전진 멈춤이 없으며, fixed의 회전 중 각가속 중앙값이 5% 이상 감소하고, 최소 참조 반경이 50% 이상 증가해야 한다. 평균 통과 시간이 기존의 1.35배+2초를 넘으면 보류한다. 개별 시드의 교환 관계는 paired_changes.json에 남겼다. harsh는 진단 범위이며 통과를 보장하지 않는다.
보류 사유 집계(중복 가능): `{'less_than_5pct_turn_smoothness_gain': 4, 'mild_clean_stable_regression': 4}`. 채택은 개발용 참조 경로에 한정하며 실제 데이터 구축 시 각 실행을 깨끗한 성공/회복/실패로 다시 구분해야 한다. 특히 north_W_right는 mild 무이탈 0/12이므로 기본 조건의 코너링 참조 및 복구 분기 검토용이다. 이를 약한 교란의 깨끗한 교사로 사용해서는 안 된다.

| 이동 | 반경 cm: 이전 → 후보 | fixed 무이탈 /12 | mild 무이탈 /12 | 상태 | 근거 |
|---|---:|---:|---:|---|---|
| center_N_left | 1.72 → 15.55 | 7 → 8 | 4 → 4 | accepted_dev | [그래프](movements/center_N_left/comparison.png) · [영상](movements/center_N_left/comparison.mp4) |
| center_N_right | 1.60 → 5.89 | 7 → 8 | 4 → 4 | accepted_dev | [그래프](movements/center_N_right/comparison.png) · [영상](movements/center_N_right/comparison.mp4) |
| center_E_left | 1.66 → 3.69 | 10 → 10 | 3 → 6 | held | [그래프](movements/center_E_left/comparison.png) · [영상](movements/center_E_left/comparison.mp4) |
| center_E_right | 1.60 → 3.10 | 9 → 10 | 1 → 3 | held | [그래프](movements/center_E_right/comparison.png) · [영상](movements/center_E_right/comparison.mp4) |
| center_S_left | 1.72 → 15.55 | 7 → 8 | 4 → 4 | accepted_dev | [그래프](movements/center_S_left/comparison.png) · [영상](movements/center_S_left/comparison.mp4) |
| center_S_right | 1.60 → 5.89 | 7 → 8 | 4 → 4 | accepted_dev | [그래프](movements/center_S_right/comparison.png) · [영상](movements/center_S_right/comparison.mp4) |
| center_W_left | 1.66 → 3.69 | 10 → 10 | 3 → 6 | held | [그래프](movements/center_W_left/comparison.png) · [영상](movements/center_W_left/comparison.mp4) |
| center_W_right | 1.60 → 3.10 | 9 → 10 | 1 → 3 | held | [그래프](movements/center_W_right/comparison.png) · [영상](movements/center_W_right/comparison.mp4) |
| north_E_left | 2.14 → 8.96 | 9 → 10 | 0 → 1 | accepted_dev | [그래프](movements/north_E_left/comparison.png) · [영상](movements/north_E_left/comparison.mp4) |
| north_S_left | 1.49 → 20.04 | 8 → 8 | 8 → 7 | held | [그래프](movements/north_S_left/comparison.png) · [영상](movements/north_S_left/comparison.mp4) |
| north_S_right | 3.17 → 21.06 | 8 → 8 | 8 → 8 | accepted_dev | [그래프](movements/north_S_right/comparison.png) · [영상](movements/north_S_right/comparison.mp4) |
| north_W_right | 1.63 → 8.60 | 9 → 10 | 0 → 0 | accepted_dev | [그래프](movements/north_W_right/comparison.png) · [영상](movements/north_W_right/comparison.mp4) |
| east_N_right | 1.64 → 21.21 | 11 → 12 | 8 → 12 | accepted_dev | [그래프](movements/east_N_right/comparison.png) · [영상](movements/east_N_right/comparison.mp4) |
| east_S_left | 1.62 → 21.44 | 12 → 12 | 9 → 4 | held | [그래프](movements/east_S_left/comparison.png) · [영상](movements/east_S_left/comparison.mp4) |
| east_W_left | 3.16 → 25.52 | 12 → 12 | 12 → 12 | accepted_dev | [그래프](movements/east_W_left/comparison.png) · [영상](movements/east_W_left/comparison.mp4) |
| east_W_right | 2.37 → 25.56 | 12 → 12 | 12 → 12 | accepted_dev | [그래프](movements/east_W_right/comparison.png) · [영상](movements/east_W_right/comparison.mp4) |
| south_N_left | 1.95 → 19.45 | 8 → 8 | 8 → 4 | held | [그래프](movements/south_N_left/comparison.png) · [영상](movements/south_N_left/comparison.mp4) |
| south_N_right | 1.50 → 7.58 | 8 → 8 | 8 → 8 | accepted_dev | [그래프](movements/south_N_right/comparison.png) · [영상](movements/south_N_right/comparison.mp4) |
| south_E_right | 1.31 → 2.69 | 9 → 10 | 0 → 4 | accepted_dev | [그래프](movements/south_E_right/comparison.png) · [영상](movements/south_E_right/comparison.mp4) |
| south_W_left | 1.38 → 2.74 | 9 → 10 | 0 → 3 | accepted_dev | [그래프](movements/south_W_left/comparison.png) · [영상](movements/south_W_left/comparison.mp4) |
| west_N_left | 1.62 → 21.44 | 12 → 12 | 9 → 4 | held | [그래프](movements/west_N_left/comparison.png) · [영상](movements/west_N_left/comparison.mp4) |
| west_E_left | 3.16 → 25.52 | 12 → 12 | 12 → 12 | accepted_dev | [그래프](movements/west_E_left/comparison.png) · [영상](movements/west_E_left/comparison.mp4) |
| west_E_right | 2.37 → 25.56 | 12 → 12 | 12 → 12 | accepted_dev | [그래프](movements/west_E_right/comparison.png) · [영상](movements/west_E_right/comparison.mp4) |
| west_S_right | 1.64 → 21.21 | 11 → 12 | 7 → 12 | accepted_dev | [그래프](movements/west_S_right/comparison.png) · [영상](movements/west_S_right/comparison.mp4) |

## 이어갈 때

teacher_registry.json의 accepted_dev만 새 코너링 참조 후보로 사용한다. held는 보류 사유를 해결하기 전까지 이전 자료와 비교 대상으로 유지한다. 경로가 넓어졌다는 이유만으로 경계 침범을 허용하지 않았다. 새 loss와 시퀀스 CNN 학습은 이번 작업에 포함하지 않았다.
원시 결과: `D:\autodrive_cnn\runs\junction_cornering_20261113`. 상세 설정·선택·모든 결과·회귀 테스트·소스/모델 해시는 protocol.json, selection_trials.json, audited_validation.json, verification.json에 연결돼 있다.
