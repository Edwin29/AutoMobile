# 교차로 학습 준비 단계 0–2 실행 결과

2026-11-13. 데스크톱 bench 시뮬레이션. 새 신경망 학습·모델 교체·Pi 접속·실차 동작·커밋/푸시는 수행하지 않았다.

## 판정

단계 0–2의 평가 계약, 실행 재현 검사, 이동별 연속 경로 탐색을 완료했다. 36개 기하학적 이동 중 기본 차체·중앙 정렬 시작에서 경계 침범 없이 안정 출구에 도달한 후보는 36/36개다. 불필요한 정지 0회 후보는 36/36개다. 중앙 좌회전/직진 8개는 미션 비허용 실험이며 실제 미션 대상은 28개다.
이 결과는 정답 자세를 이용한 오프라인 교사의 실행 가능성이다. CNN이 같은 궤적을 영상만으로 따라갈 수 있다거나 실차에서 안전하다는 증거가 아니다. 약한 복합 교란의 무이탈 통과는 아직 충분하지 않다. 학습용으로 사용할 때 깨끗한 후보·회복 분기·실패 분기를 구분해야 한다.

## 단계 0: 고정 계약

- `contract.json`: 기존 ONNX/체크포인트, pinned v4, 지도·차체·카메라 및 소스 해시, 평가 정의를 보존했다.
- 개발 시드 1301/1303/1307. 경로 매개변수의 기하 탐색 seed 2718, 속도 선택은 개발 1303만 사용했다. 최종 시험 3001/3007/3011/3019/3023/3037은 실행하지 않았다. 학습 예정 시드 2001–2020도 미사용이다.
- 출구 진행 40cm 이상, 차체 여유 0cm 이상, 횡오차 1cm 이내, 방향 오차 5° 이내를 연속 1초 유지해야 성공이다. 회복은 해당 참조 경로의 같은 횡/방향/여유 조건으로 1초 유지한다.
- 실패 및 미복귀는 분모에 남기고 회복시간은 null로 보존한다. 0.1초 고정 격자의 실제 속도·회전속도로 가속 RMS를 계산한다. 의무 정지는 불필요한 정지 비용에서 제외한다.
- 기존 배치 오차(횡 ±1cm, 종 ±2cm, 방향 ±3°)를 유지했다. 좁은 도로의 seed 1301은 출발부터 약 -0.158cm 침범한다. 이 경우 전체 무이탈 성공으로 세지 않고, 초기 정상 배치 조건부 성적도 병기한다. 중앙 정렬 시작의 별도 실행은 경로 자체의 가능성을 확인하는 용도다.

## 단계 1: 실행과 출력 검증

- 원래 `bench.junction.run`을 공통 루프로 사용했다. 수집은 관찰자이며 촬영·36ms 처리·이전 명령 유지·명령 큐·최대 20ms 물리 적분·카메라 이력·지도 가드를 바꾸지 않는다.
- A5b+v4, direct_v2, 강한 조명+약한 차체 변동의 3개 쌍에서 수집 on/off의 적용 PWM, 실제 자세·속도·경계 여유가 정확히 일치했다. 지도 추정값은 최대 약 2.53e-7 차이였고 1e-6 절대 허용오차를 통과했다. 벽시계 처리시간 로그는 일치 대상에서 제외했다.
- signed 좌/우 PWM v1의 전진·안쪽 정지·양쪽 역회전·좌우 제자리 회전·낮은 출력 회전·정지/재개를 검사했다. 실제 부호와 지도 예측 부호가 일치한다. 저출력 디더 평균과 1명령당 1갱신도 검사했다.
- 외곽 direct 출력 계약은 변경하지 않았다. 기존 역회전용 `HybridJunction.on_applied`를 확인했다. 새 신경망의 signed 출력 헤드는 아직 만들거나 학습하지 않았다.

## 단계 2: 동역학 결과

기존 원호·제자리·분할 회전을 보존하고, 접선 진입/출구와 5차 Bézier 곡선을 이용한 가변 곡률 후보를 탐색했다. 촘촘한 정적 차체 검사 후 원래 Vehicle의 지연·출력 문턱·포화·1차 응답·좌우 비대칭·역회전 밀림을 적용했다. 후보 제어기는 capture 시점 정답 자세/속도와 **명목** 차체 파라미터를 사용하며, 난수로 변한 실제 효율/출력 교란을 미리 알지 못한다.
1296회 = 36이동 × 4차체(field_estimate/sluggish/unequal_sides/field_pivot_slip) × 3조건 × 3개발 시드. 속도 선택 108회, 중앙 정렬 36회, 기존 제어기와 초기 자세를 정확히 맞춘 추가 36회는 별도다. 앞선 탐색용 12회는 최종 분모에 중복하지 않았다.

| 조건 | 안정 출구 | 전체 무이탈 안정 출구 | 초기 경계 안에서 출발한 경우의 무이탈 |
|---|---:|---:|---:|
| fixed | 432/432 | 364/432 | 364/392 |
| mild | 432/432 | 273/432 | 273/392 |
| harsh | 379/432 | 21/432 | 21/392 |

mild는 차체 mild 변동 + 횡밀림 1–2cm/방향 1–3°(0.3–0.7초) + 출력 ±5–15%(2–5초)를 함께 적용했다. harsh는 차체 harsh + 4–6cm/6–10° + ±20–35%다. 교차로를 겨냥한 외부 사건은 시작 7–8초 및 15–16초의 고정 시드 일정이며 제어기 상태와 독립이다. 원래 외곽 실험의 12–17초 일정과 다르며 실측 분포가 아니다. 노출 전 종료한 사건은 적용됐다고 세지 않는다.
조명은 정답 자세를 보는 후보의 성능을 평가할 수 없다. 별도 기존 영상 제어기 72회 중 36회는 전체 이동 기본 조건, 나머지 36회는 남·북 합류/중앙 우회전 6이동 × 3시드 × mild+강한 조명 또는 harsh+harsh 카메라로 검사했다. 두 제어기의 교란 성적은 관측 정보가 다르므로 CNN 개선 효과로 직접 비교하지 않는다.

| 기존 A5b+v4 hyb2j | 새 기준 안정 출구 | 무이탈 안정 출구 |
|---|---:|---:|
| fixed | 36/36 | 30/36 |
| mild | 1/18 | 1/18 |
| harsh | 2/18 | 0/18 |

기존 제어기의 time_limit에는 올바른 길로 나갔어도 출구 정렬 1cm/5°를 1초 유지하지 못한 경우가 포함된다. 이후 외곽 주행까지 진행한 사례도 있으므로 “전부 교차로를 못 돌았다”는 뜻이 아니다. 이전의 넓은 출구 판정과 새 안정 정렬 성공을 혼용하지 않는다.

### 같은 기본 시작 조건의 기존 동작과 후보

field_estimate/fixed/seed1303의 36이동: 기존 안정 36/36, 무이탈 30/36; 정답 자세 후보 안정 36/36, 무이탈 32/36. 불필요한 정지 횟수 합계는 기존 2, 후보 0다. 이는 오프라인 목표 경로 개선의 근거이며 센서 조건을 맞춘 정책 우열 비교가 아니다.
같은 36이동의 주행별 평균 RMS: 선형 가속 5.671 → 3.437 cm/s², 각가속 0.331 → 0.268 rad/s². 추가 진단인 제자리 회전을 포함한 전진 멈춤 시간(|v|<0.5cm/s가 0.3초 이상, 초기 1초 제외)은 평균 2.192 → 0.000초다. 이 보조 지표는 계약의 완전 정지 지표를 대체하지 않는다.
전체 동역학 결과에서 노출된 밀림 사건 1561건 중 미복귀 335건을 그대로 보존했다. 밀림 중 종료된 실패도 미복귀에 포함했다. 여러 사건/프레임을 독립 주행 표본으로 해석하지 않는다.

## 이동별 결과

각 조건 분모는 12(4차체×3시드). `opposing`은 차체가 전진하면서 일부 구간 안쪽 바퀴를 반대 방향으로 구동함을 뜻한다. 불연속 제자리 정지 단계와 구분한다. 영상은 초록=정답 자세 후보, 파랑=기존 제어기이며 초기 자세를 정확히 맞췄다. 이는 기본 차체/fixed/seed1303 비교이고 위의 1296회 분모에 추가하지 않는다.

| 이동 | 기본 실행 분류 | mild 안정/무이탈 | harsh 안정/무이탈 | 경로·영상 |
|---|---|---:|---:|---|
| center_N_straight | continuous_forward | 12/12 · 8/12 | 12/12 · 0/12 | [경로](movements/center_N_straight/paired_path.png) · [영상](movements/center_N_straight/paired_comparison.mp4) |
| center_N_left | continuous_with_opposing_wheel | 12/12 · 4/12 | 12/12 · 0/12 | [경로](movements/center_N_left/paired_path.png) · [영상](movements/center_N_left/paired_comparison.mp4) |
| center_N_right | continuous_with_opposing_wheel | 12/12 · 4/12 | 12/12 · 0/12 | [경로](movements/center_N_right/paired_path.png) · [영상](movements/center_N_right/paired_comparison.mp4) |
| center_E_straight | continuous_forward | 12/12 · 12/12 | 12/12 · 3/12 | [경로](movements/center_E_straight/paired_path.png) · [영상](movements/center_E_straight/paired_comparison.mp4) |
| center_E_left | continuous_with_opposing_wheel | 12/12 · 3/12 | 11/12 · 0/12 | [경로](movements/center_E_left/paired_path.png) · [영상](movements/center_E_left/paired_comparison.mp4) |
| center_E_right | continuous_with_opposing_wheel | 12/12 · 1/12 | 11/12 · 0/12 | [경로](movements/center_E_right/paired_path.png) · [영상](movements/center_E_right/paired_comparison.mp4) |
| center_S_straight | continuous_forward | 12/12 · 8/12 | 12/12 · 0/12 | [경로](movements/center_S_straight/paired_path.png) · [영상](movements/center_S_straight/paired_comparison.mp4) |
| center_S_left | continuous_with_opposing_wheel | 12/12 · 4/12 | 12/12 · 0/12 | [경로](movements/center_S_left/paired_path.png) · [영상](movements/center_S_left/paired_comparison.mp4) |
| center_S_right | continuous_with_opposing_wheel | 12/12 · 4/12 | 12/12 · 0/12 | [경로](movements/center_S_right/paired_path.png) · [영상](movements/center_S_right/paired_comparison.mp4) |
| center_W_straight | continuous_forward | 12/12 · 12/12 | 12/12 · 3/12 | [경로](movements/center_W_straight/paired_path.png) · [영상](movements/center_W_straight/paired_comparison.mp4) |
| center_W_left | continuous_with_opposing_wheel | 12/12 · 3/12 | 11/12 · 0/12 | [경로](movements/center_W_left/paired_path.png) · [영상](movements/center_W_left/paired_comparison.mp4) |
| center_W_right | continuous_with_opposing_wheel | 12/12 · 1/12 | 11/12 · 0/12 | [경로](movements/center_W_right/paired_path.png) · [영상](movements/center_W_right/paired_comparison.mp4) |
| north_E_straight | continuous_forward | 12/12 · 12/12 | 6/12 · 4/12 | [경로](movements/north_E_straight/paired_path.png) · [영상](movements/north_E_straight/paired_comparison.mp4) |
| north_E_left | continuous_with_opposing_wheel | 12/12 · 0/12 | 12/12 · 0/12 | [경로](movements/north_E_left/paired_path.png) · [영상](movements/north_E_left/paired_comparison.mp4) |
| north_S_left | continuous_with_opposing_wheel | 12/12 · 8/12 | 12/12 · 0/12 | [경로](movements/north_S_left/paired_path.png) · [영상](movements/north_S_left/paired_comparison.mp4) |
| north_S_right | continuous_with_opposing_wheel | 12/12 · 8/12 | 12/12 · 2/12 | [경로](movements/north_S_right/paired_path.png) · [영상](movements/north_S_right/paired_comparison.mp4) |
| north_W_straight | continuous_forward | 12/12 · 12/12 | 8/12 · 0/12 | [경로](movements/north_W_straight/paired_path.png) · [영상](movements/north_W_straight/paired_comparison.mp4) |
| north_W_right | continuous_with_opposing_wheel | 12/12 · 0/12 | 12/12 · 0/12 | [경로](movements/north_W_right/paired_path.png) · [영상](movements/north_W_right/paired_comparison.mp4) |
| east_N_straight | continuous_forward | 12/12 · 12/12 | 6/12 · 0/12 | [경로](movements/east_N_straight/paired_path.png) · [영상](movements/east_N_straight/paired_comparison.mp4) |
| east_N_right | continuous_with_opposing_wheel | 12/12 · 8/12 | 12/12 · 0/12 | [경로](movements/east_N_right/paired_path.png) · [영상](movements/east_N_right/paired_comparison.mp4) |
| east_S_straight | continuous_forward | 12/12 · 12/12 | 7/12 · 1/12 | [경로](movements/east_S_straight/paired_path.png) · [영상](movements/east_S_straight/paired_comparison.mp4) |
| east_S_left | continuous_with_opposing_wheel | 12/12 · 9/12 | 12/12 · 0/12 | [경로](movements/east_S_left/paired_path.png) · [영상](movements/east_S_left/paired_comparison.mp4) |
| east_W_left | continuous_with_opposing_wheel | 12/12 · 12/12 | 12/12 · 0/12 | [경로](movements/east_W_left/paired_path.png) · [영상](movements/east_W_left/paired_comparison.mp4) |
| east_W_right | continuous_with_opposing_wheel | 12/12 · 12/12 | 12/12 · 1/12 | [경로](movements/east_W_right/paired_path.png) · [영상](movements/east_W_right/paired_comparison.mp4) |
| south_N_left | continuous_with_opposing_wheel | 12/12 · 8/12 | 12/12 · 0/12 | [경로](movements/south_N_left/paired_path.png) · [영상](movements/south_N_left/paired_comparison.mp4) |
| south_N_right | continuous_with_opposing_wheel | 12/12 · 8/12 | 12/12 · 0/12 | [경로](movements/south_N_right/paired_path.png) · [영상](movements/south_N_right/paired_comparison.mp4) |
| south_E_straight | continuous_forward | 12/12 · 12/12 | 4/12 · 0/12 | [경로](movements/south_E_straight/paired_path.png) · [영상](movements/south_E_straight/paired_comparison.mp4) |
| south_E_right | continuous_with_opposing_wheel | 12/12 · 0/12 | 12/12 · 0/12 | [경로](movements/south_E_right/paired_path.png) · [영상](movements/south_E_right/paired_comparison.mp4) |
| south_W_straight | continuous_forward | 12/12 · 12/12 | 2/12 · 0/12 | [경로](movements/south_W_straight/paired_path.png) · [영상](movements/south_W_straight/paired_comparison.mp4) |
| south_W_left | continuous_with_opposing_wheel | 12/12 · 0/12 | 12/12 · 0/12 | [경로](movements/south_W_left/paired_path.png) · [영상](movements/south_W_left/paired_comparison.mp4) |
| west_N_straight | continuous_forward | 12/12 · 12/12 | 6/12 · 4/12 | [경로](movements/west_N_straight/paired_path.png) · [영상](movements/west_N_straight/paired_comparison.mp4) |
| west_N_left | continuous_with_opposing_wheel | 12/12 · 9/12 | 12/12 · 0/12 | [경로](movements/west_N_left/paired_path.png) · [영상](movements/west_N_left/paired_comparison.mp4) |
| west_E_left | continuous_with_opposing_wheel | 12/12 · 12/12 | 12/12 · 0/12 | [경로](movements/west_E_left/paired_path.png) · [영상](movements/west_E_left/paired_comparison.mp4) |
| west_E_right | continuous_with_opposing_wheel | 12/12 · 12/12 | 12/12 · 1/12 | [경로](movements/west_E_right/paired_path.png) · [영상](movements/west_E_right/paired_comparison.mp4) |
| west_S_straight | continuous_forward | 12/12 · 12/12 | 8/12 · 2/12 | [경로](movements/west_S_straight/paired_path.png) · [영상](movements/west_S_straight/paired_comparison.mp4) |
| west_S_right | continuous_with_opposing_wheel | 12/12 · 7/12 | 12/12 · 0/12 | [경로](movements/west_S_right/paired_path.png) · [영상](movements/west_S_right/paired_comparison.mp4) |

## 다음 단계에 남긴 조건

방향을 근본적으로 바꿔야 하는 결정은 발견하지 않았다. 연속 회전의 실행 가능성을 확인했으므로 계획의 시간 입력+방향 조건+signed 출력 방향을 유지할 수 있다. 그러나 이번 결과로 교차로 전체가 교란에 강하다고 판단하지 않는다.
단계 3에서는 후보가 깨끗하게 통과한 분기와 경계 침범 후 복구한 분기를 구분해 보존하고, 현재 강한 교란 실패를 목표 정답으로 복제하지 않아야 한다. 이 단계에서 단순 교사 오차만 줄여도 개선된다는 주장은 하지 않는다. 영상 기반 관측, 방향 명령, 적용 PWM 이력으로 가능한 회복인지 단계 3–4에서 따로 검증해야 한다.
실제 차체/트랙 오차, 새 신경망의 추론 지연, Pi의 역회전 구동 특성, 영상 기반 강한 조명 일반화, 회전 도중 STOP/재개를 포함한 전체 미션은 후속 검증 대상이다. 이번 STOP 검사는 출력 계약과 지연 응답의 단위 검사다.

## 산출물과 재실행

- 원시 물리·명령·교란·영상 해시 로그: `D:\autodrive_cnn\runs\junction_stages02_20261113`.
- `contract.json`, `parity.json`, `action_metric_checks.json`, `audit.json`, `selection_trials.json`, `audited_dynamic_results.json`, `audited_legacy_results.json`, `paired_results.json`, `failure_cases.json`, `movement_results.csv`, `execution_manifest.json`.
- 실행 모듈: `cnn_finetune.junction_stage02`, `junction_stage02_checks`, `junction_feasibility`, `junction_stage02_baselines`, `junction_stage02_report`.
- 원시 결과 폴더는 덮어쓰지 않는다. 새 반복은 OUT/REPORT를 별도 날짜·실험 ID로 지정해야 한다. 최종 시험 시드는 이 보고서에서 사용하지 않았다.
