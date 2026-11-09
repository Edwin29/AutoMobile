# AutoMobile 연구 기록

라즈베리파이 5·OV5647 단안 카메라와 엔코더/IMU 없는 4륜 차량으로 田자 트랙을 주행하는 프로젝트의 소스, 실험 결과와 개발 맥락이다. 2026-11-09 노트북 스냅샷을 정리했다.

다른 에이전트는 [프로젝트 맥락](docs/PROJECT_CONTEXT.md), [연구 과정](docs/RESEARCH_HISTORY.md), [증거와 한계](docs/EVIDENCE_AND_LIMITS.md)를 먼저 읽는다. 초기 문서는 codex/backup-before-research-20261109 브랜치에 보관했다.

## 현재 상태

- [실차 조건을 반영한 22개 환경 간섭 시험](research/real_stress_20261109/README.md): 시각 특징 오염과 누적 위치 오차가 서로 다른 정지 경로를 만드는 것을 재현했다. 실제 마지막 네 프레임과 96장 버퍼의 후보 탐색 결과도 포함한다.
- 팀원의 색 대비 C안, 독립 가상맵 외곽 제어기, 첫 합성학습 RoadNet은 서로 다른 코드다.
- 가상맵 제어기는 실차 코너를 부드럽게 통과한 사용자 관찰이 있고 이후 주행에서 정합 거부로 멈췄다. 양방향 완주 수치는 지정 조건의 시뮬레이션 결과다.
- RoadNet은 합성 480장으로 학습했다. 추가 실차 257장은 학습에 포함되지 않았다. 일부 경계 검출 개선과 횡단보도 오검출이 함께 관찰됐다.
- 국소 조향과 표식 기반 종방향 갱신은 다음 비교 후보다. 실차 검증된 통합 구조가 아니다.
- 현재 Pi 배포본, 전체 미션 통합, 실제 순간 회전 중심과 실차 연속 자세 정답은 이번 정리에서 확인하지 않았다.

## 문서

| 문서 | 내용 |
|---|---|
| [프로젝트 맥락](docs/PROJECT_CONTEXT.md) | 하드웨어·규칙·실측·다른 에이전트에 전달할 정정 설명 |
| [연구 과정](docs/RESEARCH_HISTORY.md) | 전원 진단부터 C안·가상맵·동적 보정·교차로·실차·CNN·재정의까지 |
| [증거와 한계](docs/EVIDENCE_AND_LIMITS.md) | 실차/재생/합성 구분, 정합 판정과 독립 정확도의 차이 |
| [예측 오류를 전제로 한 재설계 검토](docs/REDESIGN_REVIEW_20261109.md) | Claude 제안 검토, 국소 주행·구간 후보·미션·실행 판단 분리. 설계 단계 |
| [자료 목록](docs/DATA_CATALOG.md) | Drive, 입력 해시와 원본 영상의 접근 범위 |
| [재현 방법](docs/REPRODUCE.md) | 실행 환경, 스모크 검사, 시뮬레이션과 필요한 외부 자료 |
| [소스 목록](docs/source_manifest.json) | 원본·게시본 SHA-256과 경로 일반화 여부 |

## 코드 지도

| 경로 | 내용 |
|---|---|
| junction_logic/review_video, review_20261104, visibility_review/source | 팀원 코드와 당시 분석용 소스 스냅샷, 도면·렌더러 |
| map_localization/outer_driver | 지도 위치 보정·경로 추종·예측계수·출력 보상·Pi 실험판 |
| map_localization/c_drive_test, ideal_drive, corner_diagnosis | 기존 C안과 기준점 대조 |
| map_localization/junction_trial | 사거리·삼거리·진입로 횡단보도·가림·복구 실험 |
| map_localization/candidate_localizer | 자세 후보 탐색 |
| road_ai_v1 | 합성 데이터·Colab·RoadNet·실차 재생·원인 분리 |
| road_marks, collaborator, track_guard | 표식·미션·위상/차체 여유 설계 및 시험 |
| power_diagnostics, straight_bias_test | 모터·전원·직진 쏠림 시험 |
| camera_emulator, camera_diagnosis, line_tracer | 카메라 열화·보정·초기 분석 |
| pi_TEST_captures_20261105 | 수집한 과거 실행의 로그·설정 메타데이터 |

큰 실제 영상·사진은 [자료 목록](docs/DATA_CATALOG.md)에 접근 범위를 기록했다. Git에는 코드, 보정 배열, 합성 학습쌍, 수치 결과와 clean 체크포인트를 포함한다. 하위의 오래된 README는 당시 상태의 기록이다.

Git 날짜는 자료를 정리해 올린 날짜다. 채팅은 연구 주제별로 정리했으며 원문 덤프는 포함하지 않았다. 팀원 스냅샷과 이 채팅에서 만든 실험 코드가 혼재한다. 정밀한 저작·라이선스 이력이 확보되지 않아 새 오픈소스 라이선스를 일괄 부여하지 않았다.
