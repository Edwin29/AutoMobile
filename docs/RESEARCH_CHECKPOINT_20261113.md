# 실험 기록 체크포인트 — 2026-11-13

direct_v2 복합 교란 비교부터 loss/학습 실험, 구조 감사, 숨긴 차량 응답 식별, hyb4 첫 사전 검증까지 보관한다. 실패·미채택 결과와 중간 progress도 남긴다. 최신 상태는 이 문서와 각 최종 보고서이며, 오래된 계획·진행 JSON의 '다음 실행' 문장을 현재 실행 중이라는 뜻으로 읽지 않는다.

**현재: 기존 개발 목표는 일시중지, 새 모델 채택 없음. hyb4는 기초 구현과 48회 사전 검증까지이며 학습하지 않았다. 영상 추정 조건이 기준에 미달해 학습 확대를 보류했다.** 기존 대조군/배포 모델과 최종 평가 시드는 변경하지 않았다.

## 결과를 읽는 순서

| 주제 | 기록 | 핵심 판단 |
|---|---|---|
| direct_v2 vs A5b+v4 | [복합 교란 240회](../cnn_finetune/results/robustness_20261112/REPORT.md) | 전체 개선과 약한 복합 교란의 회귀가 함께 존재 |
| 복원 실패와 loss | [복원 진단](../cnn_finetune/results/recovery_diagnosis_20261112/REPORT.md), [loss 구현](../cnn_finetune/trajectory_loss.py) | 교사 오차와 폐루프 주행 품질을 구분 |
| 교차로 형상·교사 | [초기 단계](../cnn_finetune/results/junction_stages02_20261113/REPORT.md), [코너링](../cnn_finetune/results/junction_cornering_20261113/REPORT.md), [여유 경로](../cnn_finetune/results/goal_clearance_validation_20261113/REPORT.md) | 부드러운 참조의 기하/교사 성능과 영상 학생 성능은 다름 |
| 재귀 정책 학습 | [교차로 학습](../cnn_finetune/results/junction_learning_20261113/REPORT.md), [28이동 후보](../cnn_finetune/results/goal_all_movement_eval_20261113/SEED17_REPORT.md) | 최근 28이동 후보 정상 27/84 무이탈, 심한 이탈 21회로 미채택 |
| 독립 구조 감사 | [감사](../cnn_finetune/results/independent_design_audit_20261113/REPORT.md), [실제 코드 구조](DRIVING_ARCHITECTURE_AUDIT_20261113.md) | 인식 CNN·주행 CNN·HSV 지도·움직임 추적의 책임 구분 |
| 알려진 경로 추종 | [기존 pilot](../cnn_finetune/results/goal_path_following_pilot_20261113/REPORT.md) | 기존 영상 조건 19/24, 사전 기준 미달 |
| 영상+PWM 응답 식별 | [52사례](../cnn_finetune/results/goal_hidden_response_20261113/REPORT.md) | 진단 입력에서 가능성, 일반 주행에서는 관측 부족/악화로 미채택 |
| hyb4 범위 | [설계](HYB4_SCOPE_PROPOSAL_20261113.md), [현재 구현 상태](../hyb4/README.md) | 일반 도로 기준 정책 유지, 교차로는 방향 조건부 경로+속도 출력 후보 |
| hyb4 최신 검증 | [48회 결과](../cnn_finetune/results/hyb4_image_prerequisite_20261113/REPORT.md), [실패 환경·해석](../cnn_finetune/results/hyb4_image_prerequisite_20261113/FAILURE_CONDITIONS.md) | 정확한 입력 24/24, 기존 영상 추정 21/24 무이탈. 기준22/24 미달 |
| 노트북 자료 회수 | [복구 목록](LAPTOP_RECOVERY_20261113.md) | 79,289파일 복구. 영상·로그 대응49묶음/13,595프레임 확보, 독립 자세 정답은 아님 |

그 밖의 실험별 protocol·검사·수치·도표는 `cnn_finetune/results/*_20261112`, `*_20261113`에 보존했다. `goal_development_20261113/work_status.json`의 누적 `new_training=true`는 과거 학습을 포함하며 최신 hyb4 학습을 뜻하지 않는다. 최신 hyb4 상태는 `hyb4_image_prerequisite_20261113/implementation_status.json`이다.

## 이번 Git 보관 범위

- 이번 세션의 실험 Python 코드, 보고서, protocol, 수치 결과, 도표와 hyb4 기초 구현.
- 최신 hyb4 48회에 대한 실제 적용 PWM 기록, 물리 궤적, 사건, 개별 완료 결과를 [원시 기록 묶음](../cnn_finetune/artifacts/hyb4_20261113)에 포함했다. 20개 부드러운 회전 참조와 고정 protocol의 저장소 입력84개도 원본 바이트로 복사했다. [manifest](../cnn_finetune/artifacts/hyb4_20261113/manifest.json)에 SHA-256과 원래 위치를 기록했다. 묶음에는 Git 줄바꿈 변환을 적용하지 않는다.
- 노트북 복구 파일명·크기 대조 결과와 촬영 자료 활용 범위는 문서로 보관한다. 약9.48GB 복구 원본, 전체 학습 영상·체크포인트·과거 모든 원시 rollout은 로컬 보관이다. 이번 커밋은 모든 과거 실험의 완전한 독립 재실행 패키지가 아니다.
- 고정 protocol의 원래 절대 경로와 해시는 바꾸지 않았다. 과거 소스 버전과 현 파일이 다른 경우 해당 실험의 snapshot/manifest가 우선한다. 경로가 기록됐다는 이유로 그 로컬 파일도 Git에 포함됐다고 가정하지 않는다.

## 다른 PC에서 최신 결과 확인

저장소 루트에서 기존 bench 의존성을 설치한 Python 환경으로 실행한다.

```sh
python -m cnn_finetune.verify_research_checkpoint
python -m unittest hyb4.test_control hyb4.test_coordinator -v
```

첫 명령은 원래 D드라이브 경로 없이 Git에 포함된 파일 해시와 최신 고정 protocol의 소스/참조 해시를 검사하고, 저장된 48회 물리 궤적에서 차체 여유를 다시 계산한다. **새 시뮬레이션이나 학습이 아니다.** 두 번째는 하드웨어를 사용하지 않는 인터페이스 검사 7개다. 검사 기록은 [verification.json](../cnn_finetune/results/research_checkpoint_20261113/verification.json)에 있다.

고정 protocol 해시는 묶음의 원본 바이트로 검사한다. 실제 실행되는 저장소 소스도 묶음과 비교하되, Git checkout에서 생길 수 있는 CRLF/LF 차이만 허용한다.

hyb4의 전체 bench/Pi 등록, 일반 도로 전환·미션 연결, 새로운 CNN loss/학습, 복합 교란 및 전체 코스 검증은 남아 있다. 이번 보관 과정에서 그 작업을 재개하지 않았다.
