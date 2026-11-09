# 에이전트 안내

docs/PROJECT_CONTEXT.md, docs/RESEARCH_HISTORY.md, docs/EVIDENCE_AND_LIMITS.md를 먼저 읽는다. 하위 README는 작성 당시 상태다.

- Raspberry Pi 5 + OV5647 단안 카메라. 엔코더·IMU 없음. 코드상 PWM 듀티·방향 제어이며 스텝모터 확인 자료 없음.
- C안은 색 대비 주행 코드. track-digital-twin CNN, RoadNet, 가상맵 제어기를 혼동하지 않는다.
- 정합 통과는 독립 자세 정답이나 실제 완주율이 아니다. 시뮬레이터 정답은 채점에만 쓴다.
- 표식 상대 회전 예측계수와 출력 보상계수를 구분한다.
- real_points.json 작성자는 확인되지 않았다. oracle_test.py의 SPOTS는 Codex 육안 좌표다. 사람 검증 정답 또는 완전한 oracle로 보고하지 않는다.
- 2026-11-09 현재 차량은 사용자 손에 없다. 새 요청 없이 SSH 모터 실험을 시작하지 않는다.
- 원본 기록을 보존하고 새 결과에 입력·코드·보정 버전을 남긴다.
- 해상도·보정 여부·180도 처리·촬영 세션·차체와 차축 기준점을 확인한다.
- 비밀번호·토큰·개인 호스트·환경 폴더를 커밋하지 않는다.
- 실제 실행한 검사 범위만 보고한다. 일반 검사에서 GPIO·카메라 프로그램을 import하거나 실행하지 않는다.
