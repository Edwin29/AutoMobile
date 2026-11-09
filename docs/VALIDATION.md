# 게시본 검증 기록

2026-11-09 별도 clone에서 확인했다.

- 원본/게시본 manifest 2403개 파일의 게시본 SHA-256 일치
- Python 소스 AST 구문 및 Colab 코드 셀 compile 통과
- tools/smoke_sim.py의 기존 Driver와 CNNDriver 각 3스텝 폐루프 통과
- track_guard unittest 22개, collaborator 10개, road_marks 7개 통과
- 선택된 텍스트 파일에서 GitHub 토큰·private-key·하드코딩 자격증명 패턴 검사에 해당한 파일 없음
- 실제 영상과 Pi 하드웨어 실행, 한 바퀴 시뮬레이션의 신규 재실행은 이번 게시 검증에 포함하지 않음
- 과거 소스의 공백 스타일은 보존했다. CRLF를 허용한 git diff --check에서 기존 기록의 trailing whitespace/EOF 공백 13개가 남아 있고 기능 검증과 별도로 기록한다.

체크포인트와 합성 학습쌍은 포함했고 큰 실차 영상·사진은 DATA_CATALOG.md에 접근 범위를 기재했다. 모듈별 과거 실행 환경은 같지 않을 수 있다.
