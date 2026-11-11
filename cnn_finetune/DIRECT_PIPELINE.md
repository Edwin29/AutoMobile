# direct 파이프라인 실행 안내

`HANDOFF.md`의 3-3과 자율 진행 규칙이 우선한다.

- 교사: `direct_teacher.DirectTeacher(variant=3)`. 시뮬레이터 자세 및 차량 파라미터를 사용한다. 모델 실행에는 정답 자세가 제공되지 않는다.
- 출력: `[요청 평균 듀티/0.66, -bench 곡률/0.095]`, 곡률 상한 0.085/cm.
- 교사 확인: `python -m cnn_finetune.direct_run --variant 3 --out D:\autodrive_cnn\runs\new_teacher`
- 데이터: `collect.py`, 시드별 npz/메타데이터 JSON. `audit_data.py`로 전체 검사.
- 학습: D: 가상환경의 Python으로 `train.py` 실행. 학습/검증/시험은 시드 끝자리로 분리한다.
- 기준선 MAE: A5b 원출력을 팀 변환의 평균 듀티와 v4 명목 곡률로 변환한 뒤 direct 라벨과 비교한다. 의미가 다른 원출력 숫자를 직접 비교하지 않는다.
- 시뮬레이터: `evaluate.py`가 같은 조건에서 baseline=v4, new=direct를 환경 변수로 선택한다.
- `compare_metrics.py`: 기준 경로의 |곡률|>0.02 구간을 코너로 정의. 코너를 나온 뒤 기준 경로 방향 오차가 5도 이내가 되는 첫 프레임까지 시간을 측정한다. 끝까지 회복하지 않은 경우 별도 기록한다. 코너 프레임에서 안쪽 PWM<0.30 비율에는 0도 포함한다.

교사 후보 1/2와 과거 팀 변환 교사 파일은 재현용이다. 원본 bench/모델/지도는 수정하지 않는다.
데이터·체크포인트·로그는 저장소 밖 `D:\autodrive_cnn\`에 보관한다.
6단계 실차 사진 경로는 `D:\autodrive_cnn\pi_data\20261111`이며 이번 3단계 작업에서는 사용하지 않는다.
