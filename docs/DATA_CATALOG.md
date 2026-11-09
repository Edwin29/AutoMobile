# 데이터와 재현 자료

[Drive 폴더](https://drive.google.com/drive/folders/1Ib0vdqUi_qptSw2qgochYJUEm8T_28Be)는 소유자 접근 권한이 필요할 수 있다. 게시 과정에서 공유 권한을 변경하지 않았다.

| 파일 | 내용 |
|---|---|
| [starter](https://drive.google.com/file/d/1K-worq_CCis7M4k8sDkHGwa33333db1d/view) | 학습 코드·합성 704장·보정된 실차 진단 4장·참고선 |
| [clean 결과](https://drive.google.com/file/d/1j-gF5E_6NsXvHwVQ1p4dFM8t2vVLNNWb/view) | best.pt, metrics.json, 비교 이미지 |
| [촬영 257장](https://drive.google.com/file/d/1UNb2K1sC-EcrUyEugm52FhRqO_bDw76x/view) | PNG와 센서시각·노출·게인 CSV |
| [257장 추론](https://drive.google.com/file/d/1PSNVCsgLCoqkFLutsO9ZyuOThwx2tPev/view) | 마스크·2fps 비교 영상·통계 |
| [Colab](https://colab.research.google.com/drive/1Bjji3Gsp_CYr0fYt4zQRwbOpGEonqCCh) | 기본 ACTION=view, 학습은 새 회차로 저장 |

[data_checksums.json](data_checksums.json)에 로컬 사본 크기와 SHA-256을 기록했다. starter의 road_ai_kit/data/real은 필요하면 road_ai_v1/kit/data/real에 복원한다. 촬영 ZIP의 20261106_193448_316130 폴더는 road_ai_v1 아래에 둔다.

## 로컬에만 보관된 큰 원본

pi_TEST_captures_20261105/runs 아래의 6개 video_raw.mp4 원본은 이번 Git과 위 Drive 묶음에 포함되지 않았다. 이 저장소에는 실행 로그와 설정을 포함한다.

| 실행 | 프레임 수 |
|---|---:|
| 20261031_150910 | 364 |
| 20261031_151156 | 1633 |
| 20261031_151453 | 1042 |
| 20261031_151706 | 484 |
| 20261031_151830 | 31 |
| 20261031_151845 | 291 |

합계 3845프레임. 고정 FPS 영상 시간과 로그 t는 다를 수 있으므로 프레임 번호로 대응한다. 과거 첨부 video_raw.mp4는 151453과 해시가 같았다.

field_runs의 ccw_edges_20261106_173559_edges, ccw_edges_20261106_173933_edges, ccw_v135_20261106_174607_edges, ccw_forward127_20261106_184214_edges는 각각 정지 직전 24장이다. 실제 PNG는 로컬에 있고 이번 Git에는 index.csv와 prior·경계점 NPZ가 있다. 마지막 23.png의 t는 각각 23.07, 40.67, 61.27, 61.59초다.

## 정답 출처

- real_points.json은 일부 사진 경계의 좌표이며 작성자·검수 이력은 미확인이다.
- oracle_test.py SPOTS는 Codex가 축소 보정 사진을 보고 지정했다. 사용자 작성 정답으로 해석하지 않는다.
- train mask는 가상 도로와 단순 가림 도형의 합성 정답이다.
- 독립 실측한 연속 차량 자세 정답, 시연장 조명 데이터는 미확보다.
