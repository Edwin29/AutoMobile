"""Build final Korean report from completed paired results only."""
import json,hashlib
from pathlib import Path
import numpy as np
from .compare_metrics import extra

if __name__=='__main__':
    rows=[]
    for folder in ['direct_comparison','direct_new_comparison']:
        rows.extend(json.loads(Path(f'D:/autodrive_cnn/runs/{folder}/matrix.json').read_text()))
    assert len(rows)==20 and len({(r['arm'],r['vehicle_name'],r['direction'],r['effects_name']) for r in rows})==20
    for r in rows:r['extra']=extra(r['path'],r['direction'])
    out=Path('cnn_finetune/results');(out/'FINAL_COMPARISON.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    selection=json.loads((out/'TRAINING_SELECTION.json').read_text());selected=next(c for c in selection['candidates'] if c['run']==selection['selected_run']);metrics=selected['metrics']
    audit=json.loads((out/'DATA_AUDIT.json').read_text());onnx=json.loads((out/'FINAL_ONNX_CHECK.json').read_text())
    text=['# direct 1차 미세조정: 3단계 완료','', '시뮬레이터/합성 데이터 결과다. 실차 검증·배포·4단계 DAgger·6단계 사진 점검은 하지 않았다.','', '## 1단계 교사','', '새 예산 3/8회에서 통과. 기본 4차량 × 양방향 × realcam 8/8 무침범 완주, 최악 여유 +0.8718cm. direct 출력과 1.2초 차체 여유 예측을 사용한다. 추가 random:41/42 × 양방향 × realcam_harsh 4/4 무침범이지만 최악 여유는 +0.0380cm로 작다.','', '## 2단계 데이터','', f"생성 2/3회. {audit['count']}장, 코너 {audit['corner_fraction']:.2%}, 복귀 {audit['recovery_fraction']:.2%}, 내부 직선 {audit['inner_count']}장. 교차로 35cm 미만 0장. 최종 시드 분할: 학습22개/검증4개/시험4개, 검증·시험 각각 양방향 균등. 원시 샤드와 로그는 D:/autodrive_cnn/data/v1에 보존.", '', '## 3단계 학습','', '4/4회 사용. 첫 실행은 한 방향으로 치우친 검증 분할 때문에 최종 선정에서 제외했다. 후보2/3/4를 동일한 수정 분할로 비교해 전체·코너·복귀 중 A5b 대비 최악 MAE 비율이 가장 작은 후보2를 선택했다. A5b 가중치에서 AdamW 3e-5, 배치64, 6에포크. 시뮬레이터 결과를 이용한 재선정은 하지 않았다.', '', 'MAE는 direct 정규화 출력 기준이다. A5b는 팀 출력→명목 듀티·곡률로 변환하여 같은 의미로 비교했다.','', '| 검증 장면 | A5b MAE | 새 모델 MAE | 감소 |','|---|---:|---:|---:|']
    for k in ['all','corner','recovery']:
        b=metrics['baseline_validation'][k]['mean'];n=metrics['validation'][k]['mean'];text.append(f'| {k} | {b:.6f} | {n:.6f} | {1-n/b:.1%} |')
    text += ['',f"ONNX 재검사: 검증 샤드의 분산 표본 {onnx['samples']}장, PyTorch 최대 절대 차이 {onnx['max_abs']:.3g}. 모델 SHA-256: {selection['sha256']}", '', '## 동일 조건 시뮬레이터 비교','', 'seed41, own 36ms, 최대200초. 기본4차량×양방향 realcam 8조건 + field_estimate 양방향 harsh 2조건. baseline=BENCH_CNN_MIXER=v4, new=direct와 명시적 BENCH_CNN_ONNX.','', '| 모델 | 완주 | 무침범 완주 | 최악 여유 cm |','|---|---:|---:|---:|']
    for arm in ['baseline','new']:
        rs=[r for r in rows if r['arm']==arm];text.append(f"| {arm} | {sum(r['summary']['verified_lap'] for r in rs)}/10 | {sum(r['summary']['verified_lap'] and r['summary']['min_margin_cm']>=0 for r in rs)}/10 | {min(r['summary']['min_margin_cm'] for r in rs):.4f} |")
    text += ['', '| 차량/효과/방향 | A5b 여유 cm | 새 모델 여유 cm |','|---|---:|---:|']
    regressions=[]
    for b in [r for r in rows if r['arm']=='baseline']:
        n=next(r for r in rows if r['arm']=='new' and all(r[k]==b[k] for k in ['vehicle_name','direction','effects_name']))
        label='/'.join(b[k] for k in ['vehicle_name','effects_name','direction']);bm=b['summary']['min_margin_cm'];nm=n['summary']['min_margin_cm'];text.append(f'| {label} | {bm:.4f} | {nm:.4f} |')
        if nm<bm-1e-6:regressions.append(label)
    text += ['', '회복 시간은 기준 경로 |곡률|>0.02 코너 구간을 나온 뒤 방향 오차 5도 이내가 될 때까지다. 0 PWM도 문턱 아래로 센다. 전체 세부값과 회복 미완료 여부는 FINAL_COMPARISON.json에 있다.', '', '| 모델 | 관측 회복 사건 수 | 회복시간 중앙값 s | 코너 안쪽 바퀴 문턱 아래 비율 |','|---|---:|---:|---:|']
    for arm in ['baseline','new']:
        rs=[r for r in rows if r['arm']==arm];events=[x for r in rs for x in r['extra']['post_corner_recovery_s']];frames=sum(r['extra']['corner_frames'] for r in rs);ratio=sum(r['extra']['inner_below_floor_fraction']*r['extra']['corner_frames'] for r in rs)/max(frames,1)
        text.append(f"| {arm} | {len(events)} | {float(np.median(events)) if events else float('nan'):.3f} | {ratio:.1%} |")
    text += ['', '## 판정과 종료','', f"검증 MAE는 전체·코너·복귀 모두 개선됐다. 최소 여유가 감소한 조건은 {len(regressions)}개다(큰 이탈로 종료된 harsh 조건의 작은 차이는 종료 표본 차이도 포함한다). 따라서 '나빠지는 조건 없음'이라는 완전 합격 기준을 충족하지 못하며, 기본 모델 교체를 권고하지 않는다. 학습 예산 4회를 사용한 뒤 차선 기준에 따라 개선과 퇴보를 그대로 보고한다. 3단계 산출물과 비교를 완료했으므로 여기서 멈춘다.", '', '최종 모델: cnn_finetune/models/direct_v1/policy.onnx. 반드시 direct 실행 모드와 함께 사용하며 기존 team/v4 출력 모델로 취급하지 않는다. 실차/Pi에는 배포하지 않았다.', '', '6단계 사진 경로 D:/autodrive_cnn/pi_data/20261111은 기록만 했고 열거나 분석하지 않았다. 다른 프로그램/기존 주행 파일은 변경하지 않았다. direct 실행 코드는 원격 main 6c5a53d에서 제공된 버전을 사용했다.', '', '제한: seed41의 축소된 비교이며 여러 독립 시드에 대한 일반화나 실차 효과를 입증하지 않는다. 시험 세트 결과는 TRAINING_SELECTION.json에 보존했으며 최종 모델 선택에는 검증 세트만 사용했다.']
    (out/'DIRECT_FINAL_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(dict(regressions=regressions),ensure_ascii=False))
