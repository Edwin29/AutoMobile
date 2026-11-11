"""Freeze DAgger selection, verify export, and report final held-out results."""
import argparse,json,shutil,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('D:/autodrive_cnn');R=Path(__file__).resolve().parent/'results';M=Path(__file__).resolve().parent/'models/direct_v2'
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['export','report']);a=ap.parse_args()
    state=json.loads((R/'DAGGER_PROGRESS.json').read_text(encoding='utf-8'));selected=next(c for i in state['iterations'] for c in i['candidates'] if c['name']==state['selected']);source=Path(selected['checkpoint'])
    if a.stage=='export':
        import cv2,torch
        from .model import LanePolicy
        M.mkdir(parents=True,exist_ok=True);shutil.copy2(source/'policy.onnx',M/'policy.onnx')
        model=LanePolicy().eval();model.load_state_dict(torch.load(source/'best.pt',weights_only=True));torch.set_num_threads(1);cv2.setNumThreads(1)
        net=cv2.dnn.readNetFromONNX(str(M/'policy.onnx'));maximum=0.;count=0
        for p in sorted((ROOT/'data/v1').glob('seed_*.npz')):
            if int(p.stem.split('_')[1]) not in {104,109,118,123}:continue
            with np.load(p) as z:images=z['x']
            for i in np.linspace(0,len(images)-1,20,dtype=int):
                x=images[i:i+1].astype(np.float32)/255
                with torch.no_grad():y=model(torch.from_numpy(x)).numpy()
                net.setInput(x);maximum=max(maximum,float(np.max(np.abs(y-net.forward()))));count+=1
        assert maximum<1e-4
        check=dict(samples=count,max_abs=maximum,ok=True);(M/'onnx_check.json').write_text(json.dumps(check,indent=2))
        meta=dict(selected=state['selected'],development_seed=42,development_score=selected['development']['score'],output=['mean_duty/0.66','left_positive_curvature/0.095'],curvature_cap=.085,required_mixer='direct',preprocess='rgb_bottom_half_yuv_blur3_200x66_chw_div255',sha256=hashlib.sha256((M/'policy.onnx').read_bytes()).hexdigest(),source_checkpoint=str(source))
        (M/'policy.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta));raise SystemExit()
    from .compare_metrics import extra
    rows=json.loads((ROOT/'runs/dagger_final_comparison/matrix.json').read_text());assert len(rows)==20
    for r in rows:r['extra']=extra(r['path'],r['direction'])
    (R/'DAGGER_FINAL_COMPARISON.json').write_text(json.dumps(rows,indent=2))
    photos=json.loads((R/'DAGGER_PHOTO_CHECK.json').read_text());assert photos['total']==1072
    new=[r for r in rows if r['arm']=='new'];base=[r for r in rows if r['arm']=='baseline'];real=[r for r in new if r['effects_name']=='realcam'];events=[t for r in real for t in r['extra']['post_corner_recovery_s']];recovery=float(np.median(events)) if events else float('inf')
    clean=sum(r['summary']['verified_lap'] and r['summary']['min_margin_cm']>=0 for r in real)
    diffs=[]
    for n in real:
        b=next(b for b in base if all(b[k]==n[k] for k in ['vehicle_name','direction','effects_name']));diffs.append(n['summary']['min_margin_cm']-b['summary']['min_margin_cm'])
    passed=bool(clean>=7 and recovery<=1.49 and min(diffs)>=-.5)
    text=['# DAgger 최종 보고 — 2026-11-12','',f"최종 선정: **{state['selected']}**. 합격 기준 판정: **{'통과' if passed else '미달; 차선 기준으로 결과 보고'}**. Pi 접속·배포·실차 주행은 하지 않았다.",'','## 교사 사전 확인','','seed41의 10조건에서 variant3 교사 10/10 무침범 완주. 회복 중앙값0초는 코너 종료 시점에 이미 방향 오차5도 이내였다는 뜻이다. 교사 수정 조건에 해당하지 않아 교사와 기존36000장 정답은 유지했다.','','## 반복 및 선정','','개발은 seed42 기본4차량×양방향 realcam만 사용했다. 점수 우선순위는 무침범 수→최악 여유→회복 중앙값이다. seed41 학생 결과와 실차 사진은 선정에 사용하지 않았다.','', '| 후보 | 누적 전체 데이터 | 검증 MAE 전체/코너/복귀 | 개발 무침범 | 최악 여유 cm | 회복 중앙값 s |','|---|---:|---|---:|---:|---:|']
    for it in state['iterations']:
        for c in it['candidates']:
            v=c['metrics']['validation'];score=c['development']['score'];text.append(f"| {c['name']} | {36000+12000*it['iteration']} | {v['all']['mean']:.5f}/{v['corner']['mean']:.5f}/{v['recovery']['mean']:.5f} | {score[0]}/8 | {score[1]:.4f} | {-score[2]:.3f} |")
    text += ['',f"DAgger 3회 완료, 개발 미개선에 대한 수정 {state.get('retries_used',0)}/2회 사용. 각 반복 새12시드×1000장, beta=0.5/0.25/0, 0.5초 구간 단위 정책 선택. 제어주기36ms, 실제 적용PWM에 교사 last_pwm을 맞추고 dither는 실행 정책에 대해 한 번만 갱신했다. 정답은 항상 교사다. 차체 여유 -3cm 미만/교차로35cm 미만 프레임은 저장하지 않았다. 내부 직선10%, 실제 코너 시작 에피소드 비율≥50% 검사. DART 없음.", '', '학습은 직전 반복의 선택 모델에서 시작했고 기존 학습22시드와 누적 DAgger 자료만 사용했다. 기존 검증4/시험4시드는 학습에서 제외했다. 매 실행 내 체크포인트는 검증 MAE 최저를 사용하고 반복/수정 후보의 최종 선택은 개발 주행으로 했다.', '', '준비 중 메타데이터 직렬화와 코너 에피소드 카운터 버그를 수정했다. dagger_1 및 dagger_1_v1의 불완전/명세 미달 수집과 중단 학습은 보존했지만 선정에서 제외했다. 이들은 개발 성능 튜닝 수정과 구분한다. 유효 자료는 dagger_1_v2, dagger_2_v2, dagger_3_v2다.','','## 최종 seed41 비교','','모델을 고정한 뒤에만 10조건 최종 비교를 실행했다. harsh 2조건은 별도 관찰이며 합격 판정에서 제외했다.','', '| 조건 | A5b 여유 cm | direct_v2 여유 cm |','|---|---:|---:|']
    for b in base:
        n=next(r for r in new if all(r[k]==b[k] for k in ['vehicle_name','direction','effects_name']));label='/'.join(b[k] for k in ['vehicle_name','direction','effects_name']);text.append(f"| {label} | {b['summary']['min_margin_cm']:.4f} | {n['summary']['min_margin_cm']:.4f} |")
    text+=['',f"realcam 무침범 {clean}/8(기준7), 회복 중앙값 {recovery:.3f}초(기준≤1.49), 조건별 최소 여유 차이 최악 {min(diffs):+.4f}cm(기준≥-0.5).",'', '| 모델/효과 | 완주 | 무침범 완주 | 회복 중앙값 s | 코너 안쪽 PWM<0.30 |','|---|---:|---:|---:|---:|']
    for arm in ['baseline','new']:
        for fx in ['realcam','realcam_harsh']:
            rs=[r for r in rows if r['arm']==arm and r['effects_name']==fx];es=[v for r in rs for v in r['extra']['post_corner_recovery_s']];frames=sum(r['extra']['corner_frames'] for r in rs);ratio=sum(r['extra']['corner_frames']*r['extra']['inner_below_floor_fraction'] for r in rs)/max(frames,1)
            text.append(f"| {arm}/{fx} | {sum(r['summary']['verified_lap'] for r in rs)}/{len(rs)} | {sum(r['summary']['verified_lap'] and r['summary']['min_margin_cm']>=0 for r in rs)}/{len(rs)} | {np.median(es) if es else float('nan'):.3f} | {ratio:.1%} |")
    text+=['','## 로컬 실차 사진 점검','','2개 저장 주행, 1072장 모두 추론했다. 이미 정방향인 저장 사진에 추가 회전하지 않았다. 두 출력은 v4 할당기의 고정 명령256회 평균으로 명목 듀티·곡률로 비교했다. 실차 정답/재주행이 없으므로 정확도나 안전성 합격 판정에 쓰지 않는다. 저장 프레임 번호가 불연속이므로 급변은 인접 저장 사진 사이 차이이며 초당 변화율이 아니다. 코너 후보는 A5b 명목 |k|>0.02 기준으로 실제 코너 정답이 아니다.','', '| 주행 | 사진 수 | 평균 절대 곡률 차이 | 코너후보 곡률 차이 | A5b/new 곡률 급변 p95 |','|---|---:|---:|---:|---|']
    for p in photos['runs']:
        text.append(f"| {p['run']} | {p['count']} | {p['mean_abs_difference'][1]:.5f} | {p['corner_k_difference']:.5f} | {p['baseline_jump_p95'][1]:.5f}/{p['new_jump_p95'][1]:.5f} |")
    for p in photos['runs']:
        text+=['',f"![명목 출력 비교]({p['run']}_dagger.png)",'']
    text+=['','출력 유한값 검사 및 raw 음수 속도/범위/큰 곡률 변화 통계는 DAGGER_PHOTO_CHECK.json에 있다. 급변이 오류인지는 정답 없이 확정하지 않는다. 원본 사진은 수정하지 않았으며 원본 해시와 프레임별CSV는 D:/autodrive_cnn/logs/dagger_photos에 보존한다.','','## 산출물과 종료','','최종 모델 models/direct_v2/policy.onnx, 출력정의/해시 policy.json, PyTorch↔OpenCV 검사 onnx_check.json. direct 모드 전용이다. 결정은 DECISIONS.md, 반복 전체 수치는 DAGGER_PROGRESS.json, 최종 수치는 DAGGER_FINAL_COMPARISON.json에 있다. 승인 범위의 작업을 마쳤으므로 중단한다. 7단계 배포·실차 시험은 수행하지 않았다.']
    (R/'DAGGER_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8');print(json.dumps(dict(passed=passed,clean=clean,recovery=recovery,worst_margin_difference=min(diffs))))
