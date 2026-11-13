"""Audit saved physics, then export per-movement plots/videos and a Korean report."""
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bench.world import Track, to_world, wrap
from bench.junction import movement, ARM, heading_of
from bench.vehicle import Vehicle, PRESETS
from cnn_finetune.junction_stage02 import OUT, REPORT, ROOT, dump, metrics


def audit():
    results=json.loads((REPORT/'dynamic_results.json').read_text());selection=json.loads((REPORT/'selection.json').read_text())
    wrong=[]
    for r in results:
        key=tuple(r['key']);name='_'.join(key)
        dst=OUT/'dynamic'/(name+f"_{r['preset']}_{r['level']}_{r['seed']}_v{r['speed']}_l{r['lookahead']}")
        a=np.load(dst/'trace.npz')['samples'];path=np.load(OUT/'paths'/name/'path.npz')
        c,di,do,arm=movement(*key);events=json.loads((dst/'events.json').read_text())
        r.update(metrics(a[:,0],a[:,1:4],a[:,4],a[:,5],a[:,6],c,do,[e for e in events if e['kind']=='push'],reference=np.c_[path['points'],path['heading']]))
        r['initial_margin_cm']=float(a[0,6]);r['initially_inside']=r['initial_margin_cm']>=0
        for other in Track().data['junctions'][key[0]]['arms']:
            if other==arm:continue
            d=np.array(ARM[other]);rel=a[:,1:3]-c
            bad=(rel@d>=40)&(np.abs(rel@np.array([d[1],-d[0]]))<15)&(np.abs(wrap(a[:,3]-heading_of(d)))<np.radians(30))
            if np.any(bad):wrong.append(dict(key=key,preset=r['preset'],seed=r['seed'],level=r['level'],arm=other,time_s=float(a[np.argmax(bad),0])))
        assert r['allocator_calls']==r['commands']
    # A detected wrong exit requires rerunning with an explicit terminal; never quietly score it as success.
    dump(REPORT/'audit.json',dict(runs=len(results),wrong_exit_detections=wrong,allocator_once=True,
                                 note='Recovery recomputed from saved physics against candidate path, including all failed runs.'))
    assert not wrong,'Wrong exit requires explicit terminal rerun'
    dump(REPORT/'audited_dynamic_results.json',results)
    failures=[]
    for r in results:
        if r['clean_stable']:continue
        reason=r['outcome'] if not r['stable_success'] else 'boundary_overlap_then_stable_exit'
        failures.append(dict(key=r['key'],preset=r['preset'],level=r['level'],seed=r['seed'],reason=reason,
                             initially_inside=r['initially_inside'],initial_margin_cm=r['initial_margin_cm'],
                             min_margin_cm=r['min_margin_cm'],outside_s=r['outside_s'],
                             final_lateral_cm=r['final_lateral_cm'],final_heading_deg=r['final_heading_deg'],
                             unrecovered=r['unrecovered']))
    dump(REPORT/'failure_cases.json',failures)
    return results,selection


def count(rs):
    return f"{sum(bool(r['stable_success']) for r in rs)}/{len(rs)}",f"{sum(bool(r['clean_stable']) for r in rs)}/{len(rs)}"


def visual(key,config):
    name='_'.join(key);dst=REPORT/'movements'/name;dst.mkdir(parents=True,exist_ok=True)
    if (dst/'paired_path.png').exists() and (dst/'paired_comparison.mp4').exists():
        cap=cv2.VideoCapture(str(dst/'paired_comparison.mp4'));ok,frame=cap.read();frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
        if ok and frames>12:return dict(movement=name,video_frames=frames,dimensions=list(frame.shape[:2]))
    c,di,do,arm=movement(*key);track=Track()
    path=np.load(OUT/'paths'/name/'path.npz')
    nominal=OUT/'dynamic'/(name+f"_field_estimate_fixed_1303_v{config['speed']}_l{config['lookahead']}_legacy_pose")
    new=np.load(nominal/'trace.npz')['samples']
    old=np.load(OUT/'legacy'/(name+'_fixed_1303')/'physics.npz')['samples']
    extent=[c[0]-60,c[0]+60,c[1]-60,c[1]+60]
    xx,yy=np.meshgrid(np.linspace(extent[0],extent[1],600),np.linspace(extent[2],extent[3],600))
    inside=track.clearance(np.stack([xx,yy],axis=-1))>=0
    fig,ax=plt.subplots(figsize=(6,6));ax.imshow(inside,origin='lower',extent=extent,cmap='Greys_r',vmin=-.2,vmax=1.2)
    ax.plot(path['points'][:,0],path['points'][:,1],'--',color='orange',lw=1,label='Offline reference')
    ax.plot(old[:,1],old[:,2],color='#4779ec',lw=1.5,label='Legacy, matched initial pose')
    ax.plot(new[:,1],new[:,2],color='#00b878',lw=1.5,label='Oracle, matched initial pose')
    rim=Vehicle(PRESETS['field_estimate'],np.zeros(3)).footprint()
    for i in np.linspace(0,len(new)-1,8).astype(int):
        body=to_world(rim,new[i,1:4]);ax.plot(body[:,0],body[:,1],color='#00b878',alpha=.35,lw=.6)
    ax.set(xlim=extent[:2],ylim=extent[2:],xlabel='x (cm)',ylabel='y (cm)',title=name+' / offline simulation')
    ax.legend(fontsize=8);ax.set_aspect('equal');fig.tight_layout();fig.savefig(dst/'paired_path.png',dpi=135);plt.close(fig)
    size=640;bg=np.where(inside[::-1,:,None],np.array([65,65,65]),np.array([200,213,200])).astype(np.uint8)
    bg=cv2.resize(bg,(size,size))
    def px(points):
        points=np.asarray(points);return np.rint(np.c_[(points[:,0]-extent[0])/120*(size-1),(extent[3]-points[:,1])/120*(size-1)]).astype(np.int32)
    cv2.polylines(bg,[px(path['points'])],False,(0,150,240),1)
    cv2.polylines(bg,[px(old[:,1:3])],False,(240,120,50),1)
    cv2.polylines(bg,[px(new[:,1:3])],False,(80,220,50),1)
    video=cv2.VideoWriter(str(dst/'paired_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),12,(size,size))
    assert video.isOpened()
    for t in np.arange(0,max(new[-1,0],old[-1,0])+.1,1/12):
        image=bg.copy()
        for arr,color in [(old,(240,120,50)),(new,(80,220,50))]:
            i=min(len(arr)-1,int(np.searchsorted(arr[:,0],t)));poly=px(to_world(rim,arr[i,1:4]));cv2.polylines(image,[poly],True,color,2)
        cv2.rectangle(image,(0,0),(640,78),(25,25,25),-1)
        cv2.putText(image,name+'   t=%.2fs'%t,(10,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        cv2.putText(image,'Green: oracle / Blue: legacy / matched initial pose',(10,47),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        cv2.putText(image,'Offline physics only; finished trajectories hold last pose',(10,67),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
        video.write(image)
    video.release()
    cap=cv2.VideoCapture(str(dst/'paired_comparison.mp4'));ok,frame=cap.read();frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
    assert ok and frames>12
    return dict(movement=name,video_frames=frames,dimensions=list(frame.shape[:2]))


def main():
    results,selection=audit();legacy=json.loads((REPORT/'legacy_results.json').read_text())
    assert len(legacy)==72 and len(results)==1296
    for r in legacy:
        key=(r['junction'],r['came_from'],r['exit']);level=r['spec'].get('composite','fixed')
        dst=OUT/'legacy'/('_'.join(key)+f'_{level}_{r["seed"]}')
        z=np.load(dst/'physics.npz');a=z['samples'];c,di,do,arm=movement(*key)
        from bench.junction import load_plan_cache
        pts=load_plan_cache()[key][0];d=np.gradient(pts,axis=0)
        events=json.loads((dst/'events.json').read_text())
        r['new_metrics']=metrics(a[:,0],a[:,1:4],a[:,4],a[:,5],z['margins'],c,do,[e for e in events if e['kind']=='push'],reference=np.c_[pts,np.arctan2(d[:,0],d[:,1])])
    dump(REPORT/'audited_legacy_results.json',legacy)
    assert all(r['correct_exit']==r['new_metrics']['stable_success'] for r in legacy),'Stable scorer mismatch must be rerun'
    paths=json.loads((REPORT/'paths.json').read_text());rows=[];media=[];paired=[]
    for config in selection:
        key=tuple(config['key']);name='_'.join(key)
        baseline=next(r for r in legacy if (r['junction'],r['came_from'],r['exit'])==key and r['spec'].get('composite','fixed')=='fixed')
        paired_dir=OUT/'dynamic'/(name+f"_field_estimate_fixed_1303_v{config['speed']}_l{config['lookahead']}_legacy_pose")
        from cnn_finetune.junction_feasibility import simulate
        pair=json.loads((paired_dir/'result.json').read_text()) if paired_dir.exists() else simulate(key,'field_estimate','fixed',1303,config['speed'],config['lookahead'],initial_pose=baseline['initial_truth'])
        paired.append(pair)
        legacy_initial=np.load(OUT/'legacy'/(name+'_fixed_1303')/'physics.npz')['samples'][0,1:4]
        assert np.array_equal(np.load(paired_dir/'trace.npz')['samples'][0,1:4],legacy_initial)
        nominal=json.loads((OUT/'paths'/name/'centered_result.json').read_text())
        subsets={l:[r for r in results if r['key']==list(key) and r['level']==l] for l in ['fixed','mild','harsh']}
        status='unresolved'
        if nominal['clean_stable'] and nominal['unnecessary_stops']==0:
            status='continuous_with_opposing_wheel' if nominal['opposing_wheel_fraction']>0 else 'continuous_forward'
        rows.append(dict(movement=name,mission_allowed=key[0]!='center' or key[2]=='right',status=status,
                         nominal_margin_cm=nominal['min_margin_cm'],nominal_opposing_fraction=nominal['opposing_wheel_fraction'],
                         nominal_stop_s=nominal['unnecessary_stop_s'],
                         fixed_clean=sum(bool(r['clean_stable']) for r in subsets['fixed']),
                         mild_stable=sum(bool(r['stable_success']) for r in subsets['mild']),mild_clean=sum(bool(r['clean_stable']) for r in subsets['mild']),
                         harsh_stable=sum(bool(r['stable_success']) for r in subsets['harsh']),harsh_clean=sum(bool(r['clean_stable']) for r in subsets['harsh'])))
        media.append(visual(key,config));print(name,status,flush=True)
    dump(REPORT/'movement_results.json',rows);dump(REPORT/'media_validation.json',media);dump(REPORT/'paired_results.json',paired)
    with (REPORT/'movement_results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    groups=[]
    for level in ['fixed','mild','harsh']:
        rs=[r for r in results if r['level']==level];s,cl=count(rs);valid=[r for r in rs if r['initially_inside']];vs,vc=count(valid)
        groups.append(f'| {level} | {s} | {cl} | {vc} |')
    nominal=[json.loads((OUT/'paths'/r['movement']/'centered_result.json').read_text()) for r in rows]
    fixed_legacy=[r for r in legacy if r['spec'].get('composite','fixed')=='fixed']
    selected=paired
    ls,lc=count([r['new_metrics'] for r in fixed_legacy]);ns,nc=count(selected)
    stopold=sum(r['new_metrics']['unnecessary_stops'] for r in fixed_legacy);stopnew=sum(r['unnecessary_stops'] for r in selected)
    recovery=[e for r in results for e in r['recovery']]
    legacy_groups=[]
    for level in ['fixed','mild','harsh']:
        rs=[r['new_metrics'] for r in legacy if r['spec'].get('composite','fixed')==level];s,cl=count(rs)
        legacy_groups.append(f'| {level} | {s} | {cl} |')
    old_mean=lambda k:float(np.mean([r['new_metrics'][k] for r in fixed_legacy]))
    new_mean=lambda k:float(np.mean([r[k] for r in selected]))
    lines=['# 교차로 학습 준비 단계 0–2 실행 결과','',
           '2026-11-13. 데스크톱 bench 시뮬레이션. 새 신경망 학습·모델 교체·Pi 접속·실차 동작·커밋/푸시는 수행하지 않았다.','',
           '## 판정','',
           f'단계 0–2의 평가 계약, 실행 재현 검사, 이동별 연속 경로 탐색을 완료했다. 36개 기하학적 이동 중 기본 차체·중앙 정렬 시작에서 경계 침범 없이 안정 출구에 도달한 후보는 {sum(bool(r["clean_stable"]) for r in nominal)}/36개다. 불필요한 정지 0회 후보는 {sum(r["unnecessary_stops"]==0 for r in nominal)}/36개다. 중앙 좌회전/직진 8개는 미션 비허용 실험이며 실제 미션 대상은 28개다.',
           '이 결과는 정답 자세를 이용한 오프라인 교사의 실행 가능성이다. CNN이 같은 궤적을 영상만으로 따라갈 수 있다거나 실차에서 안전하다는 증거가 아니다. 약한 복합 교란의 무이탈 통과는 아직 충분하지 않다. 학습용으로 사용할 때 깨끗한 후보·회복 분기·실패 분기를 구분해야 한다.','',
           '## 단계 0: 고정 계약','',
           '- `contract.json`: 기존 ONNX/체크포인트, pinned v4, 지도·차체·카메라 및 소스 해시, 평가 정의를 보존했다.',
           '- 개발 시드 1301/1303/1307. 경로 매개변수의 기하 탐색 seed 2718, 속도 선택은 개발 1303만 사용했다. 최종 시험 3001/3007/3011/3019/3023/3037은 실행하지 않았다. 학습 예정 시드 2001–2020도 미사용이다.',
           '- 출구 진행 40cm 이상, 차체 여유 0cm 이상, 횡오차 1cm 이내, 방향 오차 5° 이내를 연속 1초 유지해야 성공이다. 회복은 해당 참조 경로의 같은 횡/방향/여유 조건으로 1초 유지한다.',
           '- 실패 및 미복귀는 분모에 남기고 회복시간은 null로 보존한다. 0.1초 고정 격자의 실제 속도·회전속도로 가속 RMS를 계산한다. 의무 정지는 불필요한 정지 비용에서 제외한다.',
           '- 기존 배치 오차(횡 ±1cm, 종 ±2cm, 방향 ±3°)를 유지했다. 좁은 도로의 seed 1301은 출발부터 약 -0.158cm 침범한다. 이 경우 전체 무이탈 성공으로 세지 않고, 초기 정상 배치 조건부 성적도 병기한다. 중앙 정렬 시작의 별도 실행은 경로 자체의 가능성을 확인하는 용도다.','',
           '## 단계 1: 실행과 출력 검증','',
           '- 원래 `bench.junction.run`을 공통 루프로 사용했다. 수집은 관찰자이며 촬영·36ms 처리·이전 명령 유지·명령 큐·최대 20ms 물리 적분·카메라 이력·지도 가드를 바꾸지 않는다.',
           '- A5b+v4, direct_v2, 강한 조명+약한 차체 변동의 3개 쌍에서 수집 on/off의 적용 PWM, 실제 자세·속도·경계 여유가 정확히 일치했다. 지도 추정값은 최대 약 2.53e-7 차이였고 1e-6 절대 허용오차를 통과했다. 벽시계 처리시간 로그는 일치 대상에서 제외했다.',
           '- signed 좌/우 PWM v1의 전진·안쪽 정지·양쪽 역회전·좌우 제자리 회전·낮은 출력 회전·정지/재개를 검사했다. 실제 부호와 지도 예측 부호가 일치한다. 저출력 디더 평균과 1명령당 1갱신도 검사했다.',
           '- 외곽 direct 출력 계약은 변경하지 않았다. 기존 역회전용 `HybridJunction.on_applied`를 확인했다. 새 신경망의 signed 출력 헤드는 아직 만들거나 학습하지 않았다.','',
           '## 단계 2: 동역학 결과','',
           '기존 원호·제자리·분할 회전을 보존하고, 접선 진입/출구와 5차 Bézier 곡선을 이용한 가변 곡률 후보를 탐색했다. 촘촘한 정적 차체 검사 후 원래 Vehicle의 지연·출력 문턱·포화·1차 응답·좌우 비대칭·역회전 밀림을 적용했다. 후보 제어기는 capture 시점 정답 자세/속도와 **명목** 차체 파라미터를 사용하며, 난수로 변한 실제 효율/출력 교란을 미리 알지 못한다.',
           '1296회 = 36이동 × 4차체(field_estimate/sluggish/unequal_sides/field_pivot_slip) × 3조건 × 3개발 시드. 속도 선택 108회, 중앙 정렬 36회, 기존 제어기와 초기 자세를 정확히 맞춘 추가 36회는 별도다. 앞선 탐색용 12회는 최종 분모에 중복하지 않았다.','',
           '| 조건 | 안정 출구 | 전체 무이탈 안정 출구 | 초기 경계 안에서 출발한 경우의 무이탈 |','|---|---:|---:|---:|',*groups,'',
           'mild는 차체 mild 변동 + 횡밀림 1–2cm/방향 1–3°(0.3–0.7초) + 출력 ±5–15%(2–5초)를 함께 적용했다. harsh는 차체 harsh + 4–6cm/6–10° + ±20–35%다. 교차로를 겨냥한 외부 사건은 시작 7–8초 및 15–16초의 고정 시드 일정이며 제어기 상태와 독립이다. 원래 외곽 실험의 12–17초 일정과 다르며 실측 분포가 아니다. 노출 전 종료한 사건은 적용됐다고 세지 않는다.',
           '조명은 정답 자세를 보는 후보의 성능을 평가할 수 없다. 별도 기존 영상 제어기 72회 중 36회는 전체 이동 기본 조건, 나머지 36회는 남·북 합류/중앙 우회전 6이동 × 3시드 × mild+강한 조명 또는 harsh+harsh 카메라로 검사했다. 두 제어기의 교란 성적은 관측 정보가 다르므로 CNN 개선 효과로 직접 비교하지 않는다.','',
           '| 기존 A5b+v4 hyb2j | 새 기준 안정 출구 | 무이탈 안정 출구 |','|---|---:|---:|',*legacy_groups,'',
           '기존 제어기의 time_limit에는 올바른 길로 나갔어도 출구 정렬 1cm/5°를 1초 유지하지 못한 경우가 포함된다. 이후 외곽 주행까지 진행한 사례도 있으므로 “전부 교차로를 못 돌았다”는 뜻이 아니다. 이전의 넓은 출구 판정과 새 안정 정렬 성공을 혼용하지 않는다.','',
           '### 같은 기본 시작 조건의 기존 동작과 후보','',
           f'field_estimate/fixed/seed1303의 36이동: 기존 안정 {ls}, 무이탈 {lc}; 정답 자세 후보 안정 {ns}, 무이탈 {nc}. 불필요한 정지 횟수 합계는 기존 {stopold}, 후보 {stopnew}다. 이는 오프라인 목표 경로 개선의 근거이며 센서 조건을 맞춘 정책 우열 비교가 아니다.',
           f'같은 36이동의 주행별 평균 RMS: 선형 가속 {old_mean("accel_rms_cm_s2"):.3f} → {new_mean("accel_rms_cm_s2"):.3f} cm/s², 각가속 {old_mean("angular_accel_rms_rad_s2"):.3f} → {new_mean("angular_accel_rms_rad_s2"):.3f} rad/s². 추가 진단인 제자리 회전을 포함한 전진 멈춤 시간(|v|<0.5cm/s가 0.3초 이상, 초기 1초 제외)은 평균 {old_mean("translation_pause_s"):.3f} → {new_mean("translation_pause_s"):.3f}초다. 이 보조 지표는 계약의 완전 정지 지표를 대체하지 않는다.',
           f'전체 동역학 결과에서 노출된 밀림 사건 {len(recovery)}건 중 미복귀 {sum(not e["recovered"] for e in recovery)}건을 그대로 보존했다. 밀림 중 종료된 실패도 미복귀에 포함했다. 여러 사건/프레임을 독립 주행 표본으로 해석하지 않는다.','',
           '## 이동별 결과','',
           '각 조건 분모는 12(4차체×3시드). `opposing`은 차체가 전진하면서 일부 구간 안쪽 바퀴를 반대 방향으로 구동함을 뜻한다. 불연속 제자리 정지 단계와 구분한다. 영상은 초록=정답 자세 후보, 파랑=기존 제어기이며 초기 자세를 정확히 맞췄다. 이는 기본 차체/fixed/seed1303 비교이고 위의 1296회 분모에 추가하지 않는다.','',
           '| 이동 | 기본 실행 분류 | mild 안정/무이탈 | harsh 안정/무이탈 | 경로·영상 |','|---|---|---:|---:|---|']
    for r in rows:
        n=r['movement'];lines.append(f'| {n} | {r["status"]} | {r["mild_stable"]}/12 · {r["mild_clean"]}/12 | {r["harsh_stable"]}/12 · {r["harsh_clean"]}/12 | [경로](movements/{n}/paired_path.png) · [영상](movements/{n}/paired_comparison.mp4) |')
    lines+=['','## 다음 단계에 남긴 조건','',
            '방향을 근본적으로 바꿔야 하는 결정은 발견하지 않았다. 연속 회전의 실행 가능성을 확인했으므로 계획의 시간 입력+방향 조건+signed 출력 방향을 유지할 수 있다. 그러나 이번 결과로 교차로 전체가 교란에 강하다고 판단하지 않는다.',
            '단계 3에서는 후보가 깨끗하게 통과한 분기와 경계 침범 후 복구한 분기를 구분해 보존하고, 현재 강한 교란 실패를 목표 정답으로 복제하지 않아야 한다. 이 단계에서 단순 교사 오차만 줄여도 개선된다는 주장은 하지 않는다. 영상 기반 관측, 방향 명령, 적용 PWM 이력으로 가능한 회복인지 단계 3–4에서 따로 검증해야 한다.',
            '실제 차체/트랙 오차, 새 신경망의 추론 지연, Pi의 역회전 구동 특성, 영상 기반 강한 조명 일반화, 회전 도중 STOP/재개를 포함한 전체 미션은 후속 검증 대상이다. 이번 STOP 검사는 출력 계약과 지연 응답의 단위 검사다.','',
            '## 산출물과 재실행','',
            f'- 원시 물리·명령·교란·영상 해시 로그: `{OUT}`.',
            '- `contract.json`, `parity.json`, `action_metric_checks.json`, `audit.json`, `selection_trials.json`, `audited_dynamic_results.json`, `audited_legacy_results.json`, `paired_results.json`, `failure_cases.json`, `movement_results.csv`, `execution_manifest.json`.',
            '- 실행 모듈: `cnn_finetune.junction_stage02`, `junction_stage02_checks`, `junction_feasibility`, `junction_stage02_baselines`, `junction_stage02_report`.',
            '- 원시 결과 폴더는 덮어쓰지 않는다. 새 반복은 OUT/REPORT를 별도 날짜·실험 ID로 지정해야 한다. 최종 시험 시드는 이 보고서에서 사용하지 않았다.','']
    (REPORT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    sources=[p for p in (ROOT/'cnn_finetune').glob('junction_*.py')]
    dump(REPORT/'execution_manifest.json',dict(source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
         final_models={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'cnn_finetune/models/direct_v2/policy.onnx',ROOT/'bench/_vendor/tdt_models_2026-11-01/models/lane-v5b/policy.onnx']},
         note='Initial contract predates additive feasibility scripts. This manifest describes final harness. Collection/parity only gained diagnostic tolerances; archived pilot rows retain original metrics, audited table recomputes the final metric from raw physics. Stable-exit observer reset logic was tightened during baseline execution; independently recomputed physical metrics must agree for every accepted run.'))
    print('REPORT READY',flush=True)


if __name__=='__main__':main()
