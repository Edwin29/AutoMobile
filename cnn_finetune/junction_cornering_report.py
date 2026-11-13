"""Audit paired cornering refinements and export a development-only teacher registry."""
import json
from collections import Counter
import hashlib
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cnn_finetune import junction_cornering as c


def load_trace(r):return np.load(Path(r['artifact'])/'trace.npz')['samples']


def enrich(r):
    a=load_trace(r);centre,di,do,arm=c.movement(*r['key']);grid=np.arange(1,a[-1,0],.1)
    xy=np.c_[np.interp(grid,a[:,0],a[:,1]),np.interp(grid,a[:,0],a[:,2])]
    v=np.interp(grid,a[:,0],a[:,4]);w=np.interp(grid,a[:,0],a[:,5]);alpha=np.diff(w)/.1
    near=np.linalg.norm(xy-centre,axis=1)<30
    r['turn_angular_accel_rms']=float(np.sqrt(np.mean(alpha[near[1:]]**2)))
    yaw=np.interp(grid,a[:,0],np.unwrap(a[:,3]));turned=abs(c.wrap(yaw-c.heading_of(di)))
    turning=(turned>np.radians(10))&(turned<np.radians(80))
    r['turn_distance_10_80_cm']=float(sum(np.maximum(v[turning],0))*.1)
    return r


def video_and_plot(key,config,status):
    name='_'.join(key);dst=c.REPORT/'movements'/name;dst.mkdir(parents=True,exist_ok=True)
    old=c.run(key,'old','legacy','field_estimate','fixed',1303,'validation')
    new=c.run(key,config['candidate'],config['mode'],'field_estimate','fixed',1303,'validation')
    a,b=load_trace(old),load_trace(new);centre,di,do,arm=c.movement(*key);extent=[centre[0]-55,centre[0]+55,centre[1]-55,centre[1]+55]
    xx,yy=np.meshgrid(np.linspace(extent[0],extent[1],640),np.linspace(extent[2],extent[3],640))
    inside=c.Track().clearance(np.stack([xx,yy],axis=-1))>=0
    zo=np.load(c.OUT/'paths'/name/'old.npz');zn=np.load(c.OUT/'paths'/name/(config['candidate']+'.npz'))
    fig,axes=plt.subplots(1,2,figsize=(11,5));ax=axes[0]
    ax.imshow(inside,origin='lower',extent=extent,cmap='Greys_r',vmin=-.2,vmax=1.2)
    ax.plot(a[:,1],a[:,2],color='#3c80ed',label='Previous teacher')
    ax.plot(b[:,1],b[:,2],color='#00a875',label='Cornering candidate')
    ax.plot(zn['points'][:,0],zn['points'][:,1],'--',color='orange',lw=.8,label='New reference')
    ax.set(xlim=extent[:2],ylim=extent[2:],xlabel='x (cm)',ylabel='y (cm)',title=name);ax.set_aspect('equal');ax.legend(fontsize=7)
    ax=axes[1];ax.plot(zo['s']/zo['s'][-1],abs(zo['curvature']),color='#3c80ed',label='Previous reference')
    ax.plot(zn['s']/zn['s'][-1],abs(zn['curvature']),color='#00a875',label='Cornering reference')
    ax.set(xlabel='Normalized path distance',ylabel='Absolute curvature (1/cm)',title='Lower peak = wider minimum radius');ax.legend(fontsize=8)
    fig.suptitle('Offline truth-based teachers / '+status);fig.tight_layout();fig.savefig(dst/'comparison.png',dpi=140);plt.close(fig)
    bg=np.where(inside[::-1,:,None],np.array([60,60,60]),np.array([202,216,202])).astype(np.uint8)
    def px(p):p=np.asarray(p);return np.rint(np.c_[(p[:,0]-extent[0])/110*639,(extent[3]-p[:,1])/110*639]).astype(np.int32)
    cv2.polylines(bg,[px(a[:,1:3])],False,(245,130,55),1);cv2.polylines(bg,[px(b[:,1:3])],False,(65,220,50),1)
    rim=c.Vehicle(c.PRESETS['field_estimate'],np.zeros(3)).footprint()
    writer=cv2.VideoWriter(str(dst/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),12,(640,640));assert writer.isOpened()
    for t in np.arange(0,max(a[-1,0],b[-1,0])+.1,1/12):
        image=bg.copy()
        for trace,color in [(a,(245,130,55)),(b,(65,220,50))]:
            i=min(len(trace)-1,int(np.searchsorted(trace[:,0],t)));cv2.polylines(image,[px(c.to_world(rim,trace[i,1:4]))],True,color,2)
        cv2.rectangle(image,(0,0),(640,80),(20,20,20),-1)
        for i,text in enumerate([name+' / '+status+' / %.2fs'%t,'Green: cornering / Blue: previous teacher / same initial pose',
                                 'Offline truth only; finished trajectories hold final pose']):
            cv2.putText(image,text,(8,23+23*i),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
        writer.write(image)
    writer.release();cap=cv2.VideoCapture(str(dst/'comparison.mp4'));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame in [0,n//2,n-1]:
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,img=cap.read();assert ok
    cap.release()
    return dict(key=key,frames=n,decoded_first_middle_last=True)


def main():
    rows=[enrich(r) for r in json.loads((c.REPORT/'validation.json').read_text())]
    selected=json.loads((c.REPORT/'selection.json').read_text());registry=[];pairs=[];media=[]
    for cfg in selected:
        key=tuple(cfg['key']);old=[r for r in rows if r['key']==list(key) and r['candidate']=='old']
        new=[r for r in rows if r['key']==list(key) and r['candidate']!='old'];assert len(old)==len(new)==36
        by_old={(r['preset'],r['level'],r['seed']):r for r in old}
        for r in new:
            before=by_old[(r['preset'],r['level'],r['seed'])]
            np.testing.assert_allclose(load_trace(r)[0,1:4],load_trace(before)[0,1:4],atol=1e-12,rtol=0)
            assert json.loads((Path(r['artifact'])/'events.json').read_text())==json.loads((Path(before['artifact'])/'events.json').read_text())
            assert r['allocator_calls']==r['commands']
            pairs.append(dict(key=key,preset=r['preset'],level=r['level'],seed=r['seed'],
                       stable_before=before['stable_success'],stable_after=r['stable_success'],
                       clean_before=before['clean_stable'],clean_after=r['clean_stable'],
                       push_exposure_before_s=before['push_exposure_s'],push_exposure_after_s=r['push_exposure_s'],
                       angular_accel_delta=r['turn_angular_accel_rms']-before['turn_angular_accel_rms'],
                       time_delta_s=r['simulated_s']-before['simulated_s']))
        nominal=c.run(key,cfg['candidate'],cfg['mode'],'field_estimate','fixed',1303,'nominal',True)
        reasons=[]
        for level in ['fixed','mild']:
            aa=[r for r in old if r['level']==level];bb=[r for r in new if r['level']==level]
            for metric in ['stable_success','clean_stable']:
                if sum(r[metric] for r in bb)<sum(r[metric] for r in aa):reasons.append(level+'_'+metric+'_regression')
        if not nominal['clean_stable'] or nominal['translation_pause_s']>0:reasons.append('nominal_not_clean_moving')
        aa=[r for r in old if r['level']=='fixed'];bb=[r for r in new if r['level']=='fixed']
        alpha_old=float(np.median([r['turn_angular_accel_rms'] for r in aa]));alpha_new=float(np.median([r['turn_angular_accel_rms'] for r in bb]))
        if alpha_new>=alpha_old*.95:reasons.append('less_than_5pct_turn_smoothness_gain')
        if np.mean([r['simulated_s'] for r in bb])>np.mean([r['simulated_s'] for r in aa])*1.35+2:reasons.append('too_slow')
        radius_old=old[0]['path_shape']['min_radius_cm'];radius_new=new[0]['path_shape']['min_radius_cm']
        if radius_new<radius_old*1.5:reasons.append('less_than_50pct_radius_gain')
        accepted=not reasons
        z=np.load(c.OUT/'paths'/'_'.join(key)/(cfg['candidate']+'.npz'))
        reference_speed=c.speed_profile(z['curvature'],z['s'],c.old_speed(key)) if cfg['mode']=='planned' else c.old_speed(key)/(1+5*abs(z['curvature']))
        bundle=c.OUT/'teacher_candidates'/('_'.join(key)+'.npz');bundle.parent.mkdir(exist_ok=True)
        np.savez_compressed(bundle,points=z['points'],heading=z['heading'],curvature=z['curvature'],s=z['s'],nominal_speed_cm_s=reference_speed)
        registry.append(dict(**cfg,status='accepted_dev' if accepted else 'held',reasons=reasons,
                  mission_allowed=key[0]!='center' or key[2]=='right',radius_old_cm=radius_old,radius_new_cm=radius_new,
                  fixed_turn_alpha_old=alpha_old,fixed_turn_alpha_new=alpha_new,
                  fixed_clean_old=sum(r['clean_stable'] for r in aa),fixed_clean_new=sum(r['clean_stable'] for r in bb),
                  mild_clean_old=sum(r['clean_stable'] for r in old if r['level']=='mild'),mild_clean_new=sum(r['clean_stable'] for r in new if r['level']=='mild'),
                  mild_clean_teacher_available=any(r['clean_stable'] for r in new if r['level']=='mild'),
                  rollout_use_rule='clean_stable: clean example; stable after boundary overlap: recovery branch; failure: negative example, never clean target',
                  reference=str(bundle),base_speed_cm_s=c.old_speed(key),
                  controller=dict(heading_gain=1.3,lateral_gain=.16,processing_s=.036,planned_start_accel_cm_s2=1.,action='signed left/right PWM'),
                  use='Offline reference only; label every rollout by outcome; no CNN training or deployment approval'))
        media.append(video_and_plot(key,cfg,'accepted_dev' if accepted else 'held'))
        print('_'.join(key),registry[-1]['status'],','.join(reasons),flush=True)
    c.dump(c.REPORT/'teacher_registry.json',registry);c.dump(c.REPORT/'paired_changes.json',pairs)
    c.dump(c.REPORT/'audited_validation.json',rows);c.dump(c.REPORT/'media_checks.json',media)
    accepted=[r for r in registry if r['status']=='accepted_dev'];table=[]
    for level in ['fixed','mild','harsh']:
        aa=[r for r in rows if r['candidate']=='old' and r['level']==level];bb=[r for r in rows if r['candidate']!='old' and r['level']==level]
        table.append(f'| {level} | {sum(r["stable_success"] for r in aa)}/{len(aa)} → {sum(r["stable_success"] for r in bb)}/{len(bb)} | {sum(r["clean_stable"] for r in aa)}/{len(aa)} → {sum(r["clean_stable"] for r in bb)}/{len(bb)} |')
    aa=[r for r in rows if r['candidate']=='old' and r['level']=='fixed'];bb=[r for r in rows if r['candidate']!='old' and r['level']=='fixed']
    reasons=Counter(reason for r in registry for reason in r['reasons'])
    text=['# 교차로 코너링 경로 개선 실험','',
          '2026-11-13. 이전 단계 0–2의 매우 작은 반경 회전을 더 넓고 점진적인 코너링으로 개선하는 후속 실험이다. CNN 학습과 하드웨어 동작은 수행하지 않았다.','',
          f'24개 회전 이동을 탐색했고, 개발 검증 기준을 통과한 교사 후보는 **{len(accepted)}/24개**다. 이 중 현재 미션에서 허용되는 이동은 {sum(r["mission_allowed"] for r in accepted)}개다. 나머지는 이전 후보와 함께 보존하며 새 교사로 채택하지 않는다. 중앙 좌회전 4개는 기하학 실험이며 미션에 적용하지 않는다.','',
          '## 변경과 분리 비교','',
          '- 이전 목적함수는 경계 여유를 크게 만드는 것이 우선이어서 최소 반경 약 1.5cm인 경로도 선택했다. 이번에는 약 0.5cm/1cm 여유 후보를 각각 탐색하고, 최대 곡률·곡률 변화·곡률 변화율을 줄였다. 그 결과를 촘촘한 차체 윤곽 검사로 다시 확인했다.',
          '- 진입·회전·출구를 모두 5차 Bézier로 연결하고 접속점의 곡률을 0으로 맞췄다. 미분은 해석적으로 계산했다. 기존 경로도 그대로 보존했다. 탐색은 최적해/가능 최대 반경의 증명이 아니다.',
          '- 곡률 기반 기존 속도 방식과 회전속도/횡가속 상한 및 전후방 가속 제약을 둔 미리 감속하는 속도 계획을 비교했다. 조향 저역통과 필터를 추가하지 않았다. 속도 계획이 더 나쁘면 기존 속도 방식을 선택했다.',
          '- 24이동 × (기존 1 + 새 경로 2 × 속도 2) × 2차체 × 2조건 × 개발 seed1303 = 480회 선택 평가 후 이동별 조합을 고정했다. 선택 자료도 모두 보존했다.',
          '- 검증은 24이동 × 기존/후보 2 × 4차체 × fixed/mild/harsh 3 × 3시드 = 1728회다. 이는 개발 시드 1301/1303/1307의 비교이며 선택에 사용한 1303도 포함한다. 새로운 최종 시험 일반화 성적은 아니다. 최종 시험 및 학습 시드는 미사용이다.',
          '- 모든 쌍의 시작 자세와 외부 사건 일정이 같음을 검사했다. 36ms 처리, 최대 20ms 물리 적분, 실제 구동 지연·출력 문턱·포화·역회전 밀림을 유지했다. 원래 실행과 새 공통 루프의 PWM/물리 궤적 재현 검사도 수행했다.','',
          '## 전체 후보의 검증 성적','',
          '아래에는 보류 후보도 포함한다. 실패를 숨기거나 통과 사례만 골라 전체 개선으로 보고하지 않는다. 초기 경계를 밟은 배치도 무이탈 분모에서 제외하지 않았다.','',
          '| 조건 | 안정 출구: 이전 → 후보 | 무이탈 안정 출구: 이전 → 후보 |','|---|---:|---:|',*table,'',
          f'fixed의 주행별 중앙값: 교차로 중심 30cm 안에서의 각가속 RMS {np.median([r["turn_angular_accel_rms"] for r in aa]):.3f} → {np.median([r["turn_angular_accel_rms"] for r in bb]):.3f} rad/s². 통과 시간 {np.median([r["simulated_s"] for r in aa]):.2f} → {np.median([r["simulated_s"] for r in bb]):.2f}초.',
          '밀림과 출력 증감은 이전 단계와 같은 고정 시드 합성 일정이다. 경로와 속도에 따라 사건 당시 차량 위치와 종료 전 노출량은 달라질 수 있다. 쌍별 노출 시간을 paired_changes.json에 함께 보존했다. 속도·회전속도 채널은 Vehicle의 구동 응답이며 외부에서 강제로 더한 자세 밀림은 포함하지 않으므로, 부드러움 개선 주장은 외부 밀림이 없는 fixed 비교에 한정한다. 광학 입력을 사용하지 않는 정답 자세 교사이므로 강한 조명에 대한 CNN 개선을 주장하지 않는다.','',
          '## 교사 후보의 채택 조건','',
          'fixed/mild 각각에서 안정 도달 및 무이탈 건수가 감소하지 않고, 중앙 정렬 시작에서 경계 침범·전진 멈춤이 없으며, fixed의 회전 중 각가속 중앙값이 5% 이상 감소하고, 최소 참조 반경이 50% 이상 증가해야 한다. 평균 통과 시간이 기존의 1.35배+2초를 넘으면 보류한다. 개별 시드의 교환 관계는 paired_changes.json에 남겼다. harsh는 진단 범위이며 통과를 보장하지 않는다.',
          f'보류 사유 집계(중복 가능): `{dict(reasons)}`. 채택은 개발용 참조 경로에 한정하며 실제 데이터 구축 시 각 실행을 깨끗한 성공/회복/실패로 다시 구분해야 한다. 특히 north_W_right는 mild 무이탈 0/12이므로 기본 조건의 코너링 참조 및 복구 분기 검토용이다. 이를 약한 교란의 깨끗한 교사로 사용해서는 안 된다.','',
          '| 이동 | 반경 cm: 이전 → 후보 | fixed 무이탈 /12 | mild 무이탈 /12 | 상태 | 근거 |','|---|---:|---:|---:|---|---|']
    for r in registry:
        name='_'.join(r['key']);text.append(f'| {name} | {r["radius_old_cm"]:.2f} → {r["radius_new_cm"]:.2f} | {r["fixed_clean_old"]} → {r["fixed_clean_new"]} | {r["mild_clean_old"]} → {r["mild_clean_new"]} | {r["status"]} | [그래프](movements/{name}/comparison.png) · [영상](movements/{name}/comparison.mp4) |')
    text+=['','## 이어갈 때','',
           'teacher_registry.json의 accepted_dev만 새 코너링 참조 후보로 사용한다. held는 보류 사유를 해결하기 전까지 이전 자료와 비교 대상으로 유지한다. 경로가 넓어졌다는 이유만으로 경계 침범을 허용하지 않았다. 새 loss와 시퀀스 CNN 학습은 이번 작업에 포함하지 않았다.',
           f'원시 결과: `{c.OUT}`. 상세 설정·선택·모든 결과·회귀 테스트·소스/모델 해시는 protocol.json, selection_trials.json, audited_validation.json, verification.json에 연결돼 있다.','']
    (c.REPORT/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    c.dump(c.REPORT/'verification.json',dict(paired_initial_pose_and_events=True,paired_count=len(pairs),
         accepted=len(accepted),held=24-len(accepted),source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
         for p in [Path(__file__),Path(c.__file__),c.ROOT/'cnn_finetune/test_junction_cornering.py']},
         direct_model_sha256=hashlib.sha256((c.ROOT/'cnn_finetune/models/direct_v2/policy.onnx').read_bytes()).hexdigest(),
         final_test_used=False,training_started=False))


if __name__=='__main__':main()
