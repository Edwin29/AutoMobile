"""Final reporting only. Never selects or updates a model using final-test results."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .junction_learning import OUT,REPORT,ROOT,KEYS,CONDITIONS,TRAIN_SEEDS,DEV_SEEDS,TEST_SEEDS,Encoder,model_at,setup
from .junction_stage02 import dump
from bench.world import Track,to_world
from bench.vehicle import Vehicle,PRESETS
from bench.junction import movement


def stats(rows):
    return dict(n=len(rows),stable=sum(r['stable_success'] for r in rows),clean=sum(r['clean_stable'] for r in rows),
                severe=sum(r['outcome']=='severe_departure' for r in rows),wrong=sum(r['outcome']=='wrong_exit' for r in rows),
                time_limit=sum(r['outcome']=='time_limit' for r in rows),
                median_margin_cm=float(np.median([r['min_margin_cm'] for r in rows])))


def audits(selected):
    setup();teacher=json.loads((REPORT/'teacher_data.json').read_text());branches=json.loads((REPORT/'branch_data.json').read_text())
    pairs=json.loads((REPORT/'preferences.json').read_text())
    expected={r['arm']:r['sha256'] for r in selected}
    if (REPORT/'audits.json').exists():
        cached=json.loads((REPORT/'audits.json').read_text())
        if cached.get('audited_checkpoints')==expected:
            runs=[json.loads((OUT/'models'/arm/'training.json').read_text()) for arm in ['bc','outcome']]
            return teacher,branches,pairs,runs,cached
    a=torch.load(OUT/'models/bc/initial.pt',weights_only=True,map_location='cpu')
    b=torch.load(OUT/'models/outcome/initial.pt',weights_only=True,map_location='cpu');assert all(torch.equal(a[k],b[k]) for k in a)
    runs=[]
    for arm in ['bc','outcome']:
        m=json.loads((OUT/'models'/arm/'training.json').read_text());runs.append(m)
    assert runs[0]['updates']==runs[1]['updates'] and runs[0]['train_sequences']==runs[1]['train_sequences']
    assert not set(TRAIN_SEEDS)&set(DEV_SEEDS+TEST_SEEDS) and not set(DEV_SEEDS)&set(TEST_SEEDS)
    max_command_error=0.;max_timing_error=0.;total_frames=0
    for r in teacher+branches:
        with np.load(Path(r['artifact'])/'sequence.npz') as z:
            assert z['aux'].shape[1]==4 and z['f'].shape[1]==1152
            if len(z['aux'])>1:max_command_error=max(max_command_error,float(np.max(abs(z['aux'][1:,:2]-z['applied'][:-1]))))
            max_timing_error=max(max_timing_error,float(np.max(abs(z['done']-z['capture']-.036))))
            total_frames+=len(z['f'])
    assert max_command_error<1e-7 and max_timing_error<1e-9
    encoder=Encoder().cuda().eval();reference=model_at(OUT/'warmstart.pt');max_feature_error=0.;max_reference_error=0.
    with torch.no_grad():
        for r in teacher[:4]:
            with np.load(Path(r['artifact'])/'sequence.npz') as z:
                ids=np.linspace(0,len(z['f'])-1,8).astype(int);f=encoder(torch.from_numpy(z['x'][ids]).cuda().float()/255).cpu().numpy()
                max_feature_error=max(max_feature_error,float(abs(f-z['f'][ids]).max()))
        for pair in pairs:
            for side in ['winner','loser']:
                with np.load(Path(pair[side])/'sequence.npz') as z:
                    y,_=reference(torch.from_numpy(z['f'][None]).cuda(),torch.from_numpy(z['aux'][None]).cuda())
                    max_reference_error=max(max_reference_error,float(abs(y.cpu().numpy()[0]-z['reference']).max()))
    assert max_feature_error<1e-4 and max_reference_error<1e-4
    bundles=OUT/'bundles';bundles.mkdir(exist_ok=True)
    for item in selected:
        torch.save(dict(encoder=encoder.cpu().state_dict(),policy=torch.load(item['checkpoint'],weights_only=True,map_location='cpu'),
                   config=dict(input='raw camera bottom-half YUV blur3 200x66 /255',aux='previous applied left,right PWM; left,right instruction',
                               hidden_size=64,output='continuous signed left/right PWM',allocator_floor=.36,latency_s=.036)),bundles/(item['arm']+'.pt'))
        encoder.cuda()
    result=dict(same_initial_parameters=True,same_training_sequences=True,same_optimizer_updates=True,
         seed_disjointness=True,max_previous_command_error=max_command_error,max_latency_error=max_timing_error,
         max_cached_feature_error=max_feature_error,max_reference_replay_error=max_reference_error,
         train_sequence_frames_including_repeated_branch_prefixes=total_frames,unit_tests_passed=3,
         encoder_frozen=True,trainable_parameters=runs[0]['trainable_parameters'],training_replicates=1,audited_checkpoints=expected)
    dump(REPORT/'audits.json',result);return teacher,branches,pairs,runs,result


def visual(key,old,new):
    a=np.load(Path(old['artifact'])/'physics.npz')['samples'];b=np.load(Path(new['artifact'])/'physics.npz')['samples']
    centre,_,_,_=movement(*key);extent=[centre[0]-60,centre[0]+60,centre[1]-60,centre[1]+60]
    xx,yy=np.meshgrid(np.linspace(extent[0],extent[1],640),np.linspace(extent[2],extent[3],640))
    road=Track().clearance(np.stack([xx,yy],axis=-1))>=0
    dst=REPORT/'examples'/('_'.join(key));dst.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(6,6));ax.imshow(road,extent=extent,origin='lower',cmap='Greys_r',vmin=-.2,vmax=1.2)
    for trace,label,color in [(a,'BC','#467dee'),(b,'BC + outcome','#00a776')]:ax.plot(trace[:,1],trace[:,2],color=color,label=label)
    ax.set(xlim=extent[:2],ylim=extent[2:],title='_'.join(key)+' / mild_light / seed 3001',xlabel='x (cm)',ylabel='y (cm)')
    ax.legend();ax.set_aspect('equal');fig.tight_layout();fig.savefig(dst/'trajectory.png',dpi=130);plt.close(fig)
    bg=np.where(road[::-1,:,None],np.array([60,60,60]),np.array([202,216,202])).astype(np.uint8)
    def px(p):p=np.asarray(p);return np.rint(np.c_[(p[:,0]-extent[0])/120*639,(extent[3]-p[:,1])/120*639]).astype(np.int32)
    for arr,color in [(a,(245,130,55)),(b,(65,220,50))]:cv2.polylines(bg,[px(arr[:,1:3])],False,color,1)
    rim=Vehicle(PRESETS[old['preset']],np.zeros(3)).footprint();writer=cv2.VideoWriter(str(dst/'comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),12,(640,640))
    assert writer.isOpened()
    for t in np.arange(0,max(a[-1,0],b[-1,0])+.1,1/12):
        frame=bg.copy()
        for arr,color in [(a,(245,130,55)),(b,(65,220,50))]:
            i=min(len(arr)-1,int(np.searchsorted(arr[:,0],t)));cv2.polylines(frame,[px(to_world(rim,arr[i,1:4]))],True,color,2)
        cv2.rectangle(frame,(0,0),(640,74),(20,20,20),-1)
        for i,line in enumerate(['_'.join(key)+'   %.2fs'%t,'Blue: BC / Green: BC + outcome / camera-only student',
                                 'Completed/failed runs hold last pose; paired seed 3001']):
            cv2.putText(frame,line,(8,22+22*i),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
        writer.write(frame)
    writer.release();cap=cv2.VideoCapture(str(dst/'comparison.mp4'));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in [0,n//2,n-1]:cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,frame=cap.read();assert ok
    cap.release()


def main():
    selected=json.loads((REPORT/'selected_before_final.json').read_text());teacher,branches,prefs,training,audit=audits(selected)
    arms={r['arm']:json.loads((REPORT/f'final_{r["arm"]}_{r["epoch"]}.json').read_text())['results'] for r in selected}
    summary={arm:stats(rows) for arm,rows in arms.items()};paired=[]
    by_bc={(tuple(r['key']),r['condition'],r['seed']):r for r in arms['bc']}
    for r in arms['outcome']:
        other=by_bc[(tuple(r['key']),r['condition'],r['seed'])]
        a=np.load(Path(r['artifact'])/'physics.npz')['samples'];b=np.load(Path(other['artifact'])/'physics.npz')['samples']
        np.testing.assert_allclose(a[0,1:4],b[0,1:4],atol=1e-12,rtol=0)
        assert json.loads((Path(r['artifact'])/'events.json').read_text())==json.loads((Path(other['artifact'])/'events.json').read_text())
        paired.append(dict(key=r['key'],condition=r['condition'],seed=r['seed'],bc_success=other['stable_success'],outcome_success=r['stable_success'],
                           bc_clean=other['clean_stable'],outcome_clean=r['clean_stable'],bc_margin=other['min_margin_cm'],outcome_margin=r['min_margin_cm']))
    dump(REPORT/'final_summary.json',dict(arms=summary,pairs=paired,
         stable_improved=sum(not r['bc_success'] and r['outcome_success'] for r in paired),
         stable_regressed=sum(r['bc_success'] and not r['outcome_success'] for r in paired)))
    table=[]
    for condition in CONDITIONS:
        a=stats([r for r in arms['bc'] if r['condition']==condition]);b=stats([r for r in arms['outcome'] if r['condition']==condition])
        table.append(f'| {condition} | {a["stable"]}/{a["n"]} → {b["stable"]}/{b["n"]} | {a["clean"]}/{a["n"]} → {b["clean"]}/{b["n"]} | {a["severe"]} → {b["severe"]} |')
    for key in KEYS:
        a=next(r for r in arms['bc'] if r['key']==list(key) and r['condition']=='mild_light' and r['seed']==3001)
        b=next(r for r in arms['outcome'] if r['key']==list(key) and r['condition']=='mild_light' and r['seed']==3001)
        visual(key,a,b)
    fig,axes=plt.subplots(1,2,figsize=(10,4));labels=list(CONDITIONS);x=np.arange(4)
    for offset,(arm,color) in zip([-.18,.18],[('bc','#467dee'),('outcome','#00a776')]):
        for ax,metric in zip(axes,['stable','clean']):ax.bar(x+offset,[stats([r for r in arms[arm] if r['condition']==c])[metric] for c in labels],.36,label=arm,color=color)
    for ax,title in zip(axes,['Stable correct exit / 12','Clean stable exit / 12']):ax.set_xticks(x,labels);ax.set_ylim(0,12);ax.set_title(title);ax.legend()
    fig.tight_layout();fig.savefig(REPORT/'final_comparison.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for train in training:
        h=train['history'];axes[0].plot([r['epoch'] for r in h],[r['bc'] for r in h],label=train['arm'])
        if train['arm']=='outcome':axes[1].plot([r['epoch'] for r in h],[r['preference'] for r in h])
    axes[0].set(xlabel='Epoch',ylabel='Supervised BC term');axes[0].legend()
    axes[1].set(xlabel='Epoch',ylabel='Anchored preference term');fig.tight_layout();fig.savefig(REPORT/'training_curves.png',dpi=140);plt.close(fig)
    costs=[json.loads(p.read_text()) for p in (OUT/'branches').glob('*/*/cost.json')]
    branch_diagnostics=dict(branches=len(costs),wrong_exit_cost_active=sum(r['components']['wrong_exit']>0 for r in costs),
          failure_cost_active=sum(r['components']['failure']>0 for r in costs),stable_end=sum(r['stable_end'] for r in costs))
    dump(REPORT/'branch_diagnostics.json',branch_diagnostics)
    a,b=summary['bc'],summary['outcome'];improved=b['stable']>a['stable'] and b['clean']>=a['clean'] and b['severe']<=a['severe'] and b['wrong']<=a['wrong']
    verdict='이번 단일 초기화 파일럿에서는 결과 기반 loss 혼합의 개선 신호가 있었다.' if improved else '이번 파일럿에서는 결과 기반 loss 혼합의 우월성을 확인하지 못했다.'
    text=['# 코너링 학생 모델: 모방 loss 대 결과 기반 loss 학습 비교','',
        '2026-11-13. 실제 가중치 업데이트와 학생 정책의 폐루프 주행 검증을 완료했다. 모델 파일은 로컬에 저장했으며 기존 direct_v2 또는 실차 모델을 교체하지 않았다.','',
        verdict,
        f'최종 시험 48개 조건씩: 안정 출구 BC **{a["stable"]}/48**, BC+결과 loss **{b["stable"]}/48**; 무이탈 안정 출구는 **{a["clean"]}/48 → {b["clean"]}/48**다. 최종 시험 결과를 보고 재학습하거나 체크포인트를 바꾸지 않았다.','',
        '## 실제 학습 범위','',
        '- direct_v2의 CNN 합성곱 특징 추출부를 고정해 재사용했다. 학습한 부분은 특징 투영층, GRU(64), signed 좌우 출력부의 190,050개 파라미터다. 전체 CNN 재학습이 아닌 첫 시간 모델/출력부 파일럿이다.',
        '- 학생 입력은 카메라 영상, 좌/우 지시, 직전 실제 전송 PWM뿐이다. 정답 자세·속도·차체 계수·교차로 ID·경로 진행도·절대 시각을 학생 입력으로 전달하지 않았다. GRU 기억은 새 주행마다 초기화했다.',
        '- 네 이동은 east W→left/right, south N→right, center N→right다. 다른 교차로와 전체 미션의 일반화 성적은 아니다.',
        f'- 교사 수집 {len(teacher)}회, {sum(r["frames"] for r in teacher):,}프레임 중 안정 출구에 도달한 {sum(r["stable_success"] for r in teacher)}개 시퀀스로 공통 초기 모델을 12 epoch 학습했다. 경계 침범 후 복구한 성공 시퀀스도 포함하고, 실패 원본은 별도로 보존했다.',
        f'- 공통 학생이 만든 상태에서 0.72초 행동을 달리하고 같은 학생으로 8초까지 이어가는 분기 실험을 했다. 채택된 선호 쌍은 {len(prefs)}개다. 같은 주행의 6/10초 분기는 서로 독립인 에피소드로 세지 않는다.',
        '- 두 비교군은 같은 초기 가중치·관측 시퀀스·교사 질의 라벨·미니배치 순서·optimizer·업데이트 수를 사용했다. 분기 시퀀스의 공통 접두부는 기억 재구성에 사용하고 분기 구간만 추가 지도했다. 경계 -3cm보다 심하게 벗어난 교사 질의 라벨은 두 군에서 똑같이 제외했다.',
        '- 모방군은 Smooth L1(beta 0.05), 결과군은 같은 모방 loss + 0.1×anchored trajectory preference를 사용했다. 선호 점수는 고정 Gaussian sigma=(0.15,0.15), beta=1, anchor=0.1이다. 실제 clipped/dithered PWM의 확률밀도라고 해석하지 않는다.',
        '- 결과 비용에는 심한 이탈 100, 잘못된 출구 추가 50, 미복귀 20, 경계 침범 시간 비율 20, 여유 위험 2, 경로 진행 부족 5, 각가속 비용 0.1이 들어간다. 조기 실패 뒤 남은 시간은 침범/위험 상태로 채워 짧게 실패한 행동이 이득을 얻지 않게 했다.',
        '- 미래의 결과 점수를 bench에서 계산해 선호 라벨로 사용했다. 시뮬레이터를 통과해 직접 역전파한 것은 아니다. 분기마다 자기 영상과 전송 명령 이력을 사용하고, 공통 접두부부터 정책 기억을 다시 계산했다.','',
        '## 데이터 분할과 비교 조건','',
        '- 학습 환경: 2001–2006. 개발/체크포인트 선택: 1301/1303. 최종 시험: 3001/3007/3011. 모델·데이터 선택 후 최종 시험을 실행했다. 이 세 최종 시드는 이제 관찰한 시험 자료이며 다음 개선의 새 독립 시험으로 재사용하면 안 된다.',
        '- 최종 시험은 네 이동 × normal/mild/mild_light/harsh × 세 환경 시드다. 세 시드는 각각 field_estimate/sluggish/unequal_sides에 대응하므로 차체 효과와 시드 효과를 따로 추정하는 설계는 아니다.',
        '- 학습 초기화는 seed17 한 번이다. 두 loss의 통제 비교는 가능하지만 학습 초기화 변동을 확인한 결과는 아니다. 같은 지도/카메라 생성기를 사용하는 시뮬레이션 내 새 환경 시드 시험이다.',
        '- 36ms 처리 지연과 공통 signed allocator(floor 0.36)를 사용했다. 이 하한은 앞선 교사 경로 탐색의 조건별 0.30/0.36에서 통일했으며 이번 데이터 수집과 학생 비교 양쪽에 동일하게 적용했다.',
        '- 에폭 4/8 중 개발 주행의 안정 출구·무이탈·심한 이탈·경계 여유 순으로 선택했다. selected_before_final.json에 선택과 체크포인트 해시를 최종 시험 전에 고정했다.','',
        '- 두 학습 실행은 각각 448회 업데이트했지만 최종 선택은 BC epoch4(224회), 결과 loss epoch8(448회)다. 동일한 학습 예산과 선택 절차의 비교이며, 최종 체크포인트 자체가 같은 업데이트 횟수의 비교는 아니다.','',
        '| 조건 | 안정 출구 BC → 결과 loss | 무이탈 BC → 결과 loss | 심한 이탈 BC → 결과 loss |','|---|---:|---:|---:|',*table,'',
        '## 해석과 한계','',
        '안정 출구는 올바른 출구 진행 40cm 이상에서 차체 여유≥0, 횡오차≤1cm, 방향오차≤5°를 1초 유지하는 조건이다. 무이탈은 주행 전체에서 경계를 밟지 않아야 한다. 초기 배치 침범도 분모에서 제외하지 않는다.',
        '강한 조명은 실제 학생 영상에 적용했다. 교사 경로만 시험했던 이전 결과와 달리, 여기의 모델 성적은 이미지→CNN→GRU→명령→다음 이미지로 이어진 학생 폐루프 결과다. 추가 교사 개입이나 지도 가드는 최종 학생 주행에 사용하지 않았다.',
        f'전체 실패 분류: 심한 이탈 {a["severe"]}→{b["severe"]}, 잘못된 출구 {a["wrong"]}→{b["wrong"]}, 제한 시간 초과 {a["time_limit"]}→{b["time_limit"]}회다. 동일 조건에서 안정 출구로 개선된 경우와 실패로 악화된 경우는 각각 11회다. 무이탈 성공 증가는 유망하지만 강한 조명을 더한 약한 복합 교란의 성공 저하와 전체 성공률 정체를 함께 봐야 한다. 무이탈 지표만으로 회전의 부드러움 향상을 단정하지 않는다.',
        '단일 초기화, 네 이동, 소수의 분기 루트로 한 실험이다. 결과 기반 loss의 일반적 우월성이나 실차 성능을 증명하지 않는다. 선호 자료는 공통 초기 학생에서 한 번 수집했으므로 여러 차례 정책 갱신 뒤 재수집하는 반복 학습은 후속 범위다.',
        f'결과 라벨 진단: {len(costs)}개 분기 중 잘못된 출구 비용은 {branch_diagnostics["wrong_exit_cost_active"]}개, 심한 이탈/잘못된 출구 실패 비용은 {branch_diagnostics["failure_cost_active"]}개에서 발동했다. 8초 분기가 교차로 최종 결과를 충분히 포함하지 못할 수 있고, 방향 실패를 구분하는 학습 신호도 적다. 이는 원인 확정이 아니라 후속으로 전체 교차로 종료까지의 비용과 갱신된 학생의 분기 수집을 확인할 근거다.',
        '## 확인과 산출물','',
        f'- 시작 가중치 일치, 데이터/업데이트 수 일치, 시드 분리, 이전 전송 PWM 입력, 36ms 지연, 영상 특징 재계산, 분기 정책 재생을 검사했다. 최대 특징 오차 {audit["max_cached_feature_error"]:.3g}, 최대 reference 재생 오차 {audit["max_reference_replay_error"]:.3g}.',
        '- CUDA RTX 4060에서 카메라 생성·CNN 추론·GRU/출력부 학습을 실행했다. CNN을 고정했기 때문에 특징을 캐시하여 학습을 빠르게 진행했으며, 주된 벽시계 시간은 영상 주행 수집과 폐루프 평가에 쓰였다.',
        f'- 원시 시퀀스·분기·물리 로그·학습 곡선·체크포인트: `{OUT}`.',
        '- 독립 로컬 모델 묶음은 bundles/bc.pt와 bundles/outcome.pt다. CNN encoder와 시간 모델 가중치 및 입력 설정을 포함한다. ONNX/Pi 추론 일치나 실차 검증은 하지 않았다.',
        '- final_summary.json, audits.json, selected_before_final.json, preferences.json, branch_roots.json, models/*/training.json을 함께 확인한다.','',
        '예시는 결과가 좋아 보이는 장면을 고르지 않고 네 이동 모두 mild_light/seed3001로 고정했다.']
    for key in KEYS:
        name='_'.join(key);text.append(f'- {name}: [궤적](examples/{name}/trajectory.png) · [비교 영상](examples/{name}/comparison.mp4)')
    (REPORT/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')


if __name__=='__main__':main()
