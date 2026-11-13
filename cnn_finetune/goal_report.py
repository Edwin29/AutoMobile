"""Summarize the first goal milestone without redefining acceptance."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .goal_development import ROOT,OUT,REPORT,sha
from .junction_stage02 import dump


def main():
    names=['teacher','warm','bc','outcome','converged_bc'];conditions=['normal','mild','mild_light']
    data={n:json.loads((REPORT/(n+'.json')).read_text()) for n in names}
    summary={n:{c:dict(n=len(rs),clean=sum(r['clean_stable'] for r in rs),stable=sum(r['stable_success'] for r in rs),
        wrong=sum(r['outcome']=='wrong_exit' for r in rs),severe=sum(r['outcome']=='severe_departure' for r in rs))
        for c in conditions for rs in [[r for r in data[n] if r['condition']==c]]} for n in names}
    teacher=json.loads((REPORT/'teacher_validation.json').read_text())['summary']
    recovery=json.loads((REPORT/'recovery_feasibility.json').read_text())
    train=json.loads((REPORT/'convergence_training.json').read_text())
    dump(REPORT/'milestone_summary.json',dict(goal_status='active',goal_complete=False,local_student_comparison=summary,
         teacher_validation=teacher,recovery_grid=dict(cases=recovery['cases'],clean_branch_found=recovery['clean_branch_found']),
         additional_training=dict(epochs=60,updates=train['updates'],selected_epoch=train['selected_epoch'],checkpoint=train['selected_checkpoint']),
         final_exam_used=False,all_required_movements_covered=False,full_mission_tested=False,three_initializations_tested=False,
         interpretation='Development diagnostics only. Local scoring changed prospectively; not directly comparable to previous 1cm/5deg pilot.'))
    fig,ax=plt.subplots(figsize=(10,4));x=np.arange(3)
    labels={'teacher':'Original teacher','warm':'Warmstart BC','bc':'Old BC4','outcome':'Old outcome8','converged_bc':'Additional BC60'}
    for i,n in enumerate(names):
        vals=[summary[n][c]['clean'] for c in conditions]
        ax.bar(x+(i-2)*.16,vals,.15,label=labels[n])
    ax.set_xticks(x,conditions);ax.set_ylim(0,26);ax.set_ylabel('Clean stable local exits / 24')
    ax.set_title('Development only: 4 movements x 2 seeds x 3 vehicles');ax.legend(ncol=3,fontsize=8)
    fig.tight_layout();fig.savefig(REPORT/'development_comparison.png',dpi=150);plt.close(fig)
    rows=['| 모델 | 기본 | 약한 복합 | 약한 복합+강한 조명 | 전체 |','|---|---:|---:|---:|---:|']
    ko={'teacher':'기존 교사','warm':'공통 초기 BC','bc':'기존 BC4','outcome':'기존 결과 loss8','converged_bc':'추가 BC 60 epoch'}
    for n in names:
        vals=[summary[n][c]['clean'] for c in conditions]
        rows.append('| '+ko[n]+' | '+' | '.join(f'{v}/24' for v in vals)+f' | {sum(vals)}/72 |')
    text=['# Goal 개발 첫 진단 및 최적화 기록','',
      '목표는 active이며 완료하지 않았다. 사용자와 합의한 합격선을 acceptance.json에 저장했다. 전체 미션 시험 명세와 물리적으로 복구 가능한 harsh 범위는 아직 완성되지 않았고, 최종 시험 시드는 사용하지 않았다.','',
      '## 평가를 먼저 고정한 변경','',
      '기존 1cm/5도 기준은 별도 지표로 보존했다. 새 개발용 국소 평가는 출구 진행 40cm 이후 차체가 도로 안이고 방향오차 10도 이내인 상태를 1초 유지한 뒤 첫 55cm 지점에서 종료한다. 중심선 횡오차 1cm를 성공의 필수 조건으로 삼지 않는다. 성공 전 다음 교차로까지 달리는 문제를 막는다. 이는 전체 미션 완료의 대체가 아니다. protocol.json은 결과 수집 전에 고정했다.',
      '시드 4101/4102 × 차체 3종을 완전 교차했다. 네 이동과 세 조건의 72회씩 비교이며 모든 시작 배치가 유효했다. 정책에 정답 자세를 전달하지 않았다.','',
      '## 학생 비교','',*rows,'',
      '추가 BC는 동일한 공통 초기 가중치, 동일한 기존 교사 94개 시퀀스, 동일한 CNN 동결/GRU 구조에서 60 epoch를 더 학습했다. AdamW 상태는 원본에 없어 새로 시작했고 학습률은 1e-4→1e-5로 낮췄다. 분기/선호 자료는 추가하지 않았다. 새 개발 교사 영상의 행동 MAE로 선택을 고정한 뒤 폐루프 평가했다.',
      '최종 epoch60이 선택되었지만 loss가 완전히 수렴했다고 입증한 것은 아니다. converged_bc라는 아티팩트 태그를 최적화 완료의 증명으로 해석하지 않는다. 개선은 관측됐지만 무이탈 37/72는 합격선에 크게 못 미친다. 초기화 seed17 한 번이며 기존 실차/ONNX 모델은 교체하지 않았다.','',
      '## 원인별 확인','',
      '- 교사 영상에서의 오차가 낮은 공통 초기 모델도 직접 운전하면 무이탈 22/72였다. 교사 영상 오차 하나로 모델을 선정하면 충분하지 않다.',
      '- 같은 영상/이전 PWM에서 좌·우 지시만 바꾸면 세 정책 모두 회전 명령 순서가 바뀌었다. 방향 입력을 완전히 무시한다는 가설은 지지되지 않았다. 이것은 명령 준수 성공률의 보장이 아니다.',
      '- 기존 분기 데이터의 지도학습은 0.72초 개입 구간에 집중돼 있고 이후 전체 주행을 반복 수집한 DAgger는 아니다. 학생 방문 상태의 전체 구간, 특히 출구 복구 자료를 보강할 필요가 있다.',
      '- 초기 모델에서 0.1을 곱한 선호 항의 평균 gradient norm은 약 0.720, 교사 BC는 0.161이었다. 가중치 0.1이 학습 영향 10%를 뜻하지 않는다. 체크포인트별 값과 방향 관계는 gradient_audit.json에 기록했으며 이것만으로 성능 저하 원인을 확정하지 않는다.','',
      '## 교사와 물리적 복구','',
      '기존 경로에서 제어 이득·속도 12개 조합을 검사한 576회 진단은 약한 교란의 무이탈 18/24를 개선하지 못했다. 중앙 교차로 출구의 기준 경로가 중심선에서 약 1.2cm 떨어진 채 진행하는 것이 여유 부족에 기여하는지, 출구 오프셋을 더 일찍 0으로 만드는 7개 경로/이득 조합 336회로 확인했다.',
      '개발 선택에서는 출구 35cm까지 중앙에 정렬하고 횡오차 이득을 0.08로 둔 후보가 기본·약한 교란 모두 24/24회 무이탈이었다. 선택을 고정한 뒤 새 시드 5201–5208과 차체 3종으로 576회 쌍별 검증했다.',
      '| 새 시드 교사 검증 | 기존 | 수정 후보 |','|---|---:|---:|',
      '| 기본 무이탈 | 90/96 | 96/96 |','| 약한 교란 무이탈 | 68/96 | 81/96 |',
      '| 기존 harsh 무이탈 | 0/96 | 3/96 |','| 기존 harsh 안정 출구 | 70/96 | 61/96 |','',
      '이 검증은 시작 차체가 도로 밖인 배치를 주행 전 중앙선으로 투영하는 규칙을 사전 고정하고 양쪽에 동일 적용했다. 조건별 12/96 배치가 투영됐다. 학생 72회 비교와 동일 모집단이 아니므로 직접 성능 격차로 빼지 않는다.',
      '수정 교사의 약한 교란 개선은 새 시드에서도 관측됐지만 목표 95%에 미달한다. 기존 harsh에서 안정 출구는 오히려 악화했으므로 전면 교체하지 않았다. 이 결과를 근거로 어려운 시험을 제거하거나 합격선을 낮추지 않는다.',
      f'수정 교사의 약한 교란 실패 {recovery["cases"]}개에서 밀림 시작 72ms 뒤 상태를 복제하고 0.36초 행동을 달리한 오프라인 분기를 비교했다. {recovery["clean_branch_found"]}개에서는 무이탈 대안을 찾았다. 나머지는 제한된 후보 탐색의 실패이며 물리적 불가능성의 증명이 아니다.','',
      '## 다음 작업과 판단 경계','',
      '1. 약한 교란의 잔여 실패를 경계 여유·반응 지연·차체 앞뒤 쓸림으로 분해하고, 예측 불가능한 미래 교란을 입력으로 주지 않는 복구 교사를 검토한다.',
      '2. 개선 교사의 깨끗한 시퀀스와 학생이 실제 방문한 전체 주행 상태를 구분해 수집한다. 경계 침범 경로를 깨끗한 정답으로 혼합하지 않는다.',
      '3. 충분히 학습한 BC를 기준선으로 데이터 보강과 결과 기반 loss를 비교한다. CNN 학습 해제는 별도 통제 비교로 진행하며 현재 동결이 실패 원인이라고 단정하지 않는다.',
      '4. 네 이동을 통과한 뒤 필수 이동 전체와 연결 미션, 최소 3개 학습 초기화, 독립 최종 시험으로 확대한다.',
      '5. 합격선·학생 입력 계약·기존 harsh와 새 복구 가능 범위의 의미를 바꿔야 한다면 사용자에게 먼저 보고한다.','',
      '## 보존과 검사','',
      '새 채점 경계와 기존 주행 입력/물리 prefix 동등성 시험 2개를 통과했다. 렌더링을 생략한 교사 실행도 18초 구간의 물리·전송 명령이 렌더링 버전과 정확히 일치했다. 이전 실험 모듈과 기존 모델을 수정하지 않았다.',
      f'원시 자료와 새 체크포인트: `{OUT}`. 최종 평가나 실차 검증은 아니다.']
    (REPORT/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    state=json.loads((REPORT/'work_status.json').read_text());state.update(stage='first_diagnostic_milestone_complete_goal_active',
        new_training=True,goal_complete=False,final_exam_used=False,
        latest_candidate=train['selected_checkpoint'],latest_candidate_clean=37,latest_candidate_runs=72,
        critical_findings=['Modified teacher improves mild but does not meet 95%; existing harsh remains unresolved and stable success regressed',
                           'Teacher-forced fit and local success alone are not completion; four-movement coverage only'])
    dump(REPORT/'work_status.json',state)
    snap=OUT/'source_snapshot'
    sources={}
    for p in (ROOT/'cnn_finetune').glob('goal_*.py'):
        sources[p.name]=sha(p)
        target=snap/p.name
        if target.exists():assert sha(target)==sha(p)
        else:shutil.copy2(p,target)
    dump(REPORT/'milestone_sources.json',sources)
    print('milestone report saved; goal remains active',flush=True)


if __name__=='__main__':main()
