"""Derived evidence for entry recovery feasibility and prediction error."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .goal_entry_recovery_search import REPORT as SEARCH_REPORT
from .goal_entry_forecast import REPORT as FORECAST_REPORT
from .goal_entry_robust import REPORT
from .goal_development import ROOT,sha
from .junction_stage02 import dump


def identity(r):return tuple(r['key']),r['seed'],r['preset']


def main():
    search=json.loads((SEARCH_REPORT/'results.json').read_text());cases=[r for r in search['rows'] if r['case']['root_mode']=='post_observation']
    candidates={name:json.loads((FORECAST_REPORT/(name+'.json')).read_text()) for name in ['decay','hold035']}
    candidates['duration_uncertainty']=json.loads((REPORT/'results.json').read_text())
    mismatch={};performance={}
    for name,value in candidates.items():
        rows=value['rows'];by_id={identity(r['source']):r for r in rows};actual=[];predicted=[];oracle_safe_predicted=0
        for case in cases:
            r=by_id[identity(case['case']['source'])]
            a=np.array([x['min_margin_cm'] for x in case['short_branches']])
            b=np.array(r['decision']['worst_margins'] if name=='duration_uncertainty' else r['decision']['min_margins'])
            actual.extend(a);predicted.extend(b)
            good=[x for x in case['full_branches'] if x['clean_moving']]
            oracle_safe_predicted+=bool(good and any(b[x['index']]>=.01 for x in good))
        a=np.array(actual);b=np.array(predicted)
        mismatch[name]=dict(branches=len(a),actual_short_safe=int(sum(a>=0)),predicted_short_safe=int(sum(b>=0)),
            false_safe=int(sum((b>=0)&(a<0))),false_unsafe=int(sum((b<0)&(a>=0))),
            mean_abs_min_margin_error_cm=float(np.mean(abs(a-b))),oracle_clean_case_with_predicted_safe_alternative=oracle_safe_predicted,
            scope='93 failed development roots x92 actions. Robust uses minimum across hypothetical durations, not expected realized margin.')
        results=[r['result'] for r in rows];both=[r for r in rows if r['result']['continuous_clean'] and r['source']['continuous_clean']]
        performance[name]=dict(value['summary'],outcomes={v:sum(r['outcome']==v for r in results) for v in sorted(set(r['outcome'] for r in results))},
            pause_cases=sum(r['translation_pause_s']>0 for r in results),
            both_clean_n=len(both),paired_both_clean_median_time_change_s=float(np.median([r['result']['simulated_s']-r['source']['simulated_s'] for r in both])),
            paired_both_clean_median_angular_accel_change=float(np.median([r['result']['angular_accel_rms_rad_s2']-r['source']['angular_accel_rms_rad_s2'] for r in both])))
    dump(REPORT/'prediction_error_audit.json',dict(source_sha256=sha(__file__),mismatch=mismatch,performance=performance))
    labels=['Baseline','Decay forecast','0.35s hold','Duration uncertainty'];clean=[147]+[performance[n]['clean'] for n in candidates]
    moving=[147]+[performance[n]['clean_moving'] for n in candidates]
    fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained');x=np.arange(4)
    ax.bar(x-.18,np.array(clean)/240*100,.36,label='Clean exit',color='#2563eb')
    ax.bar(x+.18,np.array(moving)/240*100,.36,label='Clean, no pause/reverse',color='#0f766e')
    for i,(a,b) in enumerate(zip(clean,moving)):
        ax.text(i-.18,a/240*100+1,str(a),ha='center',fontsize=10);ax.text(i+.18,b/240*100+1,str(b),ha='center',fontsize=10)
    ax.set_xticks(x,labels);ax.set_ylim(0,100);ax.set_ylabel('Share of 240 entry diagnostic cases (%)');ax.legend(loc='upper left')
    ax.set_title('One decision at a known shove onset; not an online student evaluation')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.savefig(REPORT/'entry_decision_comparison.png',dpi=155);plt.close(fig)
    lines=['# 진입 복구 가능성과 예측기의 선택 오류','',
        '기존 학생 모델이나 교사 레지스트리를 교체하지 않았다. 이번 작업은 진입 실패 원인을 구분하고, 안전한 교사 행동을 만들기 위한 오프라인 진단이다. 새 CNN/GRU 학습·실차 작업·최종 시험은 없다.','',
        '## 먼저 복구 조작이 존재하는가','',
        '- 기존 여유 경로 교사의 진입 사건 실패 93건 전부를 첫 밀림 관측 상태에서 분기했다. 각 상태에서 두 구간(각 7명령, 0.252초)의 평균 PWM 배율/차동 보정 92개를 검사했다.',
        '- 짧은 2.16초 결과로 차체 여유 상위 3개와 계속 전진한 후보 상위 3개, 원래 조작을 골라 출구까지 실행했다. 8,556개 짧은 분기와 402개 출구 연장이 수행됐다.',
        '- 53/93건에서 정지·후진 없이 출구까지 무이탈인 대체 조작을 실제 시뮬레이션으로 찾았다. 실패한 유한 탐색을 물리적 불가능의 증명으로 보지 않는다.',
        '- 앞선 지연 감사에서 새 명령이 작용하기 전 이탈한 6건은 원래 사건 발생 0.5초 전에도 분기했다. 552개 짧은 분기/24개 출구 연장 중 1건에서 대체 조작을 찾았다. 이는 사건 전 진입 상태를 바꾸면 결과가 달라질 수 있다는 사례이며, 미래 사건을 예측할 수 있다는 뜻이 아니다.',
        '- 모든 원본 재생, 분기 시점의 원래 제어 연장, 무수정 짧은 분기와 전체 주행이 저장된 물리 기록과 정확히 일치했다.',
        '- 위 대체 조작은 실제 미래 결과를 보고 골랐다. 온라인 정책 성능이나 확정된 학생 정답으로 사용하지 않았다.','',
        '## 현재 관측으로 조작을 고르는 진단','',
        '현재 자세·바퀴 상태·이미 전송한 명령·명목 차량 매개변수와 이미 관측된 차체 운동 잔차로 92개 후보를 예측했다. 이전 예측과 달리 미래 교사의 조작도 예측 중 변한 자세에 반응한다. 실제 미래 교란 일정·지속시간·난수는 예측 입력에 없다.','',
        '**단, 결정 시점은 오프라인에서 실제 밀림의 첫 관측으로 정렬했다. 실제 사건 탐지기와 지속적인 재계획을 포함한 온라인 제어기 성능은 아니다.** 20개 회전 × 시드 4개(6501–6504) × 차량 3종 = 동일한 개발 상태 240건이다. 기존 성공 147건도 모두 포함해 회귀를 검사했다.','',
        '| 선택 방식 | 무이탈 /240 | 무이탈·정지/후진 없음 | 새 성공 | 잃은 기존 성공 |','|---|---:|---:|---:|---:|',
        '| 기존 교사 | 147 | 147 | — | — |']
    for n in candidates:
        r=performance[n];lines.append(f"| {n} | {r['clean']} | {r['clean_moving']} | {r['gains']} | {r['losses']} |")
    lines+=['','- decay: 현재 운동 잔차를 명령 처리+명목 구동 지연 동안 유지한 뒤 시정수 0.2초로 감쇠했다.',
        '- hold035: 현재 잔차가 0.35초 더 유지된다고 예측했다.',
        '- duration_uncertainty: 0~0.7초, 0.1초 간격의 종료 가정을 모두 비교했다. 모든 가정에서 움직이며 안전한 후보 중 최악의 차체 여유가 가장 큰 후보를 선택한다. 안전 후보가 없으면 최악의 평균 침범 비용을 줄이는 조작을 선택한다. 가정 구간 사이까지 보장하는 안전 증명은 아니다.',
        '- 미래를 모르는 최초 두 예측기는 조작 차이를 최소화하는 선택 기준을 썼다. 세 번째는 지속시간 처리와 선택 기준을 함께 바꿨으므로 성능 차이를 지속시간 처리 하나의 효과라고 단정하지 않는다.',
        '- 예측·선택 파일을 먼저 저장한 뒤 실제 미래 주행을 실행해 채점했다. 학생에게 정답 자세·속도·매개변수를 새로 제공하지 않았다.','',
        '## 예측이 왜 틀렸는가','',
        '기존 실패 93개 상태 × 조작 92개 = 8,556개 분기에서, 예측 차체 여유와 실제 2.16초 주행의 여유를 대응시켰다.','',
        '| 예측 방식 | 실제 안전 분기 | 안전으로 예측했지만 실제 침범 | 위험 예측이나 실제 안전 | 여유 오차 평균(cm) |','|---|---:|---:|---:|---:|']
    for n,m in mismatch.items():lines.append(f"| {n} | {m['actual_short_safe']} | {m['false_safe']} | {m['false_unsafe']} | {m['mean_abs_min_margin_error_cm']:.3f} |")
    lines+=['','이 수치는 개발 실패 상태에 대한 진단이며 전체 상태 분포의 오차율이 아니다. 여러 지속시간의 최악값은 실제 단일 사건의 불편 추정량이 아니므로, robust 열의 절대 오차를 단순 예측 정확도 순위로 사용하지 않는다.',
        'decay와 hold035는 실제보다 안전하다고 보는 분기가 많았다. 복구 가능한 후보가 있는데도 작은 수정만 하는 위험한 후보를 고르는 사례를 확인했다. 불확실성을 더 크게 잡으면 회복 수는 늘지만 정상적으로 성공하던 사례를 손상시키거나 정지/후진하는 부작용도 생겼다.','',
        '## 검증과 판단','',
        '- 명목 물리 모델과 일정한 차체 밀림에서 36개 예측/실제 조작 비교가 통과했다. 지속시간 매개변수화는 원래 0.35초 예측과 정확히 일치했고, 0/0.7초 종료를 실제 시뮬레이터와 비교한 18개 검사도 통과했다.',
        '- 복구 가능성 53건은 교사를 개선할 근거다. 단순 교사 오차만 줄이는 학습 전에, 교사의 예측 불확실성과 조작 선택 오류를 줄여야 한다.',
        '- duration_uncertainty는 40건을 회복했지만 기존 성공 8건을 잃고, 무이탈 중 11건은 정지/후진 없는 조건을 만족하지 않았다. 따라서 새 교사나 모델로 자동 채택하지 않았다.',
        '- 이 결과로 사건 시작을 알려주는 오프라인 정렬을 제거한 관측 기반 발동과 지속적 제어를 검증했다. 후속 새 시드 비교는 `../goal_entry_reactive_20261113/REPORT.md`에 있다. 실패 원인을 구분하지 않은 채 loss 가중치만 반복 조정하지 않는다.',
        '- 전체 경로 성공률, 모든 이동의 약한 복합 교란 90%, 영상과 강한 조명, 세 학습 초기화, 레거시 대비 시간/부드러움, 사전 정의 harsh, 독립 최종 시험은 여전히 미완료다.','',
        '원본은 `D:/autodrive_cnn/runs/goal_entry_*_20261113`, 요약과 감사는 이 폴더의 `results.json`, `prediction_error_audit.json`에 있다. 그림은 `entry_decision_comparison.png`다.','']
    (REPORT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(performance),flush=True)


if __name__=='__main__':main()
