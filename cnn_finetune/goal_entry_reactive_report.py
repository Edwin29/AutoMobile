"""Prospective causal-teacher evidence and completion ledger updates."""
import json
from pathlib import Path
import shutil
import numpy as np
from .goal_entry_reactive import REPORT
from .goal_entry_guarded import REPORT as GUARDED_REPORT
from .goal_development import ROOT,sha
from .junction_stage02 import dump


def identity(r):return tuple(r['key']),r['seed'],r['preset'],r['phase']


def main():
    base_data=json.loads((REPORT/'results.json').read_text());guard=json.loads((GUARDED_REPORT/'results.json').read_text())
    rows=base_data['rows']+guard['rows'];baseline={identity(r):r for r in rows if r['profile']=='baseline'}
    summaries=dict(base_data['summary'],guarded=guard['summary']);paired={};movements=[]
    for name in ['hold035','robust','guarded']:
        paired[name]={}
        for phase in ['none','entry','middle','exit']:
            a=[r for r in rows if r['profile']==name and r['phase']==phase]
            assert len(a)==120
            for r in a:assert r['initial_sampling']==baseline[identity(r)]['initial_sampling']
            both=[r for r in a if r['continuous_clean'] and baseline[identity(r)]['continuous_clean']]
            changed=[r for r in both if r['nonzero_decisions']>0]
            times=[r['simulated_s']-baseline[identity(r)]['simulated_s'] for r in both]
            alpha=[r['angular_accel_rms_rad_s2']-baseline[identity(r)]['angular_accel_rms_rad_s2'] for r in both]
            paired[name][phase]=dict(gains=sum(r['continuous_clean'] and not baseline[identity(r)]['continuous_clean'] for r in a),
                losses=sum(not r['continuous_clean'] and baseline[identity(r)]['continuous_clean'] for r in a),
                both_clean=len(both),both_clean_intervened=len(changed),
                median_time_change_s=float(np.median(times)),p95_time_change_s=float(np.quantile(times,.95)),
                median_angular_accel_change=float(np.median(alpha)),p95_angular_accel_change=float(np.quantile(alpha,.95)),
                intervened_median_time_change_s=None if not changed else float(np.median([r['simulated_s']-baseline[identity(r)]['simulated_s'] for r in changed])),
                intervened_median_angular_accel_change=None if not changed else float(np.median([r['angular_accel_rms_rad_s2']-baseline[identity(r)]['angular_accel_rms_rad_s2'] for r in changed])))
    for key in sorted(set(tuple(r['key']) for r in rows)):
        movements.append(dict(key=key,**{name:dict(n=18,clean=sum(r['continuous_clean'] for r in rows if tuple(r['key'])==key and r['profile']==name and r['phase']!='none')) for name in summaries}))
    startup=[]
    for r in rows:
        if r['profile']!='baseline' or r['phase']!='none' or r['clean_moving']:continue
        trace=np.load(Path(r['artifact'])/'physics.npz')['physics'];mask=trace[:,4]<-.01
        startup.append(dict(key=r['key'],seed=r['seed'],preset=r['preset'],first_s=float(trace[mask,0][0]),last_s=float(trace[mask,0][-1]),
            minimum_velocity_cm_s=float(trace[:,4].min()),reverse_distance_cm=float(np.sum(np.maximum(0,-trace[:,4])*np.diff(trace[:,0],prepend=0)))))
    dump(REPORT/'comparison.json',dict(source_sha256=sha(__file__),summary=summaries,paired=paired,movements=movements,
        startup_reverse_audit=startup,paired_initial_conditions_equal=True,
        strict_forward_note='clean_moving requires no velocity below-.01cm/s anywhere, including startup. Eight no-event baseline cases fail this extra diagnostic solely during0.305-0.521s with0.006-0.028cm backwards travel. Not a definitive failure of the turn-only smoothness goal.'))
    lines=['# 관측 기반 복구 교사 — 새 시드 검증','',
        '실제 교란 시작을 알려주던 이전 분기 진단을 벗어나, 현재까지 관측된 차체 운동 잔차로만 복구를 발동하도록 연결했다. 교사는 오프라인에서 정답 자세·바퀴 속도를 이용한다. 학생에게 새 입력을 추가하거나 CNN/GRU를 학습·교체한 것은 아니다.','',
        '## 검증 범위','',
        '- 새 개발 시드 6601/6602 × 회전 20개 × 차량 3종 × 사건 없음/진입/중간/출구 × 교사 4개 = 1,920회.',
        '- 모든 조건에 약한 차량 변동이 있다. `none`은 추가 밀림·출력 사건이 없는 대조군이며 고정 매개변수 정상 조건과 같지 않다.',
        '- 사건이 있는 세 조건은 밀림과 출력 변화가 함께 작용했다. 영상/강한 조명은 사용하지 않았다.',
        '- 교사 함수는 사건 일정·단계·시작 플래그·지속시간·미래 난수를 읽지 않는다. 이미 관측한 평행이동 잔차 >0.25cm/s 또는 각도 잔차 >0.005rad/s에서 예측을 실행한다.',
        '- 선택된 수정은 14명령 동안 실행하며, 완료 후 다시 관측한다. 수정 없음으로 결정했을 때는 다음 관측에서 재평가한다.',
        '- 각 짝의 초기 자세가 일치한다. 무사건 대조군에서 세 후보 모두 복구 발동 0회였고 물리 기록은 기준 교사와 정확히 같았다.',
        '- 예측 계산의 실제 소요 시간은 모델링된 36ms에 더하지 않았다. 이 교사는 하드웨어에 배포할 제어기가 아니다.',
        '- 2개의 공유 난수 시드를 여러 조건에 교차한 결과다. 수백 개의 독립 시드나 최종 시험으로 해석하지 않는다.','',
        '## 연속 차체 기하로 검사한 무이탈','',
        '| 교사 | 무사건 /120 | 진입 /120 | 중간 /120 | 출구 /120 | 교란 있음 합계 /360 |','|---|---:|---:|---:|---:|---:|']
    for name,s in summaries.items():lines.append(f"| {name} | {s['none']['clean']} | {s['entry']['clean']} | {s['middle']['clean']} | {s['exit']['clean']} | {sum(s[p]['clean'] for p in ['entry','middle','exit'])} |")
    lines+=['','- hold035: 관측 잔차가 0.35초 더 지속된다고 예측하고, 예상 안전 조작 중 수정이 작은 후보를 선택한다.',
        '- robust: 0~0.7초의 8개 잔차 종료 가정에서 차체 여유를 비교한다. 안전한 후보가 없으면 예측 침범 비용을 줄이는 후보를 실행한다.',
        '- guarded: robust에서 어느 가정에든 위험하거나 멈출 것으로 예측된 수정은 실행하지 않고 기준 교사 조작을 유지한다. 이 규칙은 이전 6501–6504 분기 진단의 회귀 8건을 분석해 고정했다. 6601/6602의 결과가 완성되기 전에 고정했고 해당 결과는 보지 않았다.',
        '- 모든 1,920회는 지정 출구에 안정적으로 도달했다. 무이탈 실패는 도달 전 경계 침범이며, 이 표에서 출구 도달만을 성공으로 세지 않았다.','',
        '## 안전 회귀와 조작 품질','',
        '| 후보 | 새 무이탈 성공 | 잃은 기존 무이탈 | 교란 중 정지/제자리회전성 저속 구간 사례 |','|---|---:|---:|---:|']
    for name in paired:
        gain=sum(paired[name][p]['gains'] for p in ['entry','middle','exit']);loss=sum(paired[name][p]['losses'] for p in ['entry','middle','exit'])
        lines.append(f"| {name} | {gain} | {loss} | {sum(summaries[name][p]['pauses'] for p in ['entry','middle','exit'])} |")
    lines+=['','hold035는 이 표본에서 기존 성공을 잃지 않고 47건을 회복했다. 하지만 이 사실이 모든 상태의 회귀 방지나 기존 A5b+v4 대비 시간·부드러움 충족을 증명하지는 않는다.','',
        '양쪽 모두 무이탈인 사례의 시간·각가속도 차이는 `comparison.json`에 중앙값, 95백분위수, 실제 개입 사례만의 중앙값으로 분리했다. 중앙값 0만으로 개선 또는 비악화를 주장하지 않는다.','',
        '`clean_moving`은 에피소드 어디에서도 −0.01cm/s 미만의 속도가 없다는 추가 진단이다. 무사건 기준 교사 8건도 시작 후 0.305~0.521초 사이 0.006~0.028cm의 작은 뒤 이동 때문에 이 값이 false였다. 이 추가 지표를 사용자 목표의 회전 구간 부드러움 기준과 동일시하지 않는다. 기존 정지/회전성 저속 구간은 1초 이후 0.5cm/s 미만이 0.3초 이상인 경우로 계속 별도 집계했다.','',
        '## 이동별 교란 무이탈 (각18회)','',
        '| 이동 | 기준 | hold035 | robust | guarded |','|---|---:|---:|---:|---:|']
    for r in movements:lines.append(f"| {'_'.join(r['key'])} | {r['baseline']['clean']} | {r['hold035']['clean']} | {r['robust']['clean']} | {r['guarded']['clean']} |")
    lines+=['','## 다음 판단','',
        'hold035를 다음 독립 개발 검증의 고정 후보로 선택한다. 아직 교사 레지스트리나 배포 모델에는 채택하지 않는다. 28개 이동의 고정 시각 교란·정상 조건과 반복 사건을 새 시드에서 비교한 뒤 학습 데이터를 다시 수집할지 판단한다. 학생 학습에서는 영상과 강한 조명까지 포함해야 한다.',
        '전체 미션 준비도 진행했다. `../goal_mission_contract_20261113/route_draft.json`은 기존 JunctionPolicy로 검산한 32개 정적 왕복 경로와 필수28개 이동을 덮는 8개 경로 초안을 담는다. 나머지24개 방향/목표 조합은 방향 표지 종류별 한 번·조기 출발점 복귀 금지 등의 초안 제한에서 찾지 못했으며 물리적 불가능을 뜻하지 않는다. 새 채점 함수는 경로·출구 순서가 틀렸는데 무이탈로 복귀한 경우를 성공 처리하지 않는다. 정적 검사이며 실제 전체 경로 주행이나 최종 명세 고정은 아직 아니다.',
        '목표는 미완료다. 전체 경로 99%/95%/harsh80%, 이동별 약한 교란90%, 레거시 대비 부드러움·시간, 세 학습 초기화, 충분한 독립 최종 표본, 영상 기반 전체 미션 통합은 여전히 필요하다. 최종 시드 900000–999999, 학생 입력 계약과 완료 기준은 변경하지 않았다.','']
    (REPORT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('causal comparison report generated',flush=True)


if __name__=='__main__':main()
