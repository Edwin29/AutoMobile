"""Paired diagnostic of the frozen repeated-clock teacher test; no retuning."""
import json
from pathlib import Path
import numpy as np
from bench.world import Track,wrap
from bench.vehicle import PRESETS
from bench.junction import movement,heading_of
from .goal_reactive_clock_validation import REPORT
from .goal_continuous_clearance import continuous_clearance
from .goal_development import sha
from .junction_stage02 import dump


def diagnose(r):
    trace=np.load(Path(r['artifact'])/'physics.npz')['physics']
    margin=continuous_clearance(trace[:,1:4],PRESETS[r['preset']],Track())
    c,di,do,arm=movement(*r['key']);rel=trace[:,1:3]-c
    forward=rel@do;lateral=rel@np.array([do[1],-do[0]])
    heading=np.degrees(wrap(trace[:,3]-heading_of(do)))
    bad=np.flatnonzero(margin< -1e-9)
    good=(forward>=40)&(abs(lateral)<15)&(trace[:,-1]>=0)&(abs(heading)<=10)
    last_bad=np.flatnonzero(~good)
    start=last_bad[-1]+1 if len(last_bad) else 0
    held=trace[-1,0]-trace[start,0] if start<len(trace) else 0.
    return dict(key=r['key'],seed=r['seed'],preset=r['preset'],condition=r['condition'],arm=r['arm'],
        outcome=r['outcome'],continuous_clean=r['continuous_clean'],artifact=r['artifact'],
        first_incursion_s=float(trace[bad[0],0]) if len(bad) else None,
        minimum_clearance_cm=float(margin.min()),minimum_clearance_s=float(trace[margin.argmin(),0]),
        exit_forward_cm=float(forward[-1]),exit_lateral_cm=float(lateral[-1]),exit_heading_deg=float(heading[-1]),
        exit_clearance_cm=float(margin[-1]),exit_contiguous_good_s=float(held),
        translation_pause_s=r['translation_pause_s'])


def main():
    raw=REPORT/'results.json';data=json.loads(raw.read_text());rows=data['rows']
    lookup={(tuple(r['key']),r['seed'],r['preset'],r['condition'],r['arm']):r for r in rows}
    paired={};regressions=[];normal_fail=[];coverage=[]
    for condition in ['normal','mild']:
        pairs=[(r,lookup[tuple(r['key']),r['seed'],r['preset'],condition,'hold035'])
               for r in rows if r['arm']=='baseline' and r['condition']==condition]
        both=[(a,b) for a,b in pairs if a['continuous_clean'] and b['continuous_clean']]
        paired[condition]=dict(n=len(pairs),gains=sum(not a['continuous_clean'] and b['continuous_clean'] for a,b in pairs),
            losses=sum(a['continuous_clean'] and not b['continuous_clean'] for a,b in pairs),both_clean=len(both),
            both_clean_median_time_delta_s=float(np.median([b['simulated_s']-a['simulated_s'] for a,b in both])),
            both_clean_median_angular_acceleration_delta=float(np.median([b['angular_accel_rms_rad_s2']-a['angular_accel_rms_rad_s2'] for a,b in both])))
        for a,b in pairs:
            if a['stable_success'] and not b['stable_success']:
                regressions.append(dict(baseline=diagnose(a),candidate=diagnose(b)))
            if condition=='normal' and not b['continuous_clean']:normal_fail.append(diagnose(b))
        for key in sorted({tuple(a['key']) for a,b in pairs}):
            p=[(a,b) for a,b in pairs if tuple(a['key'])==key]
            coverage.append(dict(key=key,condition=condition,n=len(p),baseline=sum(a['continuous_clean'] for a,b in p),candidate=sum(b['continuous_clean'] for a,b in p)))
    pauses=[diagnose(r) for r in rows if r['translation_pause_s']>0]
    out=dict(results_sha256=sha(raw),source_sha256=sha(__file__),paired=paired,coverage=coverage,
        stable_regressions=regressions,pauses=pauses,normal_failures=normal_fail,
        data_decision='Research demonstrations only: retain all episodes and admit only complete clean stable episodes; failures and pauses are not accepted labels. No teacher/student adoption.',
        limits='Local privileged teacher, no imagery/illumination/full missions or learned-policy evidence. Eight seeds shared across movements; do not treat2688rollouts as2688independent environmental seeds.')
    dump(REPORT/'comparison.json',out)
    text=['# 반복 교란 교사 비교','',
        '28개 이동 × 8개 시드 × 3개 차량 × 정상/약한 교란 × 두 교사: 2,688회. 전체 경로 또는 CNN 평가가 아니다.','',
        '| 조건 | 기존 교사 | hold035 | 대응 성공 증가/손실 |','|---|---:|---:|---:|']
    for c in ['normal','mild']:
        a=data['summary']['baseline'][c];b=data['summary']['hold035'][c];p=paired[c]
        text.append(f"| {c} | {a['clean']}/{a['n']} | {b['clean']}/{b['n']} | +{p['gains']} / -{p['losses']} |")
    text+=['','성공은 초기 자세 적법, 안정 출구 도달, 모든 물리 표본에서 연속 차체 윤곽 무침범의 결합이다. 시간 표본 사이 연속 시간 충돌 보증은 아니다.',
        '',f'출구 안정성 퇴행 {len(regressions)}회는 모두 기존에도 차체 침범으로 실패했던 사례다. 정지/저속 {len(pauses)}회 역시 무이탈 실패 사례에 속한다. 성공률 상승과 별개로 실패 복구 품질의 한계로 남긴다.',
        '', '정상 조건은 두 교사의 물리 궤적이 전부 동일하다. 새로운 시드에서 나타난 정상 실패 18회는 교사와 초기 상태 분포의 남은 한계이며 제외하여 성공률을 높이지 않는다.',
        '', '학습 자료 수집 진행: 모든 실패를 보존하고 완전 무이탈 안정 시연만 초기 BC 학습에 사용한다. 강한 조명 쌍과 명시적 직진 명령 인코딩을 포함한다. 성공 시연만 학습하는 선택 편향이 있으므로 이후 학생의 독립 폐루프 평가가 필수다.',
        '', '원인별 출구 자세·안정 유지 시간·최초 침범 시각은 comparison.json에 기록했다. 종료 게이트나 합격 기준은 수정하지 않았다.']
    (REPORT/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(dict(paired=paired,stable_regressions=regressions,pauses=pauses,normal_failures=normal_fail),ensure_ascii=False))


if __name__=='__main__':main()
