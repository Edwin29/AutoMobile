"""Derived report only; preserves every frozen experiment and raw rollout."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .goal_clearance_paths import REPORT as SEARCH_REPORT
from .goal_clearance_validation import REPORT
from .goal_phase_disturbance import REPORT as PHASE_REPORT
from .goal_teacher_coverage import CoverageTeacher,ROOT
from .goal_continuous_clearance import continuous_clearance
from bench.vehicle import PRESETS
from bench.world import Track
from .junction_stage02 import dump
from .goal_development import sha


def initial_summary():
    runs={a:json.loads((SEARCH_REPORT/(a+'.json')).read_text()) for a in ['same_mode','planned']}
    runs['baseline']=[r['baseline'] for r in runs['same_mode']]
    out={}
    for arm,rows in runs.items():
        out[arm]={}
        for c in ['normal','mild']:
            a=[r for r in rows if r['condition']==c];turns=[r for r in a if r['key'][2]!='straight']
            out[arm][c]=dict(n=len(a),clean=sum(r['clean_stable'] for r in a),stable=sum(r['stable_success'] for r in a),
                turn_n=len(turns),turn_clean=sum(r['clean_stable'] for r in turns),
                turn_median_time_s=float(np.median([r['simulated_s'] for r in turns])),
                turn_median_angular_accel=float(np.median([r['angular_accel_rms_rad_s2'] for r in turns])),
                pauses=sum(r['translation_pause_s']>0 for r in a))
    dump(SEARCH_REPORT/'summary.json',dict(summary=out,source_sha256=sha(__file__)))
    return out


def uncertainty():
    runs={a:json.loads((REPORT/(a+'.json')).read_text()) for a in ['baseline','reserve']}
    seeds=sorted(set(r['seed'] for r in runs['baseline']));rng=np.random.default_rng(735219)
    indices=rng.integers(0,len(seeds),(20000,len(seeds)));out={}
    for condition in ['normal','mild']:
        differences=[]
        for seed in seeds:
            rates=[]
            for arm in ['baseline','reserve']:
                rows=[r for r in runs[arm] if r['condition']==condition and r['seed']==seed]
                rates.append(np.mean([r['clean_stable'] for r in rows]))
            differences.append(rates[1]-rates[0])
        boot=np.array(differences)[indices].mean(axis=1)
        out[condition]=dict(seed_rates=differences,paired_gain=float(np.mean(differences)),
            percentile_95_interval=np.quantile(boot,[.025,.975]))
    dump(REPORT/'seed_cluster_uncertainty.json',dict(seeds=seeds,resamples=20000,rng_seed=735219,
        unit='seed block containing all28 movements and3 vehicles; paired arms resampled together',results=out,
        caveat='Only eight development seed blocks. Exploratory percentile bootstrap, not final certification.'))
    return out


def plot():
    geometry={tuple(r['key']):r for r in json.loads((SEARCH_REPORT/'geometry.json').read_text())}
    track=Track();keys=[('north','S','left'),('east','W','left'),('center','E','right')]
    fig,axes=plt.subplots(2,3,figsize=(13,7.8),layout='constrained')
    for j,key in enumerate(keys):
        a=CoverageTeacher(key,'normal',6201,'field_estimate','centered');z=np.load(geometry[key]['reference'])
        p=np.r_[a.points,z['points']];lo=p.min(0)-16;hi=p.max(0)+16
        xx,yy=np.meshgrid(np.linspace(lo[0],hi[0],220),np.linspace(lo[1],hi[1],220))
        road=track.clearance(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
        ax=axes[0,j];ax.contourf(xx,yy,road>=0,levels=[-.5,.5,1.5],colors=['#d5dbe3','#fbfcfd'])
        ax.plot(a.points[:,0],a.points[:,1],label='Previous smooth path',color='#d97706')
        ax.plot(z['points'][:,0],z['points'][:,1],label='More clearance',color='#2563eb')
        ax.set_aspect('equal');ax.set_title('_'.join(key));ax.set_xlabel('x (cm)');ax.set_ylabel('y (cm)')
        ax=axes[1,j]
        for points,h,s,label,color in [(a.points,a.h,a.s,'Previous smooth path','#d97706'),(z['points'],z['heading'],z['s'],'More clearance','#2563eb')]:
            m=continuous_clearance(np.c_[points,h],PRESETS['field_estimate'],track)
            ax.plot(s/s[-1],m,label=label,color=color)
        ax.axhline(0,color='black',lw=.6);ax.set_xlabel('Normalized path distance');ax.set_ylabel('Body clearance (cm)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8)
    fig.suptitle('Static body clearance improves; closed-loop safety still needs testing',fontsize=14)
    fig.savefig(REPORT/'reference_clearance.png',dpi=155);plt.close(fig)


def main():
    initial=initial_summary();uncertainty();plot()
    v=json.loads((REPORT/'summary.json').read_text());phase=json.loads((PHASE_REPORT/'summary.json').read_text()) if (PHASE_REPORT/'summary.json').exists() else None
    lines=['# 차체 여유를 확보한 코너링 경로 — 개발 검증','',
        '교사 경로의 정적 차체 여유를 늘리면 교란 주행이 개선되는지 검사했다. 학생 CNN/GRU는 이번 실험에서 학습하거나 교체하지 않았다. 완료 기준·학생 입력 계약·최종 시험 시드는 그대로다.','',
        '## 고정한 비교','',
        '- 20개 회전의 세 구간 5차 Bezier 경로를 탐색했다. 이전 clearance-first 경로가 달성한 차체 여유의 90% 이상을 제약으로 두고 곡률·곡률 변화·길이를 최소화했다. 20개 모두 정적 검사를 통과했다.',
        '- 이는 주행 중 안전 보장이 아니다. 차체 전체의 연속 공간 기하를 경로 샘플과 물리 시간 샘플에서 검사했다. 연속 시간 충돌 증명은 아니다.',
        '- 교사 이득과 차량·교란·초기 자세는 동일하다. 기존 이동별 속도 방식 유지(same_mode)와 모든 회전 속도 계획(planned)을 비교했고, 후자는 개발 결과가 나빠 새 시드 검증 후보에서 제외했다.',
        '- 기존 교사는 이전 centered smooth teacher다. 실제 A5b+v4 레거시 전체 경로 운전기와의 비교가 아니다.','',
        '## 선택용 시드 6201/6202','',
        '| 교사 후보 | 정상 무이탈 /168 | 약한 교란 무이탈 /168 | 약한 교란 회전만 /120 |','|---|---:|---:|---:|']
    for arm,name in [('baseline','기존'),('same_mode','여유 경로·속도 방식 유지'),('planned','여유 경로·전체 속도 계획')]:
        s=initial[arm];lines.append(f"| {name} | {s['normal']['clean']} | {s['mild']['clean']} | {s['mild']['turn_clean']} |")
    lines+=['','새 후보 672개 실행 모두 연속 차체 기하 감사에서 거짓 무이탈 판정 0건.','',
        '## 후보 고정 후 새 시드 6401–6408','',
        '28개 이동 × 차량 3종 × 시드 8개 × 정상/약한 교란 × 교사 2개 = 2,688회. 동일 시드가 여러 이동·차량에 공유되므로 2,688개 독립 난수 표본으로 해석하지 않는다.','',
        '| 조건 | 기존 무이탈 | 여유 경로 무이탈 | 기존 회전만 | 여유 경로 회전만 |','|---|---:|---:|---:|---:|']
    for c,name in [('normal','정상'),('mild','약한 복합 물리 교란')]:
        a=v['summary']['baseline'][c];b=v['summary']['reserve'][c]
        lines.append(f"| {name} | {a['all']['clean']}/672 | {b['all']['clean']}/672 | {a['turns']['clean']}/480 | {b['turns']['clean']}/480 |")
    lines+=['','직선 8개 이동의 물리 기록은 두 교사가 완전히 동일했다. 모든 쌍의 초기 자세와 초기 합법화 기록이 일치했다. 각 교사·조건의 36/672 초기 자세는 기존 규칙으로 도로 안에 보정됐으며, 제외하지 않았다. 모든 새 시드 실행에서 연속 차체 기하의 거짓 무이탈 판정은 없었다.','',
        '양쪽 모두 무이탈인 회전만 비교한 중앙값 차이(후보−기존):','',
        '| 조건 | 표본 수 | 시간 차이(s) | 각가속도 RMS 차이(rad/s²) |','|---|---:|---:|---:|']
    for c in ['normal','mild']:
        p=v['paired'][c];lines.append(f"| {c} | {p['both_clean_turn_n']} | {p['median_both_clean_turn_time_change_s']:+.3f} | {p['median_both_clean_turn_angular_accel_change']:+.4f} |")
    lines+=['','불필요한 정지로 분류된 사례는 없었지만 시간과 각가속도는 악화됐다. 단순히 경로가 연속 곡선이라는 이유로 더 부드러운 실제 주행이라고 결론 내릴 수 없다. 이 비교만으로 실제 레거시 대비 목표 통과/실패를 판정할 수도 없다.','']
    if phase:
        lines+=['## 회전 단계에 교란을 배치한 진단','',
            '시드 6501–6504, 회전 20개, 차량 3종. 약한 차량 변동은 항상 켜고, 밀림과 출력 변화 한 쌍을 진입(진입 축 −15cm), 중간(45도 회전), 출구(출구 축 +15cm)에 각각 발생시켰다. 무사건 대조군도 별도 실행했다. 총 1,920회이며 각 단계는 별도 주행이다. 강한 조명과 학생 인지 능력의 검증은 아니다.','',
            '24개 구현 검사에서 기존 고정 사건 재생과 물리/명령이 수치 정밀도 내에서 일치했다. 허용 오차 1e-8/1e-10, 관측 최대 오차 8.9e-13/1.9e-14. 초기 bitwise 일치 검사는 재분할 뺄셈의 반올림 오차로 실패하여 이 수치 동등성 검사를 사용했다.','',
            '| 사건 단계 | 기존 연속 차체 무이탈 /240 | 후보 연속 차체 무이탈 /240 | 기존 사건 노출 | 후보 사건 노출 |','|---|---:|---:|---:|---:|']
        for p in ['none','entry','middle','exit']:
            a=phase['summary']['baseline'][p];b=phase['summary']['reserve'][p]
            lines.append(f"| {p} | {a['continuous_clean']} | {b['continuous_clean']} | {a['exposed']} | {b['exposed']} |")
        lines+=['','출구 사건에서 외곽 1cm 샘플 채점의 거짓 무이탈이 교사별 1건 발견됐다. 기존 north_E_left/6504/field_estimate는 +0.004490cm에서 연속 차체 −0.002362cm, 후보 south_E_right/6503/field_estimate는 +0.003212cm에서 −0.000824cm였다. 원본 채점·원시 기록을 보존하고 위 표에는 더 보수적인 연속 차체 판정을 썼다.',
            '출구 개선은 같은 사건 단계에서도 유지되므로 고정 시각을 피해 간 효과만으로 설명할 수 없다. 반면 진입과 회전 중간에서는 일관된 개선이 없다. 단계별 사건은 별도 주행이므로 전체 임무 성공률로 합산하지 않는다.','']
        audit_path=PHASE_REPORT/'entry_response_audit.json'
        if audit_path.exists():
            audit=json.loads(audit_path.read_text())
            lines+=['### 진입 실패의 반응 시간 감사','',
                '현재 자세를 즉시 읽는 낙관적인 관측을 기준으로, 밀림 후 첫 관측에서 보낸 명령이 실제 바퀴에 영향을 줄 수 있는 가장 이른 시각을 차량 지연 큐에서 측정했다. 학생에게 이 정답이나 사건 정보가 제공되는 것은 아니다.','']
            for a in ['baseline','reserve']:
                s=audit['summary'][a];lines.append(f"- {a}: 진입 실패 {s['failed_entry_cases']}건을 물리 기록과 정확히 일치하게 재생했고, {s['before_possible_response']}건은 새 관측에 대한 어떤 명령도 아직 바퀴에 영향을 줄 수 없는 시점에 이탈했다.")
            lines+=['','이는 해당 진입 상태에서 반응만으로 막을 수 없었다는 조건부 결과다. 다른 진입 자세·속도·사전 제어까지 불가능하다는 증명은 아니다. 늦게 이탈한 나머지 사례가 반드시 복구 가능하다는 의미도 아니다. 완료 기준이나 교란 세기를 낮추지 않았다.','']
    lines+=['','## 이동별 새 시드 약한 교란 결과','',
        '| 이동 | 기존 /24 | 후보 /24 |','|---|---:|---:|']
    for r in v['movements']:lines.append(f"| {'_'.join(r['key'])} | {r['baseline']['mild']['clean']} | {r['reserve']['mild']['clean']} |")
    lines+=['','## 판단과 한계','',
        '- 차체 여유를 늘린 경로가 선택 시드와 새 시드 모두에서 이탈을 줄였다. 하지만 이동별 실패와 더 느리고 급한 회전이 남았다. 새 교사나 학생 모델을 자동 채택하지 않았다.',
        '- 4개 이동 학생 실험을 28개 이동 교사 실험으로 대체해 완료로 인정하지 않는다. 영상·조명, GRU 정책, 세 학습 초기화, 연결 구간과 전체 임무, 사전 정의 harsh, 최종 시험은 미완료다.',
        '- 정상/약한 전체 임무 99%/95%, harsh 80%, 이동별 약한 90%, 부드러움·시간 목표를 유지한다. 최종 시드 900000–999999는 사용하지 않았다.',
        '- 보고서의 평균 및 bootstrap은 개발 진단이다. 시뮬레이터 범위를 넘어 실제 차량 안전성을 입증하지 않는다.','',
        '그림: `reference_clearance.png`; 실행별 자료: `baseline.json`, `reserve.json`; 단계별 자료는 `../goal_phase_disturbance_20261113/summary.json`.','']
    (REPORT/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('report generated',REPORT,flush=True)


if __name__=='__main__':main()
