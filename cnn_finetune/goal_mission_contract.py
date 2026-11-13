"""Static full-route coverage draft and a fail-closed episode scoring gate.

Names and map topology below are for offline route specification/scoring, not
actor inputs. This does not integrate a policy or prove any driving outcome.
"""
from collections import deque
import json
import math
from pathlib import Path
from bench.world import Track
from drive_v4.junction_policy import JunctionPolicy,exit_arm
from .goal_teacher_coverage import KEYS,ROOT,INVENTORY
from .goal_development import sha
from .junction_stage02 import dump

REPORT=ROOT/'cnn_finetune/results/goal_mission_contract_20261113'
LINKS={
    ('center','N'):('north','S'),('north','S'):('center','N'),
    ('center','E'):('east','W'),('east','W'):('center','E'),
    ('center','S'):('south','N'),('south','N'):('center','S'),
    ('center','W'):('west','E'),('west','E'):('center','W'),
    ('north','E'):('east','N'),('east','N'):('north','E'),
    ('east','S'):('south','E'),('south','E'):('east','S'),
    ('south','W'):('west','S'),('west','S'):('south','W'),
    ('west','N'):('north','W'),('north','W'):('west','N')}
OUTER_CW={('north','E'),('east','S'),('south','W'),('west','N')}
OUTER_CCW={('north','W'),('west','S'),('south','E'),('east','N')}


def route_for(target,direction):
    track=Track();policy=JunctionPolicy(track.data['junctions'],direction)
    start=('south','E') if direction=='cw' else ('east','S')
    finish=('east','S') if direction=='cw' else ('south','E')
    queue=deque([(start[0],start[1],direction,False,0,[])])
    visited=set()
    while queue:
        j,arm,lap,seen,used,path=queue.popleft();state=(j,arm,lap,seen,used)
        if state in visited:continue
        visited.add(state);policy.lap_direction=lap
        options=['right'] if j=='center' else policy.options(j,arm)
        default,_=policy.default(j,arm)
        for move in options:
            command=None if move==default else move
            bit=0 if command is None else {'straight':1,'left':2,'right':4}[command]
            if used & bit:continue
            ex=exit_arm(arm,move);found=seen or (j,arm,move)==tuple(target)
            step=dict(junction=j,came_from=arm,move=move,exit_arm=ex,command=command,lap_before=lap)
            new_path=path+[step]
            if (j,ex)==finish:
                if found:return dict(start_direction=direction,target=target,steps=new_path,
                    expected_passages=[[s['junction'],s['came_from'],s['exit_arm']] for s in new_path])
                continue
            nxt=LINKS[(j,ex)];new_lap='cw' if (j,ex) in OUTER_CW else 'ccw' if (j,ex) in OUTER_CCW else lap
            queue.append((*nxt,new_lap,found,used|bit,new_path))
    return None


def route_and_clearance_gate(summary,expected_passages,continuous_min_margin_cm):
    """One episode's route/clearance gate only, never full goal acceptance."""
    if not expected_passages:raise ValueError('Expected full route is required; empty route cannot pass')
    try:got=[(r['junction'],r['came_from'],r['exit_arm']) for r in summary['passages']]
    except (KeyError,TypeError):got=None
    expected=[tuple(x) for x in expected_passages]
    route_correct=got==expected
    finite=continuous_min_margin_cm is not None and math.isfinite(continuous_min_margin_cm)
    clean=bool(finite and continuous_min_margin_cm>=0)
    complete=summary.get('outcome')=='mission_complete'
    return dict(route_correct=route_correct,continuous_body_clean=clean,terminal_complete=complete,
                clean_route_success=bool(route_correct and clean and complete),
                scope='episode route and body-clearance only; actor contract, timing, interventions, aggregate confidence and all goal gates require separate evidence')


def verify_route(route):
    direction=route['start_direction'];track=Track();policy=JunctionPolicy(track.data['junctions'],direction)
    expected_start=('south','E') if direction=='cw' else ('east','S')
    finish=('east','S') if direction=='cw' else ('south','E');state=expected_start;commands=[]
    for i,step in enumerate(route['steps']):
        assert (step['junction'],step['came_from'])==state
        if step['command'] is not None:policy.push(float(i),step['command']);commands.append(step['command'])
        decision=policy.decide(float(i),*state)
        assert (decision.move,decision.exit_arm)==(step['move'],step['exit_arm'])
        edge=(step['junction'],step['exit_arm'])
        if i<len(route['steps'])-1:assert edge!=finish
        else:assert edge==finish
        if edge in OUTER_CW:policy.lap_direction='cw'
        elif edge in OUTER_CCW:policy.lap_direction='ccw'
        state=LINKS[edge]
    assert len(commands)==len(set(commands))
    assert tuple(route['target']) in {(s['junction'],s['came_from'],s['move']) for s in route['steps']}
    assert not policy.queue


def checks(routes):
    for (j,arm),(k,incoming) in LINKS.items():assert LINKS[(k,incoming)]==(j,arm)
    for r in routes:verify_route(r)
    want=[['south','E','W'],['west','S','N'],['north','W','E'],['east','N','S']]
    good=dict(outcome='mission_complete',passages=[dict(junction=j,came_from=a,exit_arm=b) for j,a,b in want],clean=True)
    assert route_and_clearance_gate(good,want,.1)['clean_route_success']
    wrong=json.loads(json.dumps(good));wrong['passages'][0]['exit_arm']='N'
    assert not route_and_clearance_gate(wrong,want,.1)['clean_route_success']
    for observed in [good['passages'][:-1],good['passages']+good['passages'][:1]]:
        assert not route_and_clearance_gate(dict(good,passages=observed),want,.1)['clean_route_success']
    for margin in [-.0001,float('nan'),None]:assert not route_and_clearance_gate(good,want,margin)['clean_route_success']
    assert not route_and_clearance_gate(dict(good,outcome='time_limit'),want,.1)['clean_route_success']
    assert not route_and_clearance_gate(dict(outcome='mission_complete'),want,.1)['clean_route_success']
    try:route_and_clearance_gate(good,[],.1)
    except ValueError:pass
    else:raise AssertionError('empty route accepted')
    return dict(reciprocal_road_links=16,policy_verified_routes=len(routes),scoring_gate_cases=10)


def main():
    routes=[];unreachable=[]
    for direction in ['cw','ccw']:
        for key in KEYS:
            r=route_for(key,direction)
            if r is None:unreachable.append(dict(direction=direction,target=key))
            else:routes.append(r)
    checked=checks(routes)
    covered={(s['junction'],s['came_from'],s['move']) for r in routes for s in r['steps']}
    assert set(KEYS)<=covered
    # Greedy compact suite is only a draft sampling scaffold, not a final test.
    remaining=set(KEYS);suite=[];available=list(enumerate(routes))
    while remaining:
        i,r=max(available,key=lambda ir:(len(remaining & {(s['junction'],s['came_from'],s['move']) for s in ir[1]['steps']}),-len(ir[1]['steps']),-ir[0]))
        coverage={(s['junction'],s['came_from'],s['move']) for s in r['steps']}
        assert remaining & coverage;suite.append(dict(route_index=i,route=r));remaining-=coverage;available=[ir for ir in available if ir[0]!=i]
    REPORT.mkdir(exist_ok=False)
    dump(REPORT/'route_draft.json',dict(source_sha256=sha(__file__),policy_sha256=sha(ROOT/'drive_v4/junction_policy.py'),
        inventory_sha256=sha(INVENTORY),routes=routes,unreachable_under_one_each_sign_rule=unreachable,compact_suite=suite,
        required_movements=KEYS,checks=checked,final_manifest_frozen=False,final_exam_used=False,
        unresolved=['No runtime driving verification','No student full-mission adapter or vision-based junction arrival/command consumption integration',
            'No final time budget, initial-pose distribution, disturbance schedule, harsh feasibility envelope or sample allocation frozen',
            'Existing bench.mission return-after300cm termination must not terminate a declared route early or force an extra unintended lap',
            'Scripted direction inputs are not validation of real sign recognition']))
    text='''# 전체 경로 검증 계약 초안

이 자료는 정적 경로 계획·채점 검사다. 주행 성공, 최종 시험 명세 고정, 모델 통합을 의미하지 않는다.

현재 `bench/mission.py`는 `clean`과 `route_correct`를 별도 필드로 반환한다. 무이탈로 출발 구역에 돌아왔더라도 잘못된 순서·출구를 택한 기록은 이번 목표의 성공으로 인정하면 안 된다. 또한 기존 기대 경로 함수는 CW만 구현되어 있다.

새 `route_and_clearance_gate`는 선언된 전체 통과 순서, 임무 종료, 차체 전체의 무이탈을 모두 요구한다. 빈 기대 경로·누락된 기록·NaN 여유는 통과하지 않는다. 이것만으로 전체 목표가 완료되는 것은 아니며 정지/회전 품질, 시간, 학생 입력 계약, 최종 표본 및 학습 초기화 등은 별도로 검증해야 한다.

CW와 CCW의 도로 연결을 모두 정의하고, 기존 JunctionPolicy로 각 경로를 검산했다. 중앙 사거리는 항상 우회전하며, 외곽 진입 기본 직진·내부 합류 기본 방향·합류 후 방향 갱신을 유지한다. 경로당 직진/좌/우 방향 명령을 각각 최대 한 번 사용한다. 정적 재생의 1초 간격은 논리 검사일 뿐 실제 명령 만료 시간을 입증하지 않는다.

경로의 교차로 이름과 지도 정보는 시험 명세·오프라인 채점용이다. 학생 입력에 넣지 않는다. 실제 교차로 도착 판단과 명령 소비를 영상으로 처리하는 통합은 아직 필요하다.

`route_draft.json`의 compact_suite는 28개 필수 이동을 포함하는 표본 배치 초안이다. 실제 연결 주행, 출발 구역 종료 규칙, 강한 조명과 복합 교란, harsh 복구 가능 범위, 제한 시간, 이동별 충분한 표본 배치는 아직 고정하지 않았다. 최종 시드 900000–999999는 사용하지 않았다.
'''
    (REPORT/'README.md').write_text(text,encoding='utf-8');print(json.dumps(dict(routes=len(routes),unreachable=len(unreachable),compact_routes=len(suite),covered=len(covered),checks=checked)),flush=True)


if __name__=='__main__':main()
