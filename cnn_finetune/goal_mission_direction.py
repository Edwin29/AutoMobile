"""Direction resolver for a continuous visual actor, without pivot control.

The caller supplies its IMAGE-derived map estimate, never simulator truth.
This component consumes queued sign commands through the existing course policy.
Only the resolved straight/left/right instruction belongs in actor inputs.
It neither steers nor makes map errors disappear; runtime integration is pending.
"""
import json
import math
import importlib.util
import sys
import numpy as np
from bench.world import Track,wrap
from bench.junction import heading_of
from .goal_development import ROOT,sha
from .junction_stage02 import dump

CONTRACT_REPORT=ROOT/'cnn_finetune/results/goal_mission_contract_20261113'
REPORT=ROOT/'cnn_finetune/results/goal_mission_direction_v2_20261113'

# Import the current pure policy without binding the drive_v4 package. The
# real image-map observer loads a pinned drive_v4 tree in its own process;
# importing this resolver must not silently redirect that import to the repo.
_spec=importlib.util.spec_from_file_location('goal_direction_policy',ROOT/'drive_v4/junction_policy.py')
_policy=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_policy;_spec.loader.exec_module(_policy)
JunctionPolicy,ARMS=_policy.JunctionPolicy,_policy.ARMS


class DirectionResolver:
    ENTER_CM=34.
    LEAVE_CM=40.
    REARM_CM=55.
    MAX_SXY=2.
    ARRIVE_LAT_CM=9.
    ARRIVE_HEAD_DEG=35.

    def __init__(self,junctions,direction):
        if direction not in ('cw','ccw'):raise ValueError(direction)
        self.junctions=junctions;self.policy=JunctionPolicy(junctions,direction)
        self.visit=None;self.active=False;self.decision=None
        self.instruction='straight';self.log=[];self.last_t=-math.inf

    def push_direction(self,t,move):
        self.policy.push(t,move)

    def _arrival(self,pose):
        candidates=[]
        for j,jd in self.junctions.items():
            rel=pose[:2]-np.asarray(jd['xy'],float)
            for arm in jd['arms']:
                a=np.asarray(ARMS[arm]);along=float(rel@a)
                lateral=abs(float(rel@np.array([a[1],-a[0]])))
                dh=abs(float(wrap(pose[2]-heading_of(-a))))
                if 0<along<=self.ENTER_CM and lateral<self.ARRIVE_LAT_CM and dh<math.radians(self.ARRIVE_HEAD_DEG):
                    candidates.append((along,j,arm))
        if not candidates:return None
        _,j,arm=min(candidates);return j,arm

    def _complete(self,t):
        d=self.decision;j=d.junction;ex=d.exit_arm
        if j!='center' and self.policy.kind(j,ex)=='outer_T':
            x,y=self.junctions[j]['xy'];a=np.asarray(ARMS[ex])
            cw=np.array([1 if y>0 else -1,0]) if abs(y)>=abs(x) else np.array([0,-1 if x>0 else 1])
            self.policy.lap_direction='cw' if cw@a>0 else 'ccw'
        self.active=False;self.instruction='straight'
        self.log.append(dict(t=t,event='estimated_exit',junction=j,exit_arm=ex,lap=self.policy.lap_direction))

    def observe(self,t,estimated_pose,sxy):
        t=float(t)
        if not math.isfinite(t) or t<self.last_t:raise ValueError('Observation times must be finite and monotonic')
        self.last_t=t
        pose=np.asarray(estimated_pose,float)
        if pose.shape!=(3,) or not np.isfinite(pose).all() or not math.isfinite(sxy) or sxy>self.MAX_SXY or sxy<0:
            return self.instruction
        if self.active:
            d=self.decision;rel=pose[:2]-np.asarray(self.junctions[d.junction]['xy'])
            out=np.asarray(ARMS[d.exit_arm]);lat=abs(float(rel@np.array([out[1],-out[0]])))
            head=abs(float(wrap(pose[2]-heading_of(out))))
            if rel@out>=self.LEAVE_CM and lat<self.ARRIVE_LAT_CM and head<math.radians(self.ARRIVE_HEAD_DEG):self._complete(t)
        if self.visit is not None and not self.active:
            centre=np.asarray(self.junctions[self.visit[0]]['xy'])
            if np.linalg.norm(pose[:2]-centre)>self.REARM_CM:self.visit=None
        if self.visit is None:
            hit=self._arrival(pose)
            if hit is not None:
                self.visit=hit;self.decision=self.policy.decide(t,*hit);self.active=True
                self.instruction=self.decision.move
                self.log.append(dict(t=t,event='estimated_arrival',**self.decision.__dict__))
        return self.instruction


def checks():
    track=Track();routes=json.loads((CONTRACT_REPORT/'route_draft.json').read_text())['routes'];checked=[]
    for ri,route in enumerate(routes):
        r=DirectionResolver(track.data['junctions'],route['start_direction']);t=0.
        for step in route['steps']:
            c=np.asarray(track.data['junctions'][step['junction']]['xy']);a=np.asarray(ARMS[step['came_from']])
            if step['command'] is not None:r.push_direction(t,step['command'])
            pose=np.r_[c+30*a,heading_of(-a)]
            old=len(r.log);r.observe(t+.01,pose,float('nan'));assert len(r.log)==old
            r.observe(t+.02,pose,2.1);assert len(r.log)==old
            move=r.observe(t+.03,pose,.5)
            assert move==step['move'] and r.decision.junction==step['junction'] and r.decision.came_from==step['came_from']
            assert r.decision.exit_arm==step['exit_arm']
            n=len(r.policy.log);r.observe(t+.04,pose,.5);assert len(r.policy.log)==n
            # Wrong-arm motion must not declare the chosen movement complete.
            wrong=np.r_[c+42*a,heading_of(a)];r.observe(t+.05,wrong,.5);assert r.active
            out=np.asarray(ARMS[step['exit_arm']]);exit_pose=np.r_[c+42*out,heading_of(out)]
            assert r.observe(t+.06,exit_pose,.5)=='straight' and not r.active
            # The next supplied estimate is already beyond this visit's rearm
            # radius. No truth lookup or expected route is passed to observe().
            t+=1.
        arrivals=[e for e in r.log if e['event']=='estimated_arrival']
        assert [(e['junction'],e['came_from'],e['exit_arm']) for e in arrivals]==[tuple(x) for x in route['expected_passages']]
        checked.append(dict(route_index=ri,passages=len(arrivals),exact=True))
    # Central rule preserves a queued command, unavailable T instruction is
    # consumed once, and noisy repeated arrival cannot consume a second sign.
    r=DirectionResolver(track.data['junctions'],'cw');r.push_direction(0,'left')
    r.observe(.1,[0,30,math.pi],.5);assert r.instruction=='right' and len(r.policy.queue)==1
    r=DirectionResolver(track.data['junctions'],'cw');r.push_direction(0,'left');r.push_direction(0,'right')
    c=np.asarray(track.data['junctions']['south']['xy']);pose=np.r_[c+[30,0],-math.pi/2]
    r.observe(.1,pose,.5);assert r.instruction=='straight' and len(r.policy.queue)==1
    r.observe(.2,pose,.5);assert len(r.policy.queue)==1
    try:r.observe(.15,pose,.5)
    except ValueError:pass
    else:raise AssertionError('Time reversal accepted')
    return dict(routes=checked,policy_edge_checks=4,finite_pose_and_uncertainty_gates=True,duplicate_visit_guard=True,
        scope='Pure direction state machine supplied synthetic map estimates; not perception, actuation, full-mission driving, or final acceptance')


if __name__=='__main__':
    before={k:v for k,v in sys.modules.items() if k=='drive_v4' or k.startswith('drive_v4.')}
    result=checks()
    assert {k:v for k,v in sys.modules.items() if k=='drive_v4' or k.startswith('drive_v4.')}==before
    REPORT.mkdir(exist_ok=False)
    dump(REPORT/'checks.json',dict(source_sha256=sha(__file__),policy_sha256=sha(ROOT/'drive_v4/junction_policy.py'),
        route_contract_sha256=sha(CONTRACT_REPORT/'route_draft.json'),map_sha256=sha(ROOT/'bench/world/track.json'),
        checks_do_not_mutate_drive_v4_package=True,**result))
    print('direction resolver checks passed',len(result['routes']),'routes',flush=True)
