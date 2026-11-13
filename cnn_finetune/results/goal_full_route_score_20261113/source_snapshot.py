"""Full-route termination/scoring: truth is confined to this offline scorer."""
import json
import math
from pathlib import Path
import numpy as np
from bench.mission import MissionScorer
from bench.world import Track
from bench.vehicle import Vehicle,PRESETS
from .goal_development import ROOT,sha
from .junction_stage02 import dump

REPORT=ROOT/'cnn_finetune/results/goal_full_route_score_20261113'


class DeclaredRouteScorer(MissionScorer):
    def __init__(self,track,rim,direction,pose0,front_cm,expected):
        super().__init__(track,rim,direction,pose0,front_cm)
        self.expected=[tuple(x) for x in expected]
        if not self.expected or any(len(x)!=3 for x in self.expected):raise ValueError('A complete nonempty declared route is required')
        self.departed=False;self.route_violation=None

    def _in_start(self,pose):
        (x0,x1),(y0,y1),yaw=self.zone
        return bool(x0<pose[0]<x1 and y0<pose[1]<y1 and math.cos(pose[2]-yaw)>.85)

    def sample(self,t,dt,pose,pwm,held):
        if self.end is not None:return
        super().sample(t,dt,pose,pwm,held)
        # The legacy travel>300cm condition does not know the declared route.
        # Discard that completion decision, preserve severe departure/stall.
        if self.end=='mission_complete':self.end=None
        if not self._in_start(pose) and self.travel>20:self.departed=True
        got=[(p['junction'],p['came_from'],p['exit_arm']) for p in self.passages]
        if got!=self.expected[:len(got)]:
            self.route_violation=dict(observed=got,expected=self.expected)
            if self.end is None:self.end='wrong_exit'
        elif self.end is None and got==self.expected and self.departed and self._in_start(pose) and self.episode is None:
            self.end='mission_complete'

    def final(self,outcome,continuous_min_margin_cm):
        summary=super().summary(outcome)
        got=[(p['junction'],p['came_from'],p['exit_arm']) for p in self.passages]
        route_correct=got==self.expected
        finite=continuous_min_margin_cm is not None and math.isfinite(continuous_min_margin_cm)
        full_clear=bool(finite and continuous_min_margin_cm>=-1e-9)
        success=bool(outcome=='mission_complete' and self.departed and route_correct and full_clear)
        summary.update(expected_passages=self.expected,route_correct=route_correct,departed_start=self.departed,
            continuous_min_clearance_cm=continuous_min_margin_cm,continuous_clean=success,clean_route_success=success,
            route_violation=self.route_violation,scope='Full declared route and body-clearance gate; aggregate success, smoothness, latency and all remaining goal gates are separate')
        return summary


def checks():
    track=Track();p=PRESETS['field_estimate'];start=track.start_pose('cw',p.front_cm,p.rear_cm)
    rim=Vehicle(p,start).footprint();expected=[('south','E','W'),('west','S','N'),('north','W','E'),('east','N','S')]
    def make():return DeclaredRouteScorer(track,rim,'cw',start,p.front_cm,expected)
    def passages(values):return [dict(junction=j,came_from=a,exit_arm=b,t=0,t_out=1) for j,a,b in values]
    a=make();a.travel=301;a.departed=True;a.sample(1,.02,start,np.ones(2)*.4,False)
    assert a.end is None  # Legacy scorer would finish at this premature return.
    b=make();b.departed=True;b.passages=passages(expected);b.travel=250
    b.sample(1,.02,start,np.ones(2)*.4,False);assert b.end=='mission_complete'
    assert b.final(b.end,.1)['clean_route_success'] # No forced extra lap just to exceed300cm.
    for actual in [expected[:-1],expected+[expected[0]], [('south','E','N')]+expected[1:]]:
        c=make();c.departed=True;c.passages=passages(actual);c.travel=301;c.sample(1,.02,start,np.ones(2)*.4,False)
        assert c.end!='mission_complete'
    for margin in [-.001,float('nan'),None]:
        c=make();c.departed=True;c.passages=passages(expected)
        assert not c.final('mission_complete',margin)['clean_route_success']
    c=make();c.passages=passages(expected);assert not c.final('mission_complete',1.)['clean_route_success']
    c=make();c.departed=True;c.passages=passages(expected);assert not c.final('time_limit',1.)['clean_route_success']
    # Exact point-on-zone-boundary and opposite heading do not count as return.
    c=make();c.departed=True;c.passages=passages(expected);pose=start.copy();pose[2]+=math.pi
    c.sample(1,.02,pose,np.ones(2)*.4,False);assert c.end is None
    try:DeclaredRouteScorer(track,rim,'cw',start,p.front_cm,[])
    except ValueError:pass
    else:raise AssertionError('Empty route accepted')
    REPORT.mkdir(exist_ok=False)
    dump(REPORT/'checks.json',dict(source_sha256=sha(__file__),legacy_source_sha256=sha(ROOT/'bench/mission.py'),
        cases=12,premature_legacy_completion_rejected=True,no_extra_lap_required=True,wrong_extra_missing_routes_rejected=True,
        finite_continuous_clearance_required=True,start_departure_and_heading_required=True,
        scope='Scorer state/termination tests with synthetic observations, not actual full-route driving or goal completion'))
    (REPORT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print('declared full-route scoring checks passed',flush=True)


if __name__=='__main__':checks()
