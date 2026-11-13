"""Shadow one-camera-processing-window motion forecasts, no control changes."""
import json
import numpy as np
from bench.vehicle import Vehicle
from bench.world import wrap,to_world
from .goal_boundary_teacher import ForecastInput
from .goal_boundary_teacher_v2 import observed_residual,PILOT
from .goal_teacher_coverage import CoverageTeacher,ROOT
from .goal_boundary_commit import REPORT
from .goal_development import sha
from .junction_stage02 import dump


def short_forecast(snapshot,residual,hold):
    car=Vehicle(snapshot.p,snapshot.pose.copy());car.speed=snapshot.wheel_speed.copy()
    car.commands=[(t,u.copy()) for t,u in snapshot.commands]
    t=snapshot.t;end=t+.036
    while t<end-1e-10:
        dt=min(.02,end-t);car.step(t,dt)
        integral=dt if hold else .2*(np.exp(-(t-snapshot.t)/.2)-np.exp(-(t+dt-snapshot.t)/.2))
        car.pose[:2]+=residual[:2]*integral;car.pose[2]=wrap(car.pose[2]+residual[2]*integral);t+=dt
    return car.pose.copy()


def main():
    dump(REPORT/'motion_audit_protocol.json',dict(source_sha256=sha(__file__),keys=PILOT,seed=6202,
        presets=['field_estimate','sluggish'],condition='mild',horizon_s=.036,
        modes=['no_residual','decay_0.2','hold_current'],
        information='current pose, wheel speed, nominal params, old PWM queue, residual from latest past pose increment',
        outcome='same unchanged corrected-teacher trajectories; forecast never controls vehicle; events used for retrospective grouping only'))
    rows=[]
    for key in PILOT:
        for preset in ['field_estimate','sluggish']:
            env=CoverageTeacher(key,'mild',6202,preset,'centered')
            while env.capture<40 and env.outcome=='time_limit':
                env.advance_to(env.capture)
                if env.outcome!='time_limit':break
                snapshot=ForecastInput.capture(env);residual=observed_residual(env)
                predictions={'no_residual':short_forecast(snapshot,np.zeros(3),False),
                             'decay_0.2':short_forecast(snapshot,residual,False),'hold_current':short_forecast(snapshot,residual,True)}
                if not env.step():break
                truth=env.car.pose.copy()
                active=any(e['kind']=='push' and e['t0']<=snapshot.t<e['t0']+e['duration'] for e in env.events)
                for name,pred in predictions.items():
                    rows.append(dict(key=key,preset=preset,t=snapshot.t,mode=name,push_active_at_capture=active,
                        position_error_cm=float(np.linalg.norm(pred[:2]-truth[:2])),heading_error_deg=float(abs(np.degrees(wrap(pred[2]-truth[2])))),
                        predicted_margin_cm=float(snapshot.track.clearance(to_world(snapshot.rim,pred)).min()),
                        actual_margin_cm=float(snapshot.track.clearance(to_world(snapshot.rim,truth)).min())))
    summary=[]
    for mode in ['no_residual','decay_0.2','hold_current']:
        for active in [False,True]:
            a=[r for r in rows if r['mode']==mode and r['push_active_at_capture']==active]
            summary.append(dict(mode=mode,push_active_at_capture=active,n=len(a),
                mean_position_error_cm=float(np.mean([r['position_error_cm'] for r in a])),
                p95_position_error_cm=float(np.quantile([r['position_error_cm'] for r in a],.95)),
                mean_heading_error_deg=float(np.mean([r['heading_error_deg'] for r in a])),
                false_safe_frames=sum(r['predicted_margin_cm']>=0 and r['actual_margin_cm']<0 for r in a)))
    dump(REPORT/'motion_audit.json',dict(summary=summary,rows=rows));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
