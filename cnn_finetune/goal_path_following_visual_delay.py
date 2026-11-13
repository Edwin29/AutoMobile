"""Bounded post-pilot diagnosis: fixed 0.40s horizon on eight sluggish cases.

The horizon is privileged calibration. Pose remains entirely image-derived.
No candidate adoption, new training, gain sweep, or compound expansion.
"""
import json
from pathlib import Path
import numpy as np
from bench.world import advance,wrap
from . import goal_path_following_pilot as pilot
from .goal_development import sha
from .junction_stage02 import dump


class DelayCalibratedControl(pilot.PathController):
    horizon=.40
    def step(self,pose,t):
        v,w=self.motion.at(t);pred=advance(np.asarray(pose),v,w,self.horizon)
        near=self.idx+int(np.argmin(np.linalg.norm(self.points[self.idx:min(len(self.points),self.idx+100)]-pred[:2],axis=1)))
        self.idx=max(self.idx-2,near);h=self.h[self.idx]
        lat=float((pred[:2]-self.points[self.idx])@np.array([np.cos(h),-np.sin(h)]));head=float(wrap(h-pred[2]))
        target=float(self.speed[self.idx])
        if self.mode=='planned':self.planned_v=min(target,self.planned_v+.036);target=self.planned_v
        omega=target*self.k[self.idx]+1.3*head-.08*lat;diff=omega*13/(2*.6)
        desired=np.array([target+diff,target-diff])/16.8;desired/=max(1,float(max(abs(desired))))
        return desired,dict(index=self.idx,lateral_cm=lat,heading_rad=head,nominal_v=v,nominal_w=w)


def main():
    pilot.setup();report=pilot.REPORT
    protocol=report/'visual_delay_diagnostic_protocol.json';assert not protocol.exists()
    # Exact nominal-command replay check before the one-parameter intervention.
    reg={tuple(r['key']):r for r in json.loads((pilot.PATH_REPORT/'geometry.json').read_text())}
    rows=json.loads((report/'vision_normal.json').read_text());rows=[r for r in rows if r['preset']=='sluggish']
    maximum=0.
    for r in rows:
        e=pilot.ReserveTeacher(tuple(r['key']),'normal',r['seed'],'sluggish','same_mode',reg)
        c=DelayCalibratedControl(e.points,e.h,e.k,e.speed,e.teacher_mode);c.horizon=.15
        for rec in json.loads((Path(r['artifact'])/'records.json').read_text()):
            y,_=c.step(rec['estimate'],rec['done']);error=float(np.max(abs(y-rec['desired'])));maximum=max(error,maximum)
            np.testing.assert_allclose(y,rec['desired'],atol=1e-12,rtol=0);c.applied(rec['done'],rec['applied'])
    dump(protocol,dict(source_sha256=sha(__file__),pilot_source_sha256=sha(pilot.__file__),
        hypothesis='The identified actuator anticipation error also contributes under image-map feedback',
        cases='same8sluggish nominal cases, both training-independent development seeds, all4movements',
        change='fixed controller prediction horizon .15->.40s; prior,nominal motion model,motor coefficients,gains,path,speed,allocator,observer unchanged',
        privileged_information='0.40s equals known simulator sluggish delay+time_constant; not an online estimate',
        nominal_action_replay_max_error=maximum,final_exam_used=False,new_training=False))
    # Isolated process: preserve frozen pilot source and route output to a new root.
    pilot.PathController=DelayCalibratedControl
    pilot.OUT=pilot.OUT/'visual_calibrated_horizon'
    results=[]
    for r in rows:
        item=pilot.run_one(tuple(r['key']),r['seed'],'sluggish','normal','vision')
        item['diagnostic']='privileged_horizon_040';item['baseline_clean']=r['continuous_clean'];results.append(item)
        dump(report/'visual_delay_progress.json',results)
        print('visual calibrated delay',len(results),'/8',item['key'],item['outcome'],item['continuous_clean'],flush=True)
    dump(report/'visual_delay_diagnostic.json',results)


if __name__=='__main__':main()
