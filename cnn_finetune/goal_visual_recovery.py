"""Visual bridge for the frozen recovery teacher and local student evaluation.

Actor input is encoded image history, previous sent PWM and explicit direction.
During actor evaluation no expert program is advanced and placeholder labels
are invalid. Future DAgger queries need a separate, verified takeover labeler.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from bench.world import Camera,Renderer
from bench.realcam import RealCam,pose_from_history
from .goal_reactive_clock_validation import ReactiveClockTeacher
from .junction_learning import Encoder,Scene,setup,CHECKPOINT
from .model import preprocess_bgr
from .goal_development import ROOT,sha
from .junction_stage02 import dump

REPORT=ROOT/'cnn_finetune/results/goal_visual_recovery_20261113'
DIRECTION={'straight':np.array([0.,0.],np.float32),'left':np.array([1.,0.],np.float32),'right':np.array([0.,1.],np.float32)}


class VisualRecoveryScene(ReactiveClockTeacher):
    def __init__(self,key,condition,seed,preset,encoder):
        super().__init__(key,condition,seed,preset)
        camera=Camera();renderer=Renderer(self.track,camera)
        self.camera=RealCam(renderer,camera,self.profile,seed);self.encoder=encoder;self.actor_evaluation=False

    @torch.no_grad()
    def step(self,policy=None,intervention=None):
        if intervention is not None:raise ValueError('Mixed-control labels require a separate verified protocol')
        self.advance_to(self.capture)
        if self.outcome!='time_limit':return False
        image=self.camera.frame(self.capture,pose_from_history(self.history),raw=True)
        x=np.rint(preprocess_bgr(image)*255).astype(np.uint8)
        feature=self.encoder(torch.from_numpy(x[None]).cuda().float()/255).cpu().numpy()[0]
        aux=np.r_[self.previous,DIRECTION[self.key[2]]].astype(np.float32)
        if policy is None:
            label=self.teacher(self.car.pose.copy(),self.car.v,self.car.omega);proposed=label.copy();ref=np.zeros(2,dtype=np.float32)
        else:
            self.actor_evaluation=True;label=np.zeros(2,dtype=np.float32)
            y,self.hidden=policy(torch.from_numpy(feature[None,None]).cuda(),torch.from_numpy(aux[None,None]).cuda(),self.hidden)
            proposed=ref=y[0,0].cpu().numpy()
        done=self.capture+.036;self.advance_to(done)
        if self.outcome!='time_limit':return False
        applied=self.alloc.apply(proposed);self.car.command(done,applied)
        self.records.append(dict(x=x,f=feature,aux=aux,label=label.astype(np.float32),label_valid=policy is None,
            action=np.asarray(proposed,np.float32),reference=ref,applied=applied.copy(),capture=self.capture,done=done,margin=self.physics[-1][-1]))
        self.previous=applied.copy();self.capture=max(self.capture+1/30,done)
        return True

    def summary(self):
        return dict(super().summary(),visual_profile=self.profile,actor_evaluation=self.actor_evaluation,
            candidate_reference_status='geometry_accepted_development_candidate',
            teacher_labels_valid=not self.actor_evaluation,
            actor_inputs=['encoded_image_history','previous_sent_pwm','direction_instruction'])

    def save(self,dst,save_images=True):
        result=Scene.save(self,dst,save_images=save_images)
        p=Path(dst)/'sequence.npz'
        with np.load(p) as z:data={k:z[k] for k in z.files}
        data['supervise']=np.array([r['label_valid'] for r in self.records],bool)
        data['label_valid']=data['supervise'].copy();np.savez_compressed(p,**data)
        dump(Path(dst)/'decisions.json',self.decisions)
        return result


def checks():
    setup();encoder=Encoder().cuda().eval();rows=[];started=time.perf_counter()
    for key in [('center','E','right'),('north','E','left'),('west','N','straight')]:
        for condition in ['normal','mild_light']:
            a=ReactiveClockTeacher(key,condition,6801,'field_estimate').finish()
            b=VisualRecoveryScene(key,condition,6801,'field_estimate',encoder).finish()
            np.testing.assert_array_equal(a.physics,b.physics)
            np.testing.assert_array_equal([r['applied'] for r in a.records],[r['applied'] for r in b.records])
            for r in b.records:np.testing.assert_array_equal(r['aux'][-2:],DIRECTION[key[2]])
            assert all(r['label_valid'] for r in b.records)
            rows.append(dict(key=key,condition=condition,frames=len(b.records),physics_exact=True,commands_exact=True,direction=DIRECTION[key[2]]))
            print('visual bridge parity',key,condition,len(b.records),'elapsed',round(time.perf_counter()-started,1),flush=True)
    # Actor evaluation must not query/advance the privileged expert at all.
    a=VisualRecoveryScene(('west','N','straight'),'normal',6801,'field_estimate',encoder)
    def forbidden(*args):raise AssertionError('actor evaluation called privileged teacher')
    a.teacher=forbidden
    def actor(features,aux,hidden):
        assert features.shape==(1,1,1152) and aux.shape==(1,1,4)
        return torch.full((1,1,2),.4,device=features.device),hidden
    a.finish(actor,until=.3);assert len(a.records)>0 and not any(r['label_valid'] for r in a.records)
    REPORT.mkdir(exist_ok=False)
    dump(REPORT/'bridge_checks.json',dict(source_sha256=sha(__file__),teacher_source_sha256=sha(ROOT/'cnn_finetune/goal_reactive_clock_validation.py'),
        encoder_checkpoint=str(CHECKPOINT),encoder_sha256=sha(CHECKPOINT),device=torch.cuda.get_device_name(0),
        rows=rows,actor_calls_expert=False,actor_feature_shape=[1152],actor_aux_shape=[4],actor_placeholder_labels_masked=True,
        no_training=True,no_final_exam=True,scope='rendering/preprocessing/feature bridge parity and actor input separation, not learned-policy performance'))
    print('visual bridge checks passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['check']);a=p.parse_args();checks()
