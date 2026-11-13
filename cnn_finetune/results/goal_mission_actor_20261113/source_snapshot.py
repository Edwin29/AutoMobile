"""Continuous recurrent actor plus image-derived direction resolution.

Requires a runner to acknowledge every actual sent PWM via on_applied().
No registration in legacy bench controllers and no hardware entry point.
"""
import json
from pathlib import Path
import numpy as np
import torch
from bench.world import Track
from .goal_mission_map import DelayedImageMap
from .goal_mission_direction import DirectionResolver,_policy
from .goal_visual_recovery import DIRECTION
from .junction_learning import Encoder,model_at,setup
from .junction_feasibility import SignedAllocator
from .model import preprocess_bgr
from .goal_development import ROOT,sha
from .junction_stage02 import dump,OUT as STAGE_OUT

REPORT=ROOT/'cnn_finetune/results/goal_mission_actor_20261113'
PATH_REPORT=ROOT/'cnn_finetune/results/goal_clearance_paths_20261113'


class ContinuousMissionActor:
    raw_camera=True

    def __init__(self,checkpoint,direction,start_hint,encoder=None,model=None,observer=None):
        self.observer=observer if observer is not None else DelayedImageMap(direction,start_hint)
        self.track=Track();self.direction=DirectionResolver(self.track.data['junctions'],direction)
        self.intr=_policy.Interrupt();self.encoder=encoder if encoder is not None else Encoder().cuda().eval()
        self.model=model if model is not None else model_at(checkpoint)
        self.alloc=SignedAllocator([.36,.36]);self.previous=np.zeros(2,np.float32);self.hidden=None
        self.pending_at=None;self.log_cursor=0;self.frames=0
        self.references={tuple(r['key']):Path(r['reference']) for r in json.loads((PATH_REPORT/'geometry.json').read_text())}
        # Outer road geometry is known map data, not environment pose or progress.
        from drive_v4.route import with_lane_offsets
        d=self.observer.visible
        self.outer={k:with_lane_offsets(d.track,d.track.outer_route(k)).points.copy() for k in ['cw','ccw']}

    def push_direction(self,t,move):self.direction.push_direction(t,move)
    def stop(self,t,reason='orchestrator'):self.intr.stop(t,reason)
    def resume(self,t,reason='orchestrator'):self.intr.resume(t,reason)

    def _routes(self):
        for event in self.direction.log[self.log_cursor:]:
            j=event['junction']
            if event['event']=='estimated_arrival':
                key=(j,event['came_from'],event['move'])
                path=(STAGE_OUT/'paths'/'_'.join(key)/'path.npz') if key[2]=='straight' else self.references[key]
                with np.load(path) as z:points=z['points'].copy()
                self.observer.set_route(points)
            else:
                ex=event['exit_arm']
                if j!='center' and self.direction.policy.kind(j,ex)=='outer_T':
                    self.observer.set_route(self.outer[self.direction.policy.lap_direction])
                else:
                    c=np.asarray(self.track.data['junctions'][j]['xy']);a=np.asarray(_policy.ARMS[ex])
                    self.observer.set_route(c+np.arange(20.,131.)[:,None]*a)
        self.log_cursor=len(self.direction.log)

    @torch.no_grad()
    def step(self,image,capture,done):
        if self.pending_at is not None:raise RuntimeError('Runner has not acknowledged the preceding actual PWM')
        estimate=self.observer.observe(image,capture,done)
        instruction=self.direction.observe(done,estimate['pose'],estimate['sxy']);self._routes()
        x=np.rint(preprocess_bgr(image)*255).astype(np.uint8)
        f=self.encoder(torch.from_numpy(x[None]).cuda().float()/255)
        aux=np.r_[self.previous,DIRECTION[instruction]].astype(np.float32)
        output,self.hidden=self.model(f[:,None],torch.from_numpy(aux[None,None]).cuda(),self.hidden)
        desired=output[0,0].cpu().numpy()
        if not np.isfinite(desired).all():raise RuntimeError('Nonfinite neural command')
        sent=np.zeros(2) if self.intr.held else self.alloc.apply(desired)
        self.pending_at=float(done);self.frames+=1
        return sent,dict(instruction=instruction,held=self.intr.held,est=estimate['pose'].tolist(),sxy=estimate['sxy'],
            map_capture=estimate['map_capture'],map_latest_published=estimate['latest_published'],
            neural_desired=desired.tolist(),reason='required_interrupt' if self.intr.held else 'continuous_neural_actor',
            actor_input_schema='image features1152,previous sentPWM2,direction2',hidden_reset=False)

    def on_applied(self,t,pwm):
        if self.pending_at is None or abs(t-self.pending_at)>1e-9:raise RuntimeError('Actual command acknowledgment timestamp mismatch')
        self.observer.on_applied(t,pwm);self.previous=np.asarray(pwm,np.float32).copy();self.pending_at=None


def checks():
    setup()
    from bench.world import Camera,Renderer
    track=Track();hint=track.start_pose('cw',14.5,10.);image=Renderer(track,Camera()).render_raw(hint)
    class Recorder(torch.nn.Module):
        def __init__(self):super().__init__();self.inputs=[]
        def forward(self,f,aux,hidden=None):
            assert f.shape==(1,1,1152) and aux.shape==(1,1,4)
            self.inputs.append(aux.cpu().numpy().copy())
            n=1 if hidden is None else float(hidden.item())+1
            return torch.tensor([[[.4,.4]]],device=f.device),torch.tensor(n,device=f.device)
    model=Recorder();actor=ContinuousMissionActor(None,'cw',hint,model=model)
    sent,info=actor.step(image,1/30,1/30+.036)
    np.testing.assert_array_equal(model.inputs[0],np.zeros((1,1,4)))
    try:actor.step(image,.1,.136)
    except RuntimeError:pass
    else:raise AssertionError('Missing sent-command acknowledgment accepted')
    # The runner can report clipping/override: next network input and map receive
    # this actual command, not the originally proposed .4,.4.
    actor.on_applied(1/30+.036,[.36,0.]);actor.stop(.08,'red');actor.stop(.08,'stop_sign')
    sent,info=actor.step(image,.1,.136);np.testing.assert_array_equal(sent,[0.,0.])
    np.testing.assert_allclose(model.inputs[1][0,0,:2],[.36,0.],atol=2e-8);actor.on_applied(.136,sent)
    actor.resume(.14,'stop_sign');assert actor.intr.held
    actor.resume(.15,'red');assert not actor.intr.held
    sent,info=actor.step(image,.17,.206);actor.on_applied(.206,sent)
    np.testing.assert_array_equal(model.inputs[2][0,0,:2],[0.,0.]);assert float(actor.hidden.item())==3
    assert actor.observer.applied[0]['pwm'].tolist()==[.36,0.]
    # A real checkpoint can run through the same interface without privileged
    # inputs; this is only an inference integration check, no driving success.
    checkpoint=Path('D:/autodrive_cnn/runs/goal_development_20261113/convergence_probe/best.pt')
    actor.model=model_at(checkpoint);actor.hidden=None
    sent,info=actor.step(image,.24,.276);actor.on_applied(.276,sent);assert np.isfinite(sent).all() and abs(sent).max()<=1
    REPORT.mkdir(exist_ok=False)
    dump(REPORT/'checks.json',dict(source_sha256=sha(__file__),checkpoint_sha256=sha(checkpoint),
        dependencies={m:sha(ROOT/'cnn_finetune'/m) for m in ['goal_mission_map.py','goal_mission_direction.py']},
        actual_command_feedback=True,required_interrupt_reasons_preserved=True,recurrent_state_preserved=True,
        missing_ack_rejected=True,actor_input_shape=[1152,4],real_checkpoint_inference=True,
        scope='Four inference frames with actual image-map core and explicit actuator acknowledgments; not a closed-loop mission',
        remaining=['Full declared-route runner and scorer','Outer-corner and connected-sequence training','Dynamic instruction-transition validation','Latency and map localization failures under perturbations']))
    (REPORT/'source_snapshot.py').write_bytes(Path(__file__).read_bytes());print('continuous mission actor interface checks passed',flush=True)


if __name__=='__main__':checks()
