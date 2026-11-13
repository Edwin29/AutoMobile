"""CNN+GRU predicts local path and speeds; no truth/map-pose input."""
import numpy as np
import torch
from torch import nn
from .control import LocalPath

POINTS=48
AUX_SCHEMA=['applied_left','applied_right','left','straight','right','maneuver_active','elapsed_s']


class TrajectoryPolicy(nn.Module):
    def __init__(self,encoder=None):
        super().__init__()
        if encoder is None:
            from cnn_finetune.model import LanePolicy
            encoder=LanePolicy().net[:11]
        self.encoder=encoder
        self.project=nn.Sequential(nn.LayerNorm(1152),nn.Linear(1152,128),nn.ELU())
        self.gru=nn.GRU(128+len(AUX_SCHEMA),64,batch_first=True)
        self.path_head=nn.Sequential(nn.Linear(64,64),nn.ELU(),nn.Linear(64,POINTS*2))
        self.speed_head=nn.Sequential(nn.Linear(64,32),nn.ELU(),nn.Linear(32,POINTS))

    def forward(self,images,aux,hidden=None):
        b,t=images.shape[:2]
        f=self.encoder(images.reshape(b*t,3,66,200)).reshape(b,t,1152)
        z,hidden=self.gru(torch.cat([self.project(f),aux],dim=-1),hidden)
        points=self.path_head(z).reshape(b,t,POINTS,2)*25.
        speed=torch.sigmoid(self.speed_head(z))*8.
        return points,speed,hidden


class Predictor:
    def __init__(self,checkpoint,device='cpu'):
        state=torch.load(checkpoint,map_location=device,weights_only=True)
        if state.get('format')!='hyb4-path-v1' or state.get('aux_schema')!=AUX_SCHEMA:
            raise ValueError('Incompatible checkpoint; PWM policies cannot be loaded as trajectories')
        if not state.get('trained',False):raise ValueError('Untrained trajectory model cannot drive')
        self.model=TrajectoryPolicy().to(device).eval();self.model.load_state_dict(state['state_dict'])
        self.device=device;self.hidden=None;self.last=None

    @torch.no_grad()
    def observe(self,image,capture,previous,direction,active):
        from cnn_finetune.model import preprocess_bgr
        if direction not in ['left','straight','right']:raise ValueError('Unknown direction')
        dt=0. if self.last is None else capture-self.last
        if dt<0:raise ValueError('Capture clock moved backward')
        aux=np.r_[previous,[direction==d for d in ['left','straight','right']],active,dt].astype(np.float32)
        x=torch.from_numpy(preprocess_bgr(image)[None,None]).to(self.device)
        a=torch.from_numpy(aux[None,None]).to(self.device)
        p,s,self.hidden=self.model(x,a,self.hidden);self.last=capture
        if not active:return None
        return LocalPath(p[0,0].cpu().numpy(),s[0,0].cpu().numpy(),float(capture))
