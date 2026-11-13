"""Observed length, rather than fragment count, sets boundary influence.

Nearby aligned fragments share a support-length estimate. Links add no points
and their gaps contribute no length. Coincident sampled evidence is capped.
"""
import numpy as np
from .contracts import EvidenceSource


def sample_evidence(fragments, cfg):
    items=[]
    for f in sorted(fragments,key=lambda f:-f.quality.get('length_cm',0))[:cfg['max_fragments']]:
        p=np.asarray(f.points_cm,float)
        if f.source!=EvidenceSource.OBSERVED or len(p)<3:continue
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
        if arc[-1]<=0:continue
        direction=p[-1]-p[0];norm=np.linalg.norm(direction)
        direction=direction/max(norm,1e-9)
        items.append((f,p,arc,direction))
    parents=list(range(len(items)))
    def root(i):
        while parents[i]!=i:i=parents[i]
        return i
    for i,(_,a,_,ta) in enumerate(items):
        for j in range(i):
            _,b,_,tb=items[j]
            if abs(ta@tb)<np.cos(np.deg2rad(cfg.get('support_join_angle_deg',20.))):continue
            distance=np.linalg.norm(a[:,None]-b[None,:],axis=2)
            ai,bj=np.unravel_index(np.argmin(distance),distance.shape)
            delta=a[ai]-b[bj]
            normal=np.array([-ta[1],ta[0]])
            if np.linalg.norm(delta)<=cfg.get('support_join_gap_cm',2.) and abs(delta@normal)<=cfg.get('support_join_normal_cm',.6):
                parents[root(i)]=root(j)
    support_cells={}
    cell_size=cfg.get('evidence_cell_cm',.5)
    for i,(_,p,arc,_) in enumerate(items):
        steps=np.diff(arc)
        measures=np.r_[steps[0]/2,(steps[:-1]+steps[1:])/2,steps[-1]/2]
        local={}
        for cell,length in zip(np.rint(p/cell_size).astype(int),measures):
            key=tuple(cell);local[key]=local.get(key,0.)+length
        aggregate=support_cells.setdefault(root(i),{})
        for key,length in local.items():aggregate[key]=max(aggregate.get(key,0.),length)
    support={key:sum(cells.values()) for key,cells in support_cells.items()}
    points=[];mass=[];groups=[]
    length_scale=cfg.get('evidence_length_cm',6.)
    reliable_length=cfg.get('reliable_support_cm',4.)
    for i,(f,p,arc,_) in enumerate(items):
        n=min(len(p),cfg['max_points_per_fragment'])
        # Uniform arclength sampling avoids dependence on pixel-chain density.
        along=np.linspace(0,arc[-1],n)
        sample=np.c_[np.interp(along,arc,p[:,0]),np.interp(along,arc,p[:,1])]
        reliability=min(1.,support[root(i)]/reliable_length)**2
        quadrature=np.ones(n);quadrature[[0,-1]]=.5
        weights=quadrature/quadrature.sum()*arc[-1]/length_scale*reliability
        points.extend(sample);mass.extend(weights);groups.extend([f.id]*n)
    points=np.asarray(points).reshape(-1,2);mass=np.asarray(mass)
    if len(points):
        bins=np.rint(points/cfg.get('evidence_cell_cm',.5)).astype(int)
        for cell in np.unique(bins,axis=0):
            indices=np.flatnonzero(np.all(bins==cell,axis=1))
            # Duplicate fragments at the same sample cell cannot multiply evidence.
            totals={g:mass[[i for i in indices if groups[i]==g]].sum() for g in {groups[i] for i in indices}}
            if len(totals)>1:mass[indices]*=max(totals.values())/sum(totals.values())
    return points,np.sqrt(mass),groups


def weighted_huber(weights):
    """Apply Huber to physical standardized residual BEFORE evidence weighting."""
    mass=np.square(weights)
    def loss(z):
        rho=np.empty((3,len(z)))
        low=z<=1.;high=~low
        rho[0,low]=z[low];rho[1,low]=1.;rho[2,low]=0.
        root=np.sqrt(z[high])
        rho[0,high]=2*root-1;rho[1,high]=1/root;rho[2,high]=-.5/(z[high]*root)
        return rho*mass[None,:]
    return loss
