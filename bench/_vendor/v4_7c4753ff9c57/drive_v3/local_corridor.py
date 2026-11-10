"""Experimental observed corridor control, independent of absolute map matching.

An outer-loop intent continues straight at T junctions using the outside edge.
Missing inside boundaries are inferred from width and are never observations.
This module does not implement sign-directed junction turns or obstacle detection.
"""
import numpy as np
from .contracts import PathDecision
from .geometry import advance,to_world,to_body


class LocalCorridorPlanner:
    def __init__(self,camera,direction='ccw',width_cm=24.):
        self.camera=camera;self.direction=direction;self.width=width_cm
        self.previous=None;self.previous_s=None;self.last_diagnostics={}
        self.route=None;self.state='LOCAL';self.completed_route=False

    def stop(self,now,clock,reason):
        self.last_diagnostics={'reason':reason,'source':'local_observation'}
        return PathDecision([],0.,now+.4,clock,reason)

    def plan_local(self,debug,capture,now,clock,v=0.,omega=0.):
        g=self.camera;colored=debug['colored'];valid=debug['valid']
        pairs=[];single=[];previous_x=0.;outer_right=self.direction=='ccw'
        # Work from near to far; skip unobserved pixels, never extrapolate an edge
        # through the camera padding. Each width/continuity gate is provisional.
        for forward in np.arange(g.lens_ahead+7,g.lens_ahead+33,.8):
            row=int(round(g.body_to_pixels([[0.,forward]])[0,1]))
            if row<1 or row>=colored.shape[0]-1:continue
            color=colored[row-1:row+2].mean(0)>=.5
            usable=valid[row]
            changes=np.diff(np.r_[False,color&usable,False].astype(int))
            runs=[(a,b) for a,b in zip(np.where(changes==1)[0],np.where(changes==-1)[0]) if b-a>=g.ppcm*.6]
            choices=[]
            for (_,a),(b,_) in zip(runs,runs[1:]):
                left=a/g.ppcm+g.lat_min;right=b/g.ppcm+g.lat_min
                width=right-left;center=(left+right)/2
                first=not pairs and not single
                if 17<=width<=34 and abs(center-previous_x)<=(20 if first else 6) and usable[a:b].mean()>.95:
                    score=abs(center-previous_x)+.3*abs(width-self.width)
                    choices.append((score,center,width,left,right))
            if choices:
                _,center,width,left,right=min(choices)
                pairs.append((center,forward,width));previous_x=center
                continue
            # At an outer-road T opening, the outside edge still bounds the
            # through road. Use map width only to infer the missing inside edge.
            edge_choices=[]
            for a,b in runs:
                if b-a<g.ppcm*2:continue
                edge=(a if outer_right else b)/g.ppcm+g.lat_min
                center=edge+(-self.width/2 if outer_right else self.width/2)
                if abs(center-previous_x)<(20 if not pairs and not single else 4):
                    edge_choices.append((abs(center-previous_x),center,edge))
            if edge_choices:
                _,center,edge=min(edge_choices)
                single.append((center,forward,self.width));previous_x=center
        samples=sorted(pairs+single,key=lambda q:q[1])
        if len(samples)<6:
            return self.stop(now,clock,'local_boundaries_missing')
        p=np.asarray(samples);span=np.ptp(p[:,1])
        if span<6:return self.stop(now,clock,'local_support_too_short')
        # Robust quadratic fit to corridor centres, not a global pose fit.
        u=(p[:,1]-p[0,1])/max(span,1.)
        keep=np.ones(len(p),bool)
        for _ in range(3):
            coeff=np.polyfit(u[keep],p[keep,0],2)
            residual=p[:,0]-np.polyval(coeff,u)
            scale=max(.35,1.4826*np.median(np.abs(residual[keep]-np.median(residual[keep]))))
            updated=np.abs(residual)<=max(1.,2.5*scale)
            if updated.sum()<6:break
            keep=updated
        rms=float(np.sqrt(np.mean(residual[keep]**2)))
        if rms>1.5:return self.stop(now,clock,'local_corridor_inconsistent')
        y=np.linspace(p[keep,1].min(),p[keep,1].max(),25)
        x=np.polyval(coeff,(y-p[0,1])/max(span,1.))
        observed=np.c_[x,y]
        # Transform capture-frame geometry into the current body frame using
        # only the last applied command. This is a short-delay prediction.
        dt=max(0.,now-capture)
        if dt>.4:return self.stop(now,clock,'local_frame_stale')
        delta=advance(np.zeros(3),v,omega,dt)
        points=to_body(observed,delta)
        # Join from the body origin to the visible corridor. The gap is inferred,
        # explicitly not observed or checked as a full-body collision guarantee.
        q=points[0];t=np.linspace(0.,1.,12,endpoint=False)
        join=np.c_[q[0]*(3*t*t-2*t*t*t),q[1]*t]
        path=np.vstack((join,points))
        if pairs:self.width=float(np.clip(.9*self.width+.1*np.median(np.array(pairs)[:,2]),20.,28.))
        self.last_diagnostics=dict(source='local_observation',global_match_required=False,
            paired_rows=len(pairs),width_inferred_rows=len(single),width_cm=self.width,
            support_cm=float(span),fit_rms_cm=rms,first_observed_body_cm=points[0].tolist(),
            intent='outer_loop_straight_at_T',next_known_junction_options=['straight','left'] if outer_right else ['straight','right'])
        observed_start_arc=float(np.linalg.norm(np.diff(join,axis=0),axis=1).sum()+np.linalg.norm(points[0]-join[-1]))
        self.last_diagnostics['lookahead_cm']=observed_start_arc+5.
        return PathDecision(path.tolist(),8.,now+.4,clock,'tracking',lookahead_cm=observed_start_arc+5.)
