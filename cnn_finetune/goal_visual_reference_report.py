"""Geometry groups and trajectory figures from frozen completed reference runs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from bench.world import Track,Camera,Renderer,to_world
from bench.vehicle import PRESETS,Vehicle
from drive_v4.junction_policy import JunctionPolicy
from .goal_all_movement_eval import REPORT
from .goal_visual_nominal_teacher_audit import REPORT as TEACHER_REPORT
from .goal_development import sha
from .junction_stage02 import dump


def main():
    data_file=REPORT/'reference_nominal.json';teacher_file=TEACHER_REPORT/'results.json'
    rows=json.loads(data_file.read_text());teachers=json.loads(teacher_file.read_text())['rows']
    track=Track();policy=JunctionPolicy(track.data['junctions']);groups={}
    for r in rows:
        j,a,m=r['key'];g=policy.kind(j,a)+('_straight' if m=='straight' else '_turn');groups.setdefault(g,[]).append(r)
    result={g:dict(n=len(a),clean=sum(r['continuous_clean'] for r in a),wrong_exit=sum(r['outcome']=='wrong_exit' for r in a)) for g,a in groups.items()}
    dump(REPORT/'reference_geometry_groups.json',dict(groups=result,source_sha256=sha(__file__),
        reference_sha256=sha(data_file),teacher_sha256=sha(teacher_file),
        inference='Outer-T turns were absent from the original four-movement training set and fail24/24 here. This supports testing coverage expansion, not a causal proof that coverage is the sole error source.',
        scope='one development seed,three vehicle settings; not24 independent environment seeds'))
    renderer=Renderer(track,Camera());tex=renderer.base_texture[:,:,::-1]
    extent=[renderer.origin,renderer.origin+tex.shape[1]/renderer.ppcm,renderer.origin,renderer.origin+tex.shape[0]/renderer.ppcm]
    keys=[('east','S','left'),('north','W','right'),('center','N','right')]
    fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    for ax,key in zip(axes,keys):
        a=next(r for r in rows if tuple(r['key'])==key and r['preset']=='field_estimate')
        b=next(r for r in teachers if tuple(r['key'])==key and r['preset']=='field_estimate')
        x=np.load(Path(a['artifact'])/'physics.npz')['samples'];y=np.load(Path(b['artifact'])/'physics.npz')['physics']
        ax.imshow(tex,origin='lower',extent=extent)
        ax.plot(y[:,1],y[:,2],color='#ffb000',lw=2,label='Privileged teacher')
        ax.plot(x[:,1],x[:,2],color='#00dfff',lw=2,label='Four-movement GRU')
        rim=Vehicle(PRESETS['field_estimate'],np.zeros(3)).footprint()
        for trace,col in [(x,'#00dfff'),(y,'#ffb000')]:
            for idx in [0,len(trace)//2,-1]:
                ax.add_patch(Polygon(to_world(rim,trace[idx,1:4]),closed=True,fill=False,edgecolor=col,lw=.9,alpha=.75))
        c=np.asarray(track.data['junctions'][key[0]]['xy']);ax.set_xlim(c[0]-62,c[0]+62);ax.set_ylim(c[1]-62,c[1]+62)
        ax.set_aspect('equal');ax.set_xlabel('x (cm)');ax.set_ylabel('y (cm)')
        ax.set_title('_'.join(key)+'\n'+a['outcome']+f"; min clearance {a['continuous_min_clearance_cm']:.3f} cm",fontsize=10)
    axes[0].legend(fontsize=8,loc='lower left');fig.suptitle('Local normal development screen: seed 6901, field_estimate. Axle paths + three body snapshots.')
    fig.savefig(REPORT/'reference_trajectories.png',dpi=150);plt.close(fig)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
