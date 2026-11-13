"""Independent continuous rectangle/rounded-road collision audit.

Positive clearance and collision sign are exact for the map's rounded boxes.
Deep obstacle-core penetration is saturated at -radius, so this must NOT replace
the old severe-departure magnitude metric. Frozen benchmark scores are retained.
"""
import json
from pathlib import Path
import numpy as np
from bench.world import Track,box_sdf,to_world
from bench.vehicle import Vehicle,PRESETS
from .goal_development import ROOT,sha
from .junction_stage02 import dump

REPORT=ROOT/'cnn_finetune/results/goal_boundary_commit_20261113'


def continuous_clearance(poses,params,track):
    poses=np.asarray(poses);co=np.cos(poses[:,2]);si=np.sin(poses[:,2])
    local=np.array([[-params.half_width_cm,-params.rear_cm],[params.half_width_cm,-params.rear_cm],
                    [params.half_width_cm,params.front_cm],[-params.half_width_cm,params.front_cm]])
    points=np.stack([poses[:,0,None]+local[None,:,0]*co[:,None]+local[None,:,1]*si[:,None],
                     poses[:,1,None]-local[None,:,0]*si[:,None]+local[None,:,1]*co[:,None]],axis=-1)
    # The outer rounded rectangle is convex; all four vertices inside implies
    # the complete filled vehicle rectangle is inside. Minimum occurs at a vertex.
    result=(-box_sdf(points,track.data['outer'],track.data['outer_radius_cm'])).min(axis=1)
    a=points;b=np.roll(points,-1,axis=1);edge=b-a;length2=(edge*edge).sum(axis=-1)
    radius=track.data['block_radius_cm']
    for box in track.data['blocks']:
        x0,x1,y0,y1=box;low=np.array([x0+radius,y0+radius]);high=np.array([x1-radius,y1-radius])
        corners=np.array([[low[0],low[1]],[high[0],low[1]],[high[0],high[1]],[low[0],high[1]]])
        # Distance between disjoint convex rectangles is attained by a vertex
        # against an edge. Include overlap/containment via separating axes.
        d1=np.linalg.norm(points-np.clip(points,low,high),axis=-1).min(axis=1)
        delta=corners[None,None]-a[:,:,None]
        fraction=np.clip((delta*edge[:,:,None]).sum(axis=-1)/length2[:,:,None],0,1)
        projected=a[:,:,None]+fraction[:,:,:,None]*edge[:,:,None]
        d2=np.linalg.norm(projected-corners[None,None],axis=-1).min(axis=(1,2))
        world_overlap=((points.max(axis=1)>=low)&(points.min(axis=1)<=high)).all(axis=1)
        relative=corners[None]-poses[:,None,:2]
        bx=relative[:,:,0]*co[:,None]-relative[:,:,1]*si[:,None]
        by=relative[:,:,0]*si[:,None]+relative[:,:,1]*co[:,None]
        body_overlap=(bx.max(axis=1)>=-params.half_width_cm)&(bx.min(axis=1)<=params.half_width_cm)
        body_overlap&=(by.max(axis=1)>=-params.rear_cm)&(by.min(axis=1)<=params.front_cm)
        distance=np.where(world_overlap&body_overlap,0,np.minimum(d1,d2))
        result=np.minimum(result,distance-radius)
    return result


def check():
    from dataclasses import replace
    track=Track();p=PRESETS['field_estimate'];rng=np.random.default_rng(55173)
    poses=np.c_[rng.uniform(-100,100,(500,2)),rng.uniform(-np.pi,np.pi,500)]
    exact=continuous_clearance(poses,p,track);dense=[]
    rim=Vehicle(p,np.zeros(3)).footprint(.025)
    for pose in poses:dense.append(float(track.clearance(to_world(rim,pose)).min()))
    dense=np.asarray(dense)
    assert np.array_equal(exact<-1e-9,dense<-1e-9)
    near=(dense>=-min(track.data['block_radius_cm'],1.))
    assert np.all(exact[near]<=dense[near]+1e-9)
    assert np.max(dense[near]-exact[near])<=.0125+1e-8
    # A rounded obstacle intrudes between two 1cm perimeter samples. All old
    # samples are outside, but the continuous bottom edge crosses the circle.
    synthetic=Track();synthetic.data=dict(outer=[-100,100,-100,100],outer_radius_cm=0.,
        blocks=[[-.7,1.3,-1,1]],block_radius_cm=1.)
    tiny=replace(p,half_width_cm=2.,front_cm=1.,rear_cm=1.)
    pose=np.array([[0.,1.99,0.]])
    sparse=synthetic.clearance(to_world(Vehicle(tiny,np.zeros(3)).footprint(1.),pose[0])).min()
    analytic=continuous_clearance(pose,tiny,synthetic)[0]
    assert sparse>0 and abs(analytic+.01)<1e-10
    assert np.isfinite(exact).all()
    dump(REPORT/'continuous_geometry_checks.json',dict(source_sha256=sha(__file__),random_seed=55173,poses=500,
        dense_sampling_cm=.025,collision_signs_agree=True,near_boundary_distance_bound_passed=True,
        sparse_outline_missed_intrusion_detected=True,
        max_near_boundary_sampling_gap_cm=float(np.max(dense[near]-exact[near]))))
    print('continuous geometry: 500 dense-outline sign and distance checks passed',flush=True)


def audit():
    track=Track();sources=[]
    # Completed pilot candidates plus any completed prospective normal cases.
    for folder,name in [('goal_boundary_teacher_20261113','pilot.json'),('goal_boundary_teacher_v2_20261113','plan.json'),
                        ('goal_boundary_teacher_v2_20261113','observed_push.json'),('goal_boundary_commit_20261113','pilot.json'),
                        ('goal_boundary_hold_20261113','pilot.json')]:
        sources += [(folder+'/'+name,r) for r in json.loads((ROOT/'cnn_finetune/results'/folder/name).read_text())]
    normal=REPORT/'all_normal.json'
    if normal.exists():sources += [('all_normal_'+r['arm'],r) for r in json.loads(normal.read_text())['rows']]
    rows=[]
    for name,r in sources:
        with np.load(Path(r['artifact'])/'physics.npz') as z:trace=z['samples']
        values=[]
        for start in range(0,len(trace),256):values.append(continuous_clearance(trace[start:start+256,1:4],PRESETS[r['preset']],track))
        margin=np.concatenate(values);minimum=float(margin.min())
        rows.append(dict(source=name,key=r['key'],condition=r['condition'],seed=r['seed'],preset=r['preset'],
            frozen_clean=r['clean_stable'],sampled_min_margin_cm=r['min_margin_cm'],continuous_min_clearance_cm=minimum,
            continuous_clean=bool(r['stable_success'] and minimum>=-1e-9 and r['initial_valid']),
            false_clean=bool(r['clean_stable'] and minimum<-1e-9)))
    summary={}
    for source in sorted(set(r['source'] for r in rows)):
        a=[r for r in rows if r['source']==source]
        summary[source]=dict(n=len(a),frozen_clean=sum(r['frozen_clean'] for r in a),continuous_clean=sum(r['continuous_clean'] for r in a),
                             false_clean=sum(r['false_clean'] for r in a))
    dump(REPORT/'continuous_clearance_audit.json',dict(source_sha256=sha(__file__),summary=summary,rows=rows,
        limitations='Continuous in vehicle footprint, sampled at unchanged physics times; not a continuous-time collision proof. Deep penetration values saturated; severe magnitude unchanged.'))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':check();audit()
