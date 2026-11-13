"""Offline truth scoring, kept outside the blinded image/PWM estimator."""
import json
from pathlib import Path
import numpy as np
from bench.world import wrap, axes
from .goal_image_response import predict, NOMINAL, fit_bank
from .goal_hidden_response_experiment import OUT, REPORT, ROOT, dump, sha


def true_values(truth,rows):
    t=truth[:,0]; poses=truth[:,1:4].copy();poses[:,2]=np.unwrap(poses[:,2]); out=[]
    for r in rows:
        a=max(0,r['start']-.028);b=max(0,r['end']-.028)
        p=np.array([np.interp(a,t,poses[:,j]) for j in range(3)])
        q=np.array([np.interp(b,t,poses[:,j]) for j in range(3)])
        d=q[2]-p[2];shift=(q[:2]-p[:2])@axes(p[2]).T
        forward=shift[0]*np.sin(d/2)+shift[1]*np.cos(d/2)
        out.append([forward/max(b-a,.001)/np.sinc(d/(2*np.pi)),d/max(b-a,.001)])
    return np.asarray(out)


def errors(pred,target):
    err=pred-target
    return dict(v_rmse_cm_s=float(np.sqrt(np.mean(err[:,0]**2))),
                w_rmse_rad_s=float(np.sqrt(np.mean(err[:,1]**2))),
                combined_rmse=float(np.sqrt(np.mean(err[:,0]**2+(13*err[:,1])**2))))


def turn_entry(meta, heading_threshold=False):
    if meta['group']!='natural':return None
    registry=json.loads((ROOT/'cnn_finetune/results/goal_clearance_paths_20261113/geometry.json').read_text())
    ref=next(r for r in registry if r['key']==meta['key'])
    z=np.load(ref['reference']);k=z['curvature'];h=np.unwrap(z['heading'])
    idx=int(np.flatnonzero(abs(h-h[0])>np.deg2rad(10) if heading_threshold else abs(k)>.005)[0])
    records=json.loads((Path(meta['source'])/'records.json').read_text())
    return next((r['done'] for r in records if r['index']>=idx),None)


def score_one(folder):
    inputs=json.loads((folder/'input.json').read_text());meta=json.loads((folder/'scorer.json').read_text())
    fits=json.loads((folder/'fits.json').read_text());complete=json.loads((folder/'complete.json').read_text())
    commands=np.asarray(inputs['commands']);obs=inputs['observations'];truth=np.load(folder/'scorer.npz')['truth']
    # Fixed regular interval grid is scored even if visual tracking failed.
    test=[dict(start=float(t),end=float(t+.2),usable=True) for t in np.arange(6.2,11.81,.2)]
    target=true_values(truth,test);nominal=errors(predict(commands,test,NOMINAL),target)
    fitted=errors(predict(commands,test,fits[-1]['best']),target)
    visual=errors(np.array([[r['v'],r['w']] for r in obs]),true_values(truth,obs)) if obs else None
    p=meta['parameters'];g=(p['cm_s_per_duty_left']+p['cm_s_per_duty_right'])/2
    actual_sum=p['actuation_delay_s']+p['time_constant_s']
    fitmeta=[]
    for f in fits:
        best=f['best'];effective=best['delay']+best['tau']
        fitmeta.append(dict(cutoff=f['cutoff'],n=f['n'],delay=best['delay'],tau=best['tau'],
                            gain=best['coefficients'][0],sum=effective,sum_error=effective-actual_sum,
                            gain_rel_error=best['coefficients'][0]/g-1,
                            plausible_sum=f.get('plausible_sum'),plausible_delay=f.get('plausible_delay'),
                            plausible_tau=f.get('plausible_tau'),insufficient=f.get('insufficient_observations',False)))
    # Named oracle diagnostic: replace only video-derived measurements with
    # exact pose increments on the same 0.2s schedule; never adopted as input.
    oracle_rows=[dict(start=float(t),end=float(t+.2),usable=True) for t in np.arange(0,5.81,.2)]
    oracle_target=true_values(truth,oracle_rows)
    for r,(v,w) in zip(oracle_rows,oracle_target):r.update(v=float(v),w=float(w))
    oracle_fit=fit_bank(commands,oracle_rows,6.)
    oracle_error=errors(predict(commands,test,oracle_fit['best']),target)
    masked=[dict(r) for r in obs]
    masked_target=true_values(truth,masked)
    for r,(v,w) in zip(masked,masked_target):r.update(v=float(v),w=float(w))
    masked_count=sum(r['end']+.070<=6 and r['usable'] and r['start']>=.2 for r in masked)
    masked_fit=fit_bank(commands,masked,6.) if masked_count>=8 else None
    masked_error=errors(predict(commands,test,masked_fit['best']),target) if masked_fit else None
    entry=turn_entry(meta);heading_entry=turn_entry(meta,True)
    return dict(name=folder.name,**meta,visual_frame_fraction=complete['visual_frame_fraction'],
                usable_intervals=complete['usable_intervals'],visual_measurement_errors=visual,
                nominal=nominal,fitted=fitted,oracle_observation_control=oracle_error,
                oracle_same_visual_mask_control=masked_error,
                improvement=1-fitted['combined_rmse']/max(nominal['combined_rmse'],1e-9),
                fits=fitmeta,turn_entry_s=entry,fit_before_turn=bool(entry is not None and 6<entry),
                posthoc_reference_10deg_s=heading_entry,
                oracle_best=oracle_fit['best'])


def summarize(rows):
    groups={}
    for group in ['natural','probe']:
        for condition in ['normal','light','shove']:
            for vehicle in ['all']+sorted(set(r['vehicle'] for r in rows if r['group']==group)):
                subset=[r for r in rows if r['group']==group and r['condition']==condition and (vehicle=='all' or r['vehicle']==vehicle)]
                if not subset:continue
                groups[f'{group}/{condition}/{vehicle}']=dict(n=len(subset),
                    median_nominal=float(np.median([r['nominal']['combined_rmse'] for r in subset])),
                    median_fitted=float(np.median([r['fitted']['combined_rmse'] for r in subset])),
                    median_improvement=float(np.median([r['improvement'] for r in subset])),
                    improved=sum(r['improvement']>0 for r in subset),
                    effective_sum_within_75ms=sum(abs(r['fits'][-1]['sum_error'])<=.07500001 for r in subset),
                    median_visual_frame_fraction=float(np.median([r['visual_frame_fraction'] for r in subset])),
                    median_usable_interval_fraction=float(np.median([r['usable_intervals']/60 for r in subset])),
                    median_oracle=float(np.median([r['oracle_observation_control']['combined_rmse'] for r in subset])),
                    before_turn=sum(r['fit_before_turn'] for r in subset))
    nuisance=[]
    for r in rows:
        if r['group']!='probe' or r['condition']=='normal':continue
        normal=next((n for n in rows if n['group']=='probe' and n['condition']=='normal' and n['vehicle']==r['vehicle'] and n['seed']==r['seed']),None)
        if normal:
            a,b=normal['fits'][-1],r['fits'][-1]
            nuisance.append(dict(name=r['name'],condition=r['condition'],vehicle=r['vehicle'],seed=r['seed'],
                lag_shift_s=b['sum']-a['sum'],gain_rel_shift=b['gain']/a['gain']-1,
                false_change=bool(abs(b['sum']-a['sum'])>.100001 or abs(b['gain']/a['gain']-1)>.15)))
    p=groups.get('probe/normal/all');nom=groups.get('probe/normal/nominal')
    gate=None
    if p and p['n']==12:
        gate=bool(p['improved']>=10 and p['median_improvement']>=.25 and nom['median_improvement']>=-.10
                  and p['median_usable_interval_fraction']>=.90 and p['effective_sum_within_75ms']>=10)
    return dict(groups=groups,nuisance=nuisance,normal_probe_gate=gate,
                final_exam_used=False,new_training=False,closed_loop_adaptation_tested=False)


if __name__=='__main__':
    rows=[]
    for folder in sorted(OUT.iterdir()):
        if folder.is_dir() and (folder/'complete.json').exists():
            row=score_one(folder);rows.append(row)
    dump(REPORT/'scores.json',rows);summary=summarize(rows);dump(REPORT/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
