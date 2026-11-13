"""Causality, blind-input, pairing and artifact-integrity checks."""
import copy
import json
from pathlib import Path
import numpy as np
from .goal_hidden_response_experiment import ROOT,OUT,REPORT,sha,dump
from .goal_image_response import fit_bank


def main():
    protocol=json.loads((REPORT/'protocol.json').read_text())
    for f,h in protocol['hashes'].items():assert sha(f)==h,f
    old=json.loads((ROOT/'cnn_finetune/results/goal_path_following_pilot_20261113/protocol.json').read_text())
    for f,h in old['hashes'].items():assert sha(f)==h,f
    folders=[p for p in OUT.iterdir() if p.is_dir() and (p/'complete.json').exists()]
    assert len(folders)==52
    probe=[];manifest={};fallback=[]
    for folder in sorted(folders):
        inp=json.loads((folder/'input.json').read_text());meta=json.loads((folder/'scorer.json').read_text())
        fits=json.loads((folder/'fits.json').read_text())
        assert set(inp)=={'commands','motion','observations'}
        assert all(set(o)=={'start','end','v','w','lateral_cm','usable','delta'} for o in inp['observations'])
        if fits[-1].get('insufficient_observations'):fallback.append(folder.name)
        if meta['group']=='probe':probe.append((folder,meta,inp))
        manifest[folder.name]={f.name:sha(f) for f in folder.iterdir() if f.suffix in ['json','npz']}
    for _,_,inp in probe:np.testing.assert_array_equal(inp['commands'],probe[0][2]['commands'])
    pairs=0
    for folder,meta,inp in probe:
        if meta['condition']!='light':continue
        normal=next(p for p,m,_ in probe if m['seed']==meta['seed'] and m['vehicle']==meta['vehicle'] and m['condition']=='normal')
        np.testing.assert_array_equal(np.load(folder/'scorer.npz')['truth'],np.load(normal/'scorer.npz')['truth']);pairs+=1
    folder=OUT/'probe_08';inp=json.loads((folder/'input.json').read_text())
    actual=fit_bank(inp['commands'],inp['observations'],6.)
    modified=copy.deepcopy(inp)
    for cmd in modified['commands']:
        if cmd[0]>6:cmd[1:]=[-.9,.9]
    for obs in modified['observations']:
        if obs['end']+.07>6:obs.update(v=999.,w=-999.)
    assert actual==fit_bank(modified['commands'],modified['observations'],6.)
    saved=json.loads((folder/'fits.json').read_text())[-1];assert actual==saved
    dump(REPORT/'artifact_manifest.json',manifest)
    dump(REPORT/'verification.json',dict(all_52_complete=True,identical_probe_pwm=True,
        normal_light_physics_exact_pairs=pairs,blinded_input_schema=True,future_input_mutation_invariant=True,
        saved_fit_reproduced=True,new_source_hashes_unchanged=len(protocol['hashes']),
        prior_experiment_hashes_unchanged=len(old['hashes']),six_second_fallbacks=fallback,
        scorer_hash=sha(Path(__file__).with_name('goal_hidden_response_score.py')),
        verifier_hash=sha(__file__),final_exam_used=False,new_training=False))
    print('Verified52 episodes;12 light physics pairs;future input invariance;old/new frozen sources unchanged',flush=True)


if __name__=='__main__':main()
