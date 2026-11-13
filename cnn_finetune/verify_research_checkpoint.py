"""Verify the published hyb4 evidence without the original D: artifact paths.

Checks hashes and recomputes footprint clearance on saved physics samples.
This is evidence verification, not a fresh rollout or neural training run.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from bench.vehicle import PRESETS
from bench.world import Track
from .goal_continuous_clearance import continuous_clearance

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'cnn_finetune/artifacts/hyb4_20261113'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest=json.loads((BUNDLE/'manifest.json').read_text(encoding='utf-8'))
    mapping={row['source_path'].replace('\\','/'):ROOT/row['repository_path'] for row in manifest['files']}
    for row in manifest['files']:
        p=ROOT/row['repository_path']
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],str(p)
    results={};protocol_hashes=0;track=Track()
    for arm,folder in [('oracle','hyb4_prerequisite_20261113'),('vision','hyb4_image_prerequisite_20261113')]:
        report=ROOT/'cnn_finetune/results'/folder
        protocol=json.loads((report/'protocol.json').read_text())
        for original,digest in protocol['hashes'].items():
            key=original.replace('\\','/')
            p=mapping[key]
            assert sha(p)==digest,str(p)
            if key.startswith('D:/Projects/AutoMobile/repo/'):
                current=ROOT/key.removeprefix('D:/Projects/AutoMobile/repo/')
                # Preserve frozen bytes in the bundle; tolerate only Git EOL conversion in checkout.
                assert current.read_bytes().replace(b'\r\n',b'\n')==p.read_bytes().replace(b'\r\n',b'\n'),str(current)
            protocol_hashes+=1
        rows=[]
        for path in sorted((BUNDLE/arm).glob('*/complete.json')):
            row=json.loads(path.read_text());z=np.load(path.parent/'physics.npz')
            clearance=continuous_clearance(np.vstack([z['initial'],z['samples'][:,1:4]]),PRESETS[row['preset']],track)
            np.testing.assert_allclose(clearance,z['continuous_clearance'],atol=1e-10,rtol=0)
            clean=bool(row['stable_success'] and row['initial_valid'] and clearance.min()>=-1e-9)
            assert clean==row['continuous_clean']
            assert abs(float(clearance.min())-row['continuous_min_clearance_cm'])<1e-10
            rows.append(row)
        expected=json.loads((report/'summary.json').read_text())
        result=dict(n=len(rows),clean=sum(r['continuous_clean'] for r in rows),stable=sum(r['stable_success'] for r in rows),
            wrong=sum(r['outcome']=='wrong_exit' for r in rows),severe=sum(r['outcome']=='severe_departure' for r in rows),
            pauses=sum(r['translation_pause_s']>0 for r in rows))
        assert all(expected[k]==v for k,v in result.items())
        assert len({(tuple(r['key']),r['seed'],r['preset']) for r in rows})==24
        results[arm]=result
    result=dict(bundle_files_verified=len(manifest['files']),frozen_protocol_hashes_verified=protocol_hashes,
        recomputed_saved_physics=True,current_protocol_inputs_match_except_line_endings=True,
        fresh_rollouts=False,new_training=False,results=results,
        verifier_sha256=sha(Path(__file__)))
    out=ROOT/'cnn_finetune/results/research_checkpoint_20261113';out.mkdir(exist_ok=True)
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
