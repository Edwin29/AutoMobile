"""Laptop-only replay of a copied image; no camera, SSH or GPIO calls."""
import hashlib
import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'reference'))
import lane_pipeline as L


def main():
    ref = ROOT / 'reference'
    img = cv2.imread(str(ref / 'cross_01.jpg'))
    with np.load(ref / 'calib_checkerboard.npz') as c:
        und, kn = L.undistort(img, c['K'], c['D'])
    with np.load(ref / 'ground.npz') as g:
        m, size, _ = L.bev_transform(dict(g), kn, img.shape[0])
    outputs = {}
    for turn in ('straight', 'left', 'right'):
        res, notes = L.analyze(und, m, size, None, turn, 'cross_v')
        outputs[turn] = {'notes': notes, 'result': None if res is None else {
            'effective_turn': res['turn'], 'road': res['road'],
            'gap_cm': res['gap'], 'curvature_per_cm': res['curv_pp'],
            'retained_edges': sorted(res['det']),
            'branch_cm': {k: L.to_cm(np.array([v]))[0].tolist() for k,v in res['branch_pts'].items()}}}
    report = {'scope': 'One historical image, not a crossing sequence or localization proof',
              'road_hint_assumption': 'cross_v',
              'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in ref.iterdir() if p.is_file()}, 'replay': outputs}
    (ROOT / 'evidence.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
