"""Checks recorded commands, independently recomputed summaries and input provenance."""
import csv
import hashlib
import json
import numpy as np
from experiment import HERE, ROOT, OUT

records=json.loads((OUT/'simulation_summary.json').read_text())
assert len(records)==22, len(records)
for r in records:
    with (OUT/(r['name']+'.csv')).open() as f: rows=list(csv.DictReader(f))
    a={k:np.array([float(row[k]) for row in rows]) for k in
       ('t','true_x','true_y','est_x','est_y','true_margin_cm','yaw_error_deg','left','right')}
    error=np.hypot(a['true_x']-a['est_x'],a['true_y']-a['est_y'])
    np.testing.assert_allclose(error.max(),r['max_position_error_cm'],atol=1e-9)
    np.testing.assert_allclose(a['true_margin_cm'].min(),r['min_true_margin_cm'],atol=1e-9)
    np.testing.assert_allclose(np.abs(a['yaw_error_deg']).max(),r['max_yaw_error_deg'],atol=1e-9)
    np.testing.assert_allclose(a['t'][-1],r['end_s'],atol=1e-9)
    np.testing.assert_allclose(np.diff(a['t']),1/15,atol=1e-10)
    assert a['left'][-1]==0 and a['right'][-1]==0, r['name']
    assert r['reason']!='timeout',r['name']
    if r['scenario']=='clean':
        assert r['reason']=='lap complete' and r['max_position_error_cm']<.30

grid=json.loads((OUT/'real_grid.json').read_text())
assert len(grid)==4 and all(r['candidates']==5577 for r in grid)
assert all(r['prior_inliers']<.55 and r['prior_median_cm']<1.8 for r in grid)
assert all(r['passing']>0 for r in grid)
audit={r['name']:r for r in json.loads((OUT/'stop_audit.json').read_text())}
assert audit['ccw_prior_lateral_20']['best_candidate_position_error_cm']<.1
assert audit['ccw_forward_persistent_2']['best_candidate_pass']
assert audit['ccw_forward_persistent_2']['best_candidate_position_error_cm']>15
assert not audit['cw_seam_20']['corrupt_at_truth_pass']
assert audit['cw_seam_20']['best_candidate_pass']
provenance=json.loads((HERE/'provenance.json').read_text())
for path,digest in provenance['hashes'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
print(f'PASS: {len(records)} runs, four real searches, truth-only audits, {len(provenance["hashes"])} hashes')
