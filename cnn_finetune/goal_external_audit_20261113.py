"""Read-only artifact audit. No training, simulator rollout, or model imports."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'cnn_finetune/results/independent_design_audit_20261113'
MANIFEST = ROOT / 'cnn_finetune/results/goal_all_movement_data_20261113/data.json'


def first_array(path, name):
    # Read just the first row of a compressed NPY, without inflating the movie.
    with zipfile.ZipFile(path) as z, z.open(name + '.npy') as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        assert not fortran and not dtype.hasobject
        count = int(np.prod(shape[1:]))
        raw = f.read(count * dtype.itemsize)
        assert len(raw) == count * dtype.itemsize
        return np.frombuffer(raw, dtype=dtype).reshape(shape[1:]).copy()


def main():
    rows = json.loads(MANIFEST.read_text())['rows']
    groups = defaultdict(list)
    totals = dict(episodes=0, frames=0, wheel_targets=0, below_allocator_floor=0,
                  turn_frames=0, change_frames=0, changes_outside_turn_weight=0)
    for row in rows:
        if not (row['eligible'] and row['translation_pause_s'] == 0):
            continue
        path = Path(row['artifact']) / 'sequence.npz'
        with np.load(path) as z:
            labels = z['label']
            aux = z['aux']
            assert z['label_valid'].all() and z['supervise'].all()
        turning = abs(labels[:, 0] - labels[:, 1]) > .15
        # Audit threshold only, not a proposed training threshold.
        changes = np.r_[False, np.max(abs(np.diff(labels, axis=0)), axis=1) > .02]
        totals['episodes'] += 1
        totals['frames'] += len(labels)
        totals['wheel_targets'] += labels.size
        totals['below_allocator_floor'] += int(((abs(labels) > 0) & (abs(labels) < .36)).sum())
        totals['turn_frames'] += int(turning.sum())
        totals['change_frames'] += int(changes.sum())
        totals['changes_outside_turn_weight'] += int((changes & ~turning).sum())
        if row['condition'] == 'normal':
            image = first_array(path, 'x')
            feature = first_array(path, 'f')
            groups[(tuple(row['key']), row['seed'])].append(dict(
                preset=row['preset'], image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                feature=feature, aux=aux[0], label=labels[0], artifact=str(path)))
    exact = []; nonexact = []
    for (key, seed), items in groups.items():
        assert len(items) == 3
        same_image = len({x['image_sha256'] for x in items}) == 1
        same_input = same_image and all(np.array_equal(items[0]['feature'], x['feature']) and
                     np.array_equal(items[0]['aux'], x['aux']) for x in items[1:])
        labels = np.array([x['label'] for x in items])
        row = dict(key=key, seed=seed, identical_first_actor_input=bool(same_input),
                   identical_first_image=bool(same_image),
                   max_target_range=float(np.ptp(labels, axis=0).max()),
                   labels={x['preset']:x['label'].tolist() for x in items},
                   artifacts=[x['artifact'] for x in items])
        (exact if same_input else nonexact).append(row)
    result = dict(manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  training_updates=0, simulator_rollouts=0, totals=totals,
                  first_frame_groups=len(groups), exact_input_groups=len(exact),
                  conflicting_exact_input_groups=sum(x['max_target_range']>1e-6 for x in exact),
                  max_conflict=max((x['max_target_range'] for x in exact), default=None),
                  exact_groups=exact, nonexact_groups=nonexact,
                  scope='First-frame target identifiability only; does not establish failure cause or impossibility of safe control.')
    OUT.mkdir(exist_ok=True)
    (OUT/'artifact_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['exact_groups','nonexact_groups']}, indent=2))
    print('Largest exact-input conflicts:', json.dumps(sorted(exact, key=lambda x:x['max_target_range'], reverse=True)[:2]))


if __name__ == '__main__':
    main()
