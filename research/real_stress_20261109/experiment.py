"""Local-only controlled stress tests. No GPIO imports. Truth is grading/rendering only.

python research/real_stress_20261109/experiment.py grid
python research/real_stress_20261109/experiment.py sim --direction ccw
"""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORE = ROOT / 'map_localization/outer_driver'
sys.path.insert(0, str(CORE))
from controller import CENTER_AHEAD, LENS_AHEAD, axes
from feedback_driver import Feedback, FixedGain
from validate_sim import Plant, DT
from calibrated_lap import calibrate

cv2.setNumThreads(1)
OUT = HERE / 'results'
OUT.mkdir(exist_ok=True)


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')


def driver_for(direction):
    d = Feedback(direction)
    d.color_value_min = 135
    d.rim = np.load(HERE / 'photo_plus5mm_rim.npy')
    d.yaw_tracker = FixedGain()
    d.right_mix = d.yaw_tracker.right = .6217314378064263
    d.record_edges = True
    return d


def candidate_grid(d, points, prior):
    # Offsets are relative to the PRIOR vehicle axes, not world axes.
    offsets = np.array([(lat, lon, yaw) for lon in np.arange(-8, 8.01, .5)
                        for lat in np.arange(-3, 3.01, .5) for yaw in range(-6, 7)], float)
    poses = np.repeat(prior[None], len(offsets), axis=0)
    poses[:, :2] += offsets[:, :2] @ axes(prior[2])
    poses[:, 2] += np.deg2rad(offsets[:, 2])
    results = []
    for chunk in np.array_split(poses, math.ceil(len(poses) / 256)):
        s, c = np.sin(chunk[:, 2]), np.cos(chunk[:, 2])
        rot = np.stack([c, -s, s, c], axis=1).reshape(-1, 2, 2)
        xy = np.einsum('pj,bjk->bpk', points, rot) + chunk[:, None, :2]
        r = np.abs(d.sample(d.sdf, xy.reshape(-1, 2))).reshape(len(chunk), -1)
        # Truncated square loss uses ALL points; outliers remain at a fixed penalty.
        results.append(np.c_[np.mean(np.minimum(r, 5.)**2, axis=1),
                                np.median(r, axis=1), np.mean(r < 2.5, axis=1)])
    scores = np.vstack(results)
    good = (scores[:, 1] <= 1.8) & (scores[:, 2] >= .55)
    best = int(np.argmin(scores[:, 0]))
    passing = np.flatnonzero(good)
    constrained = int(passing[np.argmin(scores[passing, 0])]) if len(passing) else None
    return offsets, scores, good, best, constrained


def real_grid():
    d = driver_for('ccw')
    records = []
    for folder in sorted((CORE / 'field_runs').glob('*_edges')):
        path = folder / '23.npz'
        if not path.exists():
            continue
        z = np.load(path)
        points, prior = z['points'], z['prior']
        # Check archived residuals, so a different map/version cannot silently pass.
        residual = d.sample(d.sdf, points @ axes(prior[2]) + prior[:2])
        np.testing.assert_allclose(residual, z['residual_cm'], atol=1e-5)
        start = time.perf_counter()
        offsets, scores, passed, best, constrained = candidate_grid(d, points, prior)
        row = dict(run=folder.name, n_points=len(points), candidates=len(passed),
                   prior_median_cm=float(np.median(abs(residual))),
                   prior_inliers=float(np.mean(abs(residual) < 2.5)),
                   passing=int(passed.sum()), elapsed_s=time.perf_counter()-start,
                   best_offset_lat_lon_yaw=offsets[best].tolist(),
                   best_loss_median_inliers=scores[best].tolist(), best_pass=bool(passed[best]),
                   best_passing_offset=None if constrained is None else offsets[constrained].tolist(),
                   passing_longitudinal_range=None if not passed.any() else
                       [float(offsets[passed, 1].min()), float(offsets[passed, 1].max())],
                   input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        np.savetxt(OUT / (folder.name + '_grid.csv'), np.c_[offsets, scores, passed],
                   delimiter=',', fmt='%.7f', comments='',
                   header='lateral_cm,longitudinal_cm,yaw_deg,capped_mse,median_cm,inliers,pass')
        records.append(row)
        print(json.dumps(row), flush=True)
    save(OUT / 'real_grid.json', records)


SCENARIOS = {
    'clean': {},
    'response_only': dict(motor=True),
    'exposure_66ms': dict(exposure=.066513),
    'glare': dict(glare=True),
    'shadow': dict(shadow=True),
    'seam': dict(seam=True),
    'object': dict(object=True),
    'combined': dict(motor=True, exposure=.066513, glare=True, seam=True, object=True),
    'prior_error': dict(prior_offset=True),
    'prior_lateral': dict(prior_lateral=True),
    'stall_combined': dict(stall=True, exposure=.066513, glare=True),
    'forward_persistent': dict(persistent_speed=True),
}


def pose_of(p):
    return np.r_[p.pos-CENTER_AHEAD*axes(p.yaw)[1], p.yaw]


def render_at(p, position, yaw):
    camera = position + (LENS_AHEAD-CENTER_AHEAD)*axes(yaw)[1]
    return p.world.render((*camera, yaw))


def exposure_frame(p, previous_position, previous_yaw, current, seconds):
    # Uniform exposure ending at current pose, 3 samples (including current).
    images = [current.astype(np.float32)]
    for frac in (.5*seconds/DT, seconds/DT):
        images.append(render_at(p, p.pos*(1-frac)+previous_position*frac,
                                p.yaw*(1-frac)+previous_yaw*frac).astype(np.float32))
    return np.mean(images, axis=0).astype(np.uint8)


def corrupt_image(frame, fault, strength):
    # Camera-space proxies, not measured 3-D object meshes or a fitted BRDF.
    h, w = frame.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    out = frame.astype(np.float32)
    if fault.get('glare'):
        a = .97*strength*np.exp(-.5*(((xx-.63*w)/(.19*w))**2 + ((yy-.65*h)/(.24*h))**2))
        out = out*(1-a[..., None])+np.array([230., 245., 255.])*a[..., None]
    if fault.get('shadow'):
        # Shadows on the green field can turn it into a V<130 road candidate.
        a = .62*strength*np.exp(-.5*(((xx-.28*w)/(.22*w))**2+((yy-.65*h)/(.32*h))**2))
        out *= 1-a[..., None]
    if fault.get('object'):
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, np.array([[355, 280], [435, 285], [470, 410], [350, 410]]), 255)
        a = mask[..., None]/255.*strength
        # Green/yellow sign/stand proxy: creates a non-floor edge projected onto floor.
        out = out*(1-a)+np.array([50., 170., 110.])*a
    return np.clip(out, 0, 255).astype(np.uint8)


def add_world_seam(p):
    world = p.world
    n = world.img.shape[0]
    c = (np.arange(n)+.5)/5.-125.
    x, y = np.meshgrid(c, -c)
    # A dark strip across both road and green islands, 1.0cm wide, 20cm ahead.
    center = p.pos+20*axes(p.yaw)[1]
    along = (x-center[0])*math.sin(p.yaw)+(y-center[1])*math.cos(p.yaw)
    mask = abs(along) < .5
    world.img[mask] = (32, 38, 30)


def diagnostics(d, p):
    obs = getattr(d, 'edge_observation', None)
    if obs is None:
        return dict(points=0, inliers=None, median=None, true_inliers=None, true_median=None)
    points, residual, prior = obs
    true = pose_of(p)
    true_residual = d.sample(d.sdf, points@axes(true[2])+true[:2])
    return dict(points=len(points), inliers=float(np.mean(abs(residual)<2.5)),
                median=float(np.median(abs(residual))),
                true_inliers=float(np.mean(abs(true_residual)<2.5)),
                true_median=float(np.median(abs(true_residual))))


def simulation(direction, scenario, onset, limit=210., video=False):
    fault = SCENARIOS[scenario]
    d = driver_for(direction)
    p = Plant(direction, pivot_behind_center_cm=CENTER_AHEAD,
              left_yaw_response=calibrate()['left_gain'], right_yaw_response=.6317018569280645)
    rows, min_margin, max_error, max_yaw_error = [], float('inf'), 0., 0.
    writer = None
    name = f'{direction}_{scenario}_{onset:g}'
    trace = []
    seam_added = False
    started = time.perf_counter()
    last_image = None
    for tick in range(round(limit/DT)):
        t = (tick+1)*DT
        active = onset <= t < onset+4
        if fault.get('motor'):
            # Measured response range; speed and drift amplitudes are sensitivity assumptions.
            p.left_yaw_response = .64454 if active else calibrate_cached
            p.forward_scale = 1.27 if active else 1.
            p.drift_deg_s_at_30 = -1. if active else 0.
        if fault.get('persistent_speed'):
            p.forward_scale = 1.27 if t >= onset else 1.
        if fault.get('prior_offset') and tick == math.ceil(onset/DT)-1:
            # Explicit estimator-fault control, NOT a claim about measured real slip.
            d.pose[:2] += np.array([2., 6.])@axes(d.pose[2])
            d.pose[2] += math.radians(-5.)
        if fault.get('prior_lateral') and tick == math.ceil(onset/DT)-1:
            d.pose[:2] += 3.*axes(d.pose[2])[0]
            d.pose[2] += math.radians(6.)
        if fault.get('seam') and t >= onset and not seam_added:
            add_world_seam(p)
            seam_added = True
        previous_position, previous_yaw = p.pos.copy(), p.yaw
        applied_command = p.out
        if fault.get('stall') and onset <= t < onset+2:
            # Side-speed proxy, NOT a calibrated four-wheel tire/torque model.
            p.out = (applied_command[0]*.5, applied_command[1])
        frame = p.capture_array()
        p.out = applied_command
        if active and fault.get('exposure'):
            frame = exposure_frame(p, previous_position, previous_yaw, frame, fault['exposure'])
        if active:
            # Increase over 0.5 seconds rather than an unrelated total-white flash.
            frame = corrupt_image(frame, fault, min(1., (t-onset+DT)/.5))
        command = d.step(frame, DT, p.out, rectified=True)
        p.set(*command)
        true = pose_of(p)
        err = d.pose[:2]-true[:2]
        lat, lon = err@axes(true[2]).T
        yawerr = math.degrees(math.remainder(d.pose[2]-true[2], 2*math.pi))
        rimworld = d.rim@axes(true[2])+true[:2]
        # Same photograph footprint for predicted and true margins, not old rectangular Plant rim.
        margin = float(d.sample(d.sdf, rimworld).min())
        min_margin = min(min_margin, margin)
        max_error = max(max_error, float(np.linalg.norm(err)))
        max_yaw_error = max(max_yaw_error, abs(yawerr))
        diag = diagnostics(d, p)
        row = dict(t=t, active=int(active), true_x=true[0], true_y=true[1], true_yaw=true[2],
                   est_x=d.pose[0], est_y=d.pose[1], est_yaw=d.pose[2],
                   lateral_error_cm=lat, longitudinal_error_cm=lon, yaw_error_deg=yawerr,
                   true_margin_cm=margin, forecast_cm=d.margin, left=command[0], right=command[1],
                   left_mix=d.left_mix, right_mix=d.right_mix, **diag)
        rows.append(row)
        if ((onset-2 <= t <= onset+10 and tick%3==0) or d.reason):
            trace.append((frame.copy(), dict(row), d.pose.copy(), true.copy(), d.reason))
        if d.reason:
            assert not np.any(command)
            break
    with (OUT/(name+'.csv')).open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    result = dict(name=name, direction=direction, scenario=scenario, onset=onset,
                  reason=d.reason or 'timeout', end_s=t, simulated_laps=p.travel/(2*np.pi),
                  estimated_laps=d.distance/d.arc[-1], min_true_margin_cm=min_margin,
                  max_position_error_cm=max_error, max_yaw_error_deg=max_yaw_error,
                  final_longitudinal_error_cm=float(lon), final_lateral_error_cm=float(lat),
                  wall_seconds=time.perf_counter()-started, final=diag,
                  config=fault, feedback_updates='output only; prediction gain fixed')
    if d.reason and d.reason != 'lap complete' and d.edge_observation is not None:
        pts, _, prior = d.edge_observation
        offsets, scores, passed, best, constrained = candidate_grid(d, pts, prior)
        result['stop_grid'] = dict(passing=int(passed.sum()), total=len(passed),
                                  best_offset=offsets[best].tolist(), best_scores=scores[best].tolist(),
                                  best_pass=bool(passed[best]))
        np.savez_compressed(OUT/(name+'_stop.npz'), points=pts, prior=prior, true_pose=true)
    save(OUT/(name+'.json'), result)
    if trace:
        cv2.imwrite(str(OUT/(name+'_last.jpg')), trace[-1][0])
    if video and trace:
        render_video(name, trace, p.world.img, d)
    print(json.dumps(result), flush=True)
    return result


def render_video(name, trace, world_img, d):
    path = OUT/(name+'.mp4')
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 5, (1000, 600))
    for frame, row, pose, true, reason in trace:
        out = np.full((600, 1000, 3), 24, np.uint8)
        out[:480, :640] = frame
        mapimage = cv2.resize(world_img, (360, 360))
        for state, color in [(true, (70, 230, 70)), (pose, (230, 50, 240))]:
            xy = d.rim@axes(state[2])+state[:2]
            pix = np.c_[(xy[:, 0]+125)*360/250, (125-xy[:, 1])*360/250].astype(np.int32)
            cv2.polylines(mapimage, [pix], True, color, 2)
        out[60:420, 640:] = mapimage
        inliers = '--' if row['inliers'] is None else f'{100*row["inliers"]:.1f}%'
        median = '--' if row['median'] is None else f'{row["median"]:.2f}cm'
        lines = [name, f't={row["t"]:.2f}s  active={int(row["active"])}  {reason or "RUN"}',
                 f'error along={row["longitudinal_error_cm"]:.2f}cm lat={row["lateral_error_cm"]:.2f}cm yaw={row["yaw_error_deg"]:.2f}deg',
                 f'points={int(row["points"])} inliers={inliers} (gate 55%) median={median}',
                 f'true margin={row["true_margin_cm"]:.2f}cm | GREEN=true MAGENTA=estimate']
        for i, line in enumerate(lines):
            cv2.putText(out, line, (8, 498+22*i), cv2.FONT_HERSHEY_SIMPLEX, .55, (230,230,230), 1, cv2.LINE_AA)
        w.write(out)
    for _ in range(10): w.write(out)
    w.release()


calibrate_cached = calibrate()['left_gain']

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['grid', 'sim'])
    ap.add_argument('--direction', choices=['ccw', 'cw'], default='ccw')
    ap.add_argument('--scenario', choices=list(SCENARIOS)+['all'], default='all')
    ap.add_argument('--onset', type=float, default=20.)
    ap.add_argument('--limit', type=float, default=210.)
    ap.add_argument('--video', action='store_true')
    args = ap.parse_args()
    if args.mode == 'grid': real_grid()
    else:
        for scenario in SCENARIOS if args.scenario == 'all' else [args.scenario]:
            simulation(args.direction, scenario, args.onset, args.limit, args.video)
