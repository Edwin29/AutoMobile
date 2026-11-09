"""Offline lens calibration. Input frames must share crop, resolution and rotation."""
import argparse
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np


def object_points():
    points = np.zeros((70, 3), np.float32)
    points[:, :2] = np.mgrid[0:10, 0:7].T.reshape(-1, 2)
    return points  # Unit squares: physical size is not needed for lens correction.


def fit(corners):
    return cv2.calibrateCamera([object_points() for _ in corners], corners,
                               (640, 480), None, None)


def errors(corners, matrix, distortion):
    values = []
    for points in corners:
        ok, r, t = cv2.solvePnP(object_points(), points, matrix, distortion)
        if not ok:
            values.append(float('inf'))
            continue
        prediction, _ = cv2.projectPoints(object_points(), r, t, matrix, distortion)
        values.append(float(np.sqrt(np.mean(np.sum((prediction-points)**2, axis=2)))))
    return np.array(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive')
    parser.add_argument('--track', default='track.jpg')
    parser.add_argument('--output', default='calibration')
    args = parser.parse_args()
    cv2.setNumThreads(2)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    detected, names, images = [], [], []
    with zipfile.ZipFile(args.archive) as archive:
        files = sorted(n for n in archive.namelist() if n.lower().endswith('.jpg'))
        # Temporal spacing limits near-duplicate frames; excludes nested duplicate ZIP.
        sampled = [files[i] for i in np.unique(np.linspace(0, len(files)-1, 100).astype(int))]
        for i, name in enumerate(sampled):
            im = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
            if im is None or im.shape[:2] != (480, 640):
                continue
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            ok, points = cv2.findChessboardCornersSB(gray, (10, 7), cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ok:
                detected.append(points.astype(np.float32)); names.append(name); images.append(im)
            print(f'{i+1}/{len(sampled)} detected={len(detected)}', flush=True)
    if len(detected) < 15:
        raise RuntimeError('Insufficient complete checkerboard views')
    np.savez(out/'detections.npz', corners=np.array(detected), names=np.array(names))
    # Hold out regularly spaced views before fitting; still correlated within one session.
    test_ids = list(range(0, len(detected), 5))
    train_ids = [i for i in range(len(detected)) if i not in test_ids]
    rms, k, d, _, _ = fit([detected[i] for i in train_ids])
    initial = errors([detected[i] for i in train_ids], k, d)
    med = np.median(initial)
    threshold = max(1.0, float(med + 3 * 1.4826*np.median(abs(initial-med))))
    kept = [i for i, e in zip(train_ids, initial) if e <= threshold]
    if len(kept) < 12:
        raise RuntimeError('Too few views after quality filtering')
    rms, k, d, _, _ = fit([detected[i] for i in kept])
    train_error = errors([detected[i] for i in kept], k, d)
    test_error = errors([detected[i] for i in test_ids], k, d)
    new_k, roi = cv2.getOptimalNewCameraMatrix(k, d, (640, 480), 0, (640, 480))
    mapx, mapy = cv2.initUndistortRectifyMap(k, d, None, new_k, (640, 480), cv2.CV_32FC1)
    np.savez(out/'camera_640x480_rot180.npz', camera_matrix=k, distortion=d,
             new_camera_matrix=new_k, map_x=mapx, map_y=mapy, image_size=[640,480])
    report = dict(status='provisional', model='OpenCV pinhole, 5 distortion coefficients',
                  inner_corners=[10,7], image_size=[640,480], rotation_degrees=180,
                  square_size='1 arbitrary unit; no metric distances', sampled=len(sampled),
                  detected=len(detected), training=len(kept), held_out=len(test_ids),
                  excluded=[names[i] for i in train_ids if i not in kept],
                  rms_px=rms, train_median_px=float(np.median(train_error)),
                  held_out_median_px=float(np.median(test_error)),
                  held_out_max_px=float(max(test_error)), camera_matrix=k.tolist(),
                  distortion=d.tolist(), valid_roi=[int(x) for x in roi],
                  limitations=['Held-out images come from the same capture session.',
                               'Board flatness and historical sensor crop are not independently verified.',
                               'Recalibrate after lens, focus, crop, resolution or orientation changes.'])
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for label, im in [('track', cv2.imread(args.track)), ('board', images[test_ids[len(test_ids)//2]])]:
        if im is None:
            continue
        corrected = cv2.remap(im, mapx, mapy, cv2.INTER_LINEAR)
        cv2.imwrite(str(out/f'{label}_corrected.jpg'), corrected)
        pair = np.hstack([im, corrected])
        cv2.putText(pair, 'ORIGINAL', (15,30), cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
        cv2.putText(pair, 'UNDISTORTED (alpha=0)', (655,30), cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
        cv2.imwrite(str(out/f'{label}_comparison.jpg'), pair)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
