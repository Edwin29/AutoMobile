"""Image-only planar odometry and PWM response fitting. No simulator imports.

Camera calibration is fixed input. No map, pose, vehicle label, or physical
parameter is accepted by the observer or fitter.
"""
import cv2
import numpy as np


class FloorMotion:
    def __init__(self, calibration):
        self.c = calibration
        self.previous = None
        w, h = self.c['size']
        y, x = np.mgrid[:h, :w]
        ground = self.project(np.c_[x.ravel(), y.ravel()])
        self.mask = (np.isfinite(ground).all(1) & (abs(ground[:, 0]) < 36)
                     & (ground[:, 1] > 15) & (ground[:, 1] < 65)).reshape(h, w).astype('uint8') * 255
        self.mask[-32:] = 0

    def project(self, pixels):
        c = self.c
        rect = cv2.fisheye.undistortPoints(np.asarray(pixels, float).reshape(-1, 1, 2),
                                         c['K'], c['D'], R=np.eye(3), P=c['rect_K'])
        bev = cv2.perspectiveTransform(rect, c['M']).reshape(-1, 2)
        return np.c_[bev[:, 0] / c['ppcm'] + c['lat_min'],
                     c['fwd_max'] - bev[:, 1] / c['ppcm'] + c['lens_ahead']]

    def observe(self, image, capture):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        old = self.previous
        self.previous = (gray, float(capture))
        if old is None:
            return None
        a, ta = old
        row = dict(start=ta, end=float(capture), valid=False, n=0)
        pts = cv2.goodFeaturesToTrack(a, 220, .015, 7, mask=self.mask, blockSize=7)
        if pts is None or len(pts) < 8:
            return row
        q, ok, _ = cv2.calcOpticalFlowPyrLK(a, gray, pts, None, winSize=(25, 25), maxLevel=3)
        back, ok2, _ = cv2.calcOpticalFlowPyrLK(gray, a, q, None, winSize=(25, 25), maxLevel=3)
        keep = (ok.ravel() > 0) & (ok2.ravel() > 0) & (np.linalg.norm(back[:, 0] - pts[:, 0], axis=1) < .7)
        old_g, new_g = self.project(pts[keep]), self.project(q[keep])
        if len(old_g) < 8:
            return row
        # new floor coordinates -> old floor coordinates; translation is the
        # new axle position expressed in the previous body frame.
        mat, inliers = cv2.estimateAffinePartial2D(new_g, old_g, method=cv2.RANSAC,
                                                  ransacReprojThreshold=.25, maxIters=1000,
                                                  confidence=.995, refineIters=10)
        if mat is None:
            return row
        good = inliers.ravel() > 0
        if good.sum() < 8:
            return row
        p, qg = new_g[good], old_g[good]
        u, _, vt = np.linalg.svd((p - p.mean(0)).T @ (qg - qg.mean(0)))
        rot = u @ vt
        if np.linalg.det(rot) < 0:
            u[:, -1] *= -1
            rot = u @ vt
        shift = qg.mean(0) - p.mean(0) @ rot
        theta = float(np.arctan2(rot[1, 0], rot[0, 0]))
        error = np.linalg.norm(p @ rot + shift - qg, axis=1)
        scale = float(np.hypot(mat[0, 0], mat[0, 1]))
        row.update(valid=bool(good.mean() >= .5 and np.median(error) < .15 and abs(scale - 1) < .035),
                   delta=[float(shift[0]), float(shift[1]), theta], n=int(good.sum()),
                   inlier_fraction=float(good.mean()), residual_cm=float(np.median(error)), scale=scale)
        return row


def aggregate(rows, block=3):
    out = []
    for i in range(0, len(rows) - block + 1, block):
        group = rows[i:i + block]
        if not all(r['valid'] for r in group):
            continue
        p = np.zeros(3)
        for r in group:
            dx, dy, da = r['delta']
            c, s = np.cos(p[2]), np.sin(p[2])
            p[:2] += [c * dx + s * dy, -s * dx + c * dy]
            p[2] += da
        duration = group[-1]['end'] - group[0]['start']
        forward = p[0] * np.sin(p[2] / 2) + p[1] * np.cos(p[2] / 2)
        lateral = p[0] * np.cos(p[2] / 2) - p[1] * np.sin(p[2] / 2)
        # Reject lateral impulses without consulting known event times.
        out.append(dict(start=group[0]['start'], end=group[-1]['end'],
                        v=float(forward / duration / np.sinc(p[2] / (2 * np.pi))),
                        w=float(p[2] / duration), lateral_cm=float(lateral),
                        usable=bool(abs(lateral) < .2), delta=p.tolist()))
    return out


def response_basis(commands, end, delay, tau, dt=.005):
    ts = np.arange(0, end + dt * 1.01, dt)
    commands = np.asarray(commands, float)
    idx = np.searchsorted(commands[:, 0], ts[:-1] - delay + 1e-10, side='right') - 1
    pwm = commands[np.maximum(idx, 0), 1:].copy()
    pwm[idx < 0] = 0
    pwm[abs(pwm) < .30] = 0
    state = np.zeros(2)
    integ = np.zeros((len(ts), 2))
    decay = np.exp(-dt / tau) if tau > 0 else 0
    for i, target in enumerate(pwm):
        # Exact integrated first-order hold response, independent of vehicle code.
        area = target * dt + (state - target) * tau * (1 - decay)
        state = target + (state - target) * decay
        integ[i + 1] = integ[i] + area
    return ts, integ


def basis_at(ts, integ, rows):
    # Camera exposure spans the past; use a declared 28ms mean midpoint.
    a = np.maximum(0, np.array([r['start'] for r in rows]) - .028)
    b = np.maximum(0, np.array([r['end'] for r in rows]) - .028)
    d = np.maximum(b - a, .001)
    return np.column_stack([(np.interp(b, ts, integ[:, j]) - np.interp(a, ts, integ[:, j])) / d for j in range(2)])


def fit_bank(commands, rows, cutoff):
    commands = np.asarray(commands)
    commands = commands[commands[:, 0] <= cutoff]
    fit = [r for r in rows if r['end'] + .070 <= cutoff and r['usable'] and r['start'] >= .2]
    assert len(fit) >= 8
    target = np.array([[r['v'], 13 * r['w']] for r in fit])
    prior = np.array([16.8, 0, 16.8 * .6 / 13, 0.])
    candidates = []
    for delay in np.arange(0, .201, .025):
        for tau in np.arange(0, .351, .05):
            ts, integ = response_basis(commands, cutoff, delay, tau)
            b = basis_at(ts, integ, fit)
            x = np.zeros((len(fit), 2, 4))
            x[:, 0, 0] = b.mean(1)
            x[:, 0, 1] = (b[:, 0] - b[:, 1]) / 2
            x[:, 1, 2] = 13 * (b[:, 0] - b[:, 1])
            x[:, 1, 3] = 13 * (b[:, 0] + b[:, 1])
            flat, y = x.reshape(-1, 4), target.ravel()
            reg = np.diag([.03, .3, 1., 3.])
            weight = np.ones(len(y))
            for _ in range(4):
                coef = np.linalg.lstsq(np.vstack([flat * np.sqrt(weight[:, None]), reg]),
                                      np.r_[y * np.sqrt(weight), reg @ prior], rcond=None)[0]
                coef = np.clip(coef, [10, -3, .35, -.15], [22, 3, 1.5, .15])
                residual = flat @ coef - y
                weight = np.minimum(1., .5 / np.maximum(abs(residual), 1e-8))
            err = abs(residual)
            loss = float(np.mean(np.where(err < .5, .5 * err**2, .5 * (err - .25))))
            candidates.append(dict(delay=float(delay), tau=float(tau), coefficients=coef.tolist(), loss=loss))
    candidates.sort(key=lambda r: r['loss'])
    best = candidates[0]
    plausible = [r for r in candidates if r['loss'] <= best['loss'] * 1.2 + .002]
    return dict(cutoff=cutoff, n=len(fit), best=best,
                plausible_delay=[min(r['delay'] for r in plausible), max(r['delay'] for r in plausible)],
                plausible_tau=[min(r['tau'] for r in plausible), max(r['tau'] for r in plausible)],
                plausible_sum=[min(r['delay']+r['tau'] for r in plausible), max(r['delay']+r['tau'] for r in plausible)],
                candidates=candidates)


def predict(commands, rows, model):
    ts, integ = response_basis(commands, max(r['end'] for r in rows), model['delay'], model['tau'])
    b = basis_at(ts, integ, rows)
    gain, asym, turn, bias = model['coefficients']
    return np.c_[b.mean(1) * gain + (b[:, 0] - b[:, 1]) * asym / 2,
                 (b[:, 0] - b[:, 1]) * turn + (b[:, 0] + b[:, 1]) * bias]


NOMINAL = dict(delay=.05, tau=.10, coefficients=[16.8, 0., 16.8*.6/13, 0.])
