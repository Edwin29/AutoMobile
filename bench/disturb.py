"""Camera / power disturbances, triggered near the first corner.

Definitions follow the v3-recovery field_stress study (commit 8b41dd9) so results
stay comparable: glare 4s, occlusion 3s, dropout 1.2s, blur+warm tint, a floor
seam missing from the map, power sag to 65% for 12s. Motion properties (turn
efficiency, drift, delay...) are NOT disturbances here: they belong to the
vehicle model and are set there.
"""
import cv2
import numpy as np

EFFECTS = {
    "glare": {"glare_s": 4.0},
    "occlusion": {"occlusion_s": 3.0},
    "dropout": {"dropout_s": 1.2},
    "blur_warm": {"blur": True},
    "seam": {"seam": True},
    "low_power": {"sag": 0.65, "sag_s": 12.0},
    # Strong and CONTINUING: events all lap long at random times (mean gap 5 s), random
    # place/strength/length, sometimes two at once; blur+warm tint the whole time.
    "recurring_strong": {"blur": True, "recurring": dict(mean_gap_s=5.0, start_s=3.0,
                                                          pool=["glare", "occlusion", "dropout", "low_power"],
                                                          dur_scale=[0.5, 1.5], second_prob=0.3)},
    # Realistic camera from the field logs (bench/realcam.py): sheen, creases, lens shading,
    # exposure lag, real background; 'harsh' adds shadows, walkers, long exposure, flare, noise.
    "realcam": {"realcam": "typical"},
    "realcam_harsh": {"realcam": "harsh"},
    **{f"realcam_{f}": {"realcam": f"harsh_{f}"} for f in ("light", "shadow", "exposure", "blur", "optics")},
}

BASE_S = {"glare": 4.0, "occlusion": 3.0, "dropout": 1.2, "low_power": 8.0}


def _schedule(spec, seed, horizon_s=400.0):
    rng = np.random.default_rng(seed * 7919 + 13)
    out, t = [], spec["start_s"]
    while t < horizon_s:
        kinds = [rng.choice(spec["pool"])]
        if rng.random() < spec["second_prob"]:
            kinds.append(rng.choice([k for k in spec["pool"] if k != kinds[0]]))
        for kind in kinds:
            dur = BASE_S[kind] * rng.uniform(*spec["dur_scale"])
            out.append(dict(kind=str(kind), t0=float(t), t1=float(t + dur),
                            glare_x=float(rng.uniform(150, 520)), glare_w=float(rng.uniform(0.8, 1.4)),
                            occl_x=int(rng.integers(0, 390)), occl_w=int(rng.integers(150, 300)),
                            sag=float(rng.uniform(0.5, 0.85))))
        t += float(rng.exponential(spec["mean_gap_s"])) + 0.5
    return out


def combine(names):
    p = {}
    for n in names:
        p.update(EFFECTS[n])
    return p


class Disturbances:
    def __init__(self, profile, direction, seed):
        self.p, self.direction, self.seed = dict(profile), direction, seed
        self.event_time = None
        self.schedule = _schedule(self.p["recurring"], seed) if self.p.get("recurring") else []

    def active(self, t, kind):
        return [e for e in self.schedule if e["kind"] == kind and e["t0"] <= t < e["t1"]]

    def arm(self, renderer):
        if self.p.get("seam"):
            y = -50.0 if self.direction == "ccw" else 50.0     # across the left road, absent from the map
            renderer.add_floor_line([-103.0, y], [-65.0, y])

    def update(self, t, progress, trigger_progress):
        if self.event_time is None and progress >= trigger_progress:
            self.event_time = t

    def age(self, t):
        return t - self.event_time if self.event_time is not None else -1e9

    def speed_scale(self, t):
        a = self.age(t)
        s = self.p.get("sag", 1.0) if 0 <= a < self.p.get("sag_s", 0.0) else 1.0
        for e in self.active(t, "low_power"):
            s *= e["sag"]
        return s

    def image(self, image, t, renderer, history, raw=False):
        p, a, flags = self.p, self.age(t), []
        if p.get("blur"):
            times = np.array([h[0] for h in history])
            frames = [image.astype(np.float32)]
            for lag in (0.0332565, 0.066513):              # 66.5ms field exposure
                i = max(0, int(np.searchsorted(times, t - lag, side="right") - 1))
                past = renderer.render_raw(history[i][1]) if raw else renderer.render(history[i][1])
                frames.append(past.astype(np.float32))
            img = np.mean(frames, axis=0) * np.array([.85, 1.02, 1.12], np.float32) * (1 + .10 * np.sin(t * 1.7 + self.seed))
            img = cv2.GaussianBlur(img, (7, 7), 1.4)
            rng = np.random.default_rng(self.seed * 1000003 + int(np.floor(t * 10)))
            image = np.clip(img + rng.normal(0, 6., img.shape), 0, 255).astype(np.uint8)
            flags.append("blur_warm")
        if 0 <= a < p.get("glare_s", 0.0):
            yy, xx = np.mgrid[:480, :640]
            jitter = (self.seed % 13 - 6) * 3
            alpha = .99 * np.exp(-.5 * (((xx - (435 + jitter)) / 57.6) ** 2 + ((yy - 336) / 110.4) ** 2))
            image = np.clip(image.astype(float) + (255. - image) * alpha[:, :, None], 0, 255).astype(np.uint8)
            flags.append("glare")
        if 0 <= a < p.get("occlusion_s", 0.0):
            x1, x2 = (0, 250) if self.direction == "ccw" else (390, 639)
            image = image.copy()
            cv2.rectangle(image, (x1, 160), (x2, 450), (70, 70, 70), -1)
            flags.append("occlusion")
        if 0 <= a < p.get("dropout_s", 0.0):
            image = np.full_like(image, 245)
            flags.append("dropout")
        # recurring events (random place / strength)
        for e in self.active(t, "glare"):
            yy, xx = np.mgrid[:480, :640]
            alpha = .99 * np.exp(-.5 * (((xx - e["glare_x"]) / (57.6 * e["glare_w"])) ** 2 + ((yy - 336) / 110.4) ** 2))
            image = np.clip(image.astype(float) + (255. - image) * alpha[:, :, None], 0, 255).astype(np.uint8)
            flags.append("glare*")
        for e in self.active(t, "occlusion"):
            image = image.copy()
            cv2.rectangle(image, (e["occl_x"], 160), (min(639, e["occl_x"] + e["occl_w"]), 450), (70, 70, 70), -1)
            flags.append("occlusion*")
        if self.active(t, "dropout"):
            image = np.full_like(image, 245)
            flags.append("dropout*")
        return image, flags
