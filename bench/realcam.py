"""Realistic camera, built from what the field logs show, rendered with torch (GPU if present).

What the real Pi frames show (bench/world/realcam/stats.json, ~2,500 frames, 10/30-11/10):
  * the glossy black mat reflects ceiling lights / windows: bright sheen that moves with the
    car's HEADING (the lights are far away), not with its position
  * mat creases shine at some viewing angles
  * soft focus, magenta lens shading in the corners
  * exposure / white balance take time to settle (first frames dark magenta)
  * people, chairs and other cars above the horizon
Added on top for the harsh profile: shadows of people/furniture on the floor (static and
walking), the car's own shadow, long exposure (motion blur), slow/hunting auto-exposure,
haze, sensor noise in the dark, lens flare.

Everything is applied to the RAW fisheye frame; drivers that use the rectified view get it
undistorted from the corrupted raw frame (exactly what happens on the car).
All drivers see the same camera for the same run seed.
"""
import math
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "world" / "realcam"

# per-run values are drawn uniformly from these ranges (seeded)
PROFILES = {
    # what the logs show, within their p5..p95
    "typical": dict(
        exposure_s=(0.045, 0.066), sub_samples=3, focus_sigma=(1.0, 1.5), haze=(0.0, 0.06),
        target_rel=(0.9, 1.1), ae_tau_s=(0.15, 0.3), ae_hunt=0.0, start_gain=(0.9, 1.1),
        awb_drift=0.03, start_magenta=0.0,
        vignette=(0.08, 0.15), magenta_corner=(0.10, 0.18),   # real lower half is nearly flat
        sheen_lights=2, sheen_amp=(40, 90), sheen_width=(0.15, 0.3), sheen_dist=(25, 55),
        creases=4, crease_amp=(20, 45),
        static_shadows=0, shadow_dark=(0.55, 0.8), walkers=0, own_shadow=(0.85, 0.95),
        read_noise=(1.5, 2.5), shot_noise=0.4, flare=0.0, background=True,
    ),
    # everything at once, beyond what the logs show
    "harsh": dict(
        exposure_s=(0.07, 0.11), sub_samples=4, focus_sigma=(1.6, 2.4), haze=(0.08, 0.18),
        target_rel=(0.65, 1.35), ae_tau_s=(0.4, 0.8), ae_hunt=0.25, start_gain=(0.2, 0.35),
        awb_drift=0.08, start_magenta=0.6,
        vignette=(0.25, 0.35), magenta_corner=(0.18, 0.3),
        sheen_lights=4, sheen_amp=(80, 170), sheen_width=(0.12, 0.35), sheen_dist=(20, 60),
        creases=8, crease_amp=(35, 80),
        static_shadows=10, shadow_dark=(0.3, 0.6), walkers=3, own_shadow=(0.6, 0.75),
        read_noise=(3.0, 5.0), shot_noise=1.0, flare=1.0, background=True,
    ),
}

# one harsh factor at a time on top of 'typical' (which factor breaks which driver)
FACTORS = {
    "light": ("sheen_lights", "sheen_amp", "sheen_width", "sheen_dist", "creases", "crease_amp", "flare"),
    "shadow": ("static_shadows", "shadow_dark", "walkers", "own_shadow"),
    "exposure": ("target_rel", "ae_tau_s", "ae_hunt", "start_gain", "awb_drift", "start_magenta"),
    "blur": ("exposure_s", "sub_samples", "focus_sigma"),
    "optics": ("haze", "vignette", "magenta_corner", "read_noise", "shot_noise"),
}
for _f, _keys in FACTORS.items():
    PROFILES["harsh_" + _f] = dict(PROFILES["typical"], **{k: PROFILES["harsh"][k] for k in _keys})

SCALE = 2                                                  # composition at 320x240
ROAD_SIM, GREEN_SIM = (47, 60, 50), (95, 215, 125)
# measured on the real mat: median of road / green pixels, 233 on-track Pi frames (HSV V 74 / 202)
ROAD_REAL, GREEN_REAL = (61, 74, 43), (94, 202, 84)
FLOOR_REAL = (165, 172, 170)                               # floor beyond the mat


def _torch():
    import torch
    return torch


class RealCam:
    def __init__(self, renderer, camera, profile, seed, device=None):
        torch = _torch()
        self.t = torch
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)               # many runs share the CPU
        self.rng = np.random.default_rng(seed * 104729 + 7)
        self.gen = torch.Generator(device=self.dev).manual_seed(seed * 31 + 1)
        P = PROFILES[profile] if isinstance(profile, str) else profile
        u = lambda r: float(self.rng.uniform(*r)) if isinstance(r, tuple) else r
        self.cfg = {k: u(v) for k, v in P.items()}
        self.profile = profile
        self.renderer, self.camera = renderer, camera
        self.W, self.H = camera.size
        # the scene is composed at half resolution (4x less GPU work) and enlarged before
        # sensor noise; the soft focus of the real lens hides the difference
        self.S = SCALE
        self.w, self.h = self.W // SCALE, self.H // SCALE
        self._texture(renderer)
        self._geometry(camera)
        self._lights()
        self._walkers()
        self._background()
        self.gain = self.cfg["start_gain"]
        self.awb = np.array([1.0, 1.0 - 0.35 * self.cfg["start_magenta"], 1.0])   # B,G,R gains; low G = magenta
        self.awb_target = np.ones(3)
        self.last_t = None
        self.flags = []
        self.stats = {}

    # ---- set-up ---------------------------------------------------------------------------
    def _texture(self, r):
        tex = r.texture.astype(np.float32)
        road = np.all(r.texture == ROAD_SIM, axis=-1)
        green = np.all(r.texture == GREEN_SIM, axis=-1)
        tex[road], tex[green] = ROAD_REAL, GREEN_REAL
        gloss = np.where(road, 1.0, np.where(green, 0.45, 0.25)).astype(np.float32)
        h, w = road.shape
        # low-frequency albedo variation (mat print, dirt), +-6 %
        n = self.rng.normal(0, 1, (24, 24)).astype(np.float32)
        shade = 1 + 0.06 * cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC) / 1.5
        # static shadows of people / furniture next to and over the road, soft edges
        dark = np.ones((h, w), np.float32)
        for _ in range(int(self.cfg["static_shadows"])):
            c = (self.rng.uniform(-110, 110, 2) - r.origin) * r.ppcm
            ax = self.rng.uniform(6, 25, 2) * r.ppcm
            m = np.zeros((h, w), np.float32)
            cv2.ellipse(m, (int(c[0]), int(c[1])), (int(ax[0]), int(ax[1])), float(self.rng.uniform(0, 180)), 0, 360, 1.0, -1)
            m = cv2.GaussianBlur(m, (0, 0), self.rng.uniform(2, 6) * r.ppcm)
            dark *= 1 - (1 - self.rng.uniform(*PROFILES_RANGE(self.profile, "shadow_dark"))) * m
        shade *= dark
        # creases: thin arcs on the mat, they shine only from some headings
        k = int(self.cfg["creases"])
        creases = np.zeros((h, w, max(k, 1)), np.float32)
        self.crease_dir = self.rng.uniform(-math.pi, math.pi, max(k, 1))
        self.crease_amp = self.rng.uniform(*PROFILES_RANGE(self.profile, "crease_amp"), max(k, 1)) if k else np.zeros(1)
        for i in range(k):
            m = np.zeros((h, w), np.float32)
            c = (self.rng.uniform(-100, 100, 2) - r.origin) * r.ppcm
            rad = self.rng.uniform(30, 120) * r.ppcm
            a0 = self.rng.uniform(0, 360)
            cv2.ellipse(m, (int(c[0]), int(c[1])), (int(rad), int(rad * self.rng.uniform(.3, 1))), 0, a0,
                        a0 + self.rng.uniform(20, 50), 1.0, 2)
            creases[..., i] = cv2.GaussianBlur(m, (0, 0), 1.5) * (gloss > 0.4)
        stack = np.concatenate([tex, gloss[..., None], shade[..., None], creases, np.ones((h, w, 1), np.float32)], -1)
        self.tex = self.t.from_numpy(stack).permute(2, 0, 1)[None].contiguous().to(self.dev)
        self.tex_size = (w, h)
        self.n_crease = k

    def _geometry(self, cam):
        t, W, H = self.t, self.W, self.H
        S, h, w = self.S, self.h, self.w
        def sub(a):                                      # (H*W,...) or (H,W) -> (h,w,...) pixel centres
            a = a if a.shape[:2] == (H, W) else a.reshape(H, W, *a.shape[1:])
            return np.ascontiguousarray(a[S // 2::S, S // 2::S][:h, :w])
        ground, valid = cam.raw_geometry()
        ground, valid = sub(ground).reshape(-1, 2), sub(valid)
        self.ground = t.from_numpy(ground.astype(np.float32)).to(self.dev)              # (n,2) body cm
        self.valid = t.from_numpy(valid).to(self.dev)
        gx, gy = ground[:, 0], ground[:, 1] - cam.lens_ahead
        self.bear = t.from_numpy(np.arctan2(gx, gy).reshape(h, w).astype(np.float32)).to(self.dev)
        self.dist = t.from_numpy(np.hypot(gx, gy).reshape(h, w).astype(np.float32)).to(self.dev)
        # horizontal angle of every raw pixel (for the background panorama)
        uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
        pts = np.stack([uu.ravel(), vv.ravel()], -1).reshape(-1, 1, 2)
        nrm = cv2.fisheye.undistortPoints(pts, cam.K, cam.D).reshape(-1, 2)
        az = np.arctan(np.clip(nrm[:, 0], -20, 20)).reshape(H, W)
        self.az = t.from_numpy(sub(az).astype(np.float32)).to(self.dev)
        # rectification: rectified pixel -> raw pixel (the drivers' own undistortion)
        m1, m2 = cv2.fisheye.initUndistortRectifyMap(cam.K, cam.D, np.eye(3), cam.rect_K, (W, H), cv2.CV_32FC1)
        g = np.stack([m1 / (W - 1) * 2 - 1, m2 / (H - 1) * 2 - 1], -1).astype(np.float32)
        self.rect_grid = t.from_numpy(g)[None].to(self.dev)
        # lens shading: radial fall-off, green falls faster (magenta corners, as on the real lens)
        yy, xx = np.mgrid[:H, :W].astype(np.float32)
        r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (W / 2)) ** 2
        v, mc = self.cfg["vignette"], self.cfg["magenta_corner"]
        fall = np.clip(1 - v * r2 / 1.56, 0.2, 1)
        vig = np.stack([fall * (1 + 0.3 * mc * r2), fall * (1 - mc * r2), fall * (1 + 0.5 * mc * r2)])
        self.vig = t.from_numpy(np.ascontiguousarray(vig[:, S // 2::S, S // 2::S]).astype(np.float32)).to(self.dev)
        # body-anchored own shadow (car body under the ceiling lights)
        own = np.ones((H, W), np.float32)
        oy = np.clip((yy - H * 0.78) / (H * 0.22), 0, 1)
        own -= (1 - self.cfg["own_shadow"]) * oy * np.exp(-((xx - W / 2) / (W * 0.35)) ** 2)
        self.own = t.from_numpy(sub(own)).to(self.dev)
        meter = np.exp(-(((xx - W / 2) / (W * 0.45)) ** 2 + ((yy - H * 0.55) / (H * 0.45)) ** 2))
        meter = sub(meter)
        self.meter = t.from_numpy(meter.astype(np.float32)).to(self.dev)[None]
        self.meter_sum = float(meter.sum())
        self.meter_ref = None
        s = self.cfg["focus_sigma"] / S
        k = np.arange(-int(3 * s + 1), int(3 * s + 1) + 1, dtype=np.float32)
        k = np.exp(-0.5 * (k / s) ** 2)
        self.kernel = t.from_numpy(k / k.sum()).to(self.dev)

    def _lights(self):
        n = int(self.cfg["sheen_lights"])
        R = lambda key: PROFILES_RANGE(self.profile, key)
        self.lights = [dict(az=self.rng.uniform(-math.pi, math.pi), amp=self.rng.uniform(*R("sheen_amp")),
                            w=self.rng.uniform(*R("sheen_width")), d=self.rng.uniform(*R("sheen_dist")),
                            wd=self.rng.uniform(8, 20)) for _ in range(n)]

    def _walkers(self):
        self.walkers = []
        for _ in range(int(self.cfg["walkers"])):
            a = self.rng.uniform(-120, 120, 2)
            ang = self.rng.uniform(0, 2 * math.pi)
            self.walkers.append(dict(p0=a, dir=np.array([math.cos(ang), math.sin(ang)]), speed=self.rng.uniform(20, 60),
                                     period=self.rng.uniform(8, 20), r=self.rng.uniform(12, 28),
                                     dark=self.rng.uniform(0.35, 0.6), phase=self.rng.uniform(0, 20)))

    def _background(self):
        files = sorted(ASSETS.glob("bg_*.jpg"))
        pick = self.rng.choice(len(files), 3, replace=False)
        frames = [imread(files[i]).astype(np.float32) for i in pick]
        pano = np.concatenate(frames, 1)                                       # ~360 deg strip
        self.pano = self.t.from_numpy(pano).permute(2, 0, 1)[None].to(self.dev)
        self.pano_fov = 2 * math.pi / 3                                        # each real frame ~120 deg

    # ---- per frame --------------------------------------------------------------------------
    def frame(self, t_capture, pose_at, raw=True):
        """pose_at(t) -> true pose; the frame integrates [t - exposure, t]."""
        t, cfg, W, H = self.t, self.cfg, self.w, self.h        # composition size
        dt = 0.033 if self.last_t is None else max(1e-3, t_capture - self.last_t)
        self.last_t = t_capture
        exp_s = cfg["exposure_s"] * float(np.clip(1 / max(self.gain, 1e-3), 0.8, 1.6))   # darker -> longer exposure
        n = int(cfg["sub_samples"])
        poses = np.array([pose_at(t_capture - exp_s * (1 - i / max(n - 1, 1))) for i in range(n)], np.float32)
        P = t.from_numpy(poses).to(self.dev)
        c, s = t.cos(P[:, 2]), t.sin(P[:, 2])
        gx, gy = self.ground[:, 0], self.ground[:, 1]
        wx = gx[None] * c[:, None] + gy[None] * s[:, None] + P[:, 0:1]
        wy = -gx[None] * s[:, None] + gy[None] * c[:, None] + P[:, 1:2]
        r = self.renderer
        tw, th = self.tex_size
        px = ((wx - r.origin) * r.ppcm) / (tw - 1) * 2 - 1
        py = ((wy - r.origin) * r.ppcm) / (th - 1) * 2 - 1
        grid = t.stack([px, py], -1).view(n, H, W, 2)
        samp = t.nn.functional.grid_sample(self.tex.expand(n, -1, -1, -1), grid, mode="bilinear",
                                           padding_mode="zeros", align_corners=True)
        yaw = float(poses[-1, 2])
        # sheen of far lights: depends on heading only (summed over the exposure = smeared)
        sheen = t.zeros((n, H, W), device=self.dev)
        for L in self.lights:
            rel = t.remainder(self.bear[None] - (L["az"] - P[:, 2, None, None]) + math.pi, 2 * math.pi) - math.pi
            sheen += L["amp"] * t.exp(-0.5 * (rel / L["w"]) ** 2) * t.exp(-0.5 * ((self.dist[None] - L["d"]) / L["wd"]) ** 2)
        rgb, gloss, shade = samp[:, :3], samp[:, 3], samp[:, 4]
        cover = samp[:, -1]
        img = rgb * shade[:, None] + gloss[:, None] * sheen[:, None]
        if self.n_crease:
            view = t.tensor([max(0.0, math.cos(yaw - d)) ** 6 * a for d, a in zip(self.crease_dir, self.crease_amp)],
                            device=self.dev, dtype=t.float32)
            img = img + (samp[:, 5:5 + self.n_crease] * view[None, :, None, None]).sum(1, keepdim=True)
        floor = t.tensor(FLOOR_REAL, device=self.dev, dtype=t.float32)[None, :, None, None]
        img = img + (1 - cover[:, None]) * floor
        img = img.mean(0)                                                         # motion blur
        # walking shadows (world anchored, moving)
        if self.walkers:
            mul = t.ones((H, W), device=self.dev)
            wxl, wyl = wx[-1].view(H, W), wy[-1].view(H, W)
            for k in self.walkers:
                ph = (t_capture + k["phase"]) % (2 * k["period"])
                along = k["speed"] * (ph if ph < k["period"] else 2 * k["period"] - ph) - k["speed"] * k["period"] / 2
                cx, cy = k["p0"] + k["dir"] * along
                d2 = ((wxl - cx) ** 2 + (wyl - cy) ** 2) / k["r"] ** 2
                mul = mul * (1 - (1 - k["dark"]) * t.exp(-d2 * 2))
            img = img * mul[None]
        img = img * self.own[None]
        # background above the horizon: real frames, turning with the heading
        col = ((yaw + self.az) / self.pano_fov) * W + W / 2
        pw = self.pano.shape[-1]
        col = t.remainder(col, pw)
        gy_ = t.arange(H, device=self.dev, dtype=t.float32)[:, None].expand(H, W)
        bgrid = t.stack([col / (pw - 1) * 2 - 1, gy_ / (H - 1) * 2 - 1], -1)[None]
        bg = t.nn.functional.grid_sample(self.pano, bgrid, mode="bilinear", align_corners=True)[0]
        img = t.where(self.valid[None], img, bg)
        img[:, -28 // self.S:] = t.tensor((38, 38, 42), device=self.dev, dtype=t.float32)[:, None, None]
        # optics: focus, haze, lens shading
        k = self.kernel
        pad = len(k) // 2
        x = t.nn.functional.pad(img[None], (pad, pad, pad, pad), mode="replicate")
        x = t.nn.functional.conv2d(x, k.view(1, 1, 1, -1).expand(3, 1, 1, -1), groups=3)
        x = t.nn.functional.conv2d(x, k.view(1, 1, -1, 1).expand(3, 1, -1, 1), groups=3)[0]
        x = x * (1 - cfg["haze"]) + cfg["haze"] * x.mean()
        x = x * self.vig
        # lens flare: a ghost of the strongest light in view, rainbow tinted
        if cfg["flare"] > 0:
            for L in self.lights:
                rel = math.remainder(L["az"] - yaw, 2 * math.pi)
                if abs(rel) < 0.9 and L["amp"] > 110:
                    gxp = W / 2 - rel / 0.9 * W * 0.4
                    yy = t.arange(H, device=self.dev, dtype=t.float32)[:, None]
                    xx = t.arange(W, device=self.dev, dtype=t.float32)[None]
                    blob = t.exp(-(((xx - gxp) / (35 / self.S)) ** 2 + ((yy - H * 0.35) / (60 / self.S)) ** 2))
                    tint = t.tensor((0.6, 1.0, 1.3), device=self.dev)[:, None, None]
                    x = x + cfg["flare"] * 90 * blob[None] * tint
        # auto exposure / white balance (lag; hunting in the harsh profile)
        # metering: whole frame, centre weighted (as the Pi's default AE). The reference is the
        # first frame: gain 1 = the real camera on a normal view (palette is measured), so the
        # gain only reacts to CHANGES of the scene (sheen, white floor, shadows) and to the
        # run's exposure bias.
        meas = float((x * self.meter).sum() / self.meter_sum / 3)
        if self.meter_ref is None:
            self.meter_ref = meas
        want = self.meter_ref * cfg["target_rel"] / max(meas, 1.0)
        a = 1 - math.exp(-dt / cfg["ae_tau_s"])
        self.gain = float(np.clip(math.exp(math.log(self.gain) + a * (math.log(want) - math.log(self.gain))), 0.15, 4.0))
        if cfg["ae_hunt"]:
            self.gain *= 1 + cfg["ae_hunt"] * math.sin(t_capture * 2 * math.pi / 2.7) * a
        self.awb_target = np.clip(self.awb_target + self.rng.normal(0, cfg["awb_drift"] * math.sqrt(dt), 3), 0.85, 1.15)
        self.awb += (self.awb_target - self.awb) * (1 - math.exp(-dt / 0.6))
        x = x * self.gain * t.tensor(self.awb, device=self.dev, dtype=t.float32)[:, None, None]
        if self.S != 1:
            x = t.nn.functional.interpolate(x[None], size=(self.H, self.W), mode="bilinear", align_corners=False)[0]
        # sensor noise: shot + read, amplified by the analog gain
        sig = t.sqrt(cfg["read_noise"] ** 2 * max(self.gain, 1) + cfg["shot_noise"] * x.clamp(min=0))
        x = x + sig * t.randn(x.shape, generator=self.gen, device=self.dev)
        x = x.clamp(0, 255)
        if not raw:
            x = t.nn.functional.grid_sample(x[None], self.rect_grid, mode="bilinear", padding_mode="zeros",
                                            align_corners=True)[0]
        out = x.round().to(t.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
        self.stats = dict(gain=round(self.gain, 3), meter=round(meas, 1), exposure_ms=round(exp_s * 1000, 1))
        self.flags = ["realcam_" + (self.profile if isinstance(self.profile, str) else "custom")]
        return out


def imread(path):
    """cv2.imread fails on non-ASCII paths (Korean user folders): decode from bytes."""
    return cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)


def PROFILES_RANGE(profile, key):
    v = (PROFILES[profile] if isinstance(profile, str) else profile)[key]
    return v if isinstance(v, tuple) else (v, v)


def pose_from_history(history):
    """Linear interpolation of the true pose from the 20 ms simulation history."""
    times = np.array([h[0] for h in history])
    poses = np.array([h[1] for h in history])

    def at(tq):
        i = int(np.clip(np.searchsorted(times, tq) - 1, 0, len(times) - 2))
        if tq <= times[0]:
            return poses[0]
        a = float(np.clip((tq - times[i]) / max(times[i + 1] - times[i], 1e-9), 0, 1))
        p0, p1 = poses[i], poses[i + 1]
        dyaw = math.remainder(p1[2] - p0[2], 2 * math.pi)
        return np.array([p0[0] + a * (p1[0] - p0[0]), p0[1] + a * (p1[1] - p0[1]), p0[2] + a * dyaw])
    return at
