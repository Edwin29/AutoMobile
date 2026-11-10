"""Random mixes of field_stress disturbances (no slip / wheel stall), both controllers.
Does not modify the v3-recovery files: the mixed profile is added to SCENARIOS in memory only."""
import argparse, json, random, sys
from pathlib import Path
FACTORS = {"drift_left": {"drift": -.035}, "turn_under": {"turn_proxy": True},
           "low_power": {"sag": .65, "sag_s": 12.}, "glare": {"glare_s": 4.}, "seam": {"seam": True},
           "occlusion": {"occlusion_s": 3.}, "dropout": {"dropout_s": 1.2}, "blur_warm": {"blur": True}}

def combos(n=12, seed=2026):
    rng = random.Random(seed); out, seen = [], set()
    while len(out) < n:
        k = rng.choice([2, 3, 4]); names = tuple(sorted(rng.sample(sorted(FACTORS), k)))
        if names in seen: continue
        seen.add(names); out.append((f"mix{len(out)+1:02d}", names, rng.randint(1, 99)))
    return out

def profile(names):
    p = {}
    for n in names: p.update(FACTORS[n])
    return p

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True); ap.add_argument("--controller", choices=["v3r", "v4"], required=True)
    ap.add_argument("--timing", default="same"); ap.add_argument("--combo", required=True)
    ap.add_argument("--direction", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    spec = {c[0]: c for c in combos()}[a.combo]; name, names, seed = spec
    work = Path(a.work)
    if a.controller == "v3r":
        sys.path.insert(0, str(work / "v3_recovery_lab"))
        import drive_v3.field_stress as fs
        fs.SCENARIOS[name] = profile(names)
        fs.run(a.out, name, a.direction, 0.6, seed)
    else:
        sys.path.insert(0, str(work / "github_publish" / "AutoMobile"))
        import drive_v4.compare_v3r as c
        c.SCENARIOS[name] = profile(names)
        c.run(a.out, name, a.direction, a.timing, seed)
