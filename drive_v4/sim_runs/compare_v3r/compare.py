import json, sys
from pathlib import Path
S = Path(r"C:\Users\user\AppData\Local\Temp\claude\b2a73cb441173feb\37d243ac-6618-4e4e-a585-715bfa88d552\scratchpad")
V3R = [Path(r"C:\Users\user\Documents\ChatGPT\AutoDrive 2\v3_recovery_lab\drive_v3\runs\field_stress_v2"), S/"cmp"/"desk_v3r"]
V4 = [S/"cmp"/"runs", S/"cmp"/"desk_v4"]
def load(dirs, slug):
    for d in dirs:
        p = d/slug/"summary.json"
        if p.exists(): return json.loads(p.read_text(encoding="utf-8"))
def fmt(r):
    if r is None: return "(pending)"
    ok = "PASS" if r["verified_lap"] else r["outcome"]
    unrec = sum(not e.get("recovered", True) for e in r["penetration_episodes"])
    return f"{ok:16s} corners {r['proper_corner_exits']}/4  min {r['minimum_true_margin_cm']:+6.2f}  out {r['outside_s']:5.1f}s"
single = ["baseline","drift_left","turn_under","slip","low_power","wheel_stall","glare","seam","occlusion","dropout","blur_warm"]
rows = [(s,d,17) for s in single for d in ("ccw","cw")] + [(s,d,seed) for s in ("mixed_mild","mixed_strong") for seed in (17,31,47) for d in ("ccw","cw")]
tally = {"v3r": [0,0], "v4": [0,0]}
print(f"{'scenario':24s} | {'modified v3 (0.6s)':52s} | v4 same timing (0.6s)")
for s,d,seed in rows:
    a = load(V3R, f"{s}_{d}_p0.6_s{seed}"); b = load(V4, f"{s}_{d}_same_s{seed}")
    for k,r in (("v3r",a),("v4",b)):
        if r: tally[k][0] += r["verified_lap"]; tally[k][1] += 1
    print(f"{s+'_'+d+'_s'+str(seed):24s} | {fmt(a):52s} | {fmt(b)}")
print("verified laps:", {k: f"{v[0]}/{v[1]}" for k,v in tally.items()})
print("\nv4 own timing (15fps, 70ms):")
for s in ("baseline","turn_under","slip","mixed_strong"):
    for d in ("ccw","cw"):
        print(f"  {s+'_'+d:22s} {fmt(load(V4, f'{s}_{d}_own_s17'))}")
