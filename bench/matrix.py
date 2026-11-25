"""Run a set of bench experiments in parallel (one process per run).

    python -m bench.matrix --out DIR --workers 11 [--share 0/4]   # --share k/n: run every n-th spec from k
    python -m bench.matrix --out DIR --report                      # tables from finished runs
"""
import argparse
import concurrent.futures
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

from .disturb import EFFECTS
from .vehicle import PRESETS

ROOT = Path(__file__).resolve().parents[1]


ARMS = (("v4", "own"), ("v3r", "own"), ("v4", "slow"))
# the first-corner, one-shot disturbances group C draws from (kept fixed so specs never change)
ONE_SHOT = ["blur_warm", "dropout", "glare", "low_power", "occlusion", "seam"]   # slow = v3r's latency on v4: same-latency pair


def specs():
    """A presets / B random vehicles / C random vehicles + disturbances.
    'own' = each driver's processing time, 'slow' = v3r's latency, 'fast' = v4's latency.
    Same-latency arms exist so a speed difference is not reported as an algorithm difference."""
    s = []
    for v in PRESETS:
        for d in ("ccw", "cw"):
            for ctrl, timing in ARMS:
                s.append(dict(group="A_presets", controller=ctrl, vehicle=v, effects=[], direction=d, timing=timing, seed=17))
    for v in ("ideal", "field_estimate", "weak_left_turn", "sluggish"):   # v3r at v4's latency (slow to simulate)
        for d in ("ccw", "cw"):
            s.append(dict(group="A_presets", controller="v3r", vehicle=v, effects=[], direction=d, timing="fast", seed=17))
    rng = random.Random(2026)
    for i in range(12):
        for d in ("ccw", "cw"):
            for ctrl, timing in ARMS:
                s.append(dict(group="B_random_vehicle", controller=ctrl, vehicle=f"random:{100 + i}", effects=[],
                              direction=d, timing=timing, seed=17))
    for i in range(12):
        effects = sorted(rng.sample(ONE_SHOT, rng.choice([1, 2, 3])))
        seed = rng.randint(1, 99)
        for d in ("ccw", "cw"):
            for ctrl, timing in ARMS:
                s.append(dict(group="C_random_vehicle_and_effects", controller=ctrl, vehicle=f"random:{200 + i}",
                              effects=effects, direction=d, timing=timing, seed=seed))
    for v in ("field_estimate", "random:300", "random:301"):      # D: the car changes DURING the run
        for variation in ("mild", "harsh"):
            for d in ("ccw", "cw"):
                for ctrl, timing in ARMS:
                    s.append(dict(group="D_changing_vehicle", controller=ctrl, vehicle=v, effects=[], direction=d,
                                  timing=timing, seed=17, variation=variation))
    for v in ("field_estimate", "random:400", "random:401", "random:402", "random:403"):   # E: strong AND continuing
        for d in ("ccw", "cw"):
            for ctrl, timing in ARMS:
                s.append(dict(group="E_strong_continuing", controller=ctrl, vehicle=v,
                              effects=["recurring_strong", "seam"], direction=d, timing=timing, seed=31,
                              variation="harsh"))
    for v in ("ideal", "field_estimate", "weak_left_turn", "sluggish"):    # F: the team CNNs (speed via BENCH_CNN_SPEED)
        for d in ("ccw", "cw"):
            for ctrl in ("cnnC5b", "cnnA5b", "cnnB", "cnnC", "cnnA"):
                s.append(dict(group="F_cnn_presets", controller=ctrl, vehicle=v, effects=[], direction=d,
                              timing="own", seed=17))
    # G: the best team CNN (A5b) on exactly the v4 specs of groups B-E
    for x in list(s):
        if x["group"] in ("B_random_vehicle", "C_random_vehicle_and_effects", "D_changing_vehicle",
                          "E_strong_continuing") and x["controller"] == "v4" and x["timing"] == "own":
            s.append(dict(x, group="G_cnn_" + x["group"], controller="cnnA5b"))
    # H: realistic camera from the field logs (bench/realcam.py): typical vs harsh, same specs for every driver
    for v in ("field_estimate", "weak_left_turn", "sluggish", "random:500"):
        for fx in ("realcam", "realcam_harsh"):
            for d in ("ccw", "cw"):
                for ctrl in ("cnnA5b", "cnnC5b", "v4"):    # v3r left out: slowest, already weakest (E: 0/10)
                    s.append(dict(group="H_realcam", controller=ctrl, vehicle=v, effects=[fx], direction=d,
                                  timing="own", seed=41))
    for v in ("field_estimate", "random:400"):        # harsh camera AND a car that changes during the run
        for d in ("ccw", "cw"):
            for ctrl in ("cnnA5b", "cnnC5b", "v4"):
                s.append(dict(group="H_realcam", controller=ctrl, vehicle=v, effects=["realcam_harsh"], direction=d,
                              timing="own", seed=43, variation="harsh"))
    for v in ("field_estimate", "random:500"):         # H: one harsh camera factor at a time
        for f in ("light", "shadow", "exposure", "blur", "optics"):
            for d in ("ccw", "cw"):
                for ctrl in ("cnnA5b", "cnnC5b", "v4"):
                    s.append(dict(group="H_realcam_factor", controller=ctrl, vehicle=v, effects=[f"realcam_{f}"],
                                  direction=d, timing="own", seed=41))
    # I: CNN-first hybrid (hyb2) vs CNN alone vs v4. Conditions of cnn_map_hybrid_lab's
    # outer test, then the field-log camera. Same seeds for every driver.
    I_COND = [("field_estimate", [], "fixed"), ("unequal_sides", [], "fixed"), ("weak_left_turn", [], "fixed"),
              ("high_start_duty", [], "fixed"), ("field_estimate", ["glare"], "fixed"),
              ("field_estimate", ["occlusion"], "fixed"), ("field_estimate", ["dropout"], "fixed"),
              ("field_estimate", ["whiteout_midturn"], "fixed"),
              ("field_estimate", ["blur_warm", "glare", "low_power", "occlusion"], "mild")]
    I_COND += [(v, [fx], "fixed") for v in ("field_estimate", "weak_left_turn", "sluggish", "random:500")
               for fx in ("realcam", "realcam_harsh")]
    for v, fx, var in I_COND:
        for d in ("ccw", "cw"):
            for ctrl in ("hyb2", "cnnA5b", "v4"):
                s.append(dict(group="I_hybrid", controller=ctrl, vehicle=v, effects=fx, direction=d,
                              timing="own", seed=41, variation=var))
    # Plan 3 (CNN x map cross-check): field-log camera only, several seeds.
    # Seeds 41/43 = design set (looked at while writing hyb3), 47/53 = held out.
    for seed in (41, 43, 47, 53):
        for v in ("field_estimate", "weak_left_turn", "sluggish", "random:500"):
            for fx in ("realcam", "realcam_harsh"):
                for d in ("ccw", "cw"):
                    for ctrl in ("hyb2", "hyb3", "cnnA5b"):
                        s.append(dict(group="J_xcheck", controller=ctrl, vehicle=v, effects=[fx], direction=d,
                                      timing="own", seed=seed))
    # Junction movement (bench/junction.py). First the GPT prototype's scene for comparison:
    # east T junction, arriving from the south, straight or left; its 9 conditions
    # (no 'slip': not modelled in this bench) at seed 17, three of them again at 29/43,
    # plus the field-log camera.
    K = [(v, fx, var, sd) for v, fx, var in [("field_estimate", [], "fixed"), ("unequal_sides", [], "fixed"),
                                             ("weak_left_turn", [], "fixed"), ("high_start_duty", [], "fixed"),
                                             ("field_estimate", ["glare"], "fixed"), ("field_estimate", ["occlusion"], "fixed"),
                                             ("field_estimate", ["dropout"], "fixed"),
                                             ("field_estimate", ["blur_warm", "glare", "low_power", "occlusion"], "mild")]
         for sd in (17,)]
    K += [(v, fx, var, sd) for v, fx, var in [("field_estimate", [], "fixed"), ("field_estimate", ["glare"], "fixed"),
                                             ("field_estimate", ["blur_warm", "glare", "low_power", "occlusion"], "mild")]
          for sd in (29, 43)]
    K += [(v, [fx], "fixed", sd) for v in ("field_estimate", "weak_left_turn") for fx in ("realcam", "realcam_harsh")
          for sd in (41, 43)]
    K += [("field_pivot_slip", [], "fixed", 17), ("field_pivot_slip", ["realcam"], "fixed", 41)]
    for v, fx, var, sd in K:
        for ex in ("straight", "left"):
            for ctrl in ("hyb2j", "cnnA5b"):
                s.append(dict(group="K_junction", controller=ctrl, vehicle=v, effects=fx, direction=ex, timing="own",
                              seed=sd, variation=var, scene=dict(junction="east", came_from="S", exit=ex)))
    # Central crossroad: always a right turn (course rule), from each of the four arms.
    KC = [("field_estimate", [], "fixed", 17), ("weak_left_turn", [], "fixed", 17), ("unequal_sides", [], "fixed", 17),
          ("field_estimate", ["glare"], "fixed", 17), ("field_estimate", ["blur_warm", "glare", "low_power", "occlusion"], "mild", 17),
          ("field_estimate", ["realcam"], "fixed", 41), ("field_estimate", ["realcam"], "fixed", 43),
          ("field_estimate", ["realcam_harsh"], "fixed", 41), ("field_estimate", ["realcam_harsh"], "fixed", 43),
          ("field_pivot_slip", [], "fixed", 17), ("field_pivot_slip", ["realcam"], "fixed", 41)]
    for v, fx, var, sd in KC:
        for arm in ("N", "E", "S", "W"):
            s.append(dict(group="K_junction", controller="hyb2j", vehicle=v, effects=fx, direction="right", timing="own",
                          seed=sd, variation=var, scene=dict(junction="center", came_from=arm, exit="right")))
    # Whole runs with junctions + scripted orchestrator (bench/mission.py), cw.
    for sc in ("lap_defaults", "inner_loop", "inner_loop_west", "stop_sign", "wrong_sign"):
        for v in ("field_estimate", "weak_left_turn", "unequal_sides", "field_pivot_slip"):
            for fx, sd in (([], 17), (["realcam"], 41), (["realcam"], 43), (["realcam_harsh"], 41)):
                s.append(dict(group="L_mission", controller="hyb2m", vehicle=v, effects=fx, direction="cw",
                              timing="own", seed=sd, variation="fixed", mission=sc))
    # Split pivot (2026-11-11): the narrow T merges, the crossroad and two T entries.
    for j, cf, ex in (("south", "N", "right"), ("north", "S", "right"), ("south", "E", "right"), ("east", "S", "left"),
                      ("center", "N", "right"), ("center", "E", "right"), ("center", "S", "right"), ("center", "W", "right")):
        for v in ("field_estimate", "field_pivot_slip", "weak_left_turn"):
            for fx, sd in (([], 17), (["realcam"], 41)):
                s.append(dict(group="K2_split", controller="hyb2j", vehicle=v, effects=fx, direction=ex, timing="own",
                              seed=sd, variation="fixed", scene=dict(junction=j, came_from=cf, exit=ex)))
    for x in s:
        x.setdefault("variation", "fixed")
        x["slug"] = f"{x['group']}__{x['controller']}_{x['timing']}__{x['vehicle'].replace(':', '')}__{'+'.join(x['effects']) or 'none'}__{x['direction']}"
        if x["variation"] != "fixed":
            x["slug"] += f"__{x['variation']}"
        if x["group"] in ("K_junction", "K2_split"):
            sc = x["scene"]
            x["slug"] = (f"{x['group']}__{x['controller']}__{sc['junction']}_{sc['came_from']}_{sc['exit']}__"
                         f"{x['vehicle'].replace(':', '')}__{'+'.join(x['effects']) or 'none'}__{x['variation']}__s{x['seed']}")
        if x["group"] == "L_mission":
            x["slug"] = (f"L_mission__{x['mission']}__{x['vehicle']}__{'+'.join(x['effects']) or 'none'}__s{x['seed']}")
        if x["group"] == "J_xcheck":
            x["slug"] += f"__s{x['seed']}"
    return s


def run_one(spec, out):
    dst = Path(out) / spec["slug"]
    if (dst / "summary.json").exists():
        return spec["slug"], "cached"
    if dst.exists():
        import shutil
        shutil.rmtree(dst)
    if "mission" in spec:
        cmd = [sys.executable, "-m", "bench.mission", "--controller", spec["controller"], "--vehicle", spec["vehicle"],
               "--effects", ",".join(spec["effects"]), "--scenario", spec["mission"], "--direction", spec["direction"],
               "--timing", spec["timing"], "--seed", str(spec["seed"]), "--variation", spec["variation"], "--out", str(dst)]
    elif "scene" in spec:
        sc = spec["scene"]
        cmd = [sys.executable, "-m", "bench.junction", "--controller", spec["controller"], "--vehicle", spec["vehicle"],
               "--effects", ",".join(spec["effects"]), "--junction", sc["junction"], "--came-from", sc["came_from"],
               "--exit", sc["exit"], "--timing", spec["timing"], "--seed", str(spec["seed"]),
               "--variation", spec["variation"], "--out", str(dst)]
    else:
        cmd = [sys.executable, "-m", "bench.run", "--controller", spec["controller"], "--vehicle", spec["vehicle"],
               "--effects", ",".join(spec["effects"]), "--direction", spec["direction"], "--timing", spec["timing"],
               "--seed", str(spec["seed"]), "--variation", spec["variation"], "--out", str(dst)]
    # CNN speed: every documented comparison uses 1.0 (team default 0.5 cannot start the car).
    # Forgetting the variable once (2026-11-11) silently produced a 0.5 run set; default it here.
    env = dict(os.environ)
    env.setdefault("BENCH_CNN_SPEED", "1.0")
    env = dict(env, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return spec["slug"], (r.stdout.strip().splitlines() or [""])[-1] if r.returncode == 0 else "FAIL " + r.stderr[-600:]


def report(out):
    out = Path(out)
    rows = []
    for sp in specs():
        p = out / sp["slug"] / "summary.json"
        if p.exists():
            rows.append(sp | {"r": json.loads(p.read_text(encoding="utf-8"))})
    lines = []
    for group in dict.fromkeys(r["group"] for r in rows):
        lines.append(f"\n## {group}\n")
        for key in sorted({(r["controller"], r["timing"]) for r in rows if r["group"] == group}):
            g = [r["r"] for r in rows if r["group"] == group and (r["controller"], r["timing"]) == key]
            m = np.array([x["min_margin_cm"] for x in g])
            lines.append(f"- {key[0]} ({key[1]} timing): laps {sum(x['verified_lap'] for x in g)}/{len(g)}, "
                         f"clean {sum(x['clean_lap'] for x in g)}/{len(g)}, min margin median {np.median(m):+.2f} "
                         f"worst {m.min():+.2f} cm, outside avg {np.mean([x['outside_s'] for x in g]):.1f} s, "
                         f"held {sum(x['outcome'] == 'held' for x in g)}, departed {sum(x['outcome'] == 'severe_departure' for x in g)}")
        arms = sorted({(r["controller"], r["timing"]) for r in rows if r["group"] == group})
        lines.append("\n| case | direction | " + " | ".join(f"{c} ({t})" for c, t in arms) + " |\n|---|---|"
                     + "---|" * len(arms))
        cases = dict.fromkeys((r["vehicle"], "+".join(r["effects"]) or "-", r["direction"], r["variation"])
                              for r in rows if r["group"] == group)
        for v, e, d, var in cases:
            cell = []
            for ctrl, tm in arms:
                hit = [r["r"] for r in rows if r["group"] == group and r["controller"] == ctrl and r["timing"] == tm
                       and r["vehicle"] == v and ("+".join(r["effects"]) or "-") == e and r["direction"] == d
                       and r["variation"] == var]
                if not hit:
                    cell.append("")
                    continue
                x = hit[0]
                tag = "PASS" if x["verified_lap"] else x["outcome"]
                cell.append(f"{tag} {x['min_margin_cm']:+.1f} / {x['outside_s']:.0f}s")
            label = f"{v} {e}" + ("" if var == "fixed" else f" ({var})")
            lines.append(f"| {label} | {d} | " + " | ".join(cell) + " |")
    text = "\n".join(lines)
    (out / "REPORT.md").write_text(text, encoding="utf-8")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--share", default="0/1")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--group", help="only run specs of this group")
    ap.add_argument("--controller", help="only run specs of these controllers (comma list)")
    ap.add_argument("--seeds", help="only run specs with these seeds (comma list)")
    a = ap.parse_args()
    if a.report:
        print(report(a.out))
        return
    k, n = map(int, a.share.split("/"))
    todo = [s for i, s in enumerate(specs()) if i % n == k and (not a.group or s["group"] == a.group)
            and (not a.controller or s["controller"] in a.controller.split(","))
            and (not a.seeds or str(s["seed"]) in a.seeds.split(","))]
    # longest first: v3r at v4's latency, then other v3r runs, then v4
    todo.sort(key=lambda s: (not (s["controller"] == "v3r" and s["timing"] == "fast"), s["controller"] != "v3r"))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    print(len(todo), "runs", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i, f in enumerate(concurrent.futures.as_completed([pool.submit(run_one, s, a.out) for s in todo]), 1):
            print(i, *f.result(), flush=True)


if __name__ == "__main__":
    main()
