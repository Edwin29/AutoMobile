import argparse, concurrent.futures, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import combo
ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--arms", nargs="+", required=True); ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
ARMS = {"v3r": ("v3r", "same"), "v4same": ("v4", "same"), "v4own": ("v4", "own")}
specs = [(arm, c[0], d) for arm in a.arms for c in combo.combos() for d in ("ccw", "cw")]
specs.sort(key=lambda s: s[0] != "v4own")  # long runs first
env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
def job(s):
    arm, name, d = s; ctrl, timing = ARMS[arm]; dst = out / f"{arm}_{name}_{d}"
    if (dst / "summary.json").exists(): return dst.name, "cached"
    if dst.exists():
        import shutil; shutil.rmtree(dst)
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "combo.py"), "--work", a.work, "--controller", ctrl,
                        "--timing", timing, "--combo", name, "--direction", d, "--out", str(dst)],
                       env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return dst.name, "ok" if r.returncode == 0 else "FAIL " + r.stderr[-500:]
print(len(specs), "runs", flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
    for i, f in enumerate(concurrent.futures.as_completed([pool.submit(job, s) for s in specs]), 1):
        print(i, *f.result(), flush=True)
