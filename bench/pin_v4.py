"""Freeze the current drive_v4 (+ the original drive_v3 it imports) into bench/_vendor/v4_<hash>.

    python -m bench.pin_v4        # prints the pinned folder; controllers.V4 uses the newest pin
The hash covers every .py/.json/.npz/.npy file copied, so a result names the exact code it ran.
"""
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = Path(__file__).resolve().parent / "_vendor"
PARTS = ["drive_v3", "drive_v4", "map_localization/outer_driver/track.npz"]
SKIP = {"runs", "sim_runs", "evidence", "__pycache__", "tests"}


def files():
    out = []
    for part in PARTS:
        p = ROOT / part
        if p.is_file():
            out.append(p)
            continue
        for f in sorted(p.rglob("*")):
            if f.is_file() and not (SKIP & set(f.relative_to(ROOT).parts)) and f.suffix in {".py", ".json", ".npz", ".npy"}:
                out.append(f)
    return out


def pin():
    fs = files()
    h = hashlib.sha256()
    for f in fs:
        h.update(str(f.relative_to(ROOT)).replace("\\", "/").encode())
        h.update(f.read_bytes())
    tag = h.hexdigest()[:12]
    dst = VENDOR / f"v4_{tag}"
    if not dst.exists():
        for f in fs:
            target = dst / f.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
        (dst / "PIN.txt").write_text(h.hexdigest() + "\n", encoding="utf-8")
    (VENDOR / "V4_PIN.txt").write_text(dst.name + "\n", encoding="utf-8")
    return dst


if __name__ == "__main__":
    print(pin())
