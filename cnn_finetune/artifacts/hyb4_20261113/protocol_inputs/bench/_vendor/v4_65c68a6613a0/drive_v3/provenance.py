"""Small reproducibility manifest without workstation paths or private input names."""
import hashlib
import json
from pathlib import Path
from .camera import sha256_file


def code_manifest():
    root=Path(__file__).parent
    files={p.relative_to(root).as_posix():sha256_file(p) for p in sorted(root.glob("*.py"))}
    files.update({p.relative_to(root).as_posix():sha256_file(p) for p in sorted((root/"config").glob("*.json"))})
    return {"files":files,"sha256":hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()}
