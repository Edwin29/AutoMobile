"""Versioned provisional settings; no parameter is a validated safety bound."""
from pathlib import Path
import json
import hashlib

DEFAULT = Path(__file__).parent / "config/runtime.json"


def load_settings(path=DEFAULT):
    content = Path(path).read_bytes()
    data = json.loads(content)
    if data["schema"] != 1:
        raise ValueError("Unsupported runtime configuration")
    return data, hashlib.sha256(content).hexdigest()
