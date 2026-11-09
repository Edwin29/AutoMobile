"""Validate source syntax and publication hashes without opening hardware."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    entries = json.loads((ROOT / 'docs/source_manifest.json').read_text(encoding='utf-8'))
    for entry in entries:
        file = ROOT / entry['path']
        assert hashlib.sha256(file.read_bytes()).hexdigest() == entry['published_sha256'], file
    scripts = list(ROOT.rglob('*.py'))
    for file in scripts:
        if '.venv' in file.parts:
            continue
        ast.parse(file.read_text(encoding='utf-8-sig'), filename=str(file))
    for file in (ROOT / 'road_ai_v1').glob('*.ipynb'):
        notebook = json.loads(file.read_text(encoding='utf-8-sig'))
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), str(file), 'exec')
    print(f'Validated {len(entries)} archived file hashes and Python syntax.')


if __name__ == '__main__':
    main()
