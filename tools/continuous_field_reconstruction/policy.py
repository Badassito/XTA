"""Portable semantic identity for the opt-in reconstruction package."""
from pathlib import Path
import hashlib
import importlib.metadata
import json

POLICY = 'guard-cap-weighted-cardinal-B3-v1'
ALGORITHM_VERSION = 1
ALGORITHM = POLICY
PADDING_CELLS = 3

def package_source_identity():
    root = Path(__file__).resolve().parent
    files = {}
    for path in sorted(root.rglob('*.py')):
        relative = path.relative_to(root)
        if any(part.startswith('test') or part == '__pycache__' for part in relative.parts):
            continue
        files[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    dependencies = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-image')}
    payload = {'files': files, 'dependencies': dependencies}
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    return {**payload, 'sha256': hashlib.sha256(encoded).hexdigest()}
