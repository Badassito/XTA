"""Resolve retained release fixtures from canonical History without rewriting them."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


_VERSIONS = {
    'guarded_rescue': '25.0.0',
    'job150615': '25.0.0',
    'job150615_headroom': '25.0.0',
    'projection_release': '25.0.0',
    'throughput': '25.0.1',
}


def history_archive_path(name: str, *, history_root: Path | None = None) -> Path:
    """Return a stable default path; callers still authenticate archive bytes."""
    if name not in _VERSIONS:
        raise ValueError(f'Unknown retained release fixture: {name}')
    if history_root is None:
        workspace = Path(os.environ.get('XTA_TEST_REPO') or Path(__file__).resolve().parents[1])
        history_root = workspace.parent / 'Scratch' / 'Data' / 'XTA' / 'History'
    return (Path(history_root) / '2026-10-04-cleanup' / 'release-validation-fixtures'
            / name / f'XTA_v{_VERSIONS[name]}_complete_source.zip')


def require_history_archive(name: str, expected_sha256: str, *, history_root: Path | None = None) -> Path:
    """Require the exact retained artifact so active guards cannot silently skip."""
    expected = str(expected_sha256).lower()
    if len(expected) != 64 or any(value not in '0123456789abcdef' for value in expected):
        raise ValueError('Retained release fixture requires an exact SHA256 pin')
    path = history_archive_path(name, history_root=history_root)
    if not path.is_file():
        raise FileNotFoundError(f'Required retained release fixture is missing from History: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f'Retained release fixture differs from its independent SHA256 pin: {path}')
    return path
