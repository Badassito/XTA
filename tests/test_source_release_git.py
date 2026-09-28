"""The release bundle must identify one committed set of source bytes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from tools import build_source_release


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(('git', *args), cwd=root, text=True).strip()


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / 'repo'
    root.mkdir()
    _git(root, 'init', '-q')
    _git(root, 'config', 'user.email', 'test@example.com')
    _git(root, 'config', 'user.name', 'Release test')
    (root / '.gitattributes').write_bytes(b'*.py text eol=lf\n')
    (root / 'XTA').mkdir()
    (root / 'XTA' / '__init__.py').write_bytes(b'__version__ = "1.2.3"\n')
    (root / 'XTA' / 'module.py').write_bytes(b'VALUE = 1\n')
    (root / 'GPT-6-Astra-Ultra_v1.2.3_SLURM.py').write_bytes(b'# launcher\n')
    (root / 'ARCHITECTURE.md').write_bytes(b'# package\n')
    _git(root, 'add', '.')
    _git(root, 'commit', '-qm', 'test release')
    return root


def _member(archive: Path, name: str) -> bytes:
    with zipfile.ZipFile(archive) as source:
        return source.read('XTA_v1.2.3/' + name)


def test_clean_commit_bundle_is_reproducible_and_has_a_commit_identity(tmp_path):
    root = _repository(tmp_path)
    one = build_source_release.build(root=root, output=tmp_path / 'one')
    two = build_source_release.build(root=root, output=tmp_path / 'two')
    assert hashlib.sha256(one.read_bytes()).digest() == hashlib.sha256(two.read_bytes()).digest()
    manifest = json.loads(_member(one, 'RELEASE_MANIFEST.json'))
    assert manifest['source'] == _git(root, 'rev-parse', 'HEAD')
    assert _member(one, 'XTA/module.py') == b'VALUE = 1\n'
    assert manifest['files']['XTA/module.py'] == hashlib.sha256(b'VALUE = 1\n').hexdigest()


def test_commit_bundle_reads_blob_bytes_even_with_git_archive_substitution(tmp_path):
    root = _repository(tmp_path)
    (root / '.gitattributes').write_bytes(b'*.py text eol=lf\nXTA/module.py export-subst\n')
    canonical = b'ID = "$Format:%H$"\n'
    (root / 'XTA' / 'module.py').write_bytes(canonical)
    _git(root, 'add', '.')
    _git(root, 'commit', '-qm', 'export attribute fixture')
    archive = build_source_release.build(root=root, output=tmp_path / 'release')
    assert _member(archive, 'XTA/module.py') == canonical


def test_default_refuses_dirty_or_untracked_source_and_snapshot_is_labelled(tmp_path):
    root = _repository(tmp_path)
    (root / 'XTA' / 'module.py').write_bytes(b'VALUE = 2\r\n')
    (root / 'XTA' / 'extra.py').write_bytes(b'EXTRA = True\n')
    with pytest.raises(ValueError, match='clean Git tree'):
        build_source_release.build(root=root, output=tmp_path / 'release')
    archive = build_source_release.build(root=root, output=tmp_path / 'snapshot', snapshot=True)
    manifest = json.loads(_member(archive, 'RELEASE_MANIFEST.json'))
    assert manifest['source'] == 'working-tree-snapshot'
    assert _member(archive, 'XTA/module.py') == b'VALUE = 2\r\n'
    assert _member(archive, 'XTA/extra.py') == b'EXTRA = True\n'


def test_optimized_python_still_rejects_a_mismatched_wheel(tmp_path):
    root = _repository(tmp_path)
    wheel = tmp_path / 'wrong.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('XTA/__init__.py', b'__version__ = "9.9.9"\n')
    program = (
        'from pathlib import Path; from tools.build_source_release import build; '
        f'build(root=Path({str(root)!r}), output=Path({str(tmp_path / "output")!r}), '
        f'wheel=Path({str(wheel)!r}))'
    )
    completed = subprocess.run([sys.executable, '-O', '-c', program],
                               cwd=Path(__file__).resolve().parents[1],
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert completed.returncode != 0
    assert 'Wheel package files differ' in completed.stdout
