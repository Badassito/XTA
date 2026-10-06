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
    (root / 'XTA' / '__init__.py').write_bytes(b'__version__ = "1.2.2"\n')
    (root / 'XTA' / 'module.py').write_bytes(b'VALUE = 1\n')
    (root / 'GPT-6-Astra-Ultra_v1.2.2_SLURM.py').write_bytes(b'# launcher\n')
    (root / 'ARCHITECTURE.md').write_bytes(b'# package\n')
    _git(root, 'add', '.')
    _git(root, 'commit', '-qm', 'v1.2.2: baseline')
    _git(root, 'tag', 'v1.2.2')
    (root / 'XTA' / '__init__.py').write_bytes(b'__version__ = "1.2.3"\n')
    (root / 'GPT-6-Astra-Ultra_v1.2.2_SLURM.py').rename(root / 'GPT-6-Astra-Ultra_v1.2.3_SLURM.py')
    _git(root, 'add', '.')
    _git(root, 'commit', '-qm', 'v1.2.3: test release')
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
    assert manifest['release_version_check']['release_ready'] is True
    assert manifest['release_version_check']['previous_release']['tag'] == 'v1.2.2'
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
    assert manifest['release_version_check']['release_ready'] is False
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


def test_version_gap_cannot_create_a_release_archive(tmp_path):
    root = _repository(tmp_path)
    (root / 'XTA' / '__init__.py').write_bytes(b'__version__ = "1.2.4"\n')
    (root / 'GPT-6-Astra-Ultra_v1.2.3_SLURM.py').rename(root / 'GPT-6-Astra-Ultra_v1.2.4_SLURM.py')
    _git(root, 'add', '.')
    _git(root, 'commit', '-qm', 'v1.2.4: skipped untagged development checkpoint')
    _git(root, 'tag', 'v1.2.4')
    output = tmp_path / 'blocked-release'
    with pytest.raises(ValueError, match='(?i)gap'):
        build_source_release.build(root=root, output=output)
    assert not output.exists()
    archive = build_source_release.build(root=root, output=tmp_path / 'accepted-release',
        acknowledge_gap='v1.2.2:v1.2.4', reason='User approved skipping the untagged development label')
    with zipfile.ZipFile(archive) as source:
        manifest = json.loads(source.read('XTA_v1.2.4/RELEASE_MANIFEST.json'))
    assert manifest['release_version_check']['release_ready'] is True
    assert manifest['release_version_check']['acknowledgement']


def test_version_check_cannot_certify_different_committed_payloads(tmp_path, monkeypatch):
    root = _repository(tmp_path)
    monkeypatch.setattr(build_source_release, 'check_release_version', lambda *args, **kwargs: {
        'target_version': '1.2.3', 'head_commit': 'f' * 40, 'warnings': [],
    })
    with pytest.raises(ValueError, match='Source identity changed'):
        build_source_release.build(root=root, output=tmp_path / 'mismatched-source')


def test_complete_source_bundle_includes_native_and_agent_development_files(tmp_path):
    root = _repository(tmp_path)
    members = {'AGENTS.md': b'Check release numbering.\n',
               'native/custom_kernel.cu': b'// maintained CUDA source\n',
               'native/custom_kernel.hpp': b'// maintained native header\n',
               '.github/workflows/release.yaml': b'name: release\n'}
    for name, content in members.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    snapshot = build_source_release.build(root=root, output=tmp_path / 'snapshot', snapshot=True)
    for name, content in members.items():
        assert _member(snapshot, name) == content
    _git(root, 'add', '.')
    _git(root, 'commit', '-qm', 'v1.2.3: include complete development source')
    release = build_source_release.build(root=root, output=tmp_path / 'release')
    for name, content in members.items():
        assert _member(release, name) == content


@pytest.mark.parametrize('name', ['RELEASE_MANIFEST.json', 'READ_ME_FIRST.txt'])
def test_bundle_rejects_tracked_generated_metadata_before_overwriting_it(tmp_path, name):
    root = _repository(tmp_path)
    original = b'Preserve this tracked content.\n'
    (root / name).write_bytes(original)
    _git(root, 'add', '.')
    _git(root, 'commit', '-qm', 'v1.2.3: invalid generated metadata fixture')
    with pytest.raises(ValueError, match='Reserved.*metadata'):
        build_source_release.build(root=root, output=tmp_path / 'release')
    assert (root / name).read_bytes() == original
    assert not (tmp_path / 'release').exists()


def test_complete_source_bundle_rejects_tracked_generated_paths(tmp_path):
    root = _repository(tmp_path)
    generated = root / 'build/generated.txt'
    generated.parent.mkdir()
    generated.write_bytes(b'Generated content must stay outside the maintained source.\n')
    _git(root, 'add', '.')
    _git(root, 'commit', '-qm', 'v1.2.3: invalid tracked build output')
    with pytest.raises(ValueError, match='Generated/cache path'):
        build_source_release.build(root=root, output=tmp_path / 'release')
    assert not (tmp_path / 'release').exists()
