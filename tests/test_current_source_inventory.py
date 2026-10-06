"""Current-source identity guards; numerical oracles remain in runtime suites."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace
from unittest import mock

import pytest

from tools import prepare_reconciliation_release as producer
from tools import verify_package_inventory as inventory


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'source'
    (root / 'XTA').mkdir(parents=True)
    (root / 'XTA/__init__.py').write_bytes(b'__version__ = "1.2.3"\r\n')
    (root / 'native').mkdir()
    (root / 'native/kernel.cu').write_bytes(b'// maintained native source\r\n')
    (root / 'AGENTS.md').write_bytes(b'Keep maintained source.\r\n')
    (root / 'binary.dat').write_bytes(b'\0\r\n\xff')
    return root


def record(root):
    value, _ = inventory.build_inventory(root)
    path = root / inventory.INVENTORY_PATH
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(producer._json_bytes(value))
    return value


def git(root, *args):
    return subprocess.check_output(('git', *args), cwd=root,
                                   env=dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull,
                                            GIT_CONFIG_NOSYSTEM='1'))


def commit(root):
    git(root, 'add', '--all')
    git(root, '-c', 'user.name=Inventory Test', '-c', 'user.email=inventory@example.invalid',
        'commit', '-q', '-m', 'source fixture')


def test_flat_record_covers_all_maintained_types_and_excludes_only_itself(source):
    value = record(source)
    assert set(value) == {'schema', 'digest_representation', 'version', 'files'}
    assert set(value['files']) == {'XTA/__init__.py', 'native/kernel.cu', 'AGENTS.md', 'binary.dat'}
    assert inventory.INVENTORY_PATH not in value['files']
    assert inventory.verify(source)['files'] == 4
    raw = inventory.capture_source_identity(source)
    assert raw['files'][inventory.INVENTORY_PATH] == inventory.digest((source / inventory.INVENTORY_PATH).read_bytes())


def test_shared_raw_source_reader_preserves_crlf_and_binary_exactly(source):
    record(source)
    raw = inventory.read_source_files(source)
    assert raw['AGENTS.md'] == b'Keep maintained source.\r\n'
    assert raw['binary.dat'] == b'\0\r\n\xff'
    assert inventory.INVENTORY_PATH in raw
    canonical = inventory.collect_source_files(source)
    assert canonical['AGENTS.md'] == b'Keep maintained source.\n'
    assert canonical['binary.dat'] == raw['binary.dat']


@pytest.mark.parametrize('data, expected', [(b'line\r\n', b'line\n'),
    (b'line\r', b'line\r'), (b'\0\r\n', b'\0\r\n'), (b'\xff\r\n', b'\xff\r\n')])
def test_canonical_digest_representation_preserves_binary_and_lone_cr(data, expected):
    assert inventory.canonical_source_bytes(data) == expected


@pytest.mark.parametrize('mutation, message', [('extra', 'file set'), ('missing', 'file set'),
                                              ('content', 'digest differs'), ('version', 'version differs')])
def test_source_mutations_fail_closed(source, mutation, message):
    record(source)
    if mutation == 'extra':
        (source / '.new-source').write_bytes(b'new maintained input')
    elif mutation == 'missing':
        (source / 'native/kernel.cu').unlink()
    elif mutation == 'content':
        (source / 'native/kernel.cu').write_bytes(b'changed native code')
    else:
        (source / 'XTA/__init__.py').write_bytes(b'__version__ = "1.2.4"\n')
    with pytest.raises(RuntimeError, match=message):
        inventory.verify(source)


@pytest.mark.parametrize('mutation', ['code', 'newlines', 'deleted', 'extra', 'inventory'])
def test_raw_changes_during_verification_cannot_hide_behind_canonical_bytes(source, mutation):
    record(source)
    original = inventory._inventory_for

    def mutate(payloads):
        result = original(payloads)
        if mutation == 'code':
            (source / 'AGENTS.md').write_bytes(b'Changed instructions.\r\n')
        elif mutation == 'newlines':
            (source / 'AGENTS.md').write_bytes(b'Keep maintained source.\n')
        elif mutation == 'deleted':
            (source / 'AGENTS.md').unlink()
        elif mutation == 'extra':
            (source / 'added.md').write_bytes(b'new')
        else:
            path = source / inventory.INVENTORY_PATH
            path.write_bytes(path.read_bytes() + b' ')
        return result

    with mock.patch.object(inventory, '_inventory_for', side_effect=mutate):
        with pytest.raises(RuntimeError, match='Current source changed during'):
            inventory.verify(source)


def test_independent_pass_observes_new_source_and_has_no_stale_cache(source):
    record(source)
    inventory.verify(source)
    (source / 'AGENTS.md').write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='digest differs'):
        inventory.verify(source)
    record(source)
    inventory.verify(source)


def test_source_replacement_during_initial_read_is_rejected(source):
    target = source / 'AGENTS.md'
    original = Path.read_bytes

    def read(path):
        result = original(path)
        if path == target:
            path.write_bytes(result + b'changed during read')
        return result

    with mock.patch.object(Path, 'read_bytes', read):
        with pytest.raises(RuntimeError, match='changed while reading'):
            inventory.build_inventory(source)


@pytest.mark.parametrize('name', ['../escape.py', '/absolute.py', 'C:/absolute.py',
                                'XTA\\escape.py', 'XTA/./module.py', 'XTA//module.py'])
def test_inventory_rejects_invalid_paths(source, name):
    value = record(source)
    value['files'][name] = '0' * 64
    with pytest.raises(RuntimeError, match='Invalid source member path'):
        inventory.load_inventory(producer._json_bytes(value))


@pytest.mark.parametrize('mutation', ['self', 'digest', 'representation', 'extra-field', 'case'])
def test_inventory_structure_cannot_relax_guard(source, mutation):
    value = record(source)
    if mutation == 'self':
        value['files'][inventory.INVENTORY_PATH] = '0' * 64
    elif mutation == 'digest':
        value['files']['AGENTS.md'] = 'not-a-digest'
    elif mutation == 'representation':
        value['digest_representation'] = 'trust-current-source'
    elif mutation == 'extra-field':
        value['reviewed'] = True
    else:
        value['files']['agents.md'] = value['files']['AGENTS.md']
    with pytest.raises(RuntimeError):
        inventory.load_inventory(producer._json_bytes(value))


def test_duplicate_json_keys_are_rejected():
    with pytest.raises(RuntimeError, match='Duplicate inventory JSON key'):
        inventory.load_inventory(b'{"version":"1.2.3","version":"9.9.9"}')


def test_symbolic_links_cannot_escape_source(source, tmp_path):
    destination = tmp_path / 'external'
    destination.write_bytes(b'external data')
    link = source / 'linked.py'
    try:
        link.symlink_to(destination)
    except OSError:
        pytest.skip('Host does not permit creating test symbolic links')
    with pytest.raises(RuntimeError, match='symbolic link or reparse point'):
        inventory.build_inventory(source)


def test_reparse_points_are_rejected_even_without_host_link_privilege(source):
    original = Path.lstat
    flag = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)

    def metadata(path):
        value = original(path)
        if path == source / 'native':
            return SimpleNamespace(st_mode=value.st_mode, st_file_attributes=flag)
        return value

    with mock.patch.object(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', flag, create=True), \
         mock.patch.object(Path, 'lstat', metadata):
        with pytest.raises(RuntimeError, match='symbolic link or reparse point'):
            inventory.build_inventory(source)


def test_portable_archive_metadata_and_runtime_caches_are_not_source(source):
    record(source)
    (source / 'RELEASE_MANIFEST.json').write_bytes(b'outer raw-byte manifest')
    (source / 'READ_ME_FIRST.txt').write_bytes(b'release instructions')
    (source / 'XTA/__pycache__').mkdir()
    (source / 'XTA/__pycache__/module.pyc').write_bytes(b'generated')
    inventory.verify(source)


def test_git_source_set_includes_untracked_native_files_and_pending_deletions(source):
    git(source, 'init', '-q')
    commit(source)
    record(source)
    (source / 'native/new.cuh').write_bytes(b'new maintained native input')
    with pytest.raises(RuntimeError, match='file set'):
        inventory.verify(source)
    (source / 'native/new.cuh').unlink()
    (source / 'native/kernel.cu').unlink()
    assert inventory.capture_source_identity(source)['files']['native/kernel.cu'] is None
    with pytest.raises(RuntimeError, match='file set'):
        inventory.verify(source)


@pytest.mark.parametrize('name', ['RELEASE_MANIFEST.json', 'READ_ME_FIRST.txt'])
def test_git_sources_cannot_collide_with_reserved_archive_metadata(source, name):
    git(source, 'init', '-q')
    commit(source)
    (source / name).write_bytes(b'maintained metadata collision')
    with pytest.raises(RuntimeError, match='Reserved source-bundle metadata'):
        inventory.build_inventory(source)


@pytest.mark.parametrize('name', ['__pycache__/fixture.pyc', 'XTA/__pycache__/fixture.py',
    '.pytest_cache/state.json', 'build/native.c', 'dist/readme.md', 'fixture.egg-info/PKG-INFO', 'XTA/cache.nbi'])
def test_git_tracked_generated_paths_cannot_create_bundle_file_set_ambiguity(source, name):
    git(source, 'init', '-q')
    commit(source)
    target = source / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b'generated data')
    commit(source)
    with pytest.raises(RuntimeError, match='Generated/cache path'):
        inventory.build_inventory(source)


@pytest.mark.parametrize('name', ['release_manifest.json', '__PyCache__/fixture.py'])
def test_git_source_policy_rejects_case_variants_of_reserved_and_generated_paths(source, name):
    git(source, 'init', '-q')
    commit(source)
    target = source / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b'case-variant collision')
    with pytest.raises(RuntimeError, match='cannot be maintained source'):
        inventory.build_inventory(source)


def test_extracted_source_rejects_noncanonical_case_variant_bundle_metadata(source):
    record(source)
    (source / 'release_manifest.json').write_bytes(b'ambiguous source or outer metadata')
    with pytest.raises(RuntimeError, match='Reserved source-bundle metadata'):
        inventory.verify(source)


def test_crlf_worktree_inventory_verifies_exact_lf_commit_and_extracted_archive(source, tmp_path):
    git(source, 'init', '-q')
    (source / '.gitattributes').write_bytes(b'* text=auto\n')
    commit(source)
    value = record(source)
    commit(source)
    exported = tmp_path / 'exported'
    for raw_name in git(source, 'ls-files', '-z').split(b'\0'):
        if not raw_name:
            continue
        name = raw_name.decode('utf-8')
        target = exported / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(git(source, 'show', 'HEAD:' + name))
    assert (source / 'AGENTS.md').read_bytes().endswith(b'\r\n')
    assert (exported / 'AGENTS.md').read_bytes().endswith(b'\n')
    assert not (exported / 'AGENTS.md').read_bytes().endswith(b'\r\n')
    assert inventory.verify(source)['files'] == inventory.verify(exported)['files'] == len(value['files'])
    assert inventory.collect_source_files(source) == inventory.collect_source_files(exported)


def test_draft_is_deterministic_and_does_not_mutate_source(source, tmp_path):
    record(source)
    before = inventory.capture_source_identity(source)
    first = producer.prepare(root=source, output_dir=tmp_path / 'draft-one')
    second = producer.prepare(root=source, output_dir=tmp_path / 'draft-two')
    assert first == second
    assert inventory.capture_source_identity(source) == before
    assert (tmp_path / 'draft-one/current_source_inventory.json').read_bytes() == (tmp_path / 'draft-two/current_source_inventory.json').read_bytes()


def test_write_replaces_only_flat_inventory_and_preserves_other_source(source, tmp_path):
    record(source)
    (source / 'AGENTS.md').write_bytes(b'New reviewed current instructions.\n')
    before = inventory.capture_source_identity(source)
    result = producer.prepare(root=source, output_dir=tmp_path / 'publication', write=True)
    after = inventory.capture_source_identity(source)
    assert result['written'] is True
    before['files'].pop(inventory.INVENTORY_PATH)
    after['files'].pop(inventory.INVENTORY_PATH)
    assert before == after
    inventory.verify(source)


def test_publication_failure_restores_previous_inventory(source, tmp_path):
    record(source)
    path = source / inventory.INVENTORY_PATH
    previous = path.read_bytes()
    with mock.patch.object(inventory, 'verify', side_effect=RuntimeError('failed publication')):
        with pytest.raises(RuntimeError, match='failed publication'):
            producer.prepare(root=source, output_dir=tmp_path / 'publication', write=True)
    assert path.read_bytes() == previous


@pytest.mark.parametrize('stage', ['write', 'flush', 'fsync'])
@pytest.mark.parametrize('location', ['source', 'evidence'])
def test_failed_atomic_write_preserves_existing_file_and_removes_temporary_fragment(source, tmp_path, stage, location):
    record(source)
    target = source / inventory.INVENTORY_PATH if location == 'source' else tmp_path / 'evidence/inventory.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(b'previous evidence')
    previous = target.read_bytes()
    original = producer.tempfile.NamedTemporaryFile

    class FailingHandle:
        def __init__(self, *args, **kwargs):
            self.handle = original(*args, **kwargs)
            self.name = self.handle.name

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def write(self, data):
            if stage == 'write':
                raise OSError('injected write failure')
            return self.handle.write(data)

        def flush(self):
            if stage == 'flush':
                raise OSError('injected flush failure')
            return self.handle.flush()

        def fileno(self):
            return self.handle.fileno()

    with mock.patch.object(producer.tempfile, 'NamedTemporaryFile', FailingHandle), \
         mock.patch.object(producer.os, 'fsync', side_effect=OSError('injected fsync failure')):
        with pytest.raises(OSError, match=f'injected {stage} failure'):
            producer._atomic_write(target, b'new candidate')
    assert target.read_bytes() == previous
    assert list(target.parent.glob('.inventory-*')) == []


def test_output_evidence_cannot_be_written_into_source_tree(source):
    with pytest.raises(ValueError, match='outside the source tree'):
        producer.prepare(root=source, output_dir=source / 'generated')
