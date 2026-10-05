"""One verifier pass may reuse immutable bytes, never stale files or ASTs."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from unittest import mock

import pytest

from tools import verify_package_inventory as inventory


def test_one_pass_reads_one_snapshot_and_rechecks_raw_bytes(tmp_path):
    path = tmp_path / 'source.py'
    path.write_bytes(b'VALUE = 1\r\n')
    original_read = Path.read_bytes
    reads = []

    def read_bytes(self):
        reads.append(self)
        return original_read(self)

    def verify():
        first = inventory._current_source_snapshot(path)
        assert first is inventory._current_source_snapshot(path)
        assert first[1] == 'VALUE = 1\n'
        assert first[2] == hashlib.sha256(b'VALUE = 1\n').hexdigest()
        inventory._assert_current_sources_unchanged()

    with mock.patch.object(Path, 'read_bytes', read_bytes), \
         mock.patch.object(inventory, '_verify_main', side_effect=verify):
        inventory.main()
    assert reads == [path, path]
    assert inventory._ACTIVE_SOURCE_SNAPSHOT_CACHE.get() is None


@pytest.mark.parametrize('mutation', ['code', 'newlines', 'deleted'])
def test_source_change_during_pass_cannot_be_hidden_by_snapshot(tmp_path, mutation):
    path = tmp_path / 'source.py'
    path.write_bytes(b'VALUE = 1\r\n')

    def verify():
        inventory._current_source_snapshot(path)
        if mutation == 'code':
            path.write_bytes(b'VALUE = 2\r\n')
        elif mutation == 'newlines':
            path.write_bytes(b'VALUE = 1\n')
        else:
            path.unlink()
        with pytest.raises(RuntimeError, match='Current source changed during inventory verification'):
            inventory._assert_current_sources_unchanged()
        raise RuntimeError('failed pass')

    with mock.patch.object(inventory, '_verify_main', side_effect=verify):
        with pytest.raises(RuntimeError, match='failed pass'):
            inventory.main()
    assert inventory._ACTIVE_SOURCE_SNAPSHOT_CACHE.get() is None


def test_next_independent_pass_observes_new_current_source(tmp_path):
    path = tmp_path / 'source.py'
    observed = []

    def verify():
        observed.append(inventory._current_source_snapshot(path)[1])
        inventory._assert_current_sources_unchanged()

    with mock.patch.object(inventory, '_verify_main', side_effect=verify):
        for value in (1, 2):
            path.write_text(f'VALUE = {value}\n', encoding='utf-8')
            inventory.main()
    assert observed == ['VALUE = 1\n', 'VALUE = 2\n']


def test_marker_free_modules_need_no_ast_walk():
    source = 'def function():\n    from . import dependency\n'
    tree = ast.parse(source)
    with mock.patch.object(inventory.ast, 'walk', side_effect=AssertionError('Unneeded traversal')):
        assert inventory.reviewed_local_import_seams('fixture', source, tree) == {}


def test_marker_free_fast_path_does_not_hide_a_malformed_marker():
    source = inventory.LOCAL_IMPORT_SEAM_MARKER + ' trailing text\nVALUE = 1\n'
    with pytest.raises(RuntimeError, match='marker must be the complete comment'):
        inventory.reviewed_local_import_seams('fixture', source, ast.parse(source))


def test_valid_marked_module_has_one_parent_import_traversal():
    source = 'def function():\n    ' + inventory.LOCAL_IMPORT_SEAM_MARKER + '\n    from . import dependency\n'
    tree = ast.parse(source)
    original_walk = ast.walk
    with mock.patch.object(inventory.ast, 'walk', wraps=original_walk) as walk:
        seams = inventory.reviewed_local_import_seams('fixture', source, tree)
    assert ('fixture', 'function') in seams
    assert walk.call_count == 1
