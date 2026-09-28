"""Retired workspaces remain valid until their last NumPy consumer releases them."""
from __future__ import annotations

import gc
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import weakref

import numpy as np
import pytest

from XTA import runtime


def test_retained_memmap_and_ndarray_views_survive_retirement_in_child(tmp_path):
    # A regression in this path terminates the interpreter, so keep it out of pytest.
    script = textwrap.dedent('''
        import gc
        from pathlib import Path
        import numpy as np
        from XTA.runtime import (close_memmap_array_without_flush,
                                 wait_for_retired_memmap_unlinks)

        path = Path(__import__('sys').argv[1])
        owner = np.memmap(path, mode='w+', dtype=np.uint8, shape=(8, 16))
        owner[:] = 19
        memmap_view = owner[2:6]
        ndarray_view = np.asarray(memmap_view)[1:3]
        import weakref
        mapping = weakref.ref(owner._mmap)
        close_memmap_array_without_flush(memmap_view, unlink_path=path)
        assert mapping() is not None and not mapping().closed
        del owner, memmap_view
        gc.collect()
        assert int(ndarray_view.max()) == 19
        assert mapping() is not None and not mapping().closed and path.exists()
        del ndarray_view
        gc.collect()
        wait_for_retired_memmap_unlinks(path=path)
        assert mapping() is None and not path.exists()
    ''')
    path = tmp_path / 'retired.dat'
    result = subprocess.run(
        [sys.executable, '-B', '-c', script, str(path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, encoding='utf-8', errors='replace',
        timeout=20, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_partial_open_traceback_keeps_its_view_readable_in_child(tmp_path):
    # The old rollback closed the first root while its traceback still held a view.
    script = textwrap.dedent('''
        import gc
        from pathlib import Path
        from unittest import mock
        import numpy as np
        from XTA import tta_augmentation_runtime as policy

        root = Path(__import__('sys').argv[1])
        mask, confidence = root / 'mask.dat', root / 'confidence.dat'
        mask.write_bytes(bytes([7]) * 48)
        confidence.write_bytes(bytes([9]) * 48)
        task = {'result_mask_path': str(mask), 'result_conf_path': str(confidence),
                'processing_shape': (2, 4, 6), 'slice_start': 0, 'slice_count': 2}
        owned, opened = [], []
        original = np.memmap
        def opening(*args, **kwargs):
            if opened:
                raise OSError('second mapping failed')
            value = original(*args, **kwargs)
            opened.append(value)
            return value
        with mock.patch.object(policy.np, 'memmap', side_effect=opening):
            try:
                policy._open_policy_sibling_outputs(task, shape=(2, 4, 6),
                                                    padding_count=0, shared_parent=True,
                                                    owned=owned)
            except OSError as exc:
                traceback = exc.__traceback__
                while traceback and traceback.tb_frame.f_code.co_name != '_open_policy_sibling_outputs':
                    traceback = traceback.tb_next
                assert traceback is not None
                view = traceback.tb_frame.f_locals['buffers'][0]
                assert int(view[0, 0, 0]) == 7
                assert '7' in repr(view)
                assert not opened[0]._mmap.closed and not owned
                del view, traceback
            else:
                raise AssertionError('partial open unexpectedly succeeded')
        opened.clear()
        gc.collect()
    ''')
    result = subprocess.run(
        [sys.executable, '-B', '-c', script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, encoding='utf-8', errors='replace',
        timeout=20, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_retirement_does_not_delete_replacement_file(tmp_path):
    path = tmp_path / 'replace.dat'
    owner = np.memmap(path, mode='w+', dtype=np.uint8, shape=(8,))
    view = owner[:]
    runtime.close_memmap_array_without_flush(owner, unlink_path=path)
    if os.name == 'nt':
        # A mapped file cannot be replaced on Windows until the view is gone.
        del view, owner
        gc.collect()
        runtime.wait_for_retired_memmap_unlinks(path=path)
        assert not path.exists()
        return
    path.unlink()
    path.write_bytes(b'replacement')
    del view, owner
    gc.collect()
    runtime.wait_for_retired_memmap_unlinks(path=path)
    assert path.read_bytes() == b'replacement'


def test_retirement_unlinks_its_symlink_without_deleting_target(tmp_path):
    target = tmp_path / 'target.dat'
    target.write_bytes(bytes([23]) * 8)
    link = tmp_path / 'owner-link.dat'
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f'file symlinks unavailable: {exc}')
    owner = np.memmap(link, mode='r+', dtype=np.uint8, shape=(8,))
    view = owner[:]
    runtime.close_memmap_array_without_flush(owner, unlink_path=link)
    assert int(view[0]) == 23 and link.is_symlink()
    del owner, view
    gc.collect()
    runtime.wait_for_retired_memmap_unlinks(path=link)
    assert not link.is_symlink() and target.read_bytes() == bytes([23]) * 8


def test_memfd_owner_key_is_released_after_last_view(tmp_path):
    path = tmp_path / 'owner.dat'
    owner = np.memmap(path, mode='w+', dtype=np.uint8, shape=(8,))
    owner[:] = 7
    view = owner[2:]
    key = f'memmap-retirement-test-{id(owner)}'
    with path.open('rb') as file:
        fd = os.dup(file.fileno())
    owner._workspace_memfd_owner_key = key
    runtime._register_memfd_owner(key, fd, 'memmap retirement test')
    try:
        mapping = weakref.ref(owner._mmap)
        runtime.close_memmap_array_without_flush(view)
        del owner
        gc.collect()
        assert key in runtime._MEMFD_OWNERS
        assert int(view[0]) == 7 and mapping() is not None and not mapping().closed
        del view
        gc.collect()
        assert mapping() is None and key not in runtime._MEMFD_OWNERS
    finally:
        runtime._release_memfd_owner_key(key)


def test_external_buffer_export_prevents_unsafe_unmap(tmp_path):
    path = tmp_path / 'export.dat'
    owner = np.memmap(path, mode='w+', dtype=np.uint8, shape=(8,))
    owner[:] = 11
    external = np.ndarray((8,), dtype=np.uint8, buffer=owner._mmap)
    mapping = weakref.ref(owner._mmap)
    runtime.close_memmap_array_without_flush(owner, unlink_path=path)
    del owner
    gc.collect()
    assert mapping() is not None and not mapping().closed and int(external.sum()) == 88
    with pytest.raises(TimeoutError, match='remains blocked'):
        runtime.wait_for_retired_memmap_unlinks(path=path, timeout_s=0.05)
    assert path.exists()
    del external
    gc.collect()
    runtime.wait_for_retired_memmap_unlinks(path=path)
    assert mapping() is None
    assert not path.exists()


def test_deferred_cleanup_removes_only_an_empty_owned_directory(tmp_path):
    directory = tmp_path / 'owned'
    directory.mkdir()
    path = directory / 'mapping.dat'
    owner = np.memmap(path, mode='w+', dtype=np.uint8, shape=(8,))
    view = owner[:]
    runtime.close_memmap_array_without_flush(owner, unlink_path=path)
    runtime.defer_retired_memmap_directory_cleanup(directory)
    assert directory.exists() and path.exists() and int(view[0]) == 0
    del owner, view
    gc.collect()
    runtime.wait_for_retired_memmap_unlinks(path=path)
    runtime.wait_for_retired_memmap_directory_cleanup(directory)
    assert not directory.exists()

    guarded = tmp_path / 'guarded'
    guarded.mkdir()
    unrelated = guarded / 'unrelated.txt'
    unrelated.write_text('keep')
    runtime.defer_retired_memmap_directory_cleanup(guarded)
    with pytest.raises(TimeoutError, match='remains blocked'):
        runtime.wait_for_retired_memmap_directory_cleanup(guarded, timeout_s=0.05)
    assert unrelated.read_text() == 'keep'
    unrelated.unlink()
    runtime.wait_for_retired_memmap_directory_cleanup(guarded)
    assert not guarded.exists()


def test_inherited_retirement_callback_cannot_delete_parent_scratch(tmp_path):
    path = tmp_path / 'parent.dat'
    path.write_bytes(b'parent')
    stat = path.stat()
    runtime._finish_memmap_retirement(-1, {
        'owner_pid': os.getpid() + 1,
        'owner_key': None,
        'unlink_paths': [(path, int(stat.st_dev), int(stat.st_ino))],
    })
    assert path.read_bytes() == b'parent'


def test_fork_reset_replaces_inherited_worker_state_in_child_process():
    script = textwrap.dedent('''
        from pathlib import Path
        from XTA import runtime
        old_locks = (runtime._MEMMAP_RETIREMENT_LOCK,
                     runtime._MEMMAP_UNLINK_LOCK, runtime._MEMFD_OWNER_LOCK)
        old_event = runtime._MEMMAP_UNLINK_EVENT
        runtime._MEMMAP_PENDING_UNLINKS[(Path('sentinel'), 1, 1)] = (0.0, False)
        runtime._reset_memmap_retirement_after_fork()
        assert not runtime._MEMMAP_PENDING_UNLINKS
        assert not runtime._MEMMAP_PENDING_DIRECTORIES
        assert not runtime._MEMMAP_RETIREMENTS
        assert runtime._MEMMAP_UNLINK_WORKER is None
        assert runtime._MEMMAP_UNLINK_EVENT is not old_event
        assert all(new is not old for new, old in zip(
            (runtime._MEMMAP_RETIREMENT_LOCK, runtime._MEMMAP_UNLINK_LOCK,
             runtime._MEMFD_OWNER_LOCK), old_locks))
    ''')
    result = subprocess.run(
        [sys.executable, '-B', '-c', script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, encoding='utf-8', errors='replace',
        timeout=20, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='fork is not available on Windows')
def test_fork_child_does_not_delete_parent_owned_retirement(tmp_path):
    script = textwrap.dedent('''
        import gc
        import os
        from pathlib import Path
        import numpy as np
        from XTA.runtime import (close_memmap_array_without_flush,
                                 wait_for_retired_memmap_unlinks)

        path = Path(__import__('sys').argv[1])
        owner = np.memmap(path, mode='w+', dtype=np.uint8, shape=(8,))
        view = owner[:]
        close_memmap_array_without_flush(owner, unlink_path=path)
        child = os.fork()
        if child == 0:
            del owner, view
            gc.collect()
            os._exit(0 if path.exists() else 7)
        _pid, status = os.waitpid(child, 0)
        assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
        assert path.exists() and int(view[0]) == 0
        del owner, view
        gc.collect()
        wait_for_retired_memmap_unlinks(path=path)
        assert not path.exists()
    ''')
    result = subprocess.run(
        [sys.executable, '-B', '-c', script, str(tmp_path / 'fork-parent.dat')],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, encoding='utf-8', errors='replace',
        timeout=20, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
