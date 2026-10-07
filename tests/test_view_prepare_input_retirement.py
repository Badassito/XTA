"""Rebound preparation retires the original input after its actual last owner."""
from contextlib import nullcontext, redirect_stdout
import gc
from io import StringIO
import os
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from XTA import assembly, geometry, runtime
from XTA.interpolation import CVOL_FORMAT, PreparedViewResult, materialize_raw_bbox_mask_store_workspace, write_raw_bbox_mask_store
from XTA.projection_queue import settle_prepared_view_components
from XTA.view_prepare import AdmittedViewPrepare, _array_lifetime_owner


def _result(task_view, volume):
    return PreparedViewResult('model', task_view.name, '0', 0., volume, volume, [])


def _task(root, *, keep=False, restored=False):
    root.mkdir(parents=True, exist_ok=True)
    expected = np.zeros((3, 4, 5), np.uint8)
    expected[0, 1, 2] = expected[2, 2, 3] = 1
    view = geometry.get_view_infos(*expected.shape, cartesian_views=('transverse',))[0]
    path = root / 'original.u8.dat'
    if restored:
        shadow = root / 'shadow.cvol'
        write_raw_bbox_mask_store(expected, shadow, format_name=CVOL_FORMAT, workers=1)
        original = None
    else:
        shadow = None
        original = np.memmap(path, mode='w+', dtype=np.uint8, shape=expected.shape)
        original[:] = expected
        original.flush()

    def prepare(**kwargs):
        return _result(view, np.array(kwargs['union_mm'], copy=True))

    task = AdmittedViewPrepare(
        admission=SimpleNamespace(reserve=lambda *_a:nullcontext()), transient_bytes=1,
        model_name='model', view=view, union_mm=original, confmap_mm=None,
        d1_shadow_path=shadow, union_path=path, confmap_path=None, temp_dir=root,
        dense_tiling_active=False, min_conf=0, min_radius=0, interpolation_distance=0,
        interpolation_walk_back=1, interpolation_candidates=1, interpolation_passes=1,
        interpolation_min_radius=0, interpolation_search_angle=0, keep_temp_artifacts=keep,
        slice_workers=1, interpolation_task_workers=1, component_layers_needed=True,
        precleaned_slice_cleanup=True, hole_fill_done_on_device=True, slice_meta=None,
        fuse_azimuthal_component_layers=lambda:False, component_ref_dense_retirement_active=True,
        preinterpolation_layer_already_published=False, parent_mask_ready_callback=None,
        submit_component_projection=lambda *_a,**_kw:None,
        materialize_workspace=materialize_raw_bbox_mask_store_workspace, prepare=prepare,
    )
    return task, expected


@pytest.mark.parametrize('restored', [False, True])
@pytest.mark.parametrize('keep', [False, True])
def test_rebound_success_retires_original_and_preserves_debug_input(tmp_path, restored, keep):
    task, expected = _task(tmp_path, keep=keep, restored=restored)
    path = task.union_path
    with redirect_stdout(StringIO()):
        result = task()
    mapping = result._dense_input_owner_ref
    if not restored:
        assert path.exists() and mapping() is not None
        np.testing.assert_array_equal(task.union_mm, expected)
    task = None
    gc.collect()
    assert mapping() is None
    runtime.wait_for_retired_memmap_unlinks(path=path)
    assert path.exists() is keep
    np.testing.assert_array_equal(result.final_view_volume_mm, expected)


@pytest.mark.parametrize('alias_kind', ['slice', 'frombuffer'])
def test_outside_alias_keeps_replaced_original_readable_and_on_disk(tmp_path, alias_kind):
    task, expected = _task(tmp_path)
    path = task.union_path
    alias = (task.union_mm[:] if alias_kind == 'slice' else
             np.frombuffer(task.union_mm._mmap, dtype=np.uint8).reshape(expected.shape))
    mapping = weakref.ref(task.union_mm._mmap)
    result = task()
    task = None
    gc.collect()
    assert mapping() is not None and not mapping().closed and path.exists()
    np.testing.assert_array_equal(alias, expected)
    np.testing.assert_array_equal(result.final_view_volume_mm, expected)
    alias = None
    gc.collect()
    runtime.wait_for_retired_memmap_unlinks(path=path)
    assert mapping() is None and not path.exists()


@pytest.mark.parametrize('alias_kind', ['same', 'slice', 'frombuffer'])
def test_returned_alias_is_not_mistaken_for_a_replaced_allocation(tmp_path, alias_kind):
    task, expected = _task(tmp_path)
    path = task.union_path
    calls = []

    def prepare(**kwargs):
        original = kwargs['union_mm']
        returned = (original if alias_kind == 'same' else original[:] if alias_kind == 'slice' else
                    np.frombuffer(original._mmap, dtype=np.uint8).reshape(expected.shape))
        assert _array_lifetime_owner(returned) is _array_lifetime_owner(original)
        return _result(task.view, returned)

    def close(_array, *, unlink_path):
        calls.append(unlink_path)

    task.prepare, task.close_dense = prepare, close
    result = task()
    assert not calls and path.exists()
    np.testing.assert_array_equal(result.final_view_volume_mm, expected)
    # Ordinary downstream retirement still owns an unreplaced mapping.
    runtime.close_memmap_array_without_flush(task.union_mm, unlink_path=path)
    task.prepare = None
    task = result = None
    gc.collect()
    runtime.wait_for_retired_memmap_unlinks(path=path)
    assert not path.exists()


def test_writable_nonweakrefable_buffer_preserves_prepare_contract(tmp_path):
    task, expected = _task(tmp_path)
    storage = bytearray(expected.tobytes())
    task.union_mm = np.frombuffer(storage, dtype=np.uint8).reshape(expected.shape)
    result = task()
    assert task.union_mm.flags.writeable
    np.testing.assert_array_equal(result.final_view_volume_mm, expected)
    assert isinstance(result._dense_input_owner_ref(), memoryview)
    storage[0] = 1
    assert task.union_mm[0, 0, 0] == 1 and result.final_view_volume_mm[0, 0, 0] == 0


@pytest.mark.parametrize('fault', ['replaced', 'missing', 'unverifiable'])
def test_path_identity_changes_before_scheduling_leave_the_path_untouched(tmp_path, monkeypatch, fault):
    task, expected = _task(tmp_path)
    path, real_lstat = task.union_path, Path.lstat
    changed = [False]
    original_stat = path.lstat()

    def lstat(candidate, *args, **kwargs):
        if candidate == path and (changed[0] or fault == 'unverifiable'):
            if fault == 'missing':
                raise FileNotFoundError(str(candidate))
            if fault == 'unverifiable':
                raise PermissionError(str(candidate))
            return SimpleNamespace(st_dev=original_stat.st_dev, st_ino=original_stat.st_ino + 1)
        return real_lstat(candidate, *args, **kwargs)

    def prepare(**kwargs):
        changed[0] = True
        return _result(task.view, np.array(kwargs['union_mm'], copy=True))

    task.prepare = prepare
    monkeypatch.setattr(Path, 'lstat', lstat)
    result = task()
    task.prepare = None
    task = None
    gc.collect()
    runtime.wait_for_retired_memmap_unlinks(path=path)
    # Simulate lookup/identity faults without replacing a live Windows mmap.
    assert path.exists() and path.read_bytes() == expected.tobytes()
    np.testing.assert_array_equal(result.final_view_volume_mm, expected)


def test_replacement_after_scheduling_is_preserved_by_deferred_retirement(tmp_path, monkeypatch):
    task, expected = _task(tmp_path)
    path = task.union_path
    result = task()
    if os.name == 'nt':
        real_lstat, identity = Path.lstat, path.lstat()

        def lstat(candidate, *args, **kwargs):
            if candidate == path:
                return SimpleNamespace(st_dev=identity.st_dev, st_ino=identity.st_ino + 1)
            return real_lstat(candidate, *args, **kwargs)

        # Windows cannot replace a live mapped file. Exercise the worker's
        # changed-inode branch while preserving the actual original bytes.
        monkeypatch.setattr(Path, 'lstat', lstat)
        preserved = expected.tobytes()
    else:
        path.rename(tmp_path / 'old-allocation.dat')
        path.write_bytes(b'replacement')
        preserved = b'replacement'
    task = None
    gc.collect()
    runtime.wait_for_retired_memmap_unlinks(path=path)
    assert path.read_bytes() == preserved
    np.testing.assert_array_equal(result.final_view_volume_mm, expected)


@pytest.mark.parametrize('keep', [False, True])
def test_prepare_failure_keeps_existing_alias_safe_cleanup_behavior(tmp_path, keep):
    task, expected = _task(tmp_path, keep=keep)
    path = task.union_path
    alias = task.union_mm[:]

    def prepare(**_kwargs):
        raise RuntimeError('prepare failed before handoff')

    task.prepare = prepare
    with pytest.raises(RuntimeError, match='before handoff') as caught:
        task()
    assert path.exists()
    np.testing.assert_array_equal(alias, expected)
    caught.value.__traceback__ = None
    caught = task = alias = None
    gc.collect()
    runtime.wait_for_retired_memmap_unlinks(path=path)
    assert path.exists() is keep


def test_original_retirement_does_not_block_or_damage_pending_sam_publication(tmp_path):
    from XTA.config import GIB
    from XTA.interpolation import _ByteAdmissionPool
    from tests.test_sam_async_layer_publication import (
        SelectedContext, make_queue, make_submitter, owned_task, read_layer,
    )
    shape = (3, 7, 9)
    view = geometry.get_view_infos(*shape, cartesian_views=('transverse',))[0]
    entered, release = Event(), Event()

    def publish(entry, **kwargs):
        entered.set()
        if not release.wait(10):
            raise RuntimeError('publication gate stalled')
        return assembly.materialize_sam_directional_view_layer(entry, **kwargs)

    queue = make_queue()
    assembly.set_final_source_output_shape(shape)
    task = None
    try:
        with redirect_stdout(StringIO()):
            task = owned_task(tmp_path / 'async', view, SelectedContext(tmp_path / 'evidence', shape),
                              make_submitter(queue, shape, publish), _ByteAdmissionPool(16*GIB, 'test'))
            path = task.union_path
            result = task()
            assert entered.wait(3) and not settle_prepared_view_components(result)
            expected = result.final_view_volume_mm.copy()
            task = None
            gc.collect()
            runtime.wait_for_retired_memmap_unlinks(path=path)
            assert not path.exists()
            assert all(Path(future._xta_dense_independent_publication_paths[0]).is_dir()
                       for future in result.pending_component_layers)
            release.set()
            queue.shutdown()
            assert settle_prepared_view_components(result)
            fused = np.zeros(shape, np.uint8)
            for ref in result.nrrd_layers:
                fused |= read_layer(ref, shape)
            np.testing.assert_array_equal(fused, expected)
    finally:
        release.set()
        queue.abort()
        queue.shutdown(cancel_futures=True)
        task = None
        gc.collect()
        assembly.set_final_source_output_shape(None)
