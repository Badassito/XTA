"""SAM preparation advances while bounded source-layer publication remains busy."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext, redirect_stdout
from dataclasses import replace
import gc
from io import StringIO
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import assembly, geometry, outputs
from XTA.config import GIB, TiltedViewGroup
from XTA.interpolation import CVOL_FORMAT, _ByteAdmissionPool, _DirectUnionBackingLease, write_raw_bbox_mask_store
from XTA.projection_queue import ComponentProjectionQueue, settle_prepared_view_components
from XTA.view_prepare import AdmittedViewPrepare, SamLayerProjectionSubmitter, ViewPrepareLeaseState, _array_lifetime_owner
import weakref


class SelectedContext:
    def __init__(self, root, source_shape):
        self.evidence_root = root
        self.detector_identity = 'detector'
        self.bundle_identity = 'sam-bundle'
        self.source_volume = np.zeros(source_shape, np.uint8)
        self.interpolation_entered = Event()
        self.extrapolation_entered = Event()

    def resource_scope(self, profile):
        profile._validate_owner()
        return nullcontext()

    def entries(self, volume, work_dir, kind):
        entries = []
        for direction, frame in (('forward', len(volume) // 2), ('backward', len(volume) - 1)):
            mask = np.zeros_like(volume)
            mask[frame, volume.shape[1] // 2, volume.shape[2] // 2] = 1
            path = Path(work_dir) / f'{kind}-{direction}.cvol'
            write_raw_bbox_mask_store(mask, path, format_name=CVOL_FORMAT, workers=1)
            entries.append(dict(direction=direction, path=str(path), voxel_count=int(mask.sum()),
                                policy_hash='policy', evidence_path=str(work_dir),
                                group_ids=['group'], run_ids=[direction]))
        return entries

    def interpolate(self, observations, *, work_dir, **_kwargs):
        self.interpolation_entered.set()
        entries = self.entries(observations, work_dir, 'bridge')
        merged = observations.copy()
        merged[len(observations) // 2, observations.shape[1] // 2, observations.shape[2] // 2] = 1
        merged[-1, observations.shape[1] // 2, observations.shape[2] // 2] = 1
        return merged, dict(skipped=False, added_voxels=2), entries

    def extrapolate(self, observations, *, work_dir, **_kwargs):
        self.extrapolation_entered.set()
        return observations, {}, self.entries(observations, work_dir, 'tail')


def prepare(root, view, context, submitter=None):
    volume = np.zeros((view.num_slices, view.src_h, view.src_w), np.uint8)
    volume[0, volume.shape[1] // 2, volume.shape[2] // 2] = 1
    return assembly.prepare_view_volume_after_fullframe(
        model_name='detector', view=view, union_mm=volume, confmap_mm=None,
        union_path=root / 'union.dat', confmap_path=None, temp_dir=root,
        dense_tiling_active=False, min_conf=0, min_radius=0, interpolate=2,
        interpolation_walk_back=1, interpolation_candidates=1, interpolate_passes=1,
        interpolate_min_radius=0, interpolation_search_angle=0, keep_temp=False,
        slice_workers=1, interpolation_task_workers=1, nrrd_layers_enabled=True,
        precleaned_slice_cleanup=True, hole_fill_done_on_device=True,
        preinterpolation_layer_already_published=True, interpolation_backend='sam',
        sam_context=context, extrapolation_distance=2, extrapolation_walk_back=1,
        extrapolation_min_radius=0, submit_sam_layer_projection=submitter)


def read_layer(ref, target):
    layer = outputs._open_nrrd_layer_ref(ref)
    try:
        return np.stack([outputs._read_layer_slice_in_output_shape(layer, target, index)
                         for index in range(target[0])])
    finally:
        outputs._close_nrrd_layer_source(layer)
        outputs._drop_nrrd_raw_store_chunks_ram_cache(layer)


def make_queue(**overrides):
    settings = dict(workers=2, max_pending=8, max_source_bytes=GIB,
                    max_working_bytes=16 * GIB)
    settings.update(overrides)
    return ComponentProjectionQueue(**settings)


def make_submitter(queue, target, directional=None):
    return SamLayerProjectionSubmitter(queue, target,
        directional or assembly.materialize_sam_directional_view_layer,
        assembly.materialize_sam_extrapolation_view_layer)


def test_same_parent_extrapolation_and_next_parent_progress_before_publication(tmp_path):
    target = (5, 7, 9)
    view = geometry.get_view_infos(*target, cartesian_views=('transverse',))[0]
    first, second = SelectedContext(tmp_path / 'first-evidence', target), SelectedContext(tmp_path / 'second-evidence', target)
    publishing, release = Event(), Event()
    identities = []

    def directional(entry, **kwargs):
        identities.append(kwargs['sam_context'])
        if not publishing.is_set():
            publishing.set()
            if not release.wait(10):
                raise RuntimeError('test publication handoff stalled')
        return assembly.materialize_sam_directional_view_layer(entry, **kwargs)

    queue = make_queue()
    submitter = make_submitter(queue, target, directional)
    assembly.set_final_source_output_shape(target)
    try:
        with ThreadPoolExecutor(max_workers=1) as parent, redirect_stdout(StringIO()):
            try:
                first_future = parent.submit(prepare, tmp_path / 'first', view, first, submitter)
                assert publishing.wait(3)
                first_result = first_future.result(3)
                assert first.extrapolation_entered.is_set()
                assert not settle_prepared_view_components(first_result)
                second_future = parent.submit(prepare, tmp_path / 'second', view, second, submitter)
                assert second.interpolation_entered.wait(3)
                second_result = second_future.result(3)
                assert second.extrapolation_entered.is_set()
                assert not release.is_set()
            finally:
                release.set()
        queue.shutdown()
        for result in (first_result, second_result):
            assert settle_prepared_view_components(result)
            assert len(result.nrrd_layers) == 4
            assert [(ref.mask_kind, ref.interpolation_direction or ref.extrapolation_provenance['direction'])
                    for ref in result.nrrd_layers] == [
                        ('bridge', 'forward'), ('bridge', 'backward'),
                        ('extrapolation', 'forward'), ('extrapolation', 'backward')]
            assert settle_prepared_view_components(result)
            assert len(result.nrrd_layers) == 4
        assert all(identity is not first and identity is not second for identity in identities)
        assert all(identity.source_volume.shape == target for identity in identities)
        assert all(not isinstance(identity.source_volume, np.ndarray) for identity in identities)
        stats = queue.snapshot()
        assert stats['submitted'] == stats['completed'] == 8
        assert stats['pending'] == stats['source_bytes'] == stats['working_bytes'] == 0
        assert stats['peak_pending'] <= stats['max_pending']
    finally:
        release.set()
        queue.abort()
        queue.shutdown(cancel_futures=True)
        assembly.set_final_source_output_shape(None)


def test_source_projected_layers_match_synchronous_publication_across_views(tmp_path):
    shape, target = (5, 9, 11), (7, 13, 15)
    views = geometry.get_view_infos(*shape, cartesian_views=('transverse', 'sagittal', 'coronal'),
        azimuthal_views=('transverse', 'sagittal', 'tilted_transverse'),
        azimuthal_azimuth_angles=(30., 30., 30.),
        tilt_groups=[TiltedViewGroup(('transverse',), (30.,), ('vertical',))])
    assembly.set_final_source_output_shape(target)
    try:
        with redirect_stdout(StringIO()):
            for view in views:
                root = tmp_path / view.name
                baseline = prepare(root / 'baseline', view, SelectedContext(root / 'baseline-evidence', shape))
                queue = make_queue()
                try:
                    optimized = prepare(root / 'async', view, SelectedContext(root / 'async-evidence', shape),
                                        make_submitter(queue, target))
                    queue.shutdown()
                    assert settle_prepared_view_components(optimized)
                    assert len(baseline.nrrd_layers) == len(optimized.nrrd_layers) == 4
                    for control, actual in zip(baseline.nrrd_layers, optimized.nrrd_layers):
                        assert control.key == actual.key
                        assert control.shape == actual.shape
                        assert control.native_transform == actual.native_transform
                        assert control.seed_detector_identity == actual.seed_detector_identity
                        assert control.sam_bundle_identity == actual.sam_bundle_identity
                        assert control.segment_extent_ijk == actual.segment_extent_ijk
                        np.testing.assert_array_equal(read_layer(actual, target), read_layer(control, target),
                                                      err_msg=view.name)
                finally:
                    queue.abort()
                    queue.shutdown(cancel_futures=True)
    finally:
        assembly.set_final_source_output_shape(None)


def test_publication_failure_rejects_terminal_settlement_and_returns_queue_credit(tmp_path):
    target = (5, 7, 9)
    view = geometry.get_view_infos(*target, cartesian_views=('transverse',))[0]
    publishing, release = Event(), Event()

    def fail(entry, **kwargs):
        if entry['direction'] == 'forward':
            publishing.set()
            if not release.wait(10):
                raise RuntimeError('test failed publisher stalled')
            raise ValueError('source projection failed')
        return assembly.materialize_sam_directional_view_layer(entry, **kwargs)

    queue = make_queue()
    try:
        result = prepare(tmp_path, view, SelectedContext(tmp_path / 'evidence', target),
                         make_submitter(queue, target, fail))
        assert publishing.wait(3)
        assert not settle_prepared_view_components(result)
        assert not result.nrrd_layers
        release.set()
        queue.shutdown()
        with pytest.raises(ValueError, match='source projection failed'):
            settle_prepared_view_components(result)
        assert not result.nrrd_layers
        stats = queue.snapshot()
        assert stats['failed'] == 1
        assert stats['pending'] == stats['source_bytes'] == stats['working_bytes'] == 0
    finally:
        release.set()
        queue.abort()
        queue.shutdown(cancel_futures=True)


def test_detached_extrapolation_requires_dependency_owner_before_tracking(tmp_path):
    context = SelectedContext(tmp_path / 'evidence', (3, 4, 5))
    view = geometry.get_view_infos(3, 4, 5, cartesian_views=('transverse',))[0]
    with pytest.raises(RuntimeError, match='terminal dependency owner'):
        assembly._run_sam_extrapolation(np.zeros((3, 4, 5), np.uint8),
            view=view, model_name='detector', source='fullframe', sam_context=context,
            distance=1, submit_sam_layer_projection=lambda *_args, **_kwargs: pytest.fail('orphan'))
    assert not context.extrapolation_entered.is_set()


def owned_task(root, view, context, submitter, admission):
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'union.dat'
    volume = np.memmap(path, mode='w+', shape=(view.num_slices, view.src_h, view.src_w), dtype=np.uint8)
    volume[:] = 0
    volume[0, view.src_h // 2, view.src_w // 2] = 1
    return AdmittedViewPrepare(admission=admission, transient_bytes=GIB,
        model_name='detector', view=view, union_mm=volume, confmap_mm=None,
        d1_shadow_path=None, union_path=path, confmap_path=None, temp_dir=root,
        dense_tiling_active=False, min_conf=0, min_radius=0, interpolation_distance=2,
        interpolation_walk_back=1, interpolation_candidates=1, interpolation_passes=1,
        interpolation_min_radius=0, interpolation_search_angle=0,
        keep_temp_artifacts=False, slice_workers=1, interpolation_task_workers=1,
        component_layers_needed=True, precleaned_slice_cleanup=True,
        hole_fill_done_on_device=True, slice_meta=None, fuse_azimuthal_component_layers=lambda:False,
        component_ref_dense_retirement_active=True, preinterpolation_layer_already_published=False,
        parent_mask_ready_callback=None, submit_component_projection=lambda *_a, **_kw:None,
        materialize_workspace=lambda *_a, **_kw:None, prepare=assembly.prepare_view_volume_after_fullframe,
        interpolation_backend='sam', sam_context=context, extrapolation_distance=2,
        extrapolation_walk_back=1, extrapolation_min_radius=0,
        submit_sam_layer_projection=submitter)


def test_actual_dense_retirement_readmits_staged_parent_while_publication_is_blocked(tmp_path):
    from XTA.sam_parent_staging import DeferredSamParentQueue
    target = (5, 7, 9)
    base = geometry.get_view_infos(*target, cartesian_views=('transverse',))[0]
    first_view = replace(base, name='first', physical_view_name='transverse')
    second_view = replace(base, name='second', physical_view_name='transverse')
    first = SelectedContext(tmp_path / 'first-evidence', target)
    second = SelectedContext(tmp_path / 'second-evidence', target)
    publishing, release = Event(), Event()

    def directional(entry, **kwargs):
        if not publishing.is_set():
            publishing.set()
            if not release.wait(10):
                raise RuntimeError('test staged publication stalled')
        return assembly.materialize_sam_directional_view_layer(entry, **kwargs)

    queue = make_queue()
    submitter = make_submitter(queue, target, directional)
    leases = ViewPrepareLeaseState({}, set(), {}, set(), {})
    credits = []
    admission = _ByteAdmissionPool(16 * GIB, 'test SAM parent memory')
    checkpoints, parents = ThreadPoolExecutor(1), ThreadPoolExecutor(1)
    stage = DeferredSamParentQueue(temp_dir=tmp_path / 'temp', output_dir=tmp_path / 'output',
        checkpoint_executor=checkpoints, prepare_executor=parents, leases=leases,
        dense_limit=np.prod(target), ready=lambda:True)
    try:
        with redirect_stdout(StringIO()):
            for view, context in ((first_view, first), (second_view, second)):
                task = owned_task(tmp_path / 'temp' / view.name, view, context, submitter, admission)
                key = ('detector', view.name)
                amount = int(task.union_mm.nbytes)
                leases.leases[key] = _DirectUnionBackingLease(key, amount, phase='postprocess')
                leases.postprocess_views.add(key)
                leases.postprocess_bytes[key] = amount
                stage.defer(task, amount)
                del task
            for future in list(stage.checkpoint_futures):
                future.result(3)
            resumed, _released = stage.pump()
            assert len(resumed) == 1 and len(stage.deferred) == 1
            first_future = next(iter(resumed))
            prepared = first_future.result(3)
            assert publishing.wait(3)
            assert not settle_prepared_view_components(prepared)
            assert prepared.final_view_volume_mm is not None
            assert not second.interpolation_entered.is_set()
            assert not stage.pump()[0]
            assert not leases.retire_dense_for_publication(prepared, enabled=True,
                tiled=False, keep_temp=False, retired_callback=lambda *args:credits.append(args))
            gc.collect()
            assert prepared.native_support_mm is prepared.final_view_volume_mm is None
            assert len(credits) == 1
            assert not stage.pump()[0]  # A callback cannot mutate scheduler admission itself.
            assert leases.settle_publication_retirement(*credits.pop())
            second_resumed, _released = stage.pump()
            assert len(second_resumed) == 1
            second_prepared = next(iter(second_resumed)).result(3)
            assert second.interpolation_entered.is_set() and not release.is_set()
            assert not settle_prepared_view_components(prepared)
            release.set()
            queue.shutdown()
            assert settle_prepared_view_components(prepared)
            assert settle_prepared_view_components(second_prepared)
            assert len(prepared.nrrd_layers) == len(second_prepared.nrrd_layers) == 5
    finally:
        release.set()
        queue.abort()
        queue.shutdown(cancel_futures=True)
        checkpoints.shutdown()
        parents.shutdown()
        stage.close()


def test_dense_alias_and_confidence_credit_fence_early_retirement(tmp_path):
    from concurrent.futures import Future
    target = (3, 4, 5)
    path = tmp_path / 'mask.dat'
    dense = np.memmap(path, mode='w+', shape=target, dtype=np.uint8)
    dense[:] = 1
    alias = np.frombuffer(dense._mmap, dtype=np.uint8).reshape(target)
    original_ref = weakref.ref(_array_lifetime_owner(dense))
    future = Future()
    future._xta_dense_independent_publication_paths = (tmp_path / 'selected.cvol',)
    key = ('detector', 'transverse')
    lease = _DirectUnionBackingLease(key, 2*dense.nbytes, phase='postprocess')
    leases = ViewPrepareLeaseState({key:lease}, set(), {}, {key}, {key:lease.nbytes})
    prepared = SimpleNamespace(model_name=key[0], view_name=key[1], native_support_mm=dense,
        final_view_volume_mm=dense, parent_mask_support_mm=None, parent_bridge_support_mm=None,
        pending_component_layers=[future], nrrd_layers=[], _dense_input_owner_ref=original_ref,
        _dense_input_nbytes=dense.nbytes)
    credits = []
    del dense
    assert not leases.retire_dense_for_publication(prepared, enabled=True, tiled=False,
        keep_temp=False, retired_callback=lambda *args:credits.append(args))
    assert prepared.final_view_volume_mm is not None and not credits
    assert leases.retire_input_bytes(key, lease, 60, token='confidence')
    assert not leases.retire_dense_for_publication(prepared, enabled=True, tiled=False,
        keep_temp=False, retired_callback=lambda *args:credits.append(args))
    gc.collect()
    assert prepared.native_support_mm is prepared.final_view_volume_mm is None
    assert not credits and original_ref() is not None
    assert int(alias.sum()) == 60 and path.exists()
    del alias
    gc.collect()
    assert len(credits) == 1 and original_ref() is None
    assert leases.settle_publication_retirement(*credits[0])
    assert not leases.settle_publication_retirement(*credits[0])
    assert not leases.retire_dense_for_publication(prepared, enabled=True, tiled=False,
        keep_temp=False, retired_callback=lambda *args:pytest.fail('duplicate credit'))


@pytest.mark.parametrize('reason', ['disabled', 'tiles', 'debug', 'support', 'live_ref',
                                   'unproven_child', 'source_contains_dense', 'missing_input_owner'])
def test_early_retirement_preserves_every_other_dense_consumer(tmp_path, reason):
    from concurrent.futures import Future
    path = tmp_path / 'mask.dat'
    dense = np.memmap(path, mode='w+', shape=(3, 4, 5), dtype=np.uint8)
    dense[:] = 1
    future = Future()
    future._xta_dense_independent_publication_paths = (tmp_path / 'selected.cvol',)
    key = ('detector', 'transverse')
    lease = _DirectUnionBackingLease(key, dense.nbytes, phase='postprocess')
    leases = ViewPrepareLeaseState({key:lease}, set(), {}, {key}, {key:lease.nbytes})
    prepared = SimpleNamespace(model_name=key[0], view_name=key[1], native_support_mm=dense,
        final_view_volume_mm=dense, parent_mask_support_mm=None, parent_bridge_support_mm=None,
        pending_component_layers=[future], nrrd_layers=[],
        _dense_input_owner_ref=weakref.ref(_array_lifetime_owner(dense)),
        _dense_input_nbytes=dense.nbytes)
    if reason == 'support':
        prepared.parent_mask_support_mm = dense
    elif reason == 'live_ref':
        prepared.nrrd_layers = [SimpleNamespace(live_array=dense)]
    elif reason == 'unproven_child':
        del future._xta_dense_independent_publication_paths
    elif reason == 'source_contains_dense':
        future._xta_dense_independent_publication_paths = (tmp_path,)
    elif reason == 'missing_input_owner':
        del prepared._dense_input_owner_ref
    assert not leases.retire_dense_for_publication(prepared,
        enabled=reason != 'disabled', tiled=reason == 'tiles', keep_temp=reason == 'debug',
        retired_callback=lambda *args:pytest.fail('unsafe credit'))
    assert prepared.native_support_mm is prepared.final_view_volume_mm is dense
    assert leases.leases[key] is lease and key in leases.postprocess_views
    assert path.exists() and int(dense.sum()) == 60
    prepared.native_support_mm = prepared.final_view_volume_mm = prepared.parent_mask_support_mm = None
    prepared.nrrd_layers.clear()
    dense = None
    gc.collect()
