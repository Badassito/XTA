"""No interpolation invocation proves empty tile bridge support, including SAM identity."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import assembly, geometry, interpolation


def _prepare(tmp_path, *, backend='sdf', distance=0, passes=1, augmentation_pass=0,
             callback=None, initial=None, cleanup=False, extrapolation=0, sam_context=None):
    shape = (3, 7, 9)
    path = tmp_path / 'native.dat'
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = np.memmap(path, mode='w+', dtype=np.uint8, shape=shape)
    source[:] = 0
    source[0, 2:4, 3:6] = 1
    if initial is not None:
        source[:] = initial
    view = geometry.get_view_infos(*shape, cartesian_views=('transverse',))[0]
    view = replace(view, augmentation_pass=augmentation_pass)
    result = assembly.prepare_view_volume_after_fullframe(
        model_name='detector', view=view, union_mm=source, confmap_mm=None,
        union_path=path, confmap_path=None, temp_dir=tmp_path,
        dense_tiling_active=True, min_conf=0., min_radius=0.,
        interpolate=distance, interpolation_walk_back=0, interpolation_candidates=1,
        interpolate_passes=passes, interpolate_min_radius=0., interpolation_search_angle=0.,
        keep_temp=True, slice_workers=1, interpolation_task_workers=1,
        nrrd_layers_enabled=False, interpolation_backend=backend,
        precleaned_slice_cleanup=not cleanup, hole_fill_done_on_device=not cleanup,
        parent_mask_ready_callback=callback, extrapolation_distance=extrapolation,
        sam_context=sam_context)
    return result


def _close(result):
    if result.parent_mask_support_mm is not None:
        result.parent_mask_support_mm.close()
    if result.parent_bridge_support_mm is not None:
        result.parent_bridge_support_mm.close()


@pytest.mark.parametrize('backend', ['sdf', 'sam'])
@pytest.mark.parametrize('distance,passes,augmentation_pass', [(0, 1, 0), (5, 0, 0), (5, 1, 1)])
def test_no_pass_avoids_subtraction_preserves_empty_store_and_identity(
        tmp_path, backend, distance, passes, augmentation_pass):
    from XTA import sam_integration
    saved = []
    identities = []
    ready = []
    writer = assembly._write_raw_bbox_payload_store
    publish = sam_integration.publish_sam_gate_identity

    def write(**kwargs):
        stats = writer(**kwargs)
        if kwargs.get('extra_meta') == {'operation': 'after_and_not_before'}:
            root = kwargs['store_dir']
            saved.append({name: (root / name).read_bytes()
                          for name in ('index.bin', 'chunks.bin', 'meta.json')})
        return stats

    def identity(store, **kwargs):
        result = publish(store, **kwargs)
        identities.append(result)
        return result

    def parent_ready(_model, _view, store):
        ready.append(np.stack([store.decode_slice(i) for i in range(store.shape[0])]))

    with mock.patch.object(assembly, 'subtract_volume_to_raw_bbox_store',
                           side_effect=AssertionError('No pass may scan the native delta')), \
         mock.patch.object(assembly, '_write_raw_bbox_payload_store', side_effect=write), \
         mock.patch.object(sam_integration, 'publish_sam_gate_identity', side_effect=identity):
        result = _prepare(tmp_path / 'actual', backend=backend, distance=distance, passes=passes,
                          augmentation_pass=augmentation_pass, callback=parent_ready)
    try:
        assert len(ready) == 1 and not result.interpolation_stats
        np.testing.assert_array_equal(result.final_view_volume_mm, ready[0])
        assert result.parent_bridge_support_mm is None
        assert len(saved) == 1
        reference = tmp_path / 'reference.cvol'
        from XTA.cuda_d1 import subtract_volume_to_raw_bbox_store
        subtract_volume_to_raw_bbox_store(result.final_view_volume_mm,
            result.parent_mask_support_mm, reference,
            desc='NRRD support fullframe bridges detector/transverse', workers=1)
        expected = {name: (reference / name).read_bytes() for name in saved[0]}
        assert saved[0] == expected
        if backend == 'sam':
            with mock.patch.object(sam_integration, 'publish_sam_gate_identity', publish):
                store = interpolation.RawBBoxMaskStore.open(reference)
                try:
                    assert identities == [publish(store, policy_identity='', evidence_path='')]
                finally:
                    store.close()
        else:
            assert identities == []
    finally:
        _close(result)


@pytest.mark.parametrize('added', [False, True])
def test_a_real_interpolation_call_always_retains_authoritative_subtraction(tmp_path, added):
    def interpolate(*, mask_mm, **_kwargs):
        if added:
            mask_mm[1, 3, 4] = 1
        return mask_mm, {'added_voxels': int(added)}

    with mock.patch.object(assembly, 'interpolate_view_volume_pass_maybe_process', side_effect=interpolate), \
         mock.patch.object(assembly, 'subtract_volume_to_raw_bbox_store',
                           wraps=assembly.subtract_volume_to_raw_bbox_store) as subtract:
        result = _prepare(tmp_path, distance=5)
    try:
        subtract.assert_called_once()
        if added:
            actual = np.stack([result.parent_bridge_support_mm.decode_slice(i) for i in range(3)])
            expected = np.zeros((3, 7, 9), dtype=np.uint8)
            expected[1, 3, 4] = 1
            np.testing.assert_array_equal(actual, expected)
        else:
            assert result.parent_bridge_support_mm is None
    finally:
        _close(result)


def test_no_pass_shape_mismatch_still_fails(tmp_path):
    def break_shape(_model, _view, store):
        store.shape = (2, 7, 9)
    with pytest.raises(ValueError, match='shape mismatch'):
        _prepare(tmp_path, callback=break_shape)


def test_no_pass_writer_failure_is_not_empty_success(tmp_path):
    writer = assembly._write_raw_bbox_payload_store
    def fail_empty(**kwargs):
        if kwargs.get('extra_meta') == {'operation': 'after_and_not_before'}:
            raise OSError('injected empty support write failure')
        return writer(**kwargs)
    with mock.patch.object(assembly, '_write_raw_bbox_payload_store', side_effect=fail_empty):
        with pytest.raises(OSError, match='injected empty support write failure'):
            _prepare(tmp_path)


def test_real_cpu_hole_fill_precedes_immutable_parent_snapshot(tmp_path):
    from XTA import workspace
    initial = np.zeros((3, 7, 9), dtype=np.uint8)
    initial[0, 1:6, 2:7] = 1
    initial[0, 2:5, 3:6] = 0
    expected = initial.copy()
    expected[0, 2:5, 3:6] = 1
    with mock.patch.object(workspace, '_YOLO_HOLE_FILL_ACTIVE', True), \
         mock.patch.object(assembly, 'subtract_volume_to_raw_bbox_store',
                           side_effect=AssertionError('Cleanup does not imply an interpolation pass')):
        result = _prepare(tmp_path, initial=initial, cleanup=True)
    try:
        np.testing.assert_array_equal(result.final_view_volume_mm, expected)
        parent = np.stack([result.parent_mask_support_mm.decode_slice(i) for i in range(3)])
        np.testing.assert_array_equal(parent, expected)
        assert result.parent_bridge_support_mm is None
    finally:
        _close(result)


def test_positive_sam_extrapolation_remains_outside_parent_bridge_support(tmp_path):
    tail = np.zeros((3, 7, 9), dtype=np.uint8)
    tail[2, 3, 4] = 1
    tail_path = tmp_path / 'tail.cvol'
    interpolation.write_raw_bbox_mask_store(tail, tail_path,
        format_name=interpolation.CVOL_FORMAT, desc='Controlled SAM tail', workers=1)
    observed = []
    def extrapolate(native_volume, **kwargs):
        assert not (tmp_path / 'actual' / 'tile_support' / 'detector' / 'transverse' /
                    'fullframe_bridge_support.cvol').exists()
        observed.append(np.asarray(native_volume).copy())
        return native_volume, {'added_voxels': 1}, [{'path': str(tail_path)}]
    context = SimpleNamespace(evidence_root=tmp_path / 'evidence', extrapolate=extrapolate)
    with mock.patch.object(assembly, 'subtract_volume_to_raw_bbox_store',
                           side_effect=AssertionError('Tail generation follows empty bridge support')):
        result = _prepare(tmp_path / 'actual', backend='sam', extrapolation=2, sam_context=context)
    try:
        assert len(observed) == 1
        np.testing.assert_array_equal(result.final_view_volume_mm, observed[0] | tail)
        assert result.parent_bridge_support_mm is None
        parent = np.stack([result.parent_mask_support_mm.decode_slice(i) for i in range(3)])
        np.testing.assert_array_equal(parent, observed[0])
        assert len(result.interpolation_stats) == 1
        assert result.interpolation_stats[0]['processing_role'] == 'extrapolation'
    finally:
        _close(result)
