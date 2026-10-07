"""Independent overflow/topology regressions from the job150615 investigation."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from XTA import topology


def _source_with_late_overflow() -> np.ndarray:
    # The solid middle plane joins all 65,536 isolated endpoint pixels into one
    # 26-connected object. Overflow happens only after a nonempty prior plane.
    source = np.zeros((3, 512, 512), dtype=np.uint8)
    source[0, 20:26, 30:39] = 1
    source[1] = 1
    source[2, ::2, ::2] = 1
    return source


def _canonical(store, stats) -> np.ndarray:
    luts = stats['slice_local_luts']
    return np.stack([luts.lut_for(z)[np.asarray(store[z])]
                     for z in range(store.shape[0])])


@pytest.mark.parametrize('storage', ['heap', 'sparse', 'disk'])
def test_late_overflow_keeps_every_source_pixel_and_its_global_object(tmp_path, storage):
    source = _source_with_late_overflow()
    original = source.copy()
    stats = {}
    prefix = tmp_path / storage
    narrow_path = prefix.with_suffix('.fg_labels.u16.dat')
    original_impl = topology._label_foreground_volume_streaming_impl
    attempts = []

    def observed_impl(*args, **kwargs):
        dtype = np.dtype(kwargs['_local_label_dtype_override'])
        if dtype == np.dtype(np.uint32):
            # The wide pass cannot overlap a failed disk store's allocation.
            assert not narrow_path.exists()
        attempts.append(dtype)
        return original_impl(*args, **kwargs)

    with mock.patch.dict(os.environ, {'YOLO_TTA_INTERPOLATION_LOCAL_LABEL_UINT16': '1'}), \
            mock.patch.object(topology, 'gpu_slice_labeling_enabled', return_value=False), \
            mock.patch.object(topology, 'should_use_in_memory_workspace', return_value=True), \
            mock.patch.object(topology, 'numa_interleave_memory'), \
            mock.patch.object(topology, '_label_foreground_volume_streaming_impl',
                              side_effect=observed_impl):
        store, count, paths = topology.label_foreground_volume_streaming(
            source, prefix, prefer_memory=storage != 'disk', reserve_bytes=0,
            workers=1, compact_relabel=False, component_stats_out=stats,
            sparse_local_labels=storage == 'sparse',
        )
    try:
        assert attempts == [np.dtype(np.uint16), np.dtype(np.uint32)]
        assert count == 1
        assert store.dtype == np.dtype(np.uint32)
        actual = _canonical(store, stats)
        np.testing.assert_array_equal(actual > 0, original > 0)
        np.testing.assert_array_equal(source, original)
        assert np.unique(actual[original != 0]).tolist() == [1]
        assert stats['component_counts'].tolist() == [1, 1, 65536]
        assert int(stats['root_areas'][stats['unique_roots']].sum()) == int(original.sum())
        assert stats['local_label_dtype_fallback']['overflowing_slice'] == 2
        assert stats['local_label_dtype_fallback']['overflowing_slice_components'] == 65536
        if storage == 'disk':
            assert paths == [prefix.with_suffix('.fg_labels.u32.dat')]
            assert not narrow_path.exists()
        else:
            assert paths == []
    finally:
        if isinstance(store, np.memmap):
            store._mmap.close()
        for path in paths:
            Path(path).unlink(missing_ok=True)


def test_exact_uint16_capacity_remains_narrow_and_preserves_last_id(tmp_path):
    source = np.zeros((1, 512, 512), dtype=np.uint8)
    source[0, ::2, ::2] = 1
    source[0, 510, 510] = 0
    stats = {}
    with mock.patch.dict(os.environ, {'YOLO_TTA_INTERPOLATION_LOCAL_LABEL_UINT16': '1'}), \
            mock.patch.object(topology, 'gpu_slice_labeling_enabled', return_value=False), \
            mock.patch.object(topology, 'should_use_in_memory_workspace', return_value=True):
        store, count, paths = topology.label_foreground_volume_streaming(
            source, tmp_path / 'capacity', prefer_memory=True, reserve_bytes=0,
            compact_relabel=False, component_stats_out=stats,
        )
    assert count == 65535
    assert store.dtype == np.dtype(np.uint16)
    assert int(np.asarray(store).max()) == 65535
    assert 'local_label_dtype_fallback' not in stats
    np.testing.assert_array_equal(_canonical(store, stats) != 0, source != 0)
    assert not paths


def test_concurrent_overflow_does_not_change_another_call_dtype_or_environment(tmp_path):
    sources = [_source_with_late_overflow(), np.ones((2, 13, 19), dtype=np.uint8)]

    def run(index):
        stats = {}
        result = topology.label_foreground_volume_streaming(
            sources[index], tmp_path / str(index), reserve_bytes=0, workers=1,
            compact_relabel=False, component_stats_out=stats,
        )
        return result, stats

    with mock.patch.dict(os.environ, {'YOLO_TTA_INTERPOLATION_LOCAL_LABEL_UINT16': '1'}), \
            mock.patch.object(topology, 'gpu_slice_labeling_enabled', return_value=False), \
            mock.patch.object(topology, 'should_use_in_memory_workspace', return_value=True), \
            mock.patch.object(topology, 'numa_interleave_memory'):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(run, range(2)))
        assert os.environ['YOLO_TTA_INTERPOLATION_LOCAL_LABEL_UINT16'] == '1'
        assert topology._local_label_store_dtype(False) == np.dtype(np.uint16)
    assert results[0][0][0].dtype == np.dtype(np.uint32)
    assert results[1][0][0].dtype == np.dtype(np.uint16)
    for (store, count, paths), stats in results:
        assert count == 1
        assert not paths
        assert np.unique(_canonical(store, stats)).tolist() in ([1], [0, 1])


@pytest.mark.parametrize('error', [RuntimeError('unrelated capacity word'),
                                  topology._LocalLabelCapacityError(2**32, np.uint32, z=0)])
def test_unrelated_or_wide_failures_are_not_retried(tmp_path, error):
    with mock.patch.dict(os.environ, {'YOLO_TTA_INTERPOLATION_LOCAL_LABEL_UINT16': '1'}), \
            mock.patch.object(topology, '_label_foreground_volume_streaming_impl',
                              side_effect=error) as impl:
        with pytest.raises(type(error)) as raised:
            topology.label_foreground_volume_streaming(
                np.ones((1, 2, 2), dtype=np.uint8), tmp_path / 'failure', compact_relabel=False)
    assert raised.value is error
    assert impl.call_count == 1


def _orthogonal_view(base, canonical_shape):
    from XTA.geometry import get_view_infos
    return get_view_infos(*canonical_shape, cartesian_views=(base,))[0]


def _input_from_canonical(volume, base):
    axes = {'transverse': (0, 1, 2), 'sagittal': (1, 0, 2), 'coronal': (2, 0, 1)}
    return np.ascontiguousarray(volume.transpose(axes[base]))


@pytest.mark.parametrize('base', ['transverse', 'sagittal', 'coronal'])
def test_orthogonal_footprints_restore_before_keep_one_without_fragmenting_tube(tmp_path, base):
    from XTA import media
    from XTA.d1_orthogonal_coverage import execute_orthogonal_coverage_reference
    from XTA.finalization import apply_keep_largest_objects_inplace

    canonical = np.zeros((17, 17, 17), dtype=np.uint8)
    canonical[3:14, 4:10, 4:8] = 1
    canonical[3:14, 7:13, 10:14] = 1
    canonical[3:14, 8, 8:11] = 1  # A thin bridge between two lobes.
    canonical[0, 0, 0] = 1  # One disconnected distraction, well away from the tube.
    view = _orthogonal_view(base, canonical.shape)
    inputs = _input_from_canonical(canonical, base)
    original = inputs.copy()
    shape = (13, 25, 27)  # Temporal union contraction plus awkward XY growth.
    expected_store = media.restore_mask_volume_to_original_shape(
        canonical, shape, tmp_path / 'canonical-restore.dat', prefer_memory=False,
        reserve_bytes=0, workers=1,
    )
    try:
        expected = np.array(expected_store, copy=True)
    finally:
        if isinstance(expected_store, np.memmap):
            expected_store._mmap.close()
            (tmp_path / 'canonical-restore.dat').unlink(missing_ok=True)
    actual = execute_orthogonal_coverage_reference(inputs, view, shape)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(inputs, original)
    # Sharded delivery preserves precisely the same support, including temporal
    # footprint overlap where neighboring detector shards contribute one cell.
    chunks = np.zeros(shape, dtype=np.uint8)
    for start, stop in ((0, 5), (5, 12), (12, 17)):
        chunks |= execute_orthogonal_coverage_reference(inputs[start:stop], view, shape,
                                                       slice_start=start)
    np.testing.assert_array_equal(chunks, actual)

    from scipy import ndimage
    labels, count = ndimage.label(expected, np.ones((3, 3, 3), dtype=bool))
    assert count == 2
    areas = np.bincount(labels.reshape(-1))
    areas[0] = 0
    kept_reference = labels == int(np.argmax(areas))
    with mock.patch.object(topology, 'gpu_slice_labeling_enabled', return_value=False), \
            mock.patch.object(topology, 'should_use_in_memory_workspace', return_value=True):
        stats = apply_keep_largest_objects_inplace(actual, 1, tmp_path,
                                                  reserve_bytes=0, workers=1)
    np.testing.assert_array_equal(actual != 0, kept_reference)
    assert stats['num_objects'] == 2
    assert stats['removed_objects'] == 1
    assert stats['removed_voxels'] == int(expected.sum()) - int(kept_reference.sum())


def test_actual_job_xy_ratio_uses_complete_global_footprints_with_confidence(tmp_path):
    import cv2
    from XTA.confidence_projection import score_projection_reader
    from XTA.d1_orthogonal_coverage import (
        build_orthogonal_coverage_plan, execute_orthogonal_coverage_reference,
    )
    source = np.zeros((1, 2048, 2048), np.uint8)
    source[0, 1234:1240, 1799:1805] = 1
    source[0, 1237, 1805:1810] = 1
    view = _orthogonal_view('transverse', source.shape)
    shape = (1, 3064, 3022)
    plan = build_orthogonal_coverage_plan(view, source.shape, shape)
    for length, (starts, stops) in zip(shape[1:], plan.source_ranges_tyx[1:]):
        expected_indices = np.arange(length, dtype=np.int64) * 2048 // length
        np.testing.assert_array_equal(starts, expected_indices)
        np.testing.assert_array_equal(stops, expected_indices + 1)
    expected = cv2.resize(source[0], (3022, 3064), interpolation=cv2.INTER_NEAREST)
    actual = execute_orthogonal_coverage_reference(source, view, shape)
    np.testing.assert_array_equal(actual[0], expected)
    scores = np.where(source, np.uint8(173), np.uint8(0))
    with score_projection_reader(scores, view, shape, tmp_path,
                                 memory_bytes=64 * 1024**2) as read:
        projected_scores = read(0)
    np.testing.assert_array_equal(projected_scores > 0, actual[0] > 0)
    np.testing.assert_array_equal(projected_scores, actual[0] * np.uint8(173))


@pytest.mark.parametrize('base', ['transverse', 'sagittal', 'coronal'])
def test_sparse_endpoint_and_temporal_union_addresses_match_score_projection(tmp_path, base):
    from XTA.confidence_projection import score_projection_reader
    from XTA.d1_orthogonal_coverage import execute_orthogonal_coverage_reference
    canonical = np.zeros((17, 17, 17), np.uint8)
    canonical[0, 2, 3] = 1
    canonical[-1, 14, 13] = 1
    canonical[7, 8:10, 6] = 1
    view = _orthogonal_view(base, canonical.shape)
    source = _input_from_canonical(canonical, base)
    shape = (13, 25, 27)
    actual = execute_orthogonal_coverage_reference(source, view, shape)
    with score_projection_reader(source * np.uint8(219), view, shape, tmp_path,
                                 memory_bytes=4 * 1024**2) as read:
        score_result = np.stack([read(z) for z in range(shape[0])])
    np.testing.assert_array_equal(score_result, actual * np.uint8(219))
    assert actual[0].any()
    assert actual[-1].any()


def test_projection_refuses_area_route_or_output_budget_before_output_allocation():
    from XTA import d1_orthogonal_coverage as coverage
    source = np.ones((3, 17, 19), np.uint8)
    view = _orthogonal_view('transverse', source.shape)
    with mock.patch.object(coverage.np, 'zeros', side_effect=AssertionError('output allocated')):
        with pytest.raises(ValueError, match='area contraction'):
            coverage.execute_orthogonal_coverage_reference(source, view, (2, 11, 13))
        with pytest.raises(ValueError, match='output budget'):
            coverage.execute_orthogonal_coverage_reference(source, view, (2, 25, 27), memory_mib=0)


def test_projection_address_tables_cannot_be_reenabled_or_mutated():
    from XTA.d1_orthogonal_coverage import build_orthogonal_coverage_plan
    plan = build_orthogonal_coverage_plan(_orthogonal_view('sagittal', (17, 17, 17)),
                                          (17, 17, 17), (13, 25, 27))
    for starts, stops in plan.source_ranges_tyx:
        for array in (starts, stops):
            with pytest.raises(ValueError):
                array.setflags(write=True)
            with pytest.raises(ValueError):
                array[0] = 0


@pytest.mark.parametrize('base', ['transverse', 'sagittal', 'coronal'])
def test_no_cube_noncubic_coverage_matches_actual_ordinary_layer_projection(tmp_path, base):
    from XTA import assembly
    from XTA.outputs import _read_layer_slice_in_output_shape
    from XTA.d1_orthogonal_coverage import execute_orthogonal_coverage_reference
    native_shape = (31, 43, 67)
    view = _orthogonal_view(base, native_shape)
    source = (np.random.default_rng(7470).random((view.num_slices, 29, 29)) < .07).astype(np.uint8)
    source[0, 0, 0] = source[-1, -1, -1] = 1
    path = tmp_path / 'ordinary-orthogonal.dat'
    with mock.patch.object(assembly, 'delayed_native_expansion_enabled', return_value=True):
        ordinary = assembly.project_view_volume_to_orthogonal_volume(
            source, view, path, 'ordinary no-cube layer reference',
            prefer_memory=False, reserve_bytes=0, workers=1,
        )
    try:
        expected = np.stack([_read_layer_slice_in_output_shape(ordinary, native_shape, z)
                             for z in range(native_shape[0])])
    finally:
        if isinstance(ordinary, np.memmap):
            ordinary._mmap.close()
        path.unlink(missing_ok=True)
    actual = execute_orthogonal_coverage_reference(source, view, native_shape)
    np.testing.assert_array_equal(actual, expected)


def _two_families():
    source = np.zeros((5, 240, 300), np.uint8)
    for frame in (0, 4):
        source[frame, 40:46, 40:46] = 1
        source[frame, 200:206, 240:246] = 1
    return source


def test_lazy_families_preserve_eager_geometry_without_aggregate_inventory_refusal():
    from XTA.sam_bridge_planning import SamPlanningLimits, plan_sam_bridges
    options = dict(interpolation_distance=5, interpolation_walk_back=0,
                   interpolation_min_radius=0, scope_id='independent-families')
    source = _two_families()
    eager = plan_sam_bridges(source, **options)
    assert len(eager.groups) == 2
    assert all(group.status == 'planned' for group in eager.groups)
    one_family_limit = max(int(group.crop_contract['charged_contract_bytes']) for group in eager.groups)
    limits = SamPlanningLimits(max_total_contract_bytes=one_family_limit)
    capped_eager = plan_sam_bridges(source, limits=limits, **options)
    lazy = plan_sam_bridges(source, limits=limits, lazy_contracts=True, **options)
    assert sum(group.status == 'planned' for group in capped_eager.groups) == 1
    assert len(lazy.groups) == 2
    assert all(group.status == 'planned' for group in lazy.groups)
    assert len(lazy.runs) == len(eager.runs) == 4
    assert lazy.needed_frames == eager.needed_frames
    assert dict(lazy.frame_crop_bounds) == dict(eager.frame_crop_bounds)
    reference = {group.observation_ids: group for group in eager.groups}
    for descriptor in lazy.groups:
        assert descriptor.acceptance_masks.size == descriptor.write_masks.size == 0
        baseline = reference[descriptor.observation_ids]
        with descriptor.materialize_contracts() as concrete:
            assert concrete.context_bbox_yx == baseline.context_bbox_yx
            assert concrete.frame_indices == baseline.frame_indices
            assert concrete.edges == baseline.edges
            for name in ('acceptance_masks', 'write_masks', 'known_foreground_masks', 'unrelated_masks'):
                np.testing.assert_array_equal(getattr(concrete, name), getattr(baseline, name))
            for name in ('branch_evaluation_masks', 'branch_permitted_masks',
                         'edge_write_masks', 'edge_contract_masks'):
                actual, expected = getattr(concrete, name), getattr(baseline, name)
                assert set(actual) == set(expected)
                for key in expected:
                    np.testing.assert_array_equal(actual[key], expected[key])
            # Loop variables must not retain leased mask owners into the next
            # family's admission (a real consumer must drop them as well).
            del actual, expected
        del concrete
    assert lazy.contract_lease_budget.snapshot()['live_bytes'] == 0


def test_borrowed_family_slice_keeps_byte_credit_and_prevents_untruthful_readmission():
    import gc
    from XTA.sam_bridge_planning import SamPlanningLimits, plan_sam_bridges
    options = dict(interpolation_distance=5, interpolation_walk_back=0, interpolation_min_radius=0)
    source = _two_families()
    descriptor_plan = plan_sam_bridges(source, lazy_contracts=True, **options)
    one_family_limit = max(int(group.crop_contract['charged_contract_bytes'])
                           for group in descriptor_plan.groups)
    plan = plan_sam_bridges(source, lazy_contracts=True,
                           limits=SamPlanningLimits(max_total_contract_bytes=one_family_limit), **options)
    smaller, largest = sorted(plan.groups, key=lambda group: group.crop_contract['charged_contract_bytes'])
    with smaller.materialize_contracts() as concrete:
        borrowed = concrete.acceptance_masks[0]
        assert plan.contract_lease_budget.snapshot()['live_bytes'] > 0
    del concrete
    assert plan.contract_lease_budget.snapshot()['live_bytes'] > 0
    with pytest.raises(MemoryError, match='resident lease'):
        with largest.materialize_contracts():
            pytest.fail('A retained mask owner was falsely uncharged')
    del borrowed
    gc.collect()
    assert plan.contract_lease_budget.snapshot()['live_bytes'] == 0
    with largest.materialize_contracts() as concrete:
        assert concrete.write_masks.any()
    del concrete
    assert plan.contract_lease_budget.snapshot()['live_bytes'] == 0


def test_long_tta_session_keeps_complete_endpoints_while_lta_limit_is_unchanged(tmp_path):
    from XTA.lta_sam import SamSessionPlan, SamInterpolationSessionPlan
    from XTA.sam_interpolation import prepare_sam_interpolation_pass, interpolate_sam_view_volume_pass
    from tests.test_sam_interpolation import RepeatedSeedTracker, _close
    source = np.zeros((45, 32, 32), np.uint8)
    source[0, 11:17, 11:17] = 1
    source[-1, 11:17, 11:17] = 1
    options = dict(gap_distance=44, min_radius=0, interpolation_walk_back=0)
    with pytest.raises(ValueError, match='at most 30'):
        SamSessionPlan('lta-unchanged', 0, 0, 45)
    assert SamInterpolationSessionPlan('tta-complete', 0, 0, 45).frame_count == 45
    prepared = prepare_sam_interpolation_pass(source, **options)
    assert len(prepared.runs) == 2
    assert prepared.needed_frames == tuple(range(45))
    assert prepared.runs[0].expected_frames == tuple(range(45))
    assert prepared.runs[1].expected_frames == tuple(range(44, -1, -1))
    tracker = RepeatedSeedTracker()
    merged, stats, _ = interpolate_sam_view_volume_pass(
        source, prepared_plan=prepared, runtime=tracker, work_dir=tmp_path, **options)
    try:
        assert stats['sam_generated_runs'] == 2
        assert stats['sam_selected_runs'] == 2
        assert stats['added_voxels'] == 43 * 36
        np.testing.assert_array_equal(merged[:, 11:17, 11:17], np.ones((45, 6, 6), np.uint8))
        np.testing.assert_array_equal(source[0], merged[0])
        np.testing.assert_array_equal(source[-1], merged[-1])
    finally:
        _close(merged)


def test_prepared_plan_resource_revision_rejected_before_runtime_or_directory(tmp_path):
    from XTA import sam_bridge_planning
    from XTA.sam_interpolation import prepare_sam_interpolation_pass, interpolate_sam_view_volume_pass
    source = _two_families()
    options = dict(gap_distance=5, min_radius=0, interpolation_walk_back=0)
    prepared = prepare_sam_interpolation_pass(source, **options)
    output = tmp_path / 'must-not-exist'
    runtime = mock.Mock()
    with mock.patch.object(sam_bridge_planning, 'SAM_CONTRACT_STORAGE_VERSION', 'future-resource-revision'):
        with pytest.raises(ValueError, match='planning settings'):
            interpolate_sam_view_volume_pass(source, prepared_plan=prepared, runtime=runtime,
                                            work_dir=output, **options)
    assert not output.exists()
    assert runtime.mock_calls == []


@pytest.mark.parametrize('crop_mode', ['whole', 'tiled'])
def test_known_long_session_cpu_refusal_precedes_images_and_model_admission(tmp_path, crop_mode):
    from XTA.sam_integration import SamInterpolationContext
    from XTA.sam_interpolation import SamInterpolationInfrastructureError
    source = np.zeros((200, 25, 25), np.uint8)
    source[0, 9:15, 10:16] = source[-1, 9:15, 10:16] = 1
    view = _orthogonal_view('transverse', source.shape)
    context = SamInterpolationContext(model_path='unused', device_ids=(0,),
        temp_dir=tmp_path / 'temp', evidence_root=tmp_path / 'evidence',
        source_volume=source, source_identity='synthetic immutable images',
        crop_mode=crop_mode)
    try:
        with mock.patch.object(context, 'image_provider', side_effect=AssertionError('images rendered')) as render, \
                mock.patch.object(context, '_start', side_effect=AssertionError('model admitted')) as start:
            with pytest.raises(SamInterpolationInfrastructureError, match='known CPU session buffers'):
                context.interpolate(source, view=view, scope='bounded-session',
                    gap_distance=199, min_radius=0, interpolation_walk_back=0)
        assert render.call_count == start.call_count == 0
    finally:
        context.close()


def test_tiled_iterator_drops_previous_dense_result_before_next_transfer_decode():
    import weakref
    from dataclasses import replace
    from XTA import sam_interpolation
    from XTA.sam_tracker_runtime import SamTrackerRunResult
    references = []

    class TransferBoundaryRuntime:
        def iter_results(self, _requests, **_kwargs):
            for index in range(2):
                if references:
                    assert references[-1]() is None, 'previous tiled result survives into next decode'
                mask = np.ones((4, 5), dtype=bool)
                references.append(weakref.ref(mask))
                result = SamTrackerRunResult({0: mask}, {0: 1.0}, {0: 'observed'}, {})
                yield index, result
                del result, mask

    prepared = replace(sam_interpolation.prepare_sam_interpolation_pass(np.zeros((1, 4, 5), np.uint8)),
                       tracker_jobs=(object(), object()))
    with mock.patch.object(sam_interpolation.SamPreparedInterpolationPass, 'execution_batches', return_value=((0, 1),)), \
            mock.patch.object(sam_interpolation, '_tiled_tracker_requests', return_value=()):
        stream = sam_interpolation._iterate_tiled_tracker_results(
            TransferBoundaryRuntime(), prepared, 4, {}, {}, None, None)
        first = next(stream)
        assert first[0] == 0
        assert references[0]() is not None
        del first
        second = next(stream)
        assert second[0] == 1
        del second
        with pytest.raises(StopIteration):
            next(stream)
    assert all(reference() is None for reference in references)


def _wave_task_cost(task):
    from XTA.sam_resources import cpu_session_bytes
    payload = task.payload
    x0, y0, x1, y1 = payload['crop_xyxy']
    frames = payload['frame_stop'] - payload['frame_start']
    return cpu_session_bytes(frames, (x1 - x0) * (y1 - y0))['estimated_peak_bytes']


def _wave_loaded_result(pool, path, credit, samples, *, broadcast=False):
    from XTA.sam_tracker_runtime import SamTrackerRunResult, _image_cache_summary
    task = next(task for task in pool.submitted if Path(task.payload['output_dir']) == Path(path).parent)
    payload = task.payload
    x0, y0, x1, y1 = payload['crop_xyxy']
    count = payload['frame_stop'] - payload['frame_start']
    # Observe the decode boundary after any refill. Three raw-plane buffers is
    # the declared conservative transfer margin, rather than the tiny fake's RSS.
    peak = sum(_wave_task_cost(task) for task, _ in pool.pending) + 3 * count * (y1-y0) * (x1-x0)
    assert peak <= credit, 'active SDK inputs plus completed decode exceed owned CPU-wave credit'
    samples.append(peak)
    if broadcast:
        # Metadata-equivalent full-size masks backed by one byte. No SDK, CUDA,
        # normalized input tensor, or GiB allocation occurs in this proof.
        mask = np.broadcast_to(np.array(True, dtype=bool), (y1-y0, x1-x0))
    else:
        with np.load(payload['seed_path'], allow_pickle=False) as saved:
            mask = saved['seed'].copy()
    frames = {frame: mask for frame in range(payload['frame_start'], payload['frame_stop'])}
    receipt = dict(run_id=payload['run_id'], crop_xyxy=payload['crop_xyxy'],
        frame_range=[payload['frame_start'], payload['frame_stop']],
        seed_frame=payload['seed_frame'], direction=payload['direction'],
        image_cache=_image_cache_summary(payload['image_cache']),
        seed_artifact_sha256=payload['seed_sha256'], temporary_artifact_directory=payload['output_dir'],
        request_metadata=payload.get('request_metadata', {}))
    return SamTrackerRunResult(frames, {frame: 1.0 for frame in frames},
                               {frame: 'observed' for frame in frames}, receipt)


def test_four_worker_long_sessions_spend_only_owned_no_extra_parent_wave(tmp_path):
    from XTA import sam_resources, sam_tracker_runtime
    from XTA.interpolation import _ByteAdmissionPool
    from XTA.sam_interpolation import prepare_sam_interpolation_pass, interpolate_sam_view_volume_pass
    from tests.test_sam_optimization_adversarial import _FakePool
    from tests.test_sam_interpolation import _close
    source = np.zeros((151, 40, 40), np.uint8)
    for end, (y, x) in zip((99, 119, 139, 149), ((4, 4), (4, 28), (28, 4), (28, 28))):
        source[0, y:y+4, x:x+4] = source[end, y:y+4, x:x+4] = 1
    cache = sam_tracker_runtime.materialize_interpolation_image_cache(
        np.zeros_like(source), path=tmp_path / 'images.u8',
        physical_view_id='transverse', source_identity='CPU-only wave witness')
    tracker = sam_tracker_runtime.SamInterpolationTracker(model_path='never-loaded',
        device_ids=(0, 1, 2, 3), artifact_root=tmp_path / 'runs', source_cache_ref=cache)
    workers = _FakePool()
    tracker._pool = workers
    samples = []
    pool = _ByteAdmissionPool(64 * sam_resources.GIB, 'owned-wave')
    options = dict(gap_distance=150, min_radius=0, search_angle_deg=0, interpolation_walk_back=0)
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*sam_resources.GIB,
                'no-extra-four-workers', worker_count=4, base_allowance_bytes=4*sam_resources.GIB,
                headroom_probe=lambda: 8*sam_resources.GIB) as profile:
            assert not profile.has_extra_credit
            assert pool.in_use == 4*sam_resources.GIB
            assert profile.assigned_cpu_wave_bytes == 2*sam_resources.GIB
            assert profile.metadata()['base_non_cpu_allowance_bytes'] == 2*sam_resources.GIB
            submit = workers.submit

            def checked_submit(task, *, execution_device_id):
                submit(task, execution_device_id=execution_device_id)
                peak = sum(_wave_task_cost(pending) for pending, _ in workers.pending)
                assert peak <= profile.assigned_cpu_wave_bytes
                samples.append(peak)

            workers.submit = checked_submit
            prepared = prepare_sam_interpolation_pass(source, resource_profile=profile, **options)
            wave = prepared.cpu_wave_admission
            assert wave['max_in_flight'] == 1 < len(tracker.device_ids)
            assert wave['peak_cpu_wave_estimate_bytes'] <= profile.assigned_cpu_wave_bytes
            assert len(prepared.runs) == 8
            assert sorted(len(run.expected_frames) for run in prepared.runs) == [100, 100, 120, 120, 140, 140, 150, 150]
            with mock.patch.object(sam_resources, 'admit_sam_prepared_scope',
                    wraps=sam_resources.admit_sam_prepared_scope) as admitted, \
                    mock.patch.object(sam_tracker_runtime, 'load_tracker_run_result',
                    side_effect=lambda path, **_: _wave_loaded_result(
                        workers, path, profile.assigned_cpu_wave_bytes, samples)):
                merged, stats, _ = interpolate_sam_view_volume_pass(
                    source, runtime=tracker, prepared_plan=prepared, resource_profile=profile,
                    work_dir=tmp_path / 'evidence', **options)
            assert admitted.call_args.args[2] is cache  # Omitted image_provider uses the configured source for credit.
            try:
                assert stats['sam_generated_runs'] == stats['sam_selected_runs'] == 8
                assert stats['added_voxels'] == (98+118+138+148)*16
                assert tracker.dispatch_stats['peak_in_flight'] == 1
                assert len(workers.submitted) == 8
                assert len(samples) == 16  # Every submit and every decode boundary.
                assert max(samples) <= profile.assigned_cpu_wave_bytes
                np.testing.assert_array_equal(merged[source != 0], source[source != 0])
            finally:
                _close(merged)
        assert pool.in_use == 0
    finally:
        tracker.close()


def test_near_limit_cpu_session_defers_refill_until_result_consumed(tmp_path):
    import weakref
    from XTA import sam_resources, sam_tracker_runtime
    from XTA.interpolation import _ByteAdmissionPool
    from XTA.lta_rendering import reference_existing_physical_view_cache
    from tests.test_sam_optimization_adversarial import _FakePool
    shape = (88, 1008, 1008)
    image_path = tmp_path / 'sparse-sized-images.u8'
    with image_path.open('wb') as stream:
        stream.truncate(int(np.prod(shape)))
    cache = reference_existing_physical_view_cache(image_path, shape=shape,
        physical_view_id='transverse', source_identity='metadata-equivalent CPU-only witness')
    tracker = sam_tracker_runtime.SamInterpolationTracker(model_path='never-loaded',
        device_ids=(0, 1, 2, 3), artifact_root=tmp_path / 'runs', source_cache_ref=cache)
    workers = _FakePool()
    tracker._pool = workers
    pool = _ByteAdmissionPool(64*sam_resources.GIB, 'owned-wave')
    samples, references = [], []
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*sam_resources.GIB,
                'near-limit', worker_count=4, base_allowance_bytes=4*sam_resources.GIB,
                headroom_probe=lambda: 8*sam_resources.GIB) as profile:
            single = sam_resources.cpu_session_bytes(shape[0], shape[1]*shape[2])['estimated_peak_bytes']
            raw = int(np.prod(shape))
            wave = sam_resources.cpu_wave_admission(single, raw, profile.assigned_cpu_wave_bytes, 4)
            assert single < profile.assigned_cpu_wave_bytes < single+3*raw
            assert wave['max_in_flight'] == 1
            assert wave['defer_refill_until_consumed']
            assert wave['peak_cpu_wave_estimate_bytes'] == max(single, 3*raw)
            submit = workers.submit

            def checked_submit(task, *, execution_device_id):
                # Literal boundary: the completed result owner is gone before
                # another worker receives its input credit, without later GC.
                assert all(reference() is None for reference in references)
                submit(task, execution_device_id=execution_device_id)
                assert sum(_wave_task_cost(pending) for pending, _ in workers.pending) <= profile.assigned_cpu_wave_bytes

            workers.submit = checked_submit
            seed = np.ones(shape[1:], dtype=bool)
            requests = [dict(run_id=f'long-{index}', seed_mask=seed, seed_frame=0,
                frame_start=0, frame_stop=88, direction='forward', crop_xyxy=(0, 0, 1008, 1008),
                resource_profile=profile) for index in range(3)]

            def loaded(path, **_):
                assert not workers.pending, 'next session was refilled before decoding the completed result'
                result = _wave_loaded_result(workers, path, profile.assigned_cpu_wave_bytes, samples,
                                             broadcast=True)
                references.append(weakref.ref(result.frames[0]))
                return result

            with mock.patch.object(sam_tracker_runtime, 'load_tracker_run_result', side_effect=loaded):
                iterator = tracker.iter_results(requests, max_in_flight=wave['max_in_flight'],
                    defer_refill_until_consumed=wave['defer_refill_until_consumed'])
                for expected in range(3):
                    index, result = next(iterator)
                    assert index == expected
                    assert len(workers.submitted) == expected+1
                    assert not workers.pending
                    del result
                with pytest.raises(StopIteration):
                    next(iterator)
            assert all(reference() is None for reference in references)
            assert len(samples) == len(workers.submitted) == 3
            assert tracker.dispatch_stats['peak_in_flight'] == 1
        assert pool.in_use == 0
    finally:
        tracker.close()


@pytest.mark.parametrize('crop_mode', ['whole', 'tiled'])
def test_session_within_declared_limit_but_over_physical_headroom_never_admits_model(tmp_path, crop_mode):
    from XTA import sam_resources
    from XTA.interpolation import _ByteAdmissionPool
    from XTA.sam_integration import SamInterpolationContext
    from XTA.sam_interpolation import SamInterpolationInfrastructureError
    source = np.zeros((100, 25, 25), np.uint8)
    source[0, 9:15, 10:16] = source[-1, 9:15, 10:16] = 1
    view = _orthogonal_view('transverse', source.shape)
    context = SamInterpolationContext(model_path='never-loaded', device_ids=(0, 1, 2, 3),
        temp_dir=tmp_path/'temp', evidence_root=tmp_path/'evidence',
        source_volume=source, source_identity='bounded physical CPU witness', crop_mode=crop_mode)
    pool = _ByteAdmissionPool(64*sam_resources.GIB, 'accounting floor is not RAM')
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*sam_resources.GIB,
                'one-GiB-headroom', worker_count=4, base_allowance_bytes=4*sam_resources.GIB,
                headroom_probe=lambda: sam_resources.GIB) as profile:
            estimate = sam_resources.cpu_session_bytes(100, 25*25)['estimated_peak_bytes']
            assert sam_resources.GIB < estimate < profile.assigned_session_cpu_bytes == 2*sam_resources.GIB
            assert pool.in_use == 4*sam_resources.GIB  # The inherited pool lane remains unchanged.
            assert not profile.has_extra_credit
            assert profile.assigned_cpu_wave_bytes == profile.cpu_wave_physical_residual_bytes == 0
            assert profile.metadata()['base_cpu_wave_physical_clamp_bytes'] == 2*sam_resources.GIB
            with context.resource_scope(profile), \
                    mock.patch.object(context, 'image_provider', side_effect=AssertionError('images rendered')) as render, \
                    mock.patch.object(context, '_start', side_effect=AssertionError('model admitted')) as start:
                with pytest.raises(SamInterpolationInfrastructureError, match='known CPU wave'):
                    context.interpolate(source, view=view, scope='physical-refusal',
                        gap_distance=99, min_radius=0, interpolation_walk_back=0)
            assert render.call_count == start.call_count == 0
        assert pool.in_use == 0
    finally:
        context.close()


def test_other_promised_credits_wait_for_full_physical_wave_then_release_restores_it():
    import threading
    from XTA import sam_resources
    from XTA.interpolation import _ByteAdmissionPool
    pool = _ByteAdmissionPool(64*sam_resources.GIB, 'dynamic physical headroom')
    sampled, entered = threading.Event(), threading.Event()
    def physical():
        sampled.set()
        return 6*sam_resources.GIB
    options = dict(worker_count=4, base_allowance_bytes=4*sam_resources.GIB,
                   headroom_probe=physical)
    def prepare():
        with sam_resources.admit_sam_parent_resources(pool, 4*sam_resources.GIB,
                                                     'after-other-promises', **options) as profile:
            entered.set()
            assert pool.in_use == 4*sam_resources.GIB
            record = profile.metadata()
            assert record['other_promised_bytes_at_admission'] == 0
            assert record['base_non_cpu_allowance_bytes'] == 2*sam_resources.GIB
            assert record['cpu_wave_physical_residual_bytes'] == 4*sam_resources.GIB
            assert profile.assigned_cpu_wave_bytes == 2*sam_resources.GIB
            assert profile.assigned_cpu_wave_bytes + profile.non_cpu_base_allowance_bytes + profile.other_promised_bytes <= profile.physical_headroom_bytes
            single = sam_resources.cpu_session_bytes(100, 25*25)['estimated_peak_bytes']
            assert single < profile.assigned_session_cpu_bytes
            wave = sam_resources.cpu_wave_admission(single, 100*25*25,
                profile.assigned_cpu_wave_bytes, profile.worker_count)
            assert wave['peak_cpu_wave_estimate_bytes'] <= profile.assigned_cpu_wave_bytes
            assert not profile.has_extra_credit
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pool.reserve(3*sam_resources.GIB, 'already promised elsewhere'):
            future = executor.submit(prepare)
            assert sampled.wait(2)
            assert not entered.wait(.05)
            with pool.condition:
                assert pool.in_use == 3*sam_resources.GIB
        future.result(timeout=5)
    assert pool.in_use == 0
