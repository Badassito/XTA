"""Tail publication and voting remain separate from paired bridge receipts."""
import json
import contextlib
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import assembly, geometry, interpolation, pipeline
from XTA.sam_integration import SamInterpolationContext
from XTA.outputs import NrrdLayerSink
from XTA.reconciliation import EvidenceLayer, reconcile, evidence_role
from XTA.reconciliation_io import read_layer_manifest
from XTA.reconciliation_policy import validate_policy
from XTA.reconciliation_runtime import RuntimeLayer


@pytest.mark.parametrize('base', ('transverse', 'sagittal', 'coronal'))
def test_tail_source_projection_and_manifest_preserve_raw_one_pixel_sliver(tmp_path, base):
    view = geometry.get_view_infos(7, 7, 7, cartesian_views=(base,))[0]
    source = np.zeros((7, 7, 7), np.uint8)
    source[1, 3, 5] = 1
    axes = {'transverse': (0, 1, 2), 'sagittal': (1, 0, 2), 'coronal': (2, 0, 1)}[base]
    native = source.transpose(axes).copy()
    path = tmp_path / 'tail.cvol'
    interpolation.write_raw_bbox_mask_store(native, path,
        format_name=interpolation.INTERNAL_PACKED_CVOL_FORMAT, workers=1)
    context = SimpleNamespace(bundle_identity='sam', source_volume=np.zeros(source.shape, np.uint8))
    sink = NrrdLayerSink(nrrd_dir=tmp_path/'nrrd', stem='case', output_shape_tyx=source.shape, max_workers=1)
    try:
        with mock.patch.object(assembly, 'nrrd_layer_sink', return_value=sink), \
             mock.patch.object(assembly, 'final_source_output_shape', return_value=source.shape):
            ref = assembly.materialize_sam_extrapolation_view_layer(
                dict(direction='forward', path=str(path), voxel_count=1, run_ids=['tail1'],
                     terminal_roots=['postinterp1'], evidence_path=str(tmp_path/'evidence')),
                model_name='detector', view=view, source='fullframe', sam_context=context,
                distance=5, walk_back=1, min_radius=3.)
        assert ref.mask_kind == 'extrapolation'
        assert not ref.interpolation_backend
        assert not ref.selected_bridge_connection_status
        assert not ref.proposal_selection_status
        reader = RuntimeLayer(ref, source.shape)
        try:
            np.testing.assert_array_equal(reader.read_slab(0, 7), source)
        finally:
            reader.close()
        sink.wait()
        manifest = sink.write_manifest()
    finally:
        sink.shutdown()
    saved = json.loads(manifest.read_text())['layers'][0]
    assert '_sam_extrapolation_forward' in saved['filename']
    assert 'interpolation_backend' not in saved
    assert saved['extrapolation']['seed_stage'] == 'post_interpolation_before_any_extrapolated_tail'
    assert saved['extrapolation']['terminal_min_radius'] == 3.
    with read_layer_manifest(manifest, workspace=tmp_path/'read') as layers:
        assert evidence_role(layers[0].metadata) == 'extrapolation'
        assert layers[0].proposal_bundle() is None
        np.testing.assert_array_equal(layers[0].read_slab(0, 7), source)


def test_native_additions_can_overlap_baseline_after_nonidentity_positive_restore(tmp_path):
    """Native subtraction is exact; ordinary source restore is not exclusive."""
    from XTA.outputs import _read_layer_slice_in_output_shape
    native_shape, source_shape = (4,4,4), (2,4,4)
    view = geometry.get_view_infos(*native_shape,cartesian_views=('transverse',))[0]
    baseline = np.zeros(native_shape,np.uint8)
    tail = np.zeros_like(baseline)
    baseline[0,1,1] = 1
    tail[1,1,1] = 1
    assert not np.any(baseline & tail)
    path = tmp_path/'tail.cvol'
    interpolation.write_raw_bbox_mask_store(tail,path,
        format_name=interpolation.INTERNAL_PACKED_CVOL_FORMAT,workers=1)
    context = SimpleNamespace(bundle_identity='sam',source_volume=np.zeros(native_shape,np.uint8))
    with mock.patch.object(assembly,'nrrd_layer_sink',return_value=None), \
         mock.patch.object(assembly,'final_source_output_shape',return_value=source_shape):
        ref = assembly.materialize_sam_extrapolation_view_layer(
            dict(direction='forward',path=str(path),voxel_count=1,run_ids=['tail1'],terminal_roots=['root']),
            model_name='detector',view=view,source='fullframe',sam_context=context,
            distance=2,walk_back=1,min_radius=3.)
    reader = RuntimeLayer(ref,source_shape)
    try:
        restored_tail = reader.read_slab(0,source_shape[0])
    finally:
        reader.close()
    restored_baseline = np.stack([_read_layer_slice_in_output_shape(baseline,source_shape,z)
        for z in range(source_shape[0])])
    restored_union = np.stack([_read_layer_slice_in_output_shape(baseline | tail,source_shape,z)
        for z in range(source_shape[0])])
    assert np.count_nonzero(restored_tail & restored_baseline) == 1
    np.testing.assert_array_equal(restored_tail | restored_baseline,restored_union)


def _tail(mask, name='tail', view='transverse'):
    return EvidenceLayer(name, mask.shape, dict(source='fullframe', mask_kind='extrapolation',
        physical_view_name=view, view_name=view, view_family='orthogonal'), lambda a, b: mask[a:b])


def _vote(layers, policy):
    mask = np.zeros(layers[0].shape_tyx, np.uint8)
    report = reconcile(layers, shape_tyx=mask.shape, policy=policy, memory_mib=8,
        write_slab=lambda a, b, value: mask.__setitem__(slice(a, b), value))
    return mask, report


def test_default_union_retains_tail_only_voxels_and_has_explicit_role():
    mask = np.zeros((3, 7, 9), np.uint8)
    mask[2, 3, 5] = 1
    result, report = _vote([_tail(mask)], dict(mode='union'))
    np.testing.assert_array_equal(result, mask)
    assert report['layers']['tail']['role'] == 'extrapolation'


def test_legacy_weights_inherit_bridge_and_explicit_tail_veto_remains_available():
    policy = dict(mode='weighted', grouping='views', threshold=.4, min_sources=1,
        min_prediction_sources=0, provenance_weights=dict(prediction=1., bridge=.5, mixed=.2))
    assert validate_policy(policy)['provenance_weights']['extrapolation'] == .5
    mask = np.ones((1, 2, 3), np.uint8)
    assert _vote([_tail(mask)], policy)[0].all()
    policy['provenance_weights']['extrapolation'] = 0.
    assert not _vote([_tail(mask)], policy)[0].any()


def test_custom_tail_support_is_readonly_and_caps_duplicate_view_sources():
    mask = np.ones((1, 2, 3), np.uint8)
    blocks = []
    def decide(block):
        assert not block['extrapolation_support'].flags.writeable
        blocks.append(block['extrapolation_support'].copy())
        assert not block['prediction_support'].any()
        return block['extrapolation_support'] >= 2
    result, _ = _vote([_tail(mask, 'f'), _tail(mask, 'b')], dict(grouping='views', decide=decide))
    assert not result.any()
    assert all(np.all(value == 1) for value in blocks)
    result, _ = _vote([_tail(mask, 'f'), _tail(mask, 'other', 'sagittal')],
        dict(grouping='views', decide=decide))
    assert result.all()


def _context(tmp_path, shape):
    return SamInterpolationContext(model_path='unused-test-model', device_ids=(0,),
        temp_dir=tmp_path/'temporary', evidence_root=tmp_path/'evidence',
        source_volume=np.zeros(shape, np.uint8), source_identity='input',
        detector_device_ids=(0,), interpolation_policy_enabled=False,
        policy={'sam_bridge_policy': {'unused_invalid_interpolation_field': True}})


def test_extrapolation_only_context_skips_unused_bridge_policy_and_no_job_assets(tmp_path):
    shape = (5, 12, 13)
    view = geometry.get_view_infos(*shape, cartesian_views=('transverse',))[0]
    native = np.zeros(shape, np.uint8)
    context = _context(tmp_path, shape)
    try:
        assert context.tight_crop_guard is None
        with mock.patch.object(context, 'image_provider', side_effect=AssertionError('no images')), \
             mock.patch.object(context, '_start', side_effect=AssertionError('no predictor')):
            result, stats, components = context.extrapolate(native, view=view, scope='test',
                distance=2, work_dir=tmp_path/'retained')
        assert result is native
        assert stats['skipped'] and not components
        assert not (tmp_path/'retained').exists()
        assert native.flags.writeable
    finally:
        context.close()


def test_extrapolation_context_image_failure_is_infrastructure_not_tail_stop(tmp_path):
    shape = (5, 12, 13)
    view = geometry.get_view_infos(*shape, cartesian_views=('transverse',))[0]
    native = np.zeros(shape, np.uint8)
    native[2, 3:8, 4:9] = 1
    context = _context(tmp_path, shape)
    try:
        with mock.patch.object(context, 'image_provider', side_effect=RuntimeError('renderer failed')):
            with pytest.raises(RuntimeError, match='renderer failed'):
                context.extrapolate(native, view=view, scope='test', distance=2,
                    min_radius=0., work_dir=tmp_path/'retained')
        receipt = json.loads((tmp_path/'retained'/'context_preparation_failure.json').read_text())
        assert receipt['status'] == 'infrastructure_invalid'
        assert receipt['evidence_purpose'] == 'sam_extrapolation'
        assert receipt['source_stage'] == 'post_interpolation'
        assert context._active_passes == 0
        assert native.flags.writeable
    finally:
        context.close()


@pytest.mark.parametrize('backend,distance', [('sdf', 3), ('sam', 3), ('sdf', 0)])
def test_fullframe_extrapolation_uses_completed_local_baseline_and_excludes_tails_from_gate(
        tmp_path, backend, distance):
    shape = (5, 12, 13)
    view = geometry.get_view_infos(*shape, cartesian_views=('transverse',))[0]
    original = np.zeros(shape, np.uint8)
    original[0, 4:8, 4:8] = original[2, 4:8, 4:8] = 1
    bridge = np.zeros(shape, np.uint8)
    if distance:
        bridge[1, 4:8, 4:8] = 1
    tail = np.zeros(shape, np.uint8)
    tail[3, 4, 4] = 1
    context = SimpleNamespace(evidence_root=tmp_path/'evidence', bundle_identity='sam', detector_identity='detector')
    def interpolate(observed, **kwargs):
        return observed | bridge, {'added_voxels': int(bridge.sum()), 'sam_policy_hash': 'paired'}, []
    def sdf(*, mask_mm, **kwargs):
        mask_mm |= bridge
        return mask_mm, {'added_voxels': int(bridge.sum())}
    context.interpolate = interpolate
    called = []
    def extrapolate(observed, **kwargs):
        np.testing.assert_array_equal(observed, original | bridge)
        gate = tmp_path/'tile_support'/'detector'/view.name/'fullframe_bridge_support.cvol'
        if distance:
            store = interpolation.RawBBoxMaskStore.open(gate)
            try:
                np.testing.assert_array_equal(np.stack([store.decode_slice(i) for i in range(shape[0])]), bridge)
            finally:
                store.close()
        else:
            assert not gate.exists()
        called.append(observed.copy())
        path = tmp_path/'tail.cvol'
        interpolation.write_raw_bbox_mask_store(tail, path,
            format_name=interpolation.INTERNAL_PACKED_CVOL_FORMAT, workers=1)
        return observed, {'added_voxels': 1}, [dict(direction='forward', path=str(path), voxel_count=1)]
    context.extrapolate = extrapolate
    observed = original.copy()
    with mock.patch.object(assembly, 'cleanup_view_volume_after_prediction_inplace'), \
         mock.patch.object(assembly, 'nrrd_layer_sink', return_value=None), \
         mock.patch.object(assembly, 'interpolate_view_volume_pass_maybe_process', side_effect=sdf):
        result = assembly.prepare_view_volume_after_fullframe(model_name='detector', view=view,
            union_mm=observed, confmap_mm=None, union_path=tmp_path/'original.dat', confmap_path=None,
            temp_dir=tmp_path, dense_tiling_active=True, min_conf=0., min_radius=0.,
            interpolate=distance, interpolation_walk_back=0, interpolation_candidates=1,
            interpolate_passes=1, interpolate_min_radius=0., interpolation_search_angle=0.,
            keep_temp=True, slice_workers=1, interpolation_task_workers=1,
            interpolation_backend=backend, sam_context=context,
            extrapolation_distance=2, extrapolation_min_radius=0.)
    try:
        assert len(called) == 1
        np.testing.assert_array_equal(result.final_view_volume_mm, original | bridge | tail)
        assert result.interpolation_stats[-1]['processing_role'] == 'extrapolation'
        if result.parent_bridge_support_mm is not None:
            np.testing.assert_array_equal(np.stack([result.parent_bridge_support_mm.decode_slice(i)
                for i in range(shape[0])]), bridge)
    finally:
        if result.parent_mask_support_mm is not None:
            result.parent_mask_support_mm.close()
        if result.parent_bridge_support_mm is not None:
            result.parent_bridge_support_mm.close()


def test_shared_context_real_render_planner_evidence_and_publication_without_gpu(tmp_path):
    shape = (5, 12, 13)
    view = geometry.get_view_infos(*shape, cartesian_views=('transverse',))[0]
    native = np.zeros(shape, np.uint8)
    native[2, 3:8, 4:9] = 1
    original = native.copy()
    class RepeatedSeedRuntime:
        device_ids = (0,)
        dispatch_stats = {}
        def run(self, **request):
            return SimpleNamespace(frames={frame: request['seed_mask'].copy()
                for frame in range(request['frame_start'], request['frame_stop'])},
                tracker_scores={}, observation_status={},
                receipt={'prediction_valid': True, 'coverage_complete': True})
        def close(self):
            pass
    context = _context(tmp_path, shape)
    runtime = RepeatedSeedRuntime()
    def start():
        context._runtime = runtime
    try:
        with mock.patch.object(context, '_start', side_effect=start), \
             mock.patch.object(assembly, 'nrrd_layer_sink', return_value=None):
            stats, refs = assembly._run_sam_extrapolation(native, view=view, model_name='detector',
                source='fullframe', sam_context=context, distance=2, walk_back=1, min_radius=0.,
                nrrd_layers_enabled=True)
        assert stats['added_voxels'] == 100
        assert stats['processing_role'] == 'extrapolation'
        assert len(refs) == 2
        expected = np.repeat(original[2:3], shape[0], axis=0)
        np.testing.assert_array_equal(native, expected)
        assert context.rendered_frames == shape[0]
        assert context._runtime is runtime
        assert {ref.extrapolation_provenance['direction'] for ref in refs} == {'forward', 'backward'}
        assert all(ref.extrapolation_provenance['run_ids'] for ref in refs)
        assert all(ref.extrapolation_provenance['terminal_roots'] for ref in refs)
    finally:
        context.close()


def test_extrapolation_only_parent_memory_preflight_matches_actual_reservation():
    view = geometry.get_view_infos(5, 12, 13, cartesian_views=('transverse',))[0]
    task = dict(kind='fullframe', processing_shape=(5, 12, 13), view=view)
    with mock.patch.object(pipeline, '_parent_confidence_capture_plan', return_value=None):
        expected = pipeline._parent_prepare_transient_bytes(view, 5*12*13, 5*12*13,
            interpolation_enabled=True)
        assert pipeline._policy_parent_prepare_transient_bytes([task], 5*12*13, 1,
            interpolation_distance=0, extrapolation_distance=2, retain_confidence=False) == expected


def test_extrapolation_only_parent_uses_shared_admitted_sam_lease(tmp_path):
    from tests.test_sam_parent_staging_pipeline import submission
    function, _, view, _, _, captured, _, _, _ = submission(tmp_path)
    function('model', view)
    task = captured[0][0]
    task.interpolation_backend = 'sdf'
    task.interpolation_distance = 0
    task.extrapolation_distance = 2
    profile = object()
    context = SimpleNamespace(device_ids=('cuda:0',), resource_scope=mock.Mock(
        side_effect=lambda value: contextlib.nullcontext()))
    task.sam_context = context
    with mock.patch('XTA.sam_resources.admit_sam_parent_resources',
            return_value=contextlib.nullcontext(profile)) as admit:
        with task._reservation():
            pass
    assert admit.call_args.args[0] is task.admission
    assert admit.call_args.args[1] == task.transient_bytes
    context.resource_scope.assert_called_once_with(profile)
    task.admission.reserve.assert_not_called()


def test_extrapolation_voting_preflight_accounts_for_extra_group_support_before_inference():
    from XTA.reconciliation_runtime import preflight_reconciliation
    shape = (3, 1024, 1024)
    views = geometry.get_view_infos(*shape, cartesian_views=('transverse',))
    policy = validate_policy(dict(mode='weighted', grouping='views'))
    settings = SimpleNamespace(memory_mib=166)
    plan = preflight_reconciliation(views, source_shape_tyx=shape,
        processing_shape_tyx=shape, settings=settings, policy=policy)
    assert plan['bytes_per_voxel'] == 166
    with pytest.raises(ValueError, match='at least 167 MiB'):
        preflight_reconciliation(views, source_shape_tyx=shape,
            processing_shape_tyx=shape, settings=settings, policy=policy, include_extrapolation=True)
