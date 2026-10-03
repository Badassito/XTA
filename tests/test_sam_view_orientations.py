"""SAM view routing/native output projection; no GPU or model inference."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import assembly, geometry
from XTA.config import resolve_tilted_view_groups
from XTA.sam_interpolation import _validate_view, interpolate_sam_view_volume_pass
from XTA.sam_view_geometry import sam_native_transform_record, validate_sam_view_geometry
from tests.test_sam_interpolation import RepeatedSeedTracker


SOURCE_SHAPE = (7, 9, 11)


def representative_views():
    bases = ('transverse', 'sagittal', 'coronal')
    tilted = tuple('tilted_' + name for name in bases)
    views = geometry.get_view_infos(*SOURCE_SHAPE, cartesian_views=bases,
        tilt_groups=resolve_tilted_view_groups(['transverse,sagittal,coronal:20:both']),
        azimuthal_views=(*bases, *tilted), azimuthal_azimuth_angles=(60.,)*6,
        radial_views=(*bases, *tilted), radial_min_radius=.5, radial_patch_size=8,
        spherical_views=('transverse', 'tilted_transverse'), spherical_min_radius=.5, spherical_patch_size=8)
    # Every axis, tilt sign/direction and shell face is represented; multiple
    # adjacent patches with the same recipe are covered by their geometry tests.
    selected = {}
    for view in views:
        base = (geometry.physical_view_name(view) if view.family == 'orthogonal' else
                view.tilt_base_view or view.azimuthal_base_view or view.radial_base_view)
        key = (view.family, base, view.tilt_angle_deg, view.tilt_direction, view.spherical_face)
        selected.setdefault(key, view)
    return tuple(selected.values())


VIEWS = representative_views()
VARIANTS = tuple(geometry.expand_views_into_tta_variants(VIEWS, (0., 31., 120.)))


@pytest.fixture(autouse=True)
def force_cpu_geometry(monkeypatch):
    for flag in ('YOLO_TTA_GPU_BACKPROJECT', 'YOLO_TTA_GPU_RADIAL_BACKPROJECT',
                 'YOLO_TTA_GPU_SPHERICAL_BACKPROJECT', 'YOLO_TTA_GPU_TILTED_AZIMUTHAL_BACKPROJECT'):
        monkeypatch.setenv(flag, '0')


@pytest.mark.parametrize('view', VARIANTS, ids=lambda view: view.name)
def test_every_established_tta_recipe_and_augmentation_is_admitted(view):
    validate_sam_view_geometry(view, wrap_axis=view.family == 'azimuthal')
    _validate_view(view, view.family == 'azimuthal', {'angle_deg': view.tta_angle_deg})
    native = (view.num_slices, view.src_h, view.src_w)
    record = sam_native_transform_record(view, native, SOURCE_SHAPE)
    assert record['accumulation_angle_deg'] == record['angle_deg'] == 0.
    assert record['detector_augmentation_angle_deg'] == view.tta_angle_deg
    assert record['augmentation_unwarped_before_sam']
    if view.family != 'orthogonal':
        assert record['kind'] != 'identity'
        assert record['source_reconstruction'] == 'tta_direct_native_destination_pull'
        assert record['source_projection_contract'] == 'xta.native_destination_pull/1'
        assert record['terminal_source_restore_required'] is False
    if view.family == 'azimuthal':
        assert record['azimuthal']['frame_seam'] == 'half_turn_with_mirrored_u'
    if view.family == 'radial':
        assert record['radial']['slice_direction'] == 'radius'
    if view.family == 'spherical':
        assert record['spherical']['slice_direction'] == 'radius'


@pytest.mark.parametrize('base,permutation,axes', (
    ('transverse', (0, 1, 2), ['t', 'y', 'x']),
    ('sagittal', (1, 0, 2), ['y', 't', 'x']),
    ('coronal', (1, 2, 0), ['x', 't', 'y']),
))
def test_cubic_shape_never_substitutes_identity_for_named_axis_recipe(base, permutation, axes):
    view = next(view for view in geometry.get_view_infos(7, 7, 7, cartesian_views=(base,))
                if view.family == 'orthogonal')
    record = sam_native_transform_record(view, (7, 7, 7), (7, 7, 7))
    assert record['view_to_source_axis_permutation'] == list(permutation)
    assert record['view_axes_in_source'] == axes
    assert record['kind'] == ('identity' if base == 'transverse' else 'axis_permutation')


def test_sampler_recipe_retains_roi_and_excludes_detector_augmentation():
    view = next(view for view in VIEWS if view.family == 'azimuthal' and not view.azimuthal_tilted_source)
    a, b = geometry.expand_views_into_tta_variants((view,), (0., 120.))
    shape = (view.num_slices, view.src_h, view.src_w)
    native = sam_native_transform_record(a, shape, SOURCE_SHAPE)
    rotated = sam_native_transform_record(b, shape, SOURCE_SHAPE)
    assert native['sampler_recipe'] == rotated['sampler_recipe']
    assert native['M_native_to_canvas'] == rotated['M_native_to_canvas']
    changed = sam_native_transform_record(replace(view, roi_radius=view.roi_radius + .125), shape, SOURCE_SHAPE)
    assert changed['sampler_recipe'] != native['sampler_recipe']
    assert changed['azimuthal']['roi_radius'] == view.roi_radius + .125


@pytest.mark.parametrize('view', VARIANTS, ids=lambda view: view.name)
@pytest.mark.parametrize('delayed', ('0', '1'))
def test_detector_inverse_augmentation_returns_to_the_canonical_sam_canvas(view, delayed, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_DELAY_NATIVE_EXPANSION', delayed)
    out_size = 5
    detector = geometry.build_affine(view=view.name, src_w=view.src_w, src_h=view.src_h,
                                      out_size=out_size, angle_deg=view.tta_angle_deg, pad_mode=view.pad_mode)
    restored = geometry.output_to_view_processing_affine(view, detector.M_out_to_src, out_size)
    native_to_detector = np.eye(3)
    native_to_detector[:2] = detector.M_src_to_out
    detector_to_accumulation = np.eye(3)
    detector_to_accumulation[:2] = restored
    native_to_accumulation = detector_to_accumulation @ native_to_detector
    if geometry.view_uses_inference_processing_grid(view, out_size):
        canonical = geometry.build_affine(view=view.name, src_w=view.src_w, src_h=view.src_h,
                          out_size=out_size, angle_deg=0., pad_mode=view.pad_mode).M_src_to_out
    else:
        canonical = np.array([[1., 0., 0.], [0., 1., 0.]])
    points = np.array([[1.25, 2.75, 1.], [view.src_w-1.75, view.src_h-1.25, 1.]])
    np.testing.assert_allclose((points @ native_to_accumulation.T)[:, :2], points @ canonical.T,
                               rtol=0., atol=2e-5)


@pytest.mark.parametrize('base', ('transverse', 'sagittal', 'coronal'))
@pytest.mark.parametrize('angle', (0., 31., 120.))
def test_source_projection_keeps_cartesian_voxel_witnesses_after_augmentation_restore(tmp_path, base, angle):
    physical = geometry.get_view_infos(*SOURCE_SHAPE, cartesian_views=(base,))[0]
    view = geometry.expand_views_into_tta_variants((physical,), (angle,))[0]
    source = np.zeros(SOURCE_SHAPE, np.uint8)
    source[1, 3, 8] = source[5, 7, 2] = 1
    expected_permutation = {'transverse': (0, 1, 2), 'sagittal': (1, 0, 2), 'coronal': (2, 0, 1)}[base]
    native = source.transpose(expected_permutation).copy()
    actual = assembly.project_view_volume_to_orthogonal_volume(native, view, tmp_path/'project.u8',
                'controlled SAM native reconstruction', prefer_memory=True, reserve_bytes=0,
                out_shape_tyx=SOURCE_SHAPE)
    try:
        np.testing.assert_array_equal(actual, source)
    finally:
        if isinstance(actual, np.memmap):
            actual._mmap.close()


@pytest.mark.parametrize('view', VIEWS, ids=lambda view: view.name)
def test_nonidentity_final_connection_diagnostic_does_not_claim_source_topology(view):
    if view.family == 'orthogonal' and geometry.physical_view_name(view) == 'transverse':
        return
    from XTA.tta_outputs import measure_sam_final_connections
    transform = sam_native_transform_record(view, (view.num_slices, view.src_h, view.src_w), SOURCE_SHAPE)
    ref = SimpleNamespace(interpolation_backend='sam', mask_kind='bridge', proposal_selection_status='policy_selected',
                          key=view.name, model_name='detector', proposal_evidence_path='never_read',
                          sam_run_ids=('run',), native_transform=transform)
    with mock.patch('XTA.sam_evidence.SamEvidenceBundle.open', side_effect=AssertionError('no unbounded audit')):
        result = measure_sam_final_connections([ref], np.zeros(SOURCE_SHAPE, np.uint8))
    receipt = result[('detector', view.name)]
    assert receipt['status'] == 'not_assessed'
    assert 'verified identity Transverse source transform' in receipt['reason']


@pytest.mark.parametrize('view', tuple(view for view in VIEWS if
    (view.family == 'orthogonal' or
     view.family == 'tilted' and view.tilt_angle_deg > 0 and view.tilt_direction == 'vertical' or
     view.family == 'azimuthal' and not view.azimuthal_tilted_source or
     view.family == 'radial' and not view.radial_tilted_source or
     view.family == 'spherical' and not view.spherical_tilted_source)), ids=lambda view: view.name)
def test_fake_sam_selected_directional_slots_use_native_view_geometry_and_existing_source_projector(tmp_path, view):
    # This fixture is on the canonical accumulated canvas, not an augmented
    # YOLO raster. Tracker masks retain native row/column/frame addresses.
    view = geometry.expand_views_into_tta_variants((view,), (31.,))[0]
    shape = (view.num_slices, view.src_h, view.src_w)
    if shape[0] < 3:
        pytest.skip('trajectory has no interior missing frame')
    observed = np.zeros(shape, np.uint8)
    row, column = shape[1]//2, shape[2]//2
    observed[(0, shape[0]-1), row, column] = 1
    original = observed.copy()
    tracker = RepeatedSeedTracker()
    merged, stats, components = interpolate_sam_view_volume_pass(observed, view=view,
        scope={'scope_id': view.name, 'angle_deg': view.tta_angle_deg}, runtime=tracker,
        work_dir=tmp_path/'evidence', gap_distance=shape[0], min_radius=0,
        interpolation_walk_back=0, return_bridge_components=True)
    try:
        np.testing.assert_array_equal(observed, original)
        assert stats['sam_selected_runs'] == 2
        assert {item['direction'] for item in components} == {'forward', 'backward'}
        assert np.all(merged[:, row, column])
        context = SimpleNamespace(detector_identity='detector', bundle_identity='bundle',
                                  source_volume=np.zeros(SOURCE_SHAPE, np.uint8))
        with mock.patch.object(assembly, 'nrrd_layer_sink', return_value=None), \
                mock.patch.object(assembly, 'final_source_output_shape', return_value=SOURCE_SHAPE):
            refs = [assembly.materialize_sam_directional_view_layer(dict(item), model_name='detector', view=view,
                    source='fullframe', pass_index=1, sam_context=context) for item in components]
        assert all(ref.shape == shape for ref in refs)
        assert all(ref.native_transform['view_family'] == view.family for ref in refs)
        assert all(ref.native_transform['detector_augmentation_angle_deg'] == 31. for ref in refs)
        projected = assembly.project_view_volume_to_orthogonal_volume(np.asarray(merged), view,
                    tmp_path/'source.u8', 'selected SAM view projection', out_shape_tyx=SOURCE_SHAPE,
                    prefer_memory=True, reserve_bytes=0)
        try:
            assert projected.shape == SOURCE_SHAPE
            assert projected.dtype == np.uint8
            assert set(np.unique(projected)).issubset({0, 1})
            if view.family == 'orthogonal':
                expected = np.asarray(merged).transpose(tuple(refs[0].native_transform['view_to_source_axis_permutation']))
                np.testing.assert_array_equal(projected, expected)
        finally:
            if isinstance(projected, np.memmap):
                projected._mmap.close()
    finally:
        if isinstance(merged, np.memmap):
            merged._mmap.close()


@pytest.mark.parametrize('backend', ('tiled', 'dynamic'))
@pytest.mark.parametrize('changes', ({'physical_view_id': 'sagittal'}, {'physical_view_id': 'coronal'},
    {'tta_angle_deg': 31.}, {'runtime_family': 'radial'}, {'runtime_family': 'tilted'},
    {'runtime_angle': 31.}, {'runtime_angle': float('nan')}))
def test_lta_runtime_stays_explicitly_transverse_only(backend, changes):
    from XTA.lta_execution import _require_supported_runtime
    runtime = geometry.get_view_infos(1764, 1764, 1764, cartesian_views=('transverse',))[0]
    family = changes.get('runtime_family', 'orthogonal')
    runtime = replace(runtime, family=family, tta_angle_deg=changes.get('runtime_angle', 0.))
    view = SimpleNamespace(physical_view_id=changes.get('physical_view_id', 'transverse'),
                          tta_angle_deg=changes.get('tta_angle_deg', 0.), runtime_view=runtime,
                          frame_height=1764, frame_width=1764,
                          tile_grids=(() if backend == 'dynamic' else (SimpleNamespace(tile_size=1008),)))
    plan = SimpleNamespace(bundle=SimpleNamespace(model_version='sam3.1'), sam_execution='video',
                           discovery=SimpleNamespace(target_volumes=(object(),)),
                           volumes=(SimpleNamespace(runtime_views=(view,)),), crop_backend=backend)
    with pytest.raises(ValueError, match='Transverse|transverse'):
        _require_supported_runtime(plan)


@pytest.mark.parametrize('family', ('radial', 'spherical', 'tilted', 'orthogonal'))
def test_only_azimuthal_frames_use_the_cyclic_frame_protocol(family):
    view = next(item for item in VIEWS if item.family == family)
    with pytest.raises(ValueError, match='wrapping requires an Azimuthal'):
        _validate_view(view, True, {})
