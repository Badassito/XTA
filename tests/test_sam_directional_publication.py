"""SAM source consumers must receive projected masks, and resize can collapse gaps."""
from pathlib import Path
import gc
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import assembly, geometry, interpolation
from XTA.config import resolve_tilted_view_groups
from XTA.outputs import NrrdLayerSink, _read_layer_slice_in_output_shape
from XTA.reconciliation_io import read_layer_manifest
from XTA.reconciliation_runtime import RuntimeLayer
from XTA.runtime import wait_for_retired_memmap_unlinks


def publish(tmp_path, native, view, *, sink=None, workers=1):
    path = tmp_path / 'sam_bridge_pass01_forward.cvol'
    interpolation.write_raw_bbox_mask_store(native, path,
        format_name=interpolation.INTERNAL_PACKED_CVOL_FORMAT, workers=1)
    context = SimpleNamespace(detector_identity='detector', bundle_identity='sam',
        source_volume=np.zeros((view.full_t, view.full_h, view.full_w), np.uint8))
    with mock.patch.object(assembly, 'nrrd_layer_sink', return_value=sink), \
         mock.patch.object(assembly, 'final_source_output_shape', return_value=(view.full_t, view.full_h, view.full_w)):
        ref = assembly.materialize_sam_directional_view_layer(dict(direction='forward', path=str(path),
            voxel_count=int(native.sum()), policy_hash='policy'), model_name='detector',
            view=view, source='fullframe', pass_index=1, sam_context=context, workers=workers)
    return ref, path


@pytest.mark.parametrize('base', ('sagittal', 'coronal'))
def test_cubic_public_ref_and_nrrd_export_permute_voxels_even_when_shapes_match(tmp_path, base):
    view = geometry.get_view_infos(7, 7, 7, cartesian_views=(base,))[0]
    source = np.zeros((7, 7, 7), np.uint8)
    source[1, 3, 5] = 1
    inverse = {'sagittal': (1, 0, 2), 'coronal': (2, 0, 1)}[base]
    native = source.transpose(inverse).copy()
    sink = NrrdLayerSink(nrrd_dir=tmp_path/'nrrd', stem='case', output_shape_tyx=source.shape, max_workers=1)
    try:
        ref, native_path = publish(tmp_path, native, view, sink=sink)
        assert ref.path != native_path
        assert ref.native_transform['public_backing_coordinate_space'] == 'orthogonal_processing'
        assert ref.native_transform['kind'] == 'axis_permutation'
        owner = RuntimeLayer(ref, source.shape)
        try:
            np.testing.assert_array_equal(owner.read_slab(0, 7), source)
        finally:
            owner.close()
        sink.wait()
        manifest = sink.write_manifest()
    finally:
        sink.shutdown()
    with read_layer_manifest(manifest, workspace=tmp_path/'read') as layers:
        assert len(layers) == 1
        assert layers[0].metadata['interpolation_backend'] == 'sam'
        np.testing.assert_array_equal(layers[0].read_slab(0, 7), source)


def test_empty_nonlinear_public_ref_is_source_sized_and_preserves_native_store(tmp_path):
    view = next(view for view in geometry.get_view_infos(7, 9, 11, cartesian_views=(),
        radial_views=('transverse',), radial_min_radius=.5, radial_patch_size=8) if view.family == 'radial')
    native = np.zeros((view.num_slices, view.src_h, view.src_w), np.uint8)
    with mock.patch.object(interpolation, 'materialize_raw_bbox_mask_store_workspace',
            side_effect=AssertionError('Empty slots must skip dense decoding')):
        ref, native_path = publish(tmp_path, native, view)
    assert ref.shape == (7, 9, 11)
    assert ref.segment_extent_ijk == (0, -1, 0, -1, 0, -1)
    assert native_path.is_dir()
    assert ref.native_transform['native_shape_tyx'] == list(native.shape)


def test_sam_directional_materializer_forwards_existing_projection_workers(tmp_path):
    from XTA.sparse_projection import project_azimuthal_sparse_store
    view=next(view for view in geometry.get_view_infos(7,9,11,cartesian_views=(),
        tilt_groups=resolve_tilted_view_groups(['transverse:20:vertical']),
        azimuthal_views=('tilted_transverse',),azimuthal_azimuth_angles=(45.,)) if view.family=='azimuthal')
    native=np.zeros((view.num_slices,view.src_h,view.src_w),np.uint8)
    native[1,native.shape[1]//2,native.shape[2]//2]=1
    with mock.patch('XTA.sparse_projection.project_azimuthal_sparse_store',wraps=project_azimuthal_sparse_store) as projector:
        ref,_=publish(tmp_path,native,view,workers=7)
    assert projector.call_args.kwargs['workers']==7
    assert ref.shape==(7,9,11)


@pytest.mark.parametrize('base', ('sagittal', 'coronal'))
def test_reduced_cartesian_canvas_permuted_before_single_source_restore(tmp_path, base, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_DELAY_NATIVE_EXPANSION', '1')
    view = geometry.get_view_infos(7, 9, 11, cartesian_views=(base,))[0]
    native = np.zeros((view.num_slices, 5, 5), np.uint8)
    native[2, 1, 3] = 1
    ref, _ = publish(tmp_path, native, view)
    permutation = {'sagittal': (1, 0, 2), 'coronal': (1, 2, 0)}[base]
    processing = native.transpose(permutation).copy()
    expected = np.stack([_read_layer_slice_in_output_shape(processing, (7, 9, 11), frame)
                         for frame in range(7)])
    owner = RuntimeLayer(ref, (7, 9, 11))
    try:
        assert ref.shape == processing.shape
        np.testing.assert_array_equal(owner.read_slab(0, 7), expected)
    finally:
        owner.close()


def test_failed_native_projection_decode_retires_its_dense_workspace(tmp_path):
    view = next(view for view in geometry.get_view_infos(7, 9, 11, cartesian_views=(),
        tilt_groups=resolve_tilted_view_groups(['transverse:20:vertical'])) if view.family == 'tilted')
    native = np.zeros((view.num_slices, view.src_h, view.src_w), np.uint8)
    native[1, 3, 5] = 1
    with mock.patch.object(interpolation.RawBBoxMaskStore, 'fill_decoded_slice_into',
            side_effect=RuntimeError('injected decode failure')):
        with pytest.raises(RuntimeError, match='injected decode failure'):
            publish(tmp_path, native, view)
    gc.collect()  # The injected exception's traceback can retain the workspace.
    wait_for_retired_memmap_unlinks(timeout_s=5)
    assert not list((tmp_path/'source_layers').glob('*.native.u8.dat'))


def test_production_depth_restore_maps_a_native_gap_inside_a_prediction():
    original = np.zeros((2911, 1, 1), np.uint8)
    bridge = np.zeros_like(original)
    original[1, 0, 0] = 1
    bridge[0, 0, 0] = 1
    assert not np.any(original & bridge)
    # The 2911 -> 1931 restore ORs all input frames touching an output bin.
    assert _read_layer_slice_in_output_shape(original, (1931, 1, 1), 0)[0, 0]
    assert _read_layer_slice_in_output_shape(bridge, (1931, 1, 1), 0)[0, 0]


def test_low_quality_restore_collapses_separate_source_voxels_into_one_bin():
    original = np.zeros((5, 5, 5), np.uint8)
    bridge = np.zeros_like(original)
    original[1, 1, 1] = 1
    bridge[2, 1, 1] = 1
    assert not np.any(original & bridge)
    assert _read_layer_slice_in_output_shape(original, (1, 1, 1), 0)[0, 0]
    assert _read_layer_slice_in_output_shape(bridge, (1, 1, 1), 0)[0, 0]
