"""Retained selected crops must reproduce source and two-step mirror pixels."""
import json
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import __version__, assembly, geometry, interpolation, outputs
from XTA.reconciliation_io import read_layer_manifest
from XTA.sam_evidence import selected_native_plane
from XTA.sam_policy import select_sam_proposals
from tests.test_sam_evidence_policy import build_bundle, fixture_group, fixture_run
from tools.export_sam_source_replay import export_source_replay, write_native_direction, file_sha256


def test_spooled_overlapping_groups_are_ored_once_per_slice_without_dense_volume(tmp_path):
    crops = [('B', 2, 2, (1, 2, 3, 4), np.ones((2, 2), bool)),
             ('A', 2, 2, (2, 3, 4, 5), np.ones((2, 2), bool)),
             ('C', 0, 0, (4, 5, 5, 6), np.ones((1, 1), bool))]
    stats = write_native_direction(iter(crops), tmp_path/'direction.cvol', (5, 8, 9), memory_bytes=1024)
    with closing(interpolation.RawBBoxMaskStore.open(tmp_path/'direction.cvol')) as store:
        assert int(store.decode_slice(2).sum()) == 7
        assert int(store.decode_slice(0).sum()) == 1
        assert not store.decode_slice(1).any()
    assert stats['foreground_voxels'] == 8 and stats['logical_raw_uint8_bytes'] == 360
    assert stats['replay_peak_local_raster_bytes'] < 360
    assert not list(tmp_path.glob('*.contributors.bin'))
    with pytest.raises(FileExistsError):
        write_native_direction(iter(crops), tmp_path/'direction.cvol', (5, 8, 9), memory_bytes=1024)


def reference_fixture(tmp_path):
    group, masks, raw = fixture_group()
    group['context_bbox_yx'] = (0, 0, 16, 16)
    masks = {key:np.pad(value, ((0, 4), (0, 0))) for key,value in masks.items()}
    raw = {key:np.pad(value, ((0, 4), (0, 0))) for key,value in raw.items()}
    bundle = build_bundle(tmp_path, [(fixture_run('forward', group), raw)], group=group, masks=masks,
        scope={'shape_tyx':[5, 16, 16], 'view_name':'transverse__tta_a0'})
    receipt = select_sam_proposals(bundle, {'sam_bridge_policy':{'version':4}})
    selection = tmp_path/'selection.json'
    selection.write_text(json.dumps(receipt), encoding='utf-8')
    native = np.stack([selected_native_plane(bundle, receipt, frame) for frame in range(5)])
    source_shape = (3, 18, 24)
    processing_shape = (5, 18, 24)
    original_native = np.zeros(native.shape, np.uint8)
    original_native[0] = raw[0]
    original_native[4] = raw[4]
    original_source = np.stack([outputs._read_layer_slice_in_output_shape(original_native, source_shape, z)
                                for z in range(source_shape[0])])
    view = geometry.expand_views_into_tta_variants(geometry.get_view_infos(*processing_shape,
        cartesian_views=('transverse',)), (0.,))[0]
    reference = tmp_path/'reference'
    specs, _ = outputs.resolve_low_quality_downbin_specs(['0.20'], True, source_shape)
    sink = outputs.NrrdLayerSink(nrrd_dir=reference/'nrrd', stem='fixture',
        output_shape_tyx=source_shape, max_workers=1, low_quality_specs=specs,
        low_quality_root=reference/'low_quality')
    prior_shape, prior_sink = assembly.final_source_output_shape(), outputs.nrrd_layer_sink()
    outputs.set_nrrd_layer_sink(sink)
    assembly.set_final_source_output_shape(source_shape)
    try:
        detector_path = tmp_path/'detector.cvol'
        interpolation.write_raw_bbox_mask_store(original_source, detector_path,
            format_name=interpolation.INTERNAL_PACKED_CVOL_FORMAT, workers=1)
        detector = interpolation.NrrdLayerRef(key='detector', name='detector', path=detector_path,
            shape=source_shape, dtype='uint8', storage_format=interpolation.INTERNAL_PACKED_CVOL_FORMAT,
            model_name='model', view_name=view.name, physical_view_name='transverse',
            aug_id='a0', angle_deg=0., view_family='orthogonal', source='fullframe', mask_kind='yolo',
            pass_index=0, stage='pre_interpolation')
        sink.submit_layer(detector, 'Transverse_TTA_a0_fullframe_yolo')
        context = SimpleNamespace(detector_identity='detector', bundle_identity='bundle',
            source_volume=SimpleNamespace(shape=processing_shape))
        for direction in ('forward', 'backward'):
            data = native if direction == 'forward' else np.zeros(native.shape, np.uint8)
            path = tmp_path/f'old_{direction}.cvol'
            interpolation.write_raw_bbox_mask_store(data, path,
                format_name=interpolation.INTERNAL_PACKED_CVOL_FORMAT, workers=1)
            assembly.materialize_sam_directional_view_layer(dict(direction=direction, path=str(path),
                voxel_count=int(data.sum()), policy_hash='old', evidence_path=str(bundle.directory)),
                model_name='model', view=view, source='fullframe', pass_index=1, sam_context=context)
        sink.wait()
        sink.write_manifest()
        (reference/'manifest.json').write_text(json.dumps(
            {'launcher':{'pipeline_version':'25.0.1', 'version':'25.0.1'}}), encoding='utf-8')
    finally:
        sink.shutdown()
        outputs.set_nrrd_layer_sink(prior_sink)
        assembly.set_final_source_output_shape(prior_shape)
    return bundle, selection, reference, native, source_shape, specs


def test_source_export_matches_current_projection_and_mirror_routes_without_inference(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_NRRD_GPU_MIRROR_TEE', '0')
    bundle, selection, reference, native, source_shape, specs = reference_fixture(tmp_path)
    hashes = {name:file_sha256(bundle.directory/name) for name in ('manifest.json','index.json','masks.bin')}
    with mock.patch.object(interpolation, 'materialize_raw_bbox_mask_store_workspace',
            side_effect=AssertionError('Transverse replay must not materialize a dense observation volume')):
        result = export_source_replay(bundle.directory, selection, reference, tmp_path/'replay', memory_mib=1)
    assert result['source_processing_shape_tyx'] == [5, 18, 24]
    assert result['source_shape_tyx'] == list(source_shape)
    assert result['native_shape_tyx'] == [5, 16, 16]
    assert result['replay']['original_generation_pipeline_version'] == '25.0.1'
    assert result['replay']['export_pipeline_version'] == __version__
    assert result['replay']['selection_pipeline_version'] == __version__
    assert not result['replay']['fresh_detector_run'] and not result['replay']['fresh_sam_run']
    assert (tmp_path/'replay/selection.json').read_bytes() == selection.read_bytes()
    manifest = next((tmp_path/'replay/nrrd').glob('*_nrrd_manifest.json'))
    replay_metadata = json.loads(manifest.read_text(encoding='utf-8'))['replay']
    assert (manifest.parent/replay_metadata['selection_receipt']).resolve() == (tmp_path/'replay/selection.json').resolve()
    expected = np.stack([outputs._read_layer_slice_in_output_shape(native, source_shape, z)
                         for z in range(source_shape[0])])
    with read_layer_manifest(manifest, workspace=tmp_path/'read') as layers:
        assert len(layers) == 3
        forward = next(layer for layer in layers if layer.metadata.get('interpolation_direction') == 'forward')
        backward = next(layer for layer in layers if layer.metadata.get('interpolation_direction') == 'backward')
        np.testing.assert_array_equal(forward.read_slab(0, source_shape[0]), expected)
        assert not backward.read_slab(0, source_shape[0]).any()
        assert forward.metadata['native_transform']['public_backing_coordinate_space'] == 'orthogonal_processing'
    shape = tuple(specs[0].output_shape_t_y_x)
    expected_low = np.stack([outputs._read_layer_slice_in_output_shape(expected, shape, z) for z in range(shape[0])])
    low_manifest = next((tmp_path/'replay/low_quality').glob('*/nrrd/*_nrrd_manifest.json'))
    low_replay_metadata = json.loads(low_manifest.read_text(encoding='utf-8'))['replay']
    assert (low_manifest.parent/low_replay_metadata['selection_receipt']).resolve() == (tmp_path/'replay/selection.json').resolve()
    with read_layer_manifest(low_manifest, workspace=tmp_path/'read_low') as layers:
        forward = next(layer for layer in layers if layer.metadata.get('interpolation_direction') == 'forward')
        np.testing.assert_array_equal(forward.read_slab(0, shape[0]), expected_low)
    assert hashes == {name:file_sha256(bundle.directory/name) for name in hashes}
    assert not list(tmp_path.glob('.*.source-replay-*'))
    with pytest.raises(FileExistsError):
        export_source_replay(bundle.directory, selection, reference, tmp_path/'replay', memory_mib=1)


def test_mismatched_selection_is_rejected_before_any_export(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_NRRD_GPU_MIRROR_TEE', '0')
    bundle, selection, reference, _native, _shape, _specs = reference_fixture(tmp_path)
    receipt = json.loads(selection.read_text(encoding='utf-8'))
    receipt['evidence_fingerprint'] = 'different'
    selection.write_text(json.dumps(receipt), encoding='utf-8')
    with pytest.raises(ValueError, match='does not match'):
        export_source_replay(bundle.directory, selection, reference, tmp_path/'replay', memory_mib=1)
    assert not (tmp_path/'replay').exists()
