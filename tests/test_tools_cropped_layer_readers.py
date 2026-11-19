"""Qualification, audit, profiling and fixture tools read cropped layers exactly.

Each tool's decode path runs on the same layers written by the real NrrdLayerSink
in the full-raster and cropped (segment bounding box) layouts; results must agree.
"""
from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
from types import ModuleType
from unittest import mock

import numpy as np
import pytest

from XTA import nrrd_compression, outputs
from XTA.interpolation import (
    CVOL_FORMAT,
    INTERNAL_PACKED_CVOL_FORMAT,
    NrrdLayerRef,
    write_raw_bbox_mask_store,
)
from XTA.reconciliation_io import read_layer_manifest
from tools import nrrd_reference_grid as grid

SHAPE = (11, 29, 31)
LOW_QUALITY = (outputs.LowQualityDownbinSpec('0.5', '0p5', .5, (6, 15, 16)),)
LAYOUTS = ('full', 'cropped')


@pytest.fixture
def zlib_codec(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_NRRD_GPU_MIRROR_TEE', '0')
    spec = nrrd_compression._nrrd_member_codec_spec('zlib')
    with mock.patch.object(nrrd_compression, '_require_nrrd_member_codec', return_value=spec):
        yield


def _box(box, *, seed, shape=SHAPE):
    rng = np.random.default_rng(seed)
    t0, t1, y0, y1, x0, x1 = box
    volume = np.zeros(shape, np.uint8)
    volume[t0:t1, y0:y1, x0:x1] = rng.random((t1 - t0, y1 - y0, x1 - x0)) < .5
    volume[t0, y0, x0] = volume[t1 - 1, y1 - 1, x1 - 1] = 1
    return volume


def _live(name, volume, **fields):
    return NrrdLayerRef(key=name, name=name, path=Path(name + '.live'), shape=volume.shape,
                        storage_format='live_u8', live_array=volume, **fields)


def _volumes():
    detector = _box((3, 9, 4, 20, 6, 25), seed=1)
    forward = _box((2, 6, 8, 15, 3, 14), seed=2)
    backward = _box((5, 10, 10, 27, 12, 30), seed=3)
    corner = np.zeros(SHAPE, np.uint8)
    corner[7, 28, 30] = 1
    tile = _box((1, 8, 2, 12, 20, 29), seed=4)
    return dict(detector=detector, forward=forward, backward=backward, corner=corner,
                empty=np.zeros(SHAPE, np.uint8), tile=tile,
                global_final=detector | forward | backward | corner | tile)


def _refs(root):
    volumes = _volumes()
    sam = dict(interpolation_backend='sam', proposal_selection_status='policy_selected',
               proposal_evidence_path='evidence/manifest.json', interpolation_policy_identity='policy-a')
    store_path = Path(root) / 'tile.cvol'
    with contextlib.redirect_stdout(io.StringIO()):
        stats = write_raw_bbox_mask_store(volumes['tile'], store_path,
                                          format_name=INTERNAL_PACKED_CVOL_FORMAT, desc='test', workers=1)
    return volumes, [
        ('detector', _live('detector', volumes['detector'], view_name='transverse',
                           source='fullframe', mask_kind='yolo')),
        ('forward', _live('forward', volumes['forward'], view_name='transverse', source='fullframe',
                          mask_kind='bridge', pass_index=1, stage='sam_forward',
                          interpolation_direction='forward', **sam)),
        ('backward', _live('backward', volumes['backward'], view_name='transverse', source='fullframe',
                           mask_kind='bridge', pass_index=1, stage='sam_backward',
                           interpolation_direction='backward', **sam)),
        ('corner', _live('corner', volumes['corner'], view_name='corner', source='fullframe', mask_kind='yolo')),
        ('empty', _live('empty', volumes['empty'], view_name='empty', source='fullframe', mask_kind='yolo')),
        ('tile', NrrdLayerRef(key='tile', name='tile', path=store_path, shape=SHAPE,
                              storage_format=INTERNAL_PACKED_CVOL_FORMAT,
                              segment_extent_ijk=tuple(stats['segment_extent_ijk']),
                              segment_extent_shape_tyx=SHAPE, segment_extent_source='raw_bbox_cvol_index',
                              view_name='tile', source='tile', mask_kind='yolo', tile_config_id='tile32')),
        ('Global_final_output', _live('Global_final_output', volumes['global_final'], view_name='global',
                                      source='global', mask_kind='union',
                                      stage='final_output_after_all_postprocessing',
                                      layer_role='complete_checkpoint', recomposition_op='select')),
    ]


def _publish(root, layout, monkeypatch):
    """Write one run's layers in ``layout``; returns (outputs dir, volumes, native manifest)."""
    monkeypatch.setenv('YOLO_TTA_NRRD_LAYER_ZSHARDS', '1')
    output = Path(root) / 'outputs'
    (Path(root) / 'sources').mkdir(parents=True)
    volumes, refs = _refs(Path(root) / 'sources')
    sink = outputs.NrrdLayerSink(nrrd_dir=output / 'nrrd', stem='case', output_shape_tyx=SHAPE, max_workers=2,
                                 low_quality_specs=list(LOW_QUALITY), low_quality_root=output / 'low_quality',
                                 run_id='run', layer_layout=layout)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for name, ref in refs:
                sink.submit_layer(ref, name)
            sink.wait()
        sink.write_manifest()
    finally:
        sink.shutdown()
    manifest = output / 'nrrd' / 'case_nrrd_manifest.json'
    document = json.loads(manifest.read_text())
    # The audit scopes SAM rows by these fields; keep them explicit whatever the sink copies.
    for row in document['layers']:
        if row.get('stage', '').startswith('sam_'):
            row.update(interpolation_backend='sam', proposal_selection_status='policy_selected',
                       interpolation_direction=row['stage'].removeprefix('sam_'),
                       proposal_evidence_path='evidence/manifest.json',
                       interpolation_policy_identity='policy-a')
    manifest.write_text(json.dumps(document, indent=2))
    (output / 'manifest.json').write_text(json.dumps({'status': 'complete'}))
    (Path(root) / 'pipeline.log').write_text('')
    return output, volumes, manifest


@pytest.fixture
def runs(tmp_path, zlib_codec, monkeypatch):
    return {layout: _publish(tmp_path / layout, layout, monkeypatch) for layout in LAYOUTS}


def _manifests(output):
    return [output / 'nrrd' / 'case_nrrd_manifest.json',
            output / 'low_quality' / LOW_QUALITY[0].token / 'nrrd' / 'case_nrrd_manifest.json']


def _supported_volumes(manifest):
    with read_layer_manifest(manifest, workspace=manifest.parent, include_all_layers=True) as collection:
        return {layer.metadata['filename']: (layer.read_slab(0, collection.shape_tyx[0]), layer.path,
                                             layer.metadata) for layer in collection}


def test_fixture_layers_are_really_cropped(runs):
    full, cropped = runs['full'][2], runs['cropped'][2]
    smaller = 0
    for row_full, row_cropped in zip(json.loads(full.read_text())['layers'], json.loads(cropped.read_text())['layers']):
        assert row_full['filename'] == row_cropped['filename']
        assert row_full['stored_shape_tyx'] == list(SHAPE) and 'stored_offset_tyx' not in row_full
        assert row_cropped['stored_layout'] == 'segment_bounding_box'
        smaller += np.prod(row_cropped['stored_shape_tyx']) < np.prod(SHAPE)
        assert row_cropped.get('interpolation_backend') == row_full.get('interpolation_backend')
    assert smaller == 7


def test_reference_grid_helpers_match_supported_reader(runs):
    for full_manifest, cropped_manifest in zip(_manifests(runs['full'][0]), _manifests(runs['cropped'][0])):
        expected = _supported_volumes(full_manifest)
        actual = _supported_volumes(cropped_manifest)
        assert expected.keys() == actual.keys()
        for name, (reference, full_path, _row) in expected.items():
            np.testing.assert_array_equal(actual[name][0], reference)
            digest = hashlib.sha256(reference.tobytes()).hexdigest()
            full_raster = grid.inspect_reference_raster(full_path)
            cropped_raster = grid.inspect_reference_raster(actual[name][1])
            assert not full_raster.cropped
            assert cropped_raster.cropped == (cropped_raster.stored_shape != reference.shape)
            assert cropped_raster.reference_shape == full_raster.reference_shape == reference.shape
            assert cropped_raster.reference_spatial_fields() == full_raster.reference_spatial_fields()
            assert cropped_raster.reference_segment_extent() == full_raster.reference_segment_extent()
            for path in (full_path, actual[name][1]):
                np.testing.assert_array_equal(grid.read_reference_volume(path), reference)
                assert grid.reference_payload_sha256(path) == digest
                assert grid.reference_payload_sha256(path, chunk_bytes=1) == digest
                slabs = list(grid.iter_reference_slabs(path, 3))
                assert [z for z, _ in slabs] == list(range(0, reference.shape[0], 3))
                np.testing.assert_array_equal(np.concatenate([s for _, s in slabs]), reference)
                for start, stop in ((0, 1), (2, reference.shape[0] - 2), (reference.shape[0] - 1, reference.shape[0])):
                    planes, shape = grid.read_reference_planes(path, start, stop)
                    assert shape == reference.shape
                    np.testing.assert_array_equal(planes, reference[start:stop])
        with grid.open_manifest_layers(cropped_manifest, filenames=list(expected)[::-1]) as collection:
            assert [layer.metadata['filename'] for layer in collection] == list(expected)[::-1]


def test_reference_reader_rejects_truncated_or_misplaced_payloads(tmp_path):
    def write(name, sizes, offset, payload, extent='0 4 0 3 0 2'):
        path = tmp_path / name
        path.write_bytes((f'NRRD0005\ntype: uint8\ndimension: 3\nsizes: {sizes}\nencoding: gzip\n'
                          f'space directions: (1,0,0) (0,1,0) (0,0,1)\nspace origin: (1,2,0)\n'
                          f'Segmentation_ReferenceImageExtentOffset:={offset}\n'
                          f'Segmentation_ReferenceImageExtent:={extent}\n\n').encode() + gzip.compress(payload))
        return path
    good = write('good.nrrd', '2 1 1', '1 2 0', b'\x01\x01')
    volume = grid.read_reference_volume(good)
    assert volume.shape == (3, 4, 5) and volume.sum() == 2 and volume[0, 2, 1:3].all()
    assert grid.inspect_reference_raster(good).reference_spatial_fields()['space origin'] == '(0,0,0)'
    with pytest.raises(ValueError, match='shorter'):
        grid.read_reference_volume(write('short.nrrd', '2 1 1', '1 2 0', b'\x01'))
    with pytest.raises(ValueError, match='exceeds'):
        grid.read_reference_volume(write('long.nrrd', '2 1 1', '1 2 0', b'\x01\x01\x00'))
    with pytest.raises(ValueError, match='outside reference'):
        grid.inspect_reference_raster(write('outside.nrrd', '2 1 1', '4 2 0', b'\x01\x01'))
    with pytest.raises(ValueError, match='differs'):
        grid.inspect_reference_raster(good, reference_shape_tyx=(3, 4, 6))


def test_qualify_tta_reconciliation_verdicts_match(runs):
    from tools import qualify_tta_reconciliation as qualification
    receipts, bridges = {}, {}
    for layout, (output, volumes, manifest) in runs.items():
        receipt = qualification.inspect_case(output, 'cpu_off')
        receipt.pop('output')
        receipts[layout] = receipt
        bridges[layout] = qualification.verify_bridge_evidence(output)
        for row, values in qualification.decode_manifest_layers(manifest):
            np.testing.assert_array_equal(qualification.decode_nrrd(manifest.parent / row['filename']), values)
    assert receipts['full'] == receipts['cropped']
    assert bridges['full'] == bridges['cropped']
    volumes = runs['full'][1]
    assert receipts['full']['final_sha256'] == hashlib.sha256(volumes['global_final'].tobytes()).hexdigest()
    assert receipts['full']['component_count'] == 6
    assert bridges['full']['bridge_voxels'] == int((volumes['forward'] | volumes['backward']).sum())


def test_qualify_tta_augmentation_layer_verdicts_match(runs):
    from tools.qualify_tta_augmentation import verify_published_layers
    results = {layout: verify_published_layers(manifest) for layout, (_output, _volumes, manifest) in runs.items()}
    assert results['full'] == results['cropped']
    assert results['full']['final_shape_x_y_t'] == list(reversed(SHAPE))
    assert results['full']['final_foreground'] == int(runs['full'][1]['global_final'].sum())


def test_audit_sam_bridge_overlap_matches_on_both_layouts(runs, tmp_path, zlib_codec):
    from tools import audit_sam_bridge_overlap as audit
    reports, diagnostics = {}, {}
    for layout, (output, _volumes, _manifest) in runs.items():
        destination = tmp_path / f'audit-{layout}.json'
        diagnostic = tmp_path / f'new-voxels-{layout}' / 'sam_new_voxels.seg.nrrd'
        argv = ['audit_sam_bridge_overlap.py', str(output), str(destination),
                '--new-voxels-output', str(diagnostic)]
        with mock.patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
            audit.main()
        report = json.loads(destination.read_text())
        report.pop('seconds')
        reports[layout] = report
        diagnostics[layout] = grid.read_reference_volume(diagnostic)
        voxels = report.pop('new_voxels_diagnostic')['voxel_count']
        assert voxels == int(diagnostics[layout].sum())
    assert reports['full'] == reports['cropped']
    np.testing.assert_array_equal(diagnostics['full'], diagnostics['cropped'])
    volumes = runs['full'][1]
    detector, union = volumes['detector'] != 0, (volumes['forward'] | volumes['backward']) != 0
    assert reports['full']['shape_tyx'] == list(SHAPE)
    assert reports['full']['totals']['detector_voxels'] == int(detector.sum())
    assert reports['full']['totals']['bridge_union_voxels'] == int(union.sum())
    assert reports['full']['totals']['overlap_union_voxels'] == int((detector & union).sum())
    np.testing.assert_array_equal(diagnostics['full'], (union & ~detector).astype(np.uint8))
    cropped_detector = runs['cropped'][2].parent / 'case_detector.seg.nrrd'
    with pytest.raises(ValueError, match='cropped box'):
        audit.open_payload(cropped_detector)


def test_audit_pta_native_plane_readers_match(runs):
    from tools import audit_pta_native_boundary as boundary
    from tools import audit_pta_native_downbin as downbin
    for grid_index in range(2):
        full_manifest = _manifests(runs['full'][0])[grid_index]
        cropped_manifest = _manifests(runs['cropped'][0])[grid_index]
        for name in ('case_Global_final_output.seg.nrrd', 'case_corner.seg.nrrd', 'case_empty.seg.nrrd'):
            full_path, cropped_path = full_manifest.parent / name, cropped_manifest.parent / name
            reference = grid.read_reference_volume(full_path)
            for t in range(reference.shape[0]):
                expected, actual = boundary.plane(full_path, t), boundary.plane(cropped_path, t)
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_array_equal(actual[0], reference[t] > 0)
                assert actual[1:] == expected[1:]
                assert actual[2] == hashlib.sha256(reference[t].tobytes()).hexdigest()
            for start, stop in ((0, 2), (3, reference.shape[0])):
                expected, actual = downbin.read_planes(full_path, start, stop), downbin.read_planes(cropped_path, start, stop)
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_array_equal(actual[0], reference[start:stop])
                assert actual[1] == expected[1] == reference.shape


def test_profile_tta_nrrd_contention_hashes_reference_grid(runs):
    from tools.profile_tta_nrrd_contention import _decoded_nrrd_sha256
    for full_manifest, cropped_manifest in zip(_manifests(runs['full'][0]), _manifests(runs['cropped'][0])):
        for name, (reference, full_path, _row) in _supported_volumes(full_manifest).items():
            digest = hashlib.sha256(reference.tobytes()).hexdigest()
            assert _decoded_nrrd_sha256(full_path) == _decoded_nrrd_sha256(cropped_manifest.parent / name) == digest


def test_profile_nrrd_output_cache_keys_include_placement(tmp_path):
    from tools import profile_nrrd_output as profiler
    payload = gzip.compress(b'\x01\x01', mtime=0)
    paths = []
    for index, offset in enumerate(('0 0 0', '3 1 2', '3 1 2')):
        path = tmp_path / f'layer-{index}.seg.nrrd'
        path.write_bytes((f'NRRD0005\ntype: uint8\ndimension: 3\nsizes: 2 1 1\nencoding: gzip\n'
                          f'content: copy {index}\n'
                          f'Segmentation_ReferenceImageExtentOffset:={offset}\n'
                          f'Segmentation_ReferenceImageExtent:=0 4 0 2 0 3\n\n').encode() + payload)
        paths.append(path)
    decoded, encoded, count = profiler._verify_nrrd_files(paths, {})
    assert len(set(encoded.values())) == 1 and count == 2
    assert decoded[paths[0].name] != decoded[paths[1].name] == decoded[paths[2].name]
    expected = np.zeros((4, 3, 5), np.uint8)
    expected[2, 1, 3:5] = 1
    assert decoded[paths[1].name] == hashlib.sha256(expected.tobytes()).hexdigest()


def _fake_deflate():
    fake = ModuleType('deflate')
    fake.gzip_compress = lambda payload, level: gzip.compress(bytes(payload),
        compresslevel=min(9, max(1, int(level))), mtime=0)
    return fake


@pytest.mark.parametrize('layout', LAYOUTS)
def test_profile_nrrd_output_verifies_sink_layers_in_each_layout(tmp_path, monkeypatch, layout):
    from tools import profile_nrrd_output as profiler
    volume = _box((2, 9, 5, 21, 7, 26), seed=11)
    store = tmp_path / 'fixture.cvol'
    with contextlib.redirect_stdout(io.StringIO()):
        stats = write_raw_bbox_mask_store(volume, store, format_name=CVOL_FORMAT, desc='test', workers=1)
    receipt = tmp_path / 'receipt.json'
    receipt.write_text(json.dumps(dict(shape_tyx=list(SHAPE), segment_extent_ijk=list(stats['segment_extent_ijk']),
                                       decoded_sha256=hashlib.sha256(volume.tobytes()).hexdigest())))
    fake = _fake_deflate()
    monkeypatch.setitem(sys.modules, 'deflate', fake)
    with mock.patch.dict('os.environ', {
            'YOLO_TTA_NRRD_MEMBER_CODEC': 'libdeflate', 'YOLO_TTA_NRRD_GZIP_WORKERS': '2',
            'YOLO_TTA_NRRD_FILL_WORKERS': '2', 'YOLO_TTA_NRRD_GPU_MIRROR_TEE': '0'}):
        load = profiler.ExistingCvolFixture(store, receipt)
        spec = outputs.resolve_low_quality_downbin_specs('0.5', True, SHAPE)[0][0]
        mirror_expected = profiler._expected_mirror_sha256(load.ref, SHAPE, spec.output_shape_t_y_x)
        result = profiler._run_mode(mode='off', root=tmp_path, evidence_root=tmp_path, load=load, jobs=2,
                                    sink_workers=1, deflate_module=fake, mirror_spec=spec,
                                    expected_mirror=mirror_expected, layer_layout=layout)
        assert profiler.plan(output_dir=tmp_path, deps_dir=tmp_path, shape=SHAPE, jobs=1, sink_workers=1,
                             gzip_workers=1, fill_workers=1)['nrrd_layer_layout'] == 'cropped'
    assert result['all_exact'] and result['nrrd_layer_layout'] == layout
    assert set(result['output_sha256'].values()) == {load.expected}
    assert set(result['mirror_sha256'].values()) == {mirror_expected}
    stored = grid.inspect_reference_raster(next((tmp_path / 'nrrd-off').glob('*.seg.nrrd')))
    assert stored.cropped == (layout == 'cropped')
    assert stored.reference_shape == SHAPE


def test_qualify_spherical_records_match_across_layouts(runs):
    from tools import qualify_spherical_optimization as spherical
    full_dir, cropped_dir = runs['full'][0], runs['cropped'][0]
    records = {layout: spherical.decoded_layers(output) for layout, (output, _v, _m) in runs.items()}
    assert records['full'] == records['cropped']
    assert len(records['full']) == 7
    record = records['full']['case_corner.seg.nrrd']
    assert record['shape'] == list(SHAPE) and record['spatial_header']['sizes'] == '31 29 11'
    assert record['spatial_header']['space origin'] == '(0,0,0)'
    # Force the bounded difference audit through read_mask on both layouts.
    candidate = {name: dict(item, sha256='changed') for name, item in records['cropped'].items()}
    comparison = spherical.compare_masks(full_dir, cropped_dir, records['full'], candidate)
    assert comparison['all_equal'] and comparison['changed_voxels_across_layers'] == 0
    for name in records['full']:
        np.testing.assert_array_equal(spherical.read_mask(cropped_dir / 'nrrd' / name),
                                      spherical.read_mask(full_dir / 'nrrd' / name))


def test_nrrd_output_fixture_expands_cropped_layers(runs, tmp_path, monkeypatch):
    from tools import nrrd_output_fixture as fixture
    monkeypatch.setattr(fixture, 'SCRATCH', tmp_path)
    keys = ('shape_tyx', 'logical_bytes', 'segment_extent_ijk', 'source_segment_extent_ijk',
            'decoded_sha256', 'cvol_decoded_sha256')
    for name in ('Global_final_output', 'corner', 'tile', 'detector'):
        receipts = {}
        for layout, (output, _volumes, _manifest) in runs.items():
            with contextlib.redirect_stdout(io.StringIO()):
                receipts[layout] = fixture.convert(output / 'nrrd' / f'case_{name}.seg.nrrd',
                                                   tmp_path / f'{layout}-{name}')
        assert {key: receipts['full'][key] for key in keys} == {key: receipts['cropped'][key] for key in keys}
        reference = runs['full'][1]['global_final' if name == 'Global_final_output' else name]
        assert receipts['cropped']['decoded_sha256'] == hashlib.sha256(reference.tobytes()).hexdigest()
        assert receipts['full']['source_stored_offset_tyx'] == [0, 0, 0]
        assert np.prod(receipts['cropped']['source_stored_shape_tyx']) < np.prod(SHAPE)


def _slicer_geometry(shape_tyx):
    t, height, width = shape_tyx
    return '1;0;0;0;0;1;0;0;0;0;1;0;0;0;0;1;' + f'0;{width - 1};0;{height - 1};0;{t - 1};'


def _write_slicer_save(path, stored, offset_xyz, *, geometry=None, oversampling='1'):
    """Write a seg.nrrd shaped like Slicer's own save: reference offset only, no XTA extent."""
    origin = ','.join(str(value) for value in offset_xyz)
    parameters = '' if geometry is None else (
        f'Segmentation_ConversionParameters:=Collapse labelmaps|1|Merge&Oversampling factor|{oversampling}|'
        f'Factor&Reference image geometry|{geometry}|Image geometry&\n')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((
        'NRRD0004\ntype: unsigned char\ndimension: 3\nspace: left-posterior-superior\n'
        f'sizes: {" ".join(str(value) for value in reversed(stored.shape))}\n'
        'space directions: (1,0,0) (0,1,0) (0,0,1)\nkinds: domain domain domain\nencoding: gzip\n'
        f'space origin: ({origin})\nSegment0_ID:=Segment_1\nSegment0_LabelValue:=1\nSegment0_Layer:=0\n'
        f'{parameters}Segmentation_MasterRepresentation:=Binary labelmap\n'
        f'Segmentation_ReferenceImageExtentOffset:={" ".join(str(value) for value in offset_xyz)}\n\n'
    ).encode() + gzip.compress(np.ascontiguousarray(stored).tobytes(), mtime=0))
    return path


def test_slicer_offset_only_saves_resolve_their_reference_grid(tmp_path):
    from tools import audit_pta_native_boundary as boundary
    from tools import audit_pta_native_downbin as downbin
    from tools import benchmark_topology_runs as topology_benchmark
    from tools import qualify_spherical_optimization as spherical
    reference = np.zeros((3, 4, 5), np.uint8)
    reference[1, 2, 3] = reference[2, 3, 1] = reference[2, 3, 3] = reference[2, 2, 2] = 1
    geometry = _slicer_geometry(reference.shape)
    saves = {
        # Slicer saved the whole reference grid at offset zero.
        'whole': _write_slicer_save(tmp_path / 'whole' / 'nrrd' / 'mask.seg.nrrd', reference, (0, 0, 0),
                                    geometry=geometry),
        # No saved geometry: a box at the origin is its own grid (HEAD tools read it so).
        'no_geometry': _write_slicer_save(tmp_path / 'no_geometry' / 'nrrd' / 'mask.seg.nrrd', reference,
                                          (0, 0, 0)),
        # Slicer cropped the save to the segment box at x=1, y=2, t=1.
        'cropped': _write_slicer_save(tmp_path / 'cropped' / 'nrrd' / 'mask.seg.nrrd', reference[1:3, 2:4, 1:4],
                                      (1, 2, 1), geometry=geometry),
    }
    digest = hashlib.sha256(reference.tobytes()).hexdigest()
    records = {}
    for label, path in saves.items():
        raster = grid.inspect_reference_raster(path)
        assert raster.reference_shape == reference.shape
        assert raster.cropped == (label == 'cropped')
        assert raster.reference_spatial_fields() == {'sizes': '5 4 3', 'space origin': '(0,0,0)'}
        np.testing.assert_array_equal(grid.read_reference_volume(path), reference)
        assert grid.reference_payload_sha256(path) == grid.reference_payload_sha256(path, chunk_bytes=1) == digest
        for t in range(reference.shape[0]):
            mask, shape, plane_digest = boundary.plane(path, t)
            np.testing.assert_array_equal(mask, reference[t] > 0)
            assert shape == reference.shape
            assert plane_digest == hashlib.sha256(reference[t].tobytes()).hexdigest()
        planes, shape = downbin.read_planes(path, 1, 3)
        np.testing.assert_array_equal(planes, reference[1:3])
        np.testing.assert_array_equal(spherical.read_mask(path), reference != 0)
        records[label] = spherical.decoded_layers(path.parent.parent)
        pairs = topology_benchmark.real_slice_pairs(path)
        assert [name for name, _a, _b in pairs] == ['real_z2']
        if label != 'whole':
            expected = topology_benchmark.real_slice_pairs(saves['whole'])
            for (name, a, b), (expected_name, expected_a, expected_b) in zip(pairs, expected, strict=True):
                assert name == expected_name
                np.testing.assert_array_equal(a, expected_a)
                np.testing.assert_array_equal(b, expected_b)
    assert records['whole'] == records['no_geometry'] == records['cropped']
    box = reference[1:3, 2:4, 1:4]
    with pytest.raises(ValueError, match='needs its reference extent'):
        grid.inspect_reference_raster(_write_slicer_save(tmp_path / 'unplaced.seg.nrrd', box, (1, 2, 1)))
    with pytest.raises(ValueError, match='Oversampled'):
        grid.inspect_reference_raster(_write_slicer_save(tmp_path / 'oversampled.seg.nrrd', box, (1, 2, 1),
                                                         geometry=geometry, oversampling='2'))
    with pytest.raises(ValueError, match='Invalid Slicer reference image geometry'):
        grid.inspect_reference_raster(_write_slicer_save(tmp_path / 'malformed.seg.nrrd', box, (1, 2, 1),
                                                         geometry=geometry.rpartition(';0;')[0] + ';'))
    with pytest.raises(ValueError, match='outside reference'):
        grid.inspect_reference_raster(_write_slicer_save(tmp_path / 'outside.seg.nrrd', box, (3, 2, 1),
                                                         geometry=geometry))
    with pytest.raises(ValueError, match='differs'):
        grid.inspect_reference_raster(saves['cropped'], reference_shape_tyx=(3, 4, 6))


def test_benchmark_topology_real_pairs_match_across_layouts(runs):
    from tools import benchmark_topology_runs as topology_benchmark
    checked = 0
    for full_manifest, cropped_manifest in zip(_manifests(runs['full'][0]), _manifests(runs['cropped'][0])):
        for name in ('case_Global_final_output.seg.nrrd', 'case_detector.seg.nrrd', 'case_tile.seg.nrrd'):
            reference = grid.read_reference_volume(full_manifest.parent / name)
            active = set(np.flatnonzero(reference.any(axis=(1, 2))).tolist())
            expected = topology_benchmark.real_slice_pairs(full_manifest.parent / name)
            actual = topology_benchmark.real_slice_pairs(cropped_manifest.parent / name)
            assert expected and len(actual) == len(expected)
            for (label, a, b), (expected_label, expected_a, expected_b) in zip(actual, expected):
                assert label == expected_label and int(label.removeprefix('real_z')) in active
                assert a.dtype == expected_a.dtype and b.dtype == expected_b.dtype
                np.testing.assert_array_equal(a, expected_a)
                np.testing.assert_array_equal(b, expected_b)
                checked += 1
    assert checked >= 6


def test_audit_sam_bridge_overlap_native_audit_matches_on_both_layouts(runs, tmp_path):
    from XTA.confidence_evidence import write_confidence_evidence
    from tools import audit_sam_bridge_overlap as audit
    volumes = runs['full'][1]
    # Native-grid evidence equal to the output grid keeps the restoration an identity.
    native = dict(detector=np.roll(volumes['detector'], 1, axis=2),
                  forward=np.roll(volumes['forward'], -1, axis=1),
                  backward=np.roll(volumes['backward'], 1, axis=0))
    reports = {}
    for layout, (output, _volumes, manifest) in runs.items():
        document = json.loads(manifest.read_text())
        for row in document['layers']:
            if row.get('interpolation_backend') == 'sam':
                row['native_transform'] = dict(view_family='orthogonal', view_name='transverse')
        manifest.write_text(json.dumps(document, indent=2))
        evidence = output / 'reconciliation_evidence'
        write_confidence_evidence(evidence / 'detector', SHAPE,
                                  lambda z: native['detector'][z] * np.uint8(200),
                                  layer_key='detector', model_name='model')
        (evidence / 'manifest.json').write_text(json.dumps(dict(
            schema='xta.confidence_evidence/1',
            layers=[dict(directory='detector', layer_key='detector', model_name='model')])))
        for direction in ('forward', 'backward'):
            with contextlib.redirect_stdout(io.StringIO()):
                write_raw_bbox_mask_store(native[direction],
                                          output / 'nrrd' / 'evidence' / f'sam_bridge_pass01_{direction}.cvol',
                                          format_name=CVOL_FORMAT, desc='test', workers=1)
        destination = tmp_path / f'audit-native-{layout}.json'
        argv = ['audit_sam_bridge_overlap.py', str(output), str(destination), '--audit-native']
        with mock.patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
            audit.main()
        report = json.loads(destination.read_text())
        report.pop('seconds')
        reports[layout] = report
    assert reports['full'] == reports['cropped']
    detector, union = volumes['detector'] != 0, (volumes['forward'] | volumes['backward']) != 0
    native_detector, native_union = native['detector'] != 0, (native['forward'] | native['backward']) != 0
    assert reports['full']['native_audit'] == dict(
        shape_tyx=list(SHAPE), confidence_support_voxels=int(native_detector.sum()),
        bridge_union_voxels=int(native_union.sum()), overlap_voxels=int((native_detector & native_union).sum()),
        source_restored_original_voxels=int(native_detector.sum()),
        source_restored_overlap_voxels=int((native_detector & union).sum()),
        source_detector_original_intersection_voxels=int((native_detector & detector).sum()))
    assert reports['full']['native_audit']['source_restored_overlap_voxels'] > 0
    assert 0 < reports['full']['native_audit']['source_detector_original_intersection_voxels'] < int(detector.sum())
    assert reports['full']['totals']['overlap_union_voxels'] == int((detector & union).sum())
