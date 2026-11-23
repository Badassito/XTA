"""Cropped per-layer NRRD storage.

Sink-published layers store only their segment bounding box (Slicer
``Segmentation_ReferenceImageExtentOffset`` convention) yet decode to exactly the
full-grid writer's reference content; ``NrrdLayerSink(layer_layout='full')`` writes
the historical raster byte for byte.
"""
from __future__ import annotations

import contextlib
import gzip
import io
import json
from pathlib import Path
import sys
from unittest import mock

import numpy as np
import pytest

from XTA import nrrd_compression, outputs
from XTA.interpolation import (
    CVOL_FORMAT,
    INTERNAL_PACKED_CVOL_FORMAT,
    NrrdLayerRef,
    RawBBoxMaskStore,
    write_raw_bbox_mask_store,
)
from XTA.reconciliation_io import _read_header, read_layer_manifest
from XTA.nrrd_spans import stream_native_crop_spans

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import xta_native_filter as native_filter  # noqa: E402

SHAPE = (11, 29, 31)
LOW_QUALITY = (
    outputs.LowQualityDownbinSpec('0.5', '0p5', .5, (6, 15, 16)),
    outputs.LowQualityDownbinSpec('0.33', '0p33', .33, (4, 10, 10)),
)


@pytest.fixture
def zlib_codec(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_NRRD_GPU_MIRROR_TEE', '0')
    monkeypatch.setenv('YOLO_TTA_NRRD_LAYER_ZSHARD_MIN_SLICES', '2')
    spec = nrrd_compression._nrrd_member_codec_spec('zlib')
    with mock.patch.object(nrrd_compression, '_require_nrrd_member_codec', return_value=spec):
        yield


def _box_volume(shape, box, *, seed):
    rng = np.random.default_rng(seed)
    t0, t1, y0, y1, x0, x1 = box
    volume = np.zeros(shape, np.uint8)
    volume[t0:t1, y0:y1, x0:x1] = rng.random((t1 - t0, y1 - y0, x1 - x0)) < .5
    volume[t0, y0, x0] = volume[t1 - 1, y1 - 1, x1 - 1] = 1
    return volume


def _live(name, volume, **fields):
    return NrrdLayerRef(key=name, name=name, path=Path(name + '.live'), shape=volume.shape,
                        storage_format='live_u8', live_array=volume, view_name=name,
                        source=fields.pop('source', 'fullframe'), mask_kind=fields.pop('mask_kind', 'yolo'),
                        **fields)


def _store_ref(root, name, volume, fmt, **fields):
    path = Path(root) / f'{name}.cvol'
    with contextlib.redirect_stdout(io.StringIO()):
        stats = write_raw_bbox_mask_store(volume, path, format_name=fmt, desc='test', workers=1)
    return NrrdLayerRef(key=name, name=name, path=path, shape=volume.shape, storage_format=fmt,
                        segment_extent_ijk=tuple(stats['segment_extent_ijk']),
                        segment_extent_shape_tyx=volume.shape, segment_extent_source='raw_bbox_cvol_index',
                        view_name=name, source='tile', mask_kind='yolo', tile_config_id=name)


def _layers(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    live = _box_volume(SHAPE, (3, 9, 4, 20, 6, 25), seed=1)
    globally = _box_volume(SHAPE, (0, 11, 0, 29, 1, 31), seed=2)
    corner = np.zeros(SHAPE, np.uint8)
    corner[7, 28, 30] = 1
    cvol = _box_volume(SHAPE, (2, 10, 8, 27, 3, 14), seed=3)
    cvol[5] = 0
    small = _box_volume((6, 15, 17), (1, 5, 3, 11, 2, 13), seed=4)
    memmap_volume = _box_volume((5, 19, 20), (1, 4, 2, 13, 7, 16), seed=5)
    memmap_path = root / 'memmap.raw'
    memmap_volume.tofile(memmap_path)
    restored = _store_ref(root, 'restored', small, CVOL_FORMAT)
    return [
        ('live', _live('live', live)),
        ('global', _live('global', globally, source='global', mask_kind='union',
                         stage='final_output_after_all_postprocessing')),
        ('corner', _live('corner', corner)),
        ('empty', _live('empty', np.zeros(SHAPE, np.uint8))),
        ('cvol_raw', _store_ref(root, 'cvol_raw', cvol, CVOL_FORMAT)),
        ('cvol_packed', _store_ref(root, 'cvol_packed', cvol, INTERNAL_PACKED_CVOL_FORMAT)),
        ('restored', NrrdLayerRef(**{**restored.__dict__, 'source': 'fullframe', 'mask_kind': 'bridge',
                                     'pass_index': 1})),
        ('memmap', NrrdLayerRef(key='memmap', name='memmap', path=memmap_path, shape=memmap_volume.shape,
                                storage_format='raw_u8',
                                segment_extent_ijk=outputs.compute_segment_extent_zyx(memmap_volume),
                                segment_extent_shape_tyx=memmap_volume.shape,
                                segment_extent_source='raw_layer_materialization_scan',
                                view_name='memmap', source='fullframe', mask_kind='bridge', pass_index=2)),
    ]


def _publish(root, layout, monkeypatch, *, z_shards=1, layers=None):
    monkeypatch.setenv('YOLO_TTA_NRRD_LAYER_ZSHARDS', str(z_shards))
    root = Path(root)
    sink = outputs.NrrdLayerSink(nrrd_dir=root / 'nrrd', stem='case', output_shape_tyx=SHAPE, max_workers=2,
                                 low_quality_specs=list(LOW_QUALITY), low_quality_root=root / 'low_quality',
                                 run_id='run', layer_layout=layout)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for name, ref in (layers if layers is not None else _layers(root / 'sources')):
                sink.submit_layer(ref, name)
            sink.wait()
        sink.write_manifest()
    finally:
        sink.shutdown()
    decoded = {}
    manifests = [('native', root / 'nrrd' / 'case_nrrd_manifest.json')] + [
        (spec.token, root / 'low_quality' / spec.token / 'nrrd' / 'case_nrrd_manifest.json')
        for spec in LOW_QUALITY]
    for grid, path in manifests:
        with read_layer_manifest(path, workspace=root, include_all_layers=True) as collection:
            for layer in collection:
                decoded[(grid, layer.metadata['suffix'])] = (
                    layer.read_slab(0, collection.shape_tyx[0]), layer.metadata, layer.path,
                    layer.offset_tyx, layer.stored_shape_tyx, json.loads(path.read_text()))
    return decoded


@pytest.mark.parametrize('z_shards', [1, 3])
def test_cropped_layers_decode_identically_to_full_grid(tmp_path, zlib_codec, monkeypatch, z_shards):
    full = _publish(tmp_path / 'full', 'full', monkeypatch, z_shards=z_shards)
    cropped = _publish(tmp_path / 'cropped', 'cropped', monkeypatch, z_shards=z_shards)
    assert full.keys() == cropped.keys() and len(full) == 8 * 3
    smaller = 0
    for key, (full_array, full_row, full_path, *_rest) in full.items():
        array, row, path, offset, stored, manifest = cropped[key]
        np.testing.assert_array_equal(array, full_array, err_msg=str(key))
        assert row['stored_offset_tyx'] == list(offset) and row['stored_shape_tyx'] == list(stored)
        assert row['stored_layout'] == 'segment_bounding_box' and manifest['stored_layout'] == 'segment_bounding_box'
        assert row['output_shape_tyx'] == full_row['output_shape_tyx']
        assert row['backing_shape_tyx'] == full_row['backing_shape_tyx']
        assert row['empty_segment'] == full_row['empty_segment'] == (not full_array.any())
        if full_array.any():
            nz = np.argwhere(full_array)
            assert np.all(nz.min(0) >= offset) and np.all(nz.max(0) < np.add(offset, stored))
        smaller += int(np.prod(stored)) < int(np.prod(full_row['stored_shape_tyx']))
    assert smaller >= 18
    # Native layers with known tight extents store exactly their foreground box.
    for name in ('live', 'cvol_raw', 'cvol_packed'):
        array, row, *_ = cropped[('native', name)]
        nz = np.argwhere(array)
        assert row['stored_offset_tyx'] == nz.min(0).tolist()
        assert row['stored_shape_tyx'] == (nz.max(0) - nz.min(0) + 1).tolist()


def test_cropped_headers_follow_slicer_reference_offset_convention(tmp_path, zlib_codec, monkeypatch):
    cropped = _publish(tmp_path, 'cropped', monkeypatch)
    _, row, path, offset, stored, _ = cropped[('native', 'live')]
    header, _ = _read_header(path)
    t0, y0, x0 = offset
    assert header['sizes'].split() == [str(v) for v in reversed(stored)]
    assert header['space origin'] == f'({x0},{y0},{t0})'
    assert header['space directions'] == '(1,0,0) (0,1,0) (0,0,1)'
    assert header['Segmentation_ReferenceImageExtentOffset'] == f'{x0} {y0} {t0}'
    assert header['Segmentation_ReferenceImageExtent'] == f'0 {SHAPE[2]-1} 0 {SHAPE[1]-1} 0 {SHAPE[0]-1}'
    st, sy, sx = stored
    assert header['Segment0_Extent'] == f'0 {sx-1} 0 {sy-1} 0 {st-1}'
    assert f'stored_offset_tyx=({t0},{y0},{x0})' in header['content']
    assert f'reference_shape_tyx=({SHAPE[0]},{SHAPE[1]},{SHAPE[2]})' in header['content']
    # Slicer aborts on a nonempty 1x1x1 crop: a corner voxel widens toward its neighbour.
    _, row, path, offset, stored, _ = cropped[('native', 'corner')]
    header, _ = _read_header(path)
    assert stored == (1, 1, 2) and offset == (7, 28, 29)
    assert header['Segment0_Extent'] == '1 1 0 0 0 0'
    # An empty layer keeps its empty extent on one zero voxel at the grid origin.
    _, row, path, offset, stored, _ = cropped[('native', 'empty')]
    header, _ = _read_header(path)
    assert stored == (1, 1, 1) and offset == (0, 0, 0) and row['empty_segment'] is True
    assert header['sizes'] == '1 1 1' and header['Segment0_Extent'] == '0 -1 0 -1 0 -1'
    assert header['Segmentation_ReferenceImageExtentOffset'] == '0 0 0'


def test_full_layout_kill_switch_writes_historical_rasters(tmp_path, zlib_codec, monkeypatch):
    full = _publish(tmp_path, 'full', monkeypatch)
    for (grid, _suffix), (_array, row, path, offset, stored, manifest) in full.items():
        header, _ = _read_header(path)
        expected = SHAPE if grid == 'native' else next(s.output_shape_t_y_x for s in LOW_QUALITY if s.token == grid)
        assert stored == tuple(expected) and offset == (0, 0, 0)
        assert header['space origin'] == '(0,0,0)'
        assert not any(key.startswith('Segmentation_ReferenceImage') for key in header)
        assert 'stored_offset_tyx' not in row and 'stored_layout' not in row
        assert 'stored_layout' not in manifest and 'stored_offset' not in header['content']


def test_invalid_layout_fails_before_any_layer_is_queued(tmp_path):
    with pytest.raises(ValueError, match='Unsupported NRRD layer layout'):
        outputs.NrrdLayerSink(nrrd_dir=tmp_path / 'nrrd', stem='case', output_shape_tyx=SHAPE, max_workers=1,
                              layer_layout='bounding')
    sink = outputs.NrrdLayerSink(nrrd_dir=tmp_path / 'default', stem='case', output_shape_tyx=SHAPE,
                                 max_workers=1)
    try:
        assert sink.layer_layout == 'cropped'
    finally:
        sink.shutdown()


def test_understated_extent_republishes_full_raster_without_losing_foreground(
        tmp_path, zlib_codec, monkeypatch, capsys):
    volume = _box_volume(SHAPE, (2, 9, 3, 26, 4, 28), seed=8)
    # Correct t range (so streaming reads every foreground slice) but a too-small X/Y box.
    wrong = NrrdLayerRef(**{**_live('wrong', volume).__dict__,
                            'segment_extent_ijk': (10, 20, 8, 15, 2, 8),
                            'segment_extent_shape_tyx': SHAPE, 'segment_extent_source': 'test'})
    layers = [('wrong', wrong)]
    full = _publish(tmp_path / 'full', 'full', monkeypatch, layers=layers)
    capsys.readouterr()
    cropped = _publish(tmp_path / 'cropped', 'cropped', monkeypatch, layers=layers)
    for key, (full_array, *_rest) in full.items():
        array, row, path, offset, stored, _ = cropped[key]
        np.testing.assert_array_equal(array, full_array)
        assert row['stored_layout'] == 'full_reference' and row['stored_offset_tyx'] == [0, 0, 0]
        assert row['stored_shape_tyx'] == row['output_shape_tyx']
    np.testing.assert_array_equal(cropped[('native', 'wrong')][0], volume)


class _Recorder:
    minimum_input_bytes = 1

    def __init__(self):
        self.parts = []

    def write(self, data):
        self.parts.append(bytes(memoryview(data).cast('B')))
        return len(self.parts[-1])

    def write_canonical_zeros(self, count):
        self.parts.append(bytes(int(count)))
        return int(count)

    def write_zeros(self, count):
        return self.write_canonical_zeros(count)


def test_crop_adapter_matches_direct_crop_for_arbitrary_write_boundaries():
    rng = np.random.default_rng(17)
    for _ in range(300):
        shape = tuple(int(v) for v in rng.integers(1, (6, 9, 12)))
        lo = [int(rng.integers(0, n)) for n in shape]
        hi = [int(rng.integers(a, n)) + 1 for a, n in zip(lo, shape)]
        volume = np.zeros(shape, np.uint8)
        box = tuple(slice(a, b) for a, b in zip(lo, hi))
        volume[box] = rng.random(volume[box].shape) < .4
        z0 = int(rng.integers(0, shape[0]))
        z1 = int(rng.integers(z0, shape[0])) + 1
        stream = volume[z0:z1].reshape(-1)
        cuts = sorted(set(int(v) for v in rng.integers(0, stream.size + 1, int(rng.integers(0, 12)))))
        recorder = _Recorder()
        writer = outputs._CroppedNrrdPayloadWriter(
            recorder, shape, tuple(lo), tuple(b - a for a, b in zip(lo, hi)),
            logical_start=z0 * shape[1] * shape[2], logical_stop=z1 * shape[1] * shape[2])
        for a, b in zip([0] + cuts, cuts + [stream.size]):
            piece = stream[a:b]
            if piece.size and not piece.any() and rng.random() < .7:
                getattr(writer, ('write_zeros', 'write_aligned_zeros', 'write_canonical_zeros')[
                    int(rng.integers(0, 3))])(piece.size)
            elif rng.random() < .5:
                writer.write(memoryview(np.ascontiguousarray(piece)))
            else:
                writer.write_owned_known_nonzero(piece.copy())
        writer.finish()
        t_lo, t_hi = max(z0, lo[0]), min(z1, hi[0])
        expected = (volume[t_lo:t_hi, lo[1]:hi[1], lo[2]:hi[2]].tobytes() if t_hi > t_lo else b'')
        assert b''.join(recorder.parts) == expected


def test_native_crop_span_batch_preserves_reference_observer_coordinates(tmp_path):
    shape = (9, 19, 23)
    volume = _box_volume(shape, (3, 7, 6, 11, 8, 14), seed=7)
    volume[4] = 0
    ref = _store_ref(tmp_path, 'cropped', volume, INTERNAL_PACKED_CVOL_FORMAT)
    observed = np.zeros_like(volume)

    def observe(z, y, x, crop):
        observed[z, y:y + crop.shape[0], x:x + crop.shape[1]] = crop

    sink = io.BytesIO()
    codec = nrrd_compression._nrrd_member_codec_spec('zlib')
    store = RawBBoxMaskStore.open(ref.path, mmap_payload=True)
    try:
        with nrrd_compression._MemberParallelGzipPayloadWriter(sink, codec_spec=codec) as inner:
            writer = outputs._CroppedNrrdPayloadWriter(
                inner, shape, (2, 4, 5), (6, 9, 11), logical_start=19 * 23,
            )
            with mock.patch.object(writer, '_write_reference_bytes', side_effect=AssertionError):
                stream_native_crop_spans(store, writer, 1, 9, 46, observe, batch_bytes=92)
                assert writer.finish() == 6 * 9 * 11
            assert inner._writer_stats['owned_member_batches'] > 1
    finally:
        store.close()
    np.testing.assert_array_equal(observed, volume)
    assert gzip.decompress(sink.getvalue()) == volume[2:8, 4:13, 5:16].tobytes()


def test_sharded_layer_sync_failure_preserves_previous_output(tmp_path, zlib_codec, monkeypatch):
    ref = _live('sharded', _box_volume(SHAPE, (0, 11, 4, 9, 2, 14), seed=9))
    target = tmp_path / 'sharded.nrrd'
    target.write_bytes(b'previous output')
    monkeypatch.setattr(outputs, '_nrrd_zshard_capacity', lambda: 3)
    with mock.patch.object(outputs.os, 'fsync', side_effect=OSError('file sync failed')):
        with pytest.raises(OSError, match='file sync failed'):
            outputs.write_single_layer_nrrd_from_ref(ref, SHAPE, target, layout='full', z_shards=3)
    assert target.read_bytes() == b'previous output'
    assert list(tmp_path.iterdir()) == [target]


def test_crop_adapter_rejects_outside_foreground_and_incomplete_streams():
    shape = (4, 5, 6)
    volume = np.zeros(shape, np.uint8)
    volume[1:3, 1:4, 2:5] = 1
    for position in ((0, 0, 0), (1, 0, 3), (2, 2, 1), (2, 2, 5), (3, 4, 5), (1, 4, 2)):
        bad = volume.copy()
        bad[position] = 1
        writer = outputs._CroppedNrrdPayloadWriter(_Recorder(), shape, (1, 1, 2), (2, 3, 3))
        with pytest.raises(outputs.NrrdCropExtentViolation):
            writer.write(bad.reshape(-1))
    writer = outputs._CroppedNrrdPayloadWriter(_Recorder(), shape, (1, 1, 2), (2, 3, 3))
    writer.write(volume[:2].reshape(-1))
    with pytest.raises(RuntimeError, match='incomplete'):
        writer.finish()


def test_native_filter_reads_cropped_layers_on_their_reference_grid(tmp_path, zlib_codec, monkeypatch):
    full = _publish(tmp_path / 'full', 'full', monkeypatch)
    cropped = _publish(tmp_path / 'cropped', 'cropped', monkeypatch)
    for key in full:
        reference_shape = full[key][0].shape
        expected = native_filter.Nrrd(full[key][2], reference_shape)
        actual = native_filter.Nrrd(cropped[key][2], reference_shape)
        assert actual.shape == expected.shape == reference_shape
        np.testing.assert_array_equal(actual.ijk_to_ras, expected.ijk_to_ras)
        np.testing.assert_array_equal(actual.array(), expected.array())
        for (za, a), (zb, b) in zip(actual.slabs(3), expected.slabs(3)):
            assert za == zb
            np.testing.assert_array_equal(a, b)
        # The reference extent alone recovers the grid without a manifest shape.
        assert native_filter.Nrrd(cropped[key][2]).shape == reference_shape
        with pytest.raises(ValueError, match='shape mismatch'):
            native_filter.Nrrd(cropped[key][2], tuple(n + 1 for n in reference_shape))


def _cropped_native_writer(path, array, name):
    volume = np.ascontiguousarray(array, dtype=np.uint8)
    ref = NrrdLayerRef(key=name, name=name, path=Path(path).with_suffix('.live'), shape=volume.shape,
                       storage_format='live_u8', live_array=volume)
    with contextlib.redirect_stdout(io.StringIO()):
        outputs.write_single_layer_nrrd_from_ref(ref, volume.shape, Path(path), segment_name=name,
                                                 layout='cropped')


def _native_filter_fixtures():
    import importlib.util
    path = Path(__file__).with_name('test_native_filter.py')
    spec = importlib.util.spec_from_file_location('native_filter_fixtures_for_cropped_layers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_transfer_from_cropped_layers_matches_full_grid_layers(tmp_path, zlib_codec):
    fixtures = _native_filter_fixtures()
    results = {}
    for layout in ('full', 'cropped'):
        root = tmp_path / layout
        patch = (contextlib.nullcontext() if layout == 'full'
                 else mock.patch.object(fixtures, 'write', _cropped_native_writer))
        with patch:
            args, good, _bad = fixtures.complex_fixture(root)
        if layout == 'cropped':
            header, _ = _read_header(Path(args.full_manifest).parent / 'volume_component.seg.nrrd')
            assert 'Segmentation_ReferenceImageExtentOffset' in header
        with contextlib.redirect_stdout(io.StringIO()):
            assert native_filter.run(args) == 0
        output = Path(args.output)
        report = native_filter.load_json(output / 'report.json')
        results[layout] = (native_filter.Nrrd(output / 'volume.nrrd').array(), report['status'],
                           report['final_global_voxels'], (output / 'components.tsv').read_text())
        np.testing.assert_array_equal(results[layout][0], good)
    assert results['full'][1:] == results['cropped'][1:]
    np.testing.assert_array_equal(results['full'][0], results['cropped'][0])


def test_reconciliation_rejects_manifest_offset_that_disagrees_with_header(tmp_path, zlib_codec, monkeypatch):
    volume = _box_volume(SHAPE, (3, 9, 4, 20, 6, 25), seed=21)
    _publish(tmp_path, 'cropped', monkeypatch, layers=[('live', _live('live', volume))])
    manifest_path = tmp_path / 'nrrd' / 'case_nrrd_manifest.json'
    document = json.loads(manifest_path.read_text())
    document['layers'][0]['stored_offset_tyx'][2] += 1
    manifest_path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='stored offset disagrees'):
        read_layer_manifest(manifest_path, workspace=tmp_path, include_all_layers=True)
