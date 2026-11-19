"""Readers of TTA layer NRRDs give identical results for cropped and full-grid layers.

Each layer is published twice by the real NrrdLayerSink (``layer_layout`` ``full``
and ``cropped``) and read through the PTA mask loader and the Slicer-workflow
cut tools. Cropped files store only their segment box at
``Segmentation_ReferenceImageExtentOffset`` and must decode to the same reference grid.
"""
from __future__ import annotations

import contextlib
import gzip
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from unittest import mock

import numpy as np
import pytest

from XTA import nrrd_compression, outputs, pta_inputs
from XTA.interpolation import NrrdLayerRef
from XTA.pta_workers import WarningLog
from tools import xta_cut_axial_pair, xta_cut_nrrd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import xta_native_filter as native_filter  # noqa: E402

SHAPE = (11, 29, 31)
LOW_QUALITY = outputs.LowQualityDownbinSpec('0.5', '0p5', .5, (6, 15, 16))
GRIDS = {'native': SHAPE, 'preview': LOW_QUALITY.output_shape_t_y_x}


@pytest.fixture
def zlib_codec(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_NRRD_GPU_MIRROR_TEE', '0')
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


def _single(shape, position):
    volume = np.zeros(shape, np.uint8)
    volume[position] = 1
    return volume


def _volumes(shape=SHAPE):
    t, h, w = shape
    return {
        'interior': _box_volume(shape, (3, 9, 4, 20, 6, 25), seed=1),
        'low_edges': _box_volume(shape, (0, 4, 0, 7, 0, 9), seed=2),
        'high_edges': _box_volume(shape, (t - 4, t, h - 9, h, w - 9, w), seed=3),
        'whole': _box_volume(shape, (0, t, 0, h, 0, w), seed=4),
        'first_voxel': _single(shape, (0, 0, 0)),
        'last_voxel': _single(shape, (t - 1, h - 1, w - 1)),
        'empty': np.zeros(shape, np.uint8),
    }


def _publish(root, layout, monkeypatch, volumes, shape=SHAPE, low_quality=(LOW_QUALITY,)):
    """Publish every volume with the stock sink; return {(grid, name): (path, manifest row)}."""
    root = Path(root)
    sink = outputs.NrrdLayerSink(nrrd_dir=root / 'nrrd', stem='case', output_shape_tyx=shape, max_workers=2,
                                 low_quality_specs=list(low_quality), low_quality_root=root / 'low_quality',
                                 run_id='run', layer_layout=layout)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for name, volume in volumes.items():
                ref = NrrdLayerRef(key=name, name=name, path=root / f'{name}.live', shape=volume.shape,
                                   storage_format='live_u8', live_array=volume, view_name=name,
                                   source='fullframe', mask_kind='yolo')
                sink.submit_layer(ref, name)
            sink.wait()
        sink.write_manifest()
    finally:
        sink.shutdown()
    manifests = [('native', root / 'nrrd' / 'case_nrrd_manifest.json')] + [
        ('preview', root / 'low_quality' / spec.token / 'nrrd' / 'case_nrrd_manifest.json')
        for spec in low_quality]
    published = {}
    for grid, path in manifests:
        for row in json.loads(path.read_text())['layers']:
            published[(grid, row['suffix'])] = (path.parent / row['filename'], row)
    return published


def _header_fields(path):
    header = Path(path).read_bytes().split(b'\n\n', 1)[0].decode()
    fields = {}
    for line in header.splitlines()[1:]:
        if line.startswith('#'):
            continue
        separator = ':=' if ':=' in line else ': '
        key, value = line.split(separator, 1)
        fields[key] = value
    return fields


def _rewrite_header(source, destination, edit):
    header, payload = Path(source).read_bytes().split(b'\n\n', 1)
    lines = edit(header.decode().splitlines())
    Path(destination).write_bytes(('\n'.join(lines) + '\n\n').encode() + payload)
    return Path(destination)


@pytest.fixture
def published(tmp_path, zlib_codec, monkeypatch):
    volumes = _volumes()
    full = _publish(tmp_path / 'full', 'full', monkeypatch, volumes)
    cropped = _publish(tmp_path / 'cropped', 'cropped', monkeypatch, volumes)
    assert full.keys() == cropped.keys() and len(full) == 2 * len(volumes)
    for key, (path, row) in cropped.items():
        # The fixture really exercises cropped storage, including edge boxes and widened voxels.
        assert row['stored_layout'] == 'segment_bounding_box', key
        if key[1] != 'whole':
            assert row['stored_shape_tyx'] != full[key][1]['stored_shape_tyx'], key
    return volumes, full, cropped


def _load(path, expected, **options):
    warnings, summaries = WarningLog(), []
    with contextlib.redirect_stdout(io.StringIO()):
        if 'source_plane_indices' in options:
            mask, matched = pta_inputs.load_indexed_nrrd_mask_for_volume(
                path, expected, warnings=warnings, plane_summary_out=summaries, **options)
        else:
            mask = pta_inputs.load_nrrd_mask_for_volume(
                path, expected, warnings=warnings, plane_summary_out=summaries, **options)
            matched = None
    summary = tuple(np.asarray(value).copy() for value in summaries[0]) if summaries else None
    return mask, matched, summary, dict(warnings.counts), {k: list(v) for k, v in warnings.examples.items()}


def _assert_same_load(actual, expected, label):
    np.testing.assert_array_equal(actual[0], expected[0], err_msg=label)
    assert actual[1] == expected[1], label
    if actual[2] is not None and expected[2] is not None:
        for a, b in zip(actual[2], expected[2]):
            np.testing.assert_array_equal(a, b, err_msg=label)
    assert actual[3:] == expected[3:], label


def _pta_cases(grid_shape):
    t, h, w = grid_shape
    indices = [-1] + [min(t - 1, max(0, 2 * z - 1)) for z in range(1, t + 1)] + [t + 3]
    return {
        'exact': (grid_shape, {}),
        'force_larger': ((t + 2, h + 3, w + 1), {'force': True}),
        'force_smaller': ((t - 1, h - 2, w - 3), {'force': True}),
        'indexed': ((len(indices), h, w), {'source_plane_indices': indices}),
        'indexed_force': ((len(indices), h + 2, w - 1), {'source_plane_indices': indices, 'force': True}),
    }


def test_pta_mask_loader_reads_cropped_layers_like_full_grid_layers(published, monkeypatch):
    volumes, full, cropped = published
    box_reads = []
    original = pta_inputs._read_cropped_nrrd_planes
    monkeypatch.setattr(pta_inputs, '_read_cropped_nrrd_planes',
                        lambda *args, **kwargs: box_reads.append(args[6]) or original(*args, **kwargs))
    for (grid, name), (full_path, _row) in full.items():
        cropped_path = cropped[(grid, name)][0]
        if grid == 'native':
            assert np.array_equal(_load(full_path, SHAPE)[0], volumes[name])
        for case, (expected, options) in _pta_cases(GRIDS[grid]).items():
            label = f'{grid}/{name}/{case}'
            reference = _load(full_path, expected, **options)
            _assert_same_load(_load(cropped_path, expected, **options), reference, label)
            # The general pynrrd path places the box identically.
            with monkeypatch.context() as patch:
                patch.setattr(pta_inputs, '_read_canonical_uint8_nrrd', lambda *a, **k: None)
                general = _load(cropped_path, expected, **options)
            np.testing.assert_array_equal(general[0], reference[0], err_msg=label)
            assert general[1] == reference[1], label
    # Every non-whole cropped layer used the box reader, both direct and resampled.
    assert box_reads.count(True) == 2 * (len(volumes) - 1)
    assert box_reads.count(False) == 2 * (len(volumes) - 1) * 4


def _slicer_geometry(reference_xyz):
    matrix = '-1;0;0;0;0;-1;0;0;0;0;1;0;0;0;0;1'
    extent = ';'.join(f'0;{size - 1}' for size in reference_xyz)
    return ('Segmentation_ConversionParameters:=Smoothing factor|0.5|help&Reference image geometry|'
            f'{matrix};{extent};|Image geometry description string.&Oversampling factor|1|help&')


def test_pta_mask_loader_accepts_slicer_saved_boxes(published, tmp_path):
    """Slicer saves keep only the offset; its reference geometry, when set, names the grid."""
    _volumes_, full, cropped = published
    for (grid, name), (cropped_path, _row) in cropped.items():
        shape = GRIDS[grid]
        reference_xyz = tuple(reversed(shape))
        # Same file names keep the loader's warning text comparable.
        for variant in ('offset', 'geometry'):
            (tmp_path / variant / grid).mkdir(parents=True, exist_ok=True)
        offset_only = _rewrite_header(cropped_path, tmp_path / 'offset' / grid / cropped_path.name, lambda lines: [
            line for line in lines if not line.startswith('Segmentation_ReferenceImageExtent:=')])
        with_geometry = _rewrite_header(offset_only, tmp_path / 'geometry' / grid / cropped_path.name,
                                        lambda lines: lines + [_slicer_geometry(reference_xyz)])
        for case, (expected, options) in _pta_cases(shape).items():
            reference = _load(full[(grid, name)][0], expected, **options)
            label = f'{grid}/{name}/{case}'
            _assert_same_load(_load(with_geometry, expected, **options), reference, label)
        # Without a saved geometry the box is pasted into the expected image grid.
        _assert_same_load(_load(offset_only, shape), _load(full[(grid, name)][0], shape), f'{grid}/{name}')
        expected, options = _pta_cases(shape)['indexed']
        if _header_fields(offset_only)['Segmentation_ReferenceImageExtentOffset'] != '0 0 0':
            with pytest.raises(ValueError, match='no source depth'):
                _load(offset_only, expected, **options)
        else:
            # A zero offset leaves source-plane numbering alone: it reads as a complete raster.
            (tmp_path / 'no_offset' / grid).mkdir(parents=True, exist_ok=True)
            complete = _rewrite_header(offset_only, tmp_path / 'no_offset' / grid / cropped_path.name, lambda lines: [
                line for line in lines if not line.startswith('Segmentation_ReferenceImageExtentOffset:=')])
            assert _outcome(offset_only, expected, **options) == _outcome(complete, expected, **options)
            if name == 'whole':
                _assert_same_load(_load(offset_only, expected, **options),
                                  _load(full[(grid, name)][0], expected, **options), f'{grid}/{name}/indexed')


def _outcome(path, expected, **options):
    """Comparable load result, or the error with the file path removed."""
    try:
        mask, matched, summary, counts, examples = _load(path, expected, **options)
    except ValueError as error:
        return 'error', str(error).replace(str(path), '<path>')
    return 'ok', mask.tobytes(), mask.shape, matched, tuple(a.tobytes() for a in summary or ()), counts, examples


def _write_tyx_nrrd(path, volume, extra_lines=(), *, directions=True, encoding='gzip'):
    t, h, w = volume.shape
    lines = ['NRRD0004', 'type: unsigned char', 'dimension: 3', f'sizes: {w} {h} {t}', f'encoding: {encoding}']
    if directions:
        lines += ['space: left-posterior-superior', 'space directions: (1,0,0) (0,1,0) (0,0,1)',
                  'kinds: domain domain domain', 'space origin: (0,0,0)']
    payload = volume.tobytes()
    Path(path).write_bytes(('\n'.join([*lines, *extra_lines]) + '\n\n').encode()
                           + (gzip.compress(payload) if encoding == 'gzip' else payload))
    return Path(path)


_ZERO_OFFSET = 'Segmentation_ReferenceImageExtentOffset:=0 0 0'
_EMPTY_SLICER_GEOMETRY = ('Segmentation_ConversionParameters:=Reference image geometry||desc&'
                          'Oversampling factor|1|desc&')


@pytest.mark.parametrize('general', [False, True], ids=['fast', 'pynrrd'])
def test_pta_zero_offset_without_reference_grid_reads_complete_rasters_as_before(tmp_path, monkeypatch, general):
    """A zero offset with no extent or geometry places a box only when it can crop the image grid."""
    if general:
        monkeypatch.setattr(pta_inputs, '_read_canonical_uint8_nrrd', lambda *a, **k: None)
    rng = np.random.default_rng(5)
    stored = (rng.random((12, 20, 22)) < .3).astype(np.uint8)
    for directions in (True, False):
        folder = tmp_path / f'directions_{directions}'
        (folder / 'control').mkdir(parents=True)
        (folder / 'zero').mkdir()
        control = _write_tyx_nrrd(folder / 'control' / 'mask.seg.nrrd', stored, directions=directions)
        zero = _write_tyx_nrrd(folder / 'zero' / 'mask.seg.nrrd', stored, [_EMPTY_SLICER_GEOMETRY, _ZERO_OFFSET],
                               directions=directions)
        cases = [((12, 20, 22), {}), ((12, 20, 22), {'force': True}),
                 ((10, 18, 20), {'force': True}), ((13, 19, 22), {'force': True}),
                 ((6, 20, 22), {'source_plane_indices': [-1, 0, 3, 7, 11, 15]}),
                 ((6, 21, 20), {'source_plane_indices': [-1, 0, 3, 7, 11, 15], 'force': True})]
        if not directions:
            cases.append(((14, 22, 24), {'force': True}))
        for expected, options in cases:
            label = f'directions={directions} {expected} {options}'
            result = _outcome(zero, expected, **options)
            # Shape-only axis detection may itself be ambiguous under tolerance; it must agree.
            assert result[0] == 'ok' or not (directions or not options), (label, result)
            assert result == _outcome(control, expected, **options), label
    # With axis directions, a box that fits inside the image grid is a Slicer crop at the origin:
    # it is pasted, never stretched (Slicer's offset-only saves always crop to the segments).
    zero = tmp_path / 'directions_True' / 'zero' / 'mask.seg.nrrd'
    pasted = np.zeros((14, 22, 24), np.uint8)
    pasted[:12, :20, :22] = stored
    for options in ({}, {'force': True}):
        np.testing.assert_array_equal(_load(zero, (14, 22, 24), **options)[0], pasted)
    # A nonzero offset still needs axis directions and, for indexed planes, a declared depth.
    shifted = _write_tyx_nrrd(tmp_path / 'shifted.seg.nrrd', stored, ['Segmentation_ReferenceImageExtentOffset:=1 0 0'],
                              directions=False)
    with pytest.raises(ValueError, match='neither a reference extent nor axis directions'):
        _load(shifted, (12, 20, 23))
    shifted = _write_tyx_nrrd(tmp_path / 'shifted_depth.seg.nrrd', stored,
                              ['Segmentation_ReferenceImageExtentOffset:=0 0 1'])
    with pytest.raises(ValueError, match='no source depth'):
        _load(shifted, (6, 20, 22), source_plane_indices=[-1, 0, 3, 7, 11, 15])


# Header layout written by Slicer 5.12.4 when re-saving an XTA layer without a reference volume.
_SLICER_CONVERSION_PARAMETERS = (
    'Segmentation_ConversionParameters:=Decimation factor|0.0|Desired reduction in the total number of polygons. '
    'Range: 0.0 (no decimation) to 1.0 (as much simplification as possible).&Smoothing factor|0.5|Smoothing '
    'factor. Range: 0.0 (no smoothing) to 1.0 (strong smoothing).&Reference image geometry||Image geometry '
    'description string determining the geometry of the labelmap that is created in course of conversion.&'
    'Oversampling factor|1|Determines the oversampling of the reference image geometry.&Crop to reference image '
    'geometry|0|Crop the model to the extent of reference geometry.&')


def test_pta_mask_loader_reads_slicer_resave_header(tmp_path):
    shape = (5, 7, 9)
    truth = np.zeros(shape, np.uint8)
    truth[1:3, 2:5, 3:8] = 1
    path = tmp_path / 'slicer_resave.seg.nrrd'
    path.write_bytes((
        'NRRD0004\n# Complete NRRD file format specification at:\n# http://teem.sourceforge.net/nrrd/format.html\n'
        'type: unsigned char\ndimension: 3\nspace: left-posterior-superior\nsizes: 5 3 2\n'
        'space directions: (1,0,0) (0,1,0) (0,0,1)\nkinds: domain domain domain\nencoding: raw\n'
        'space origin: (3,2,1)\nSegment0_Extent:=0 4 0 2 0 1\nSegment0_ID:=Segment_1\nSegment0_LabelValue:=1\n'
        'Segment0_Layer:=0\nSegment0_Name:=a\nSegmentation_ContainedRepresentationNames:=Binary labelmap|\n'
        f'{_SLICER_CONVERSION_PARAMETERS}\nSegmentation_MasterRepresentation:=Binary labelmap\n'
        'Segmentation_ReferenceImageExtentOffset:=3 2 1\n\n').encode() + bytes([1]) * 30)
    np.testing.assert_array_equal(_load(path, shape)[0], truth)
    with mock.patch.object(pta_inputs, '_read_canonical_uint8_nrrd', lambda *a, **k: None):
        np.testing.assert_array_equal(_load(path, shape)[0], truth)


def test_pta_mask_loader_slicer_empty_and_invalid_boxes(published, tmp_path):
    _volumes_, _full, cropped = published
    t, h, w = SHAPE
    # Slicer 5.12 saves an empty segment as a 1x1x1 image with offset zero and RAS-like axes.
    empty = tmp_path / 'slicer_empty.seg.nrrd'
    empty.write_bytes((
        'NRRD0004\ntype: unsigned char\ndimension: 3\nspace: left-posterior-superior\nsizes: 1 1 1\n'
        'space directions: (-1,0,0) (0,-1,0) (0,0,1)\nkinds: domain domain domain\nencoding: raw\n'
        'space origin: (0,0,0)\nSegment0_Extent:=0 -1 0 -1 0 -1\nSegment0_LabelValue:=1\n'
        'Segmentation_ReferenceImageExtentOffset:=0 0 0\n\n').encode() + b'\0')
    mask, *_ = _load(empty, SHAPE)
    assert mask.shape == SHAPE and not mask.any()
    source = cropped[('native', 'high_edges')][0]
    offset_only = lambda lines: [line for line in lines if not line.startswith('Segmentation_ReferenceImageExtent:=')]
    shifted = _rewrite_header(source, tmp_path / 'shifted.seg.nrrd', lambda lines: [
        'Segmentation_ReferenceImageExtentOffset:=23 20 8' if line.startswith('Segmentation_ReferenceImageExtentOffset')
        else line for line in offset_only(lines)])
    with pytest.raises(ValueError, match='lies outside its reference grid'):
        _load(shifted, SHAPE)
    # A declared reference grid keeps the full-grid --force tolerance.
    with pytest.raises(ValueError, match='tolerance'):
        _load(cropped[('native', 'interior')][0], (t, h, w - 7), force=True)
    oversampled = _rewrite_header(source, tmp_path / 'oversampled.seg.nrrd', lambda lines: offset_only(lines) + [
        _slicer_geometry((w, h, t)).replace('Oversampling factor|1|', 'Oversampling factor|2|')])
    with pytest.raises(ValueError, match='Oversampled'):
        _load(oversampled, SHAPE)
    bad_extent = _rewrite_header(source, tmp_path / 'bad_extent.seg.nrrd', lambda lines: [
        'Segmentation_ReferenceImageExtent:=1 30 0 28 0 10' if line.startswith('Segmentation_ReferenceImageExtent:=')
        else line for line in lines])
    with pytest.raises(ValueError, match='start at voxel zero'):
        _load(bad_extent, SHAPE)


def _decoded(path):
    """Independent reference-grid decode of a cut output with the supported XTA reader."""
    layer = native_filter.Nrrd(path)
    return layer.array(), layer.shape


def _global_extent(path):
    fields = _header_fields(path)
    extent = [int(v) for v in fields['Segment0_Extent'].split()]
    if any(extent[2 * axis + 1] < extent[2 * axis] for axis in range(3)):
        return None
    offset = [int(v) for v in fields.get('Segmentation_ReferenceImageExtentOffset', '0 0 0').split()]
    return [value + offset[index // 2] for index, value in enumerate(extent)]


def _cut(path, destination, mode, reference_column):
    info = xta_cut_nrrd.probe_nrrd(path)
    first = max(0, min(info.width, reference_column - info.origin_xyz[0]))
    options = {} if info.reference_xyz is None else {'reference_first_removed': reference_column}
    audit = xta_cut_nrrd.write_nrrd(path, destination, info, first, mode, **options)
    return audit


def test_cut_nrrd_cropped_layers_match_full_grid_cuts(published, tmp_path):
    _volumes_, full, cropped = published
    placements = []
    for (grid, name), (full_path, _row) in full.items():
        cropped_path = cropped[(grid, name)][0]
        info = xta_cut_nrrd.probe_nrrd(cropped_path)
        width = GRIDS[grid][2]
        assert info.reference_xyz == tuple(reversed(GRIDS[grid])) and info.origin_xyz == tuple(
            reversed(cropped[(grid, name)][1]['stored_offset_tyx']))
        assert xta_cut_nrrd.probe_nrrd(full_path).reference_xyz is None
        # Every column on the preview grid; native samples the grid plus each box boundary.
        start, stop = info.origin_xyz[0], info.origin_xyz[0] + info.width
        columns = set(range(width + 1)) if grid == 'preview' else set(range(0, width + 1, 5)) | {
            v for edge in (start, stop) for v in (edge - 1, edge, edge + 1)} | {width}
        for mode in ('zero', 'crop'):
            for column in sorted(c for c in columns if (0 if mode == 'zero' else 1) <= c <= width):
                label = f'{grid}/{name}/{mode}/{column}'
                root = tmp_path / 'cuts' / grid / name / mode / str(column)
                root.mkdir(parents=True)
                expected_audit = _cut(full_path, root / 'full.seg.nrrd', mode, column)
                audit = _cut(cropped_path, root / 'cropped.seg.nrrd', mode, column)
                expected, expected_shape = _decoded(root / 'full.seg.nrrd')
                actual, actual_shape = _decoded(root / 'cropped.seg.nrrd')
                assert actual_shape == expected_shape, label
                np.testing.assert_array_equal(actual, expected, err_msg=label)
                assert _global_extent(root / 'cropped.seg.nrrd') == _global_extent(root / 'full.seg.nrrd'), label
                fields = _header_fields(root / 'cropped.seg.nrrd')
                sizes = [int(v) for v in fields['sizes'].split()]
                offset = [int(v) for v in fields['Segmentation_ReferenceImageExtentOffset'].split()]
                assert fields['space origin'] == '(' + ','.join(map(str, offset)) + ')', label
                assert fields['Segmentation_ReferenceImageExtent'] == ' '.join(
                    f'0 {n - 1}' for n in reversed(expected_shape)), label
                assert audit['reference_shape_tyx'] == list(expected_shape), label
                assert audit['output_shape_tyx'] == list(reversed(sizes)), label
                assert audit['stored_offset_tyx'] == list(reversed(offset)), label
                assert expected_audit['output_shape_tyx'] == list(expected_shape), label
                if sizes == [1, 1, 1]:
                    # Slicer aborts on a nonempty one-voxel segmentation.
                    assert _global_extent(root / 'cropped.seg.nrrd') is None and offset == [0, 0, 0], label
                assert fields['content'] == _header_fields(root / 'full.seg.nrrd')['content'], label
                placements.append((mode, int(np.prod(sizes)), _global_extent(root / 'cropped.seg.nrrd') is None,
                                   info.width - xta_cut_nrrd.probe_nrrd(root / 'cropped.seg.nrrd').width))
    # The sweep covers widened one-voxel crops and boxes emptied or removed by the plane.
    assert any(mode == 'crop' and size == 2 and not empty for mode, size, empty, _ in placements)
    assert any(mode == 'crop' and size == 1 and empty for mode, size, empty, _ in placements)
    assert any(mode == 'crop' and size > 2 and narrower > 0 for mode, size, _, narrower in placements)


def test_cut_nrrd_rejects_inconsistent_cropped_placement(published, tmp_path):
    _volumes_, _full, cropped = published
    path = cropped[('native', 'interior')][0]
    info = xta_cut_nrrd.probe_nrrd(path)
    with pytest.raises(ValueError, match='reference-grid column'):
        xta_cut_nrrd.write_nrrd(path, tmp_path / 'a.nrrd', info, info.width, 'crop')
    with pytest.raises(ValueError, match='disagree'):
        xta_cut_nrrd.write_nrrd(path, tmp_path / 'b.nrrd', info, 3, 'crop', reference_first_removed=3)
    with pytest.raises(ValueError, match='valid output width'):
        xta_cut_nrrd.write_nrrd(path, tmp_path / 'c.nrrd', info, 0, 'crop', reference_first_removed=0)
    moved = _rewrite_header(path, tmp_path / 'moved.seg.nrrd', lambda lines: [
        'space origin: (0,0,0)' if line.startswith('space origin') else line for line in lines])
    with pytest.raises(ValueError, match='must equal its integer space origin'):
        xta_cut_nrrd.probe_nrrd(moved)
    outside = _rewrite_header(path, tmp_path / 'outside.seg.nrrd', lambda lines: [
        'Segmentation_ReferenceImageExtent:=0 20 0 28 0 10' if line.startswith('Segmentation_ReferenceImageExtent:=')
        else line for line in lines])
    with pytest.raises(ValueError, match='exceeds its reference extent'):
        xta_cut_nrrd.probe_nrrd(outside)
    assert not any(tmp_path.glob('[abc].nrrd'))


def test_axial_pair_grid_comes_from_the_reference_extent(published):
    _volumes_, full, cropped = published
    native, low = tuple(reversed(SHAPE)), tuple(reversed(GRIDS['preview']))
    for grid in ('native', 'preview'):
        for name in ('interior', 'high_edges', 'empty'):
            info = xta_cut_axial_pair.probe_nrrd(cropped[(grid, name)][0])
            full_info = xta_cut_axial_pair.probe_nrrd(full[(grid, name)][0])
            for first_native, first_preview in ((1, 1), (12, 6), (30, 15), (31, 16)):
                chosen = xta_cut_axial_pair.nrrd_cut(info, None, native, low, first_native, first_preview)
                expected = xta_cut_axial_pair.nrrd_cut(full_info, None, native, low, first_native, first_preview)
                assert chosen[0] == expected[0] == grid
                column = first_native if grid == 'native' else first_preview
                assert chosen[1] == max(0, min(info.width, column - info.origin_xyz[0]))
            other = 'preview' if grid == 'native' else 'native'
            with pytest.raises(ValueError, match='differs from the'):
                xta_cut_axial_pair.nrrd_cut(info, other, native, low, 12, 6)


@pytest.mark.skipif(not (shutil.which('ffmpeg') and shutil.which('ffprobe')), reason='FFmpeg and ffprobe are required')
def test_axial_pair_cli_cuts_cropped_layers_like_full_grid_layers(tmp_path, zlib_codec, monkeypatch):
    shape = (9, 12, 11)
    pixels = ((np.arange(np.prod(shape), dtype=np.uint16) * 37 + 19) % 256).astype(np.uint8).reshape(shape)
    video = tmp_path / 'sample.mkv'
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', '11x12',
                    '-r', '60', '-i', 'pipe:0', '-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'gray',
                    '-color_range', 'pc', str(video)], input=pixels.tobytes(), check=True, capture_output=True)
    volumes = {
        'straddle': _box_volume(shape, (2, 7, 3, 11, 3, 9), seed=11),
        'beyond': _box_volume(shape, (1, 4, 0, 5, 7, 11), seed=12),
        'one_wide': _box_volume(shape, (4, 5, 6, 7, 5, 7), seed=13),
        'empty': np.zeros(shape, np.uint8),
    }
    volumes['one_wide'][4, 6, 5:7] = 1
    layouts = {layout: _publish(tmp_path / layout, layout, monkeypatch, volumes, shape=shape, low_quality=())
               for layout in ('full', 'cropped')}
    cli = Path(__file__).resolve().parents[1] / 'tools' / 'xta_cut_axial_pair.py'
    for name in volumes:
        for mode in ('zero', 'crop') if name in ('straddle', 'beyond') else ('crop',):
            results = {}
            for layout, layers in layouts.items():
                source = layers[('native', name)][0]
                output = tmp_path / 'cli' / f'{name}_{mode}_{layout}'
                # Preview X=1.5 maps exactly to native X=5: native columns 0..5 are retained.
                process = subprocess.run([sys.executable, '-B', str(cli), '--mkv', str(video), '--nrrd', str(source),
                                          '--ras', '-1.5', '-3', '1', '--preview-scale', '0.5', '--mode', mode,
                                          '--output-dir', str(output)], text=True, capture_output=True)
                assert process.returncode == 0, process.stdout + process.stderr
                plan = json.loads((output / 'cut.json').read_text())
                results[layout] = (_decoded(output / source.name), plan)
            (expected, expected_shape), full_plan = results['full']
            (actual, actual_shape), plan = results['cropped']
            assert actual_shape == expected_shape == ((9, 12, 6) if mode == 'crop' else shape)
            np.testing.assert_array_equal(actual, expected, err_msg=f'{name}/{mode}')
            reference = volumes[name].copy()
            reference[:, :, 6:] = 0
            np.testing.assert_array_equal(expected, reference[:, :, :expected_shape[2]])
            assert plan['mkv_output']['shape_tyx'] == full_plan['mkv_output']['shape_tyx']
            assert plan['nrrd_grid'] == full_plan['nrrd_grid'] == 'native'
            assert plan['nrrd_reference_xyz'] == [11, 12, 9] and 'nrrd_reference_xyz' not in full_plan
            assert plan['nrrd_output']['first_removed_reference_i'] == 6
