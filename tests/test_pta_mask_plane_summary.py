"""NRRD read-time plane summaries replace full-mask rescans without changing results."""
import itertools
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from XTA import pta, pta_inputs
from XTA.pta_config import parse_pta_args
from XTA.pta_runtime import build_runtime_options


def write_mask(tmp_path, source, *, encoding='gzip', name='mask.nrrd'):
    nrrd = pytest.importorskip('nrrd')
    path = tmp_path / name
    nrrd.write(str(path), np.ascontiguousarray(source).transpose(2, 1, 0),
               header={'encoding': encoding, 'space directions': np.eye(3)},
               compression_level=1)
    return path


def load_with_summary(path, shape, **kwargs):
    summaries = []
    if kwargs.get('source_plane_indices') is not None:
        mask, _ = pta_inputs.load_indexed_nrrd_mask_for_volume(
            path, shape, warnings=pta.WarningLog(), plane_summary_out=summaries, **kwargs)
    else:
        mask = pta_inputs.load_nrrd_mask_for_volume(
            path, shape, warnings=pta.WarningLog(), plane_summary_out=summaries, **kwargs)
    return mask, summaries


def load_general_reader(path, shape, **kwargs):
    """The pre-existing full pynrrd reader: an independent reference for mask bytes."""
    with mock.patch.object(pta_inputs, '_read_canonical_uint8_nrrd', return_value=None):
        mask, summaries = load_with_summary(path, shape, **kwargs)
    assert summaries == []
    return mask


def assert_summary_matches_full_scans(mask, summary):
    assert summary.has_foreground.dtype == np.bool_ and summary.has_foreground.shape == (len(mask),)
    np.testing.assert_array_equal(summary.has_foreground, np.any(mask, axis=(1, 2)))
    np.testing.assert_array_equal(summary.first_foreground, [int(np.argmax(plane)) for plane in mask])


def assert_same_mask(actual, expected):
    assert actual.dtype == expected.dtype == np.uint8
    assert actual.shape == expected.shape and actual.flags.c_contiguous
    assert actual.tobytes() == expected.tobytes()


def source_cases():
    rng = np.random.default_rng(1107)
    sparse = (rng.random((9, 13, 11)) < 0.02).astype(np.uint8) * rng.integers(1, 256, (9, 13, 11), dtype=np.uint8)
    first = np.zeros((4, 5, 6), np.uint8)
    first[:, 0, 0] = 3
    last = np.zeros((4, 5, 6), np.uint8)
    last[:, -1, -1] = 255
    alternating = np.zeros((6, 3, 4), np.uint8)
    alternating[::2, 1, 2] = 1
    return {
        'single_zero': np.zeros((1, 1, 1), np.uint8),
        'single_full': np.full((1, 1, 1), 255, np.uint8),
        'thin': rng.integers(0, 2, (3, 1, 5), dtype=np.uint8),
        'empty': np.zeros((9, 13, 11), np.uint8),
        'full': np.full((9, 13, 11), 7, np.uint8),
        'sparse_values': sparse,
        'first_voxel': first,
        'last_voxel': last,
        'alternating_planes': alternating,
        'dense_random': rng.integers(0, 256, (5, 6, 7), dtype=np.uint8),
    }


@pytest.mark.parametrize('encoding', ['raw', 'gzip'])
@pytest.mark.parametrize('case', sorted(source_cases()))
def test_direct_read_bytes_and_summary_match_full_reader(tmp_path, encoding, case):
    source = source_cases()[case]
    path = write_mask(tmp_path, source, encoding=encoding)
    mask, summaries = load_with_summary(path, source.shape)
    assert_same_mask(mask, (source > 0).astype(np.uint8))
    assert_same_mask(mask, load_general_reader(path, source.shape))
    assert len(summaries) == 1
    assert_summary_matches_full_scans(mask, summaries[0])


@pytest.mark.parametrize('shape', [
    (3000, 2, 3),     # every plane in one 8 MiB group
    (17, 1025, 1024), # groups of 7 planes, last group partial
    (2, 2897, 2897),  # planes above 8 MiB are read in two pieces
])
def test_plane_groups_and_large_planes_keep_bytes_and_summary(tmp_path, shape):
    rng = np.random.default_rng(sum(shape))
    source = np.zeros(shape, np.uint8)
    flat = source.reshape(shape[0], -1)
    for plane in range(0, shape[0], 3):
        flat[plane, rng.integers(0, flat.shape[1], 2)] = rng.integers(1, 256, 2, dtype=np.uint8)
    flat[-1, -1] = 9
    path = write_mask(tmp_path, source)
    mask, summaries = load_with_summary(path, shape)
    assert_same_mask(mask, (source > 0).astype(np.uint8))
    assert_summary_matches_full_scans(mask, summaries[0])


@pytest.mark.parametrize('deltas', list(itertools.product((-3, -1, 0, 2, 3), repeat=3)))
def test_forced_resample_summary_matches_its_written_planes(tmp_path, deltas):
    rng = np.random.default_rng(31)
    source = ((rng.random((7, 11, 17)) < 0.08) * rng.integers(1, 256, (7, 11, 17))).astype(np.uint8)
    source[2] = 0
    source[4, -1, -1] = 1
    path = write_mask(tmp_path, source)
    target = tuple(size + delta for size, delta in zip(source.shape, deltas))
    mask, summaries = load_with_summary(path, target, force=True)
    assert_same_mask(mask, load_general_reader(path, target, force=True))
    assert_summary_matches_full_scans(mask, summaries[0])


@pytest.mark.parametrize('target_shape', [(9, 11, 17), (9, 13, 14)])
def test_indexed_mapping_summary_marks_unmatched_planes_empty(tmp_path, target_shape):
    source = np.ones((5, 11, 17), np.uint8)
    source[1] = 0
    source[3, :4] = 0
    path = write_mask(tmp_path, source)
    planes = (-2, -1, 0, 1, 2, 3, 4, 5, 6)
    mask, summaries = load_with_summary(path, target_shape, force=True, source_plane_indices=planes)
    assert_same_mask(mask, load_general_reader(path, target_shape, force=True, source_plane_indices=planes))
    assert_summary_matches_full_scans(mask, summaries[0])
    assert summaries[0].has_foreground.tolist() == [False, False, True, False, True, True, True, False, False]


@pytest.mark.parametrize('case', ['signed', 'noncanonical'])
def test_general_reader_layouts_provide_no_summary(tmp_path, case):
    nrrd = pytest.importorskip('nrrd')
    source = np.arange(5 * 6 * 7, dtype=np.int16).reshape(5, 6, 7) - 30
    path = tmp_path / 'mask.nrrd'
    if case == 'signed':
        nrrd.write(str(path), source.transpose(2, 1, 0), header={'space directions': np.eye(3)})
    else:
        nrrd.write(str(path), (source > 0).astype(np.uint8),
                   header={'space directions': np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0]])})
    mask, summaries = load_with_summary(path, source.shape)
    np.testing.assert_array_equal(mask, source > 0)
    assert summaries == []


def test_failed_read_provides_no_summary(tmp_path):
    path = tmp_path / 'short.nrrd'
    path.write_bytes(b'NRRD0005\ntype: uint8\ndimension: 3\nsizes: 4 3 2\nencoding: raw\n'
                     b'space directions: (1,0,0) (0,1,0) (0,0,1)\n\n' + bytes(23))
    summaries = []
    with pytest.raises(ValueError, match='Unexpected EOF'):
        pta_inputs.load_nrrd_mask_for_volume(path, (2, 3, 4), warnings=pta.WarningLog(),
                                             plane_summary_out=summaries)
    assert summaries == []


def _source(mask, summary=None, *, matched=None, label_source='nrrd'):
    return pta.SourceVolume(
        input_dir=Path('dataset'), stem='sample', kind='sequence', image_paths=[], video_path=None,
        labels_by_frame={}, segmentation_nrrd_path=Path('sample.nrrd'), mask_volume=mask,
        volume_class='fully_labeled' if matched is None else 'partially_labeled',
        label_source=label_source, input_start_index=0, encoded_indices=tuple(range(len(mask))),
        volume=np.arange(mask.size, dtype=np.uint8).reshape(mask.shape), fps=1.0,
        matched_frame_indices=matched, mask_plane_summary=summary,
    )


def _summary(mask):
    return pta_inputs.MaskPlaneSummary(
        np.any(mask, axis=(1, 2)), np.asarray([np.argmax(plane) for plane in mask], np.intp))


@pytest.mark.parametrize('matched', [None, (0, 2, 3, 6)])
def test_states_and_anchors_from_summary_match_full_scans_without_scanning(matched):
    rng = np.random.default_rng(5)
    mask = (rng.random((8, 9, 10)) < 0.03).astype(np.uint8)
    mask[1] = 0
    mask[3, -1, -1] = 1
    summary = _summary(mask)
    src = _source(mask, matched=matched)
    expected_states = pta.derive_annotation_states(src, mask)
    expected_anchors = pta.collect_foreground_slice_anchors(src, mask, expected_states, warnings=pta.WarningLog())
    with mock.patch.object(pta.np, 'any', side_effect=AssertionError('full mask scan')), \
            mock.patch.object(pta.np, 'argmax', side_effect=AssertionError('plane scan')):
        states = pta.derive_annotation_states(src, mask, plane_summary=summary)
        anchors = pta.collect_foreground_slice_anchors(
            src, mask, states, warnings=pta.WarningLog(), plane_summary=summary)
    assert states == expected_states
    assert anchors == expected_anchors


def _prepare(source, output):
    args = build_runtime_options(parse_pta_args(['--input', 'dataset', '--save', 'voxel_volume']))
    return pta.prepare_loaded_source(
        source, args=args, warnings=pta.WarningLog(), workers=1, out_dir=output, tile_configs=(),
        channel_variants=pta.expand_channel_variants(pta.resolve_channel_formats(args.channel_format)),
        write_side_effects=False,
    )


def test_prepare_uses_reader_summary_only_for_the_array_it_describes(tmp_path):
    source = np.zeros((6, 9, 9), np.uint8)
    source[1:5, 2:7, 3:6] = 4
    source[2, 0, 8] = 1
    path = write_mask(tmp_path, source)
    mask, summaries = load_with_summary(path, source.shape)
    seen = []
    real_states, real_anchors = pta.derive_annotation_states, pta.collect_foreground_slice_anchors

    def states(src, mask_u8, **kwargs):
        seen.append(('states', kwargs.get('plane_summary')))
        return real_states(src, mask_u8, **kwargs)

    def anchors(src, mask_u8, annotation_states, **kwargs):
        seen.append(('anchors', kwargs.get('plane_summary')))
        return real_anchors(src, mask_u8, annotation_states, **kwargs)

    results = {}
    with mock.patch.object(pta, 'derive_annotation_states', side_effect=states), \
            mock.patch.object(pta, 'collect_foreground_slice_anchors', side_effect=anchors):
        for name, volume, summary in (
            ('summary', mask, summaries[0]),
            ('scan', mask.copy(), None),
            ('converted', np.asfortranarray(mask), summaries[0]),
        ):
            seen.clear()
            source = _source(volume, summary)
            results[name] = _prepare(source, tmp_path / name)
            expected = summaries[0] if name == 'summary' else None
            assert [kind for kind, _ in seen] == ['states', 'anchors']
            assert all(value is expected for _, value in seen)
            assert source.mask_plane_summary is None
    for name in ('scan', 'converted'):
        assert results[name].annotation_states == results['summary'].annotation_states
        assert results[name].foreground_preservation_stats == results['summary'].foreground_preservation_stats
        assert results[name].voxel_final == results['summary'].voxel_final
        assert results[name].mask_for_render.tobytes() == results['summary'].mask_for_render.tobytes()
        assert results[name].volume_for_render.tobytes() == results['summary'].volume_for_render.tobytes()


def test_summary_is_used_once_so_a_later_mask_write_is_rescanned(tmp_path):
    source = np.zeros((6, 9, 9), np.uint8)
    source[1:5, 2:7, 3:6] = 1
    mask, summaries = load_with_summary(write_mask(tmp_path, source), source.shape)
    src = _source(mask, summaries[0])
    first = _prepare(src, tmp_path / 'first')
    assert src.mask_plane_summary is None
    assert first.annotation_states == (pta.ANNOTATION_BACKGROUND,) + (pta.ANNOTATION_FOREGROUND,) * 4 \
        + (pta.ANNOTATION_BACKGROUND,)
    src.mask_volume[:] = 0  # a stale summary would still report planes 1-4 as foreground
    src.mask_volume[0, 8, 8] = 1
    with mock.patch.object(pta, 'collect_foreground_slice_anchors', wraps=pta.collect_foreground_slice_anchors) as anchors:
        again = _prepare(src, tmp_path / 'again')
    assert anchors.call_args.kwargs['plane_summary'] is None
    assert again.annotation_states == (pta.ANNOTATION_FOREGROUND,) + (pta.ANNOTATION_BACKGROUND,) * 5
    assert again.foreground_preservation_stats['input_foreground_transverse_slices'] == 1


def test_sequence_load_attaches_reader_summary_to_source(tmp_path):
    source = np.zeros((4, 5, 6), np.uint8)
    source[2, 3, 4] = 1
    path = write_mask(tmp_path, source)
    spec = pta.VolumeInputSpec(
        input_dir=tmp_path, stem='sample', kind='sequence',
        image_paths_by_index={index: tmp_path / f'{index}.png' for index in range(4)},
        video_path=None, labels_by_index={}, segmentation_nrrd_path=path,
        volume_class='fully_labeled', label_source='nrrd', input_start_index=0,
        encoded_indices=(0, 1, 2, 3), frame_count_hint=4, fps_hint=1.0)
    with mock.patch.object(pta, 'load_image_sequence_to_memory',
                           return_value=(np.zeros(source.shape, np.uint8), None)):
        loaded = pta.load_source_volume_from_spec(spec, warnings=pta.WarningLog(), workers=1)
    assert_summary_matches_full_scans(loaded.mask_volume, loaded.mask_plane_summary)
    assert loaded.mask_plane_summary.has_foreground.tolist() == [False, False, True, False]


@pytest.mark.parametrize('workers', [1, 2])
def test_video_load_attaches_reader_summary_on_serial_and_parallel_paths(tmp_path, workers):
    source = np.zeros((4, 5, 6), np.uint8)
    source[1, 0, 5] = 2
    path = write_mask(tmp_path, source)
    spec = pta.VolumeInputSpec(
        input_dir=tmp_path, stem='sample', kind='video', image_paths_by_index={},
        video_path=tmp_path / 'sample.mkv', labels_by_index={}, segmentation_nrrd_path=path,
        volume_class='fully_labeled', label_source='nrrd', input_start_index=None,
        encoded_indices=(0, 1, 2, 3), frame_count_hint=4, fps_hint=7.0,
        video_info_hint={'num_frames': 4, 'height': 5, 'width': 6, 'fps': 7.0})
    with mock.patch.object(pta, 'decode_video_gray8_to_memory',
                           return_value=(np.zeros(source.shape, np.uint8), 7.0, None)):
        loaded = pta.load_source_volume_from_spec(spec, warnings=pta.WarningLog(), workers=workers)
    assert_same_mask(loaded.mask_volume, (source > 0).astype(np.uint8))
    assert_summary_matches_full_scans(loaded.mask_volume, loaded.mask_plane_summary)
    assert loaded.mask_plane_summary.first_foreground[1] == 5


@pytest.mark.parametrize('target, span, calls', [((9, 14, 19), 'pta.mask.nrrd_normalize', 1),
                                                  ((10, 14, 21), 'pta.mask.nrrd_resample', 9)])
def test_mask_write_stages_have_spans(tmp_path, monkeypatch, target, span, calls):
    from XTA import runtime
    monkeypatch.setenv('YOLO_TTA_TELEMETRY', '1')
    monkeypatch.setenv('YOLO_TTA_TELEMETRY_SYSTEM_SAMPLER', '0')
    recorder = runtime.RuntimeTelemetry()
    recorder.flush_seconds = 1e9
    monkeypatch.setattr(pta_inputs, 'runtime_telemetry', lambda: recorder)
    source = (np.random.default_rng(0).random((9, 14, 19)) < 0.2).astype(np.uint8)
    mask = pta_inputs.load_nrrd_mask_for_volume(write_mask(tmp_path, source), target,
                                                warnings=pta.WarningLog(), force=True)
    expected = source
    if target != source.shape:
        # The endpoint-aligned map writes target planes 4 and 5 in one source-plane span.
        source_z = [0, 1, 2, 3, 4, 4, 5, 6, 7, 8]
        source_x = [0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 15, 16, 17, 18]
        expected = source[np.ix_(source_z, range(source.shape[1]), source_x)]
    assert_same_mask(mask, expected)
    phases = recorder.snapshot()['phases']
    assert phases['pta.mask.nrrd_payload_read']['calls'] == 1
    assert phases[span]['calls'] == calls
