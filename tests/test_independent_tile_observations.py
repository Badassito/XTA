"""Independent tile observations keep aligned masks and raw measured confidence."""

from dataclasses import replace
import threading
from unittest import mock

import numpy as np
import pytest

from XTA import assembly, geometry
from XTA.confidence_evidence import configure_confidence_evidence
from XTA.confidence_tiles import (
    capture_consolidated_tile_confidence,
    capture_tile_confidence,
    retire_tile_confidence,
)
from XTA.interpolation import (
    CTILE_FORMAT,
    CTILE_INDEX_DTYPE,
    INTERNAL_PACKED_CVOL_FORMAT,
    DeferredTilePostprocessResult,
    RawBBoxMaskStore,
    TilePostprocessTask,
    write_raw_bbox_mask_store,
)
from XTA.tile_observations import accumulate_independent_tile_observation, merge_tile_config_store


@pytest.fixture
def processing_view():
    source_shape = assembly.final_source_output_shape()
    assembly.set_final_source_output_shape(None)
    try:
        yield geometry.get_view_infos(3, 7, 9, cartesian_views=('transverse',))[0]
    finally:
        configure_confidence_evidence()
        assembly.set_final_source_output_shape(source_shape)


def _tile(root, view, name, crop, mask, *, scores=None, config='cfg'):
    path = root / (name + '.ctile')
    write_raw_bbox_mask_store(mask, path, format_name=CTILE_FORMAT)
    fields = dict(model_name='test', view_name=view.name, aug_id=view.tta_aug_id,
                  angle_deg=view.tta_angle_deg, config_id=config, tile_id=name, parent_crop=crop,
                  tile_mask_path=path)
    if scores is not None:
        task = TilePostprocessTask(**fields, tile_mask_mm=mask, tile_confmap_mm=scores, tile_confmap_path=None)
        capture_tile_confidence(task, view=view, work_dir=root)
    return DeferredTilePostprocessResult(**fields, tile_shape=mask.shape, storage_format=CTILE_FORMAT)


@pytest.mark.parametrize('deferred', [False, True])
def test_independent_crops_or_all_observations_and_max_only_measured_scores(
    tmp_path, processing_view, deferred,
):
    configure_confidence_evidence(tmp_path/'output', enabled=True, defer_projection=deferred)
    accumulator = np.zeros((3, 7, 9), np.uint8)
    locks = [threading.Lock(), threading.Lock()]
    expected_mask = np.zeros_like(accumulator)
    expected_scores = np.zeros_like(accumulator)
    mask_a = np.zeros((3, 3, 4), np.uint8)
    mask_a[0, 1, 1:4] = 1
    mask_a[2, 2, 0] = 1
    scores_a = np.zeros_like(mask_a)
    scores_a[0, 1, 1:4] = [72, 151, 0]
    scores_a[2, 0, 0] = 239  # Outside the cleaned observation.
    mask_b = np.zeros_like(mask_a)
    mask_b[0, 0, 0:2] = 1
    mask_b[1, 2, 3] = 1
    scores_b = mask_b * np.uint8(126)
    specimens = [('tile-b', (2,5,3,7), mask_b, scores_b), ('tile-a', (1,4,1,5), mask_a, scores_a)]
    for name, crop, mask, scores in specimens:
        tile = _tile(tmp_path, processing_view, name, crop, mask, scores=scores)
        y0, y1, x0, x1 = crop
        expected_mask[:, y0:y1, x0:x1] |= mask
        score_crop = expected_scores[:, y0:y1, x0:x1]
        np.maximum(score_crop, np.where(mask, scores, 0), out=score_crop)
        with mock.patch.object(RawBBoxMaskStore, 'decode_slice', side_effect=AssertionError('dense read')):
            accumulate_independent_tile_observation(
                tile, tile_accumulator_mm=accumulator, tile_accumulator_locks=locks,
            )
        assert not tile.tile_mask_path.exists()
        with pytest.raises(RuntimeError, match='already consumed'):
            accumulate_independent_tile_observation(tile, tile_accumulator_mm=accumulator)

    other = _tile(tmp_path, processing_view, 'different-config', (1,4,1,5), mask_a,
                  scores=mask_a*np.uint8(255), config='other')
    accumulate_independent_tile_observation(other, tile_accumulator_mm=np.zeros_like(accumulator))
    other_angle = replace(processing_view, name=processing_view.name + '_angle_17', tta_angle_deg=17)
    angle_tile = _tile(tmp_path, other_angle, 'different-angle', (1,4,1,5), mask_a,
                       scores=mask_a*np.uint8(254))
    accumulate_independent_tile_observation(angle_tile, tile_accumulator_mm=np.zeros_like(accumulator))

    np.testing.assert_array_equal(accumulator, expected_mask)
    reference = capture_consolidated_tile_confidence(
        accumulator, view=processing_view, model_name='test', config_id='cfg',
        category='independent', stage='cfg_pre_tile_interpolation', temp_dir=tmp_path,
    )
    assert reference.metadata['provenance']['tile_acceptance'] == 'independent'
    retire_tile_confidence('test', processing_view.name, 'cfg')
    retire_tile_confidence('test', processing_view.name, 'other')
    retire_tile_confidence('test', other_angle.name, 'cfg')
    assert not list((tmp_path/'tile_confidence').iterdir())
    if deferred:
        with reference.source_reader(tmp_path/'projection') as read:
            score, known = read(0, 3)
    else:
        score, known = reference.read(0, 3)
    np.testing.assert_array_equal(score, expected_scores)
    np.testing.assert_array_equal(known, expected_scores != 0)


def test_ctile_decode_failure_releases_mask_and_measurements(tmp_path, processing_view):
    configure_confidence_evidence(tmp_path/'output', enabled=True)
    accumulator = np.zeros((3, 7, 9), np.uint8)
    mask = np.ones((3, 2, 3), np.uint8)
    tile = _tile(tmp_path, processing_view, 'bad', (1,3,2,5), mask, scores=mask*np.uint8(90))
    index_path = tile.tile_mask_path/'index.bin'
    records = np.fromfile(index_path, dtype=CTILE_INDEX_DTYPE)
    records['payload_nbytes'][0] += 1
    records.tofile(index_path)
    with pytest.raises(ValueError, match='logical bbox byte count mismatch'):
        accumulate_independent_tile_observation(tile, tile_accumulator_mm=accumulator)
    assert not tile.tile_mask_path.exists()
    assert not list((tmp_path/'tile_confidence').iterdir())
    assert not accumulator.any()


def test_ctile_out_of_bounds_declines_before_writing(tmp_path, processing_view):
    configure_confidence_evidence()
    mask = np.ones((3, 2, 3), np.uint8)
    tile = _tile(tmp_path, processing_view, 'shift', (1,3,2,5), mask)
    accumulator = np.zeros((3, 7, 9), np.uint8)
    with pytest.raises(ValueError, match='crop or shape'):
        accumulate_independent_tile_observation(
            replace(tile, parent_crop=(6,8,2,5)), tile_accumulator_mm=accumulator,
        )
    assert not tile.tile_mask_path.exists()
    assert not accumulator.any()


def test_merge_config_reads_crops_under_destination_lock_and_leaves_source(tmp_path):
    source = np.zeros((3, 7, 9), np.uint8)
    source[1, 0:1, 2:5] = 255
    source[2, 6, 7:9] = 1
    path = tmp_path/'config.cvol'
    write_raw_bbox_mask_store(source, path, format_name=INTERNAL_PACKED_CVOL_FORMAT)
    destination = np.zeros_like(source)[:, ::-1, :]
    destination[0, 5, 8] = 1
    expected = destination | (source != 0)
    lock = threading.Lock()
    decode = RawBBoxMaskStore.decode_slice_crop

    def checked_decode(store, *args, **kwargs):
        assert lock.locked()
        return decode(store, *args, **kwargs)

    with mock.patch.object(RawBBoxMaskStore, 'decode_slice', side_effect=AssertionError('dense read')), \
         mock.patch.object(RawBBoxMaskStore, 'decode_slice_crop', checked_decode):
        merge_tile_config_store(path, destination_mm=destination, destination_lock=lock, workers=2)
    np.testing.assert_array_equal(destination, expected)
    assert not lock.locked()
    assert path.is_dir()

    with pytest.raises(ValueError, match='destination processing grid'):
        merge_tile_config_store(path, destination_mm=destination[:, :, :-1], destination_lock=lock)
    np.testing.assert_array_equal(destination, expected)
