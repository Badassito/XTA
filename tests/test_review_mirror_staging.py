from contextlib import closing
from unittest.mock import patch

import numpy as np
import pytest

from XTA import outputs
from XTA.interpolation import (
    CVOL_FORMAT,
    INTERNAL_PACKED_CVOL_FORMAT,
    NrrdLayerRef,
    RawBBoxMaskStore,
    write_raw_bbox_mask_store,
)


def _store_ref(tmp_path, volume, format_name, *, declared_shape=None):
    store_path = tmp_path / 'source.cvol'
    write_raw_bbox_mask_store(volume, store_path, format_name=format_name, desc='test', workers=1)
    return NrrdLayerRef(
        key='test', name='test', path=store_path, shape=declared_shape or volume.shape,
        storage_format=format_name,
    )


def _stage_mirror(root, ref, full_shape, mirror_shape):
    native_path = root / 'native.nrrd'
    mirror_path = root / 'mirror.nrrd'
    review_path = root / 'mirror.cvol'
    receipt = {}
    result = outputs.write_layer_nrrd_with_low_quality_mirrors(
        ref, full_shape, native_path, [(mirror_shape, mirror_path)],
        layout='cropped', write_native=False,
        mirror_review_paths={str(mirror_path): review_path}, mirror_nrrd_paths=(),
        plan_receipt=receipt,
    )
    assert result == native_path
    assert not native_path.exists()
    assert not mirror_path.exists()
    assert set(receipt) == {str(mirror_path)}
    return review_path, receipt[str(mirror_path)]


@pytest.mark.parametrize('format_name', (CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT))
@pytest.mark.parametrize('mirror_shape', ((4, 5, 11), (15, 27, 60)))
@pytest.mark.parametrize('empty', (False, True))
def test_identity_review_mirror_bytes_match_two_step_tee(tmp_path, format_name, mirror_shape, empty):
    volume = np.zeros((9, 17, 49), dtype=np.uint8)
    if not empty:
        volume[1, 1:7, 2:11] = 1
        volume[2, 5:11, 22:41] = 1
        volume[4, 9:17, 40:49] = 1
        volume[7, 2:11:2, 1:45:3] = 1
    ref = _store_ref(tmp_path, volume, format_name)
    with patch.object(outputs, '_try_stage_identity_review_mirror', return_value=False):
        expected_path, expected_plan = _stage_mirror(tmp_path / 'two-step', ref, volume.shape, mirror_shape)

    with patch.object(RawBBoxMaskStore, 'iter_native_sparse_members',
                      side_effect=AssertionError('review-only staging inflated the full native raster')):
        actual_path, actual_plan = _stage_mirror(tmp_path / 'direct', ref, volume.shape, mirror_shape)

    assert actual_plan == expected_plan
    for filename in ('chunks.bin', 'index.bin', 'meta.json'):
        assert (actual_path / filename).read_bytes() == (expected_path / filename).read_bytes()

    with closing(RawBBoxMaskStore.open(actual_path)) as store:
        assert store.shape == mirror_shape
        restored = np.stack([store.decode_slice(z) for z in range(store.shape[0])])
        assert int(store.meta['stats']['foreground_voxels']) == np.count_nonzero(restored)
        assert tuple(store.meta['segment_extent_ijk']) == outputs.compute_segment_extent_zyx(restored)
        assert bool(np.any(restored)) == (not empty)


def test_review_mirror_retains_intermediate_temporal_union(tmp_path):
    volume = np.zeros((2, 1, 600), dtype=np.uint8)
    volume[0, 0, 1] = volume[1, 0, 2] = 1
    full_shape = (1, 1, 600)
    mirror_shape = (1, 1, 1)
    # An identity claim on a ref is insufficient. The sealed store grid differs.
    ref = _store_ref(tmp_path, volume, INTERNAL_PACKED_CVOL_FORMAT, declared_shape=full_shape)
    direct_path = tmp_path / 'direct.cvol'
    outputs._stage_layer_raster(ref, mirror_shape, direct_path)
    composed_path, _ = _stage_mirror(tmp_path / 'composed', ref, full_shape, mirror_shape)
    with closing(RawBBoxMaskStore.open(direct_path)) as direct, \
            closing(RawBBoxMaskStore.open(composed_path)) as composed:
        assert direct.decode_slice(0).item() == 0
        assert composed.decode_slice(0).item() == 1


def test_identity_review_mirror_discards_partial_store(tmp_path, monkeypatch):
    volume = np.ones((2, 5, 11), dtype=np.uint8)
    ref = _store_ref(tmp_path, volume, CVOL_FORMAT)
    real_iter = RawBBoxMaskStore.iter_restored_sparse_members

    def fail_after_first_member(self, *args, **kwargs):
        yield next(iter(real_iter(self, *args, **kwargs)))
        raise RuntimeError('injected mirror restore failure')

    monkeypatch.setattr(RawBBoxMaskStore, 'iter_restored_sparse_members', fail_after_first_member)
    staging_root = tmp_path / 'staging'
    with pytest.raises(RuntimeError, match='injected mirror restore failure'):
        _stage_mirror(staging_root, ref, volume.shape, (1, 3, 7))
    assert not (staging_root / 'mirror.cvol').exists()
    assert not (staging_root / 'native.nrrd').exists()
    assert ref.path.is_dir()


@pytest.mark.parametrize('payload_limit', [64, 15])
def test_review_packing_matches_scalar_store(tmp_path, monkeypatch, payload_limit):
    from XTA import review_layer_sources as review

    shape = (11, 4, 17)  # A packed plane occupies at most 12 bytes.
    volume = np.random.default_rng(176205).integers(0, 4, shape, dtype=np.uint8)
    volume[1] = volume[4:6] = volume[8] = 0
    volume[0, 1, 0] = 255
    monkeypatch.setattr(review, '_REVIEW_PACK_PAYLOAD_BYTES', payload_limit)
    reference = review.PackedReviewRaster(tmp_path / 'scalar', shape)
    actual = review.PackedReviewRaster(tmp_path / 'batched', shape)
    try:
        with patch.object(actual.writer, '_pwrite_all', wraps=actual.writer._pwrite_all) as writes:
            for start, stop in ((0, 4), (4, 6), (6, 11)):
                if start == 4:
                    reference.writer.consume_empty_range(start, stop-start)
                    actual.write_zeros(volume[start:stop].nbytes)
                else:
                    reference.writer.consume(start, volume[start:stop])
                    assert actual.write(volume[start:stop]) == volume[start:stop].nbytes
            assert actual.position == shape[0]
            if payload_limit == 64:
                assert writes.call_count == 2
            assert actual.finalize() == reference.finalize()
        for filename in ('chunks.bin', 'index.bin', 'meta.json'):
            assert (tmp_path/'scalar'/filename).read_bytes() == (tmp_path/'batched'/filename).read_bytes()
    finally:
        reference.discard()
        actual.discard()


def test_review_write_checks_full_span_before_first_split(tmp_path, monkeypatch):
    from XTA import review_layer_sources as review

    monkeypatch.setattr(review, 'MAX_DENSE_PACKED_BLOCK_SLICES', 2)
    raster = review.PackedReviewRaster(tmp_path/'layer', (3, 2, 9))
    try:
        with patch.object(raster.writer, 'consume_encoded_fields', wraps=raster.writer.consume_encoded_fields) as put:
            with pytest.raises(ValueError, match='out of range'):
                raster.write(np.ones((4, 2, 9), np.uint8))
            put.assert_not_called()
        assert raster.position == 0 and not raster.writer.failed
        raster.write(np.ones((3, 2, 9), np.uint8))
        raster.finalize()
    finally:
        raster.discard()


def test_failed_review_batch_rejects_finalization(tmp_path, monkeypatch):
    from XTA import review_layer_sources as review

    monkeypatch.setattr(review, 'MAX_DENSE_PACKED_BLOCK_SLICES', 2)
    raster = review.PackedReviewRaster(tmp_path/'layer', (4, 3, 9))
    original = raster.writer._pwrite_all
    def append(fd, data, offset):
        if offset:
            raise OSError('injected second append')
        original(fd, data, offset)
    monkeypatch.setattr(raster.writer, '_pwrite_all', append)
    try:
        with pytest.raises(OSError, match='second append'):
            raster.write(np.ones((4, 3, 9), np.uint8))
        assert raster.writer.failed and raster.position == 0
        with pytest.raises(RuntimeError, match='invalidated'):
            raster.finalize()
        assert not (tmp_path/'layer'/'meta.json').exists()
    finally:
        raster.discard()


def test_review_oversized_plane_uses_bounded_scalar_writer(tmp_path, monkeypatch):
    from XTA import review_layer_sources as review

    monkeypatch.setattr(review, '_REVIEW_PACK_PAYLOAD_BYTES', 11)
    raster = review.PackedReviewRaster(tmp_path/'layer', (2, 4, 17))
    try:
        with patch.object(raster.writer, 'consume', wraps=raster.writer.consume) as consume:
            raster.write(np.ones((2, 4, 17), np.uint8))
            assert consume.call_count == 1
        raster.finalize()
    finally:
        raster.discard()
