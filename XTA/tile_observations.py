"""Stream cleaned tile observations into their own aligned processing-view grid."""

from contextlib import closing
from operator import index
from traceback import clear_frames

import numpy as np

from .assembly import (
    _delete_tile_result_storage,
    _or_sparse_tile_crop_into_parent,
    load_waiting_tile_result_from_raw_store,
)
from .confidence_tiles import discard_unpublished_tile_confidence, retain_independent_tile_confidence
from .interpolation import CTILE_FORMAT, CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT, RawBBoxMaskStore
from .interpolation import DeferredTilePostprocessResult
from .runtime import choose_slice_parallel_workers, parallel_for_indices_chunked


def _destination_shape(destination):
    if (not isinstance(destination, np.ndarray) or destination.ndim != 3
            or destination.dtype != np.uint8 or not destination.flags.writeable):
        raise ValueError('Tile placement requires a writable TYX uint8 destination')
    return destination.shape


def _merge_slice(store, z, destination, y_offset, x_offset, locks):
    try:
        # The decoded crop owns its bytes, so an in-flight OR does not keep
        # an exported payload mapping alive when the source is retired.
        decoded = store.decode_slice_crop(z, dtype=np.bool_)
    except BaseException as error:
        # A failed raw decode can retain a borrowed payload in its traceback.
        # Release those unwound locals before closing/unlinking on Windows.
        clear_frames(error.__traceback__)
        raise
    if decoded is None:
        return
    y0, x0, y1, x1, crop = decoded
    _or_sparse_tile_crop_into_parent(
        destination, locks, idx=z, y0=y0 + y_offset, y1=y1 + y_offset,
        x0=x0 + x_offset, x1=x1 + x_offset, crop_u8=crop,
    )


def accumulate_independent_tile_observation(
    result, *, tile_accumulator_mm, tile_accumulator_locks=None,
):
    """Consume one closed cleaned CTILE without filtering by a parent observation.

    Own the tile store through success or failure. Original measurements remain
    available to the configuration publisher after the mask store retires.
    """
    if not isinstance(result, DeferredTilePostprocessResult):
        raise TypeError('Independent tile placement requires a closed CTILE result')
    owns_mask = True
    try:
        depth, height, width = _destination_shape(tile_accumulator_mm)
        py0, py1, px0, px1 = map(index, result.parent_crop)
        if (not (0 <= py0 < py1 <= height and 0 <= px0 < px1 <= width)
                or result.tile_shape != (depth, py1 - py0, px1 - px0)):
            raise ValueError('Independent tile crop or shape differs from its processing grid')
        with retain_independent_tile_confidence(result):
            opened = load_waiting_tile_result_from_raw_store(result)
            with closing(opened.tile_mask_store) as store:
                if store.meta.get('format') != CTILE_FORMAT:
                    raise ValueError('Independent tile result has the wrong stored format')
                for z in range(depth):
                    _merge_slice(store, z, tile_accumulator_mm, py0, px0, tile_accumulator_locks)
            owns_mask = False
            _delete_tile_result_storage(result)
    except BaseException:
        try:
            if owns_mask:
                _delete_tile_result_storage(result)
        finally:
            discard_unpublished_tile_confidence(result)
        raise


def merge_tile_config_store(path, *, destination_mm, destination_lock, workers=1):
    """OR a settled config store into the exact same grid, leaving path ownership to the caller."""
    shape = _destination_shape(destination_mm)
    with closing(RawBBoxMaskStore.open(path, mmap_payload=True)) as store:
        if store.shape != shape or store.meta.get('format') not in (CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT):
            raise ValueError('Tile configuration store differs from the destination processing grid')

        def merge_slice(z):
            _merge_slice(store, z, destination_mm, 0, 0, None)

        with destination_lock:
            parallel_for_indices_chunked(
                shape[0], merge_slice, max_workers=choose_slice_parallel_workers(int(workers), shape[0]),
                desc='Merge settled tile configuration', show_progress=False,
            )
