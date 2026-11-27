"""Private packed staging for the selected editable review grids."""
from contextlib import closing
from pathlib import Path

import numpy as np

from .interpolation import INTERNAL_PACKED_CVOL_FORMAT, IncrementalRawBBoxMaskStoreWriter, RawBBoxMaskStore
from .packed_dense_source import DENSE_PACKED_CONTROL_BYTES, MAX_DENSE_PACKED_BLOCK_SLICES, encode_dense_fields


# The input belongs to the caller. Bound the extra packed payload and control
# arrays together, including the writer's private snapshot of the numeric rows.
_REVIEW_PACK_PAYLOAD_BYTES = 1024**2 - DENSE_PACKED_CONTROL_BYTES


class DiscardedRaster:
    """Allow a mirror consumer to use the regular restore without a native file."""

    def write(self, payload):
        return memoryview(payload).nbytes

    def write_zeros(self, size):
        return size


class PackedReviewRaster:
    """Convert the ordered restore stream into a sealed, sparse binary layer."""

    def __init__(self, destination, shape):
        path = Path(destination)
        if path.exists():
            raise FileExistsError(f'Review layer staging already exists: {path}')
        self.shape = tuple(shape)
        self.slice_bytes = self.shape[1] * self.shape[2]
        plane_bound = int(self.shape[1]) * ((int(self.shape[2]) + 7) // 8)
        self._batch_slices = min(MAX_DENSE_PACKED_BLOCK_SLICES,
                                 _REVIEW_PACK_PAYLOAD_BYTES // max(1, plane_bound))
        self.position = 0
        self.writer = IncrementalRawBBoxMaskStoreWriter(
            shape=self.shape, store_dir=path, format_name=INTERNAL_PACKED_CVOL_FORMAT,
            desc='Editable review layer',
        )

    def write(self, payload):
        data = np.frombuffer(payload, dtype=np.uint8).reshape(-1, *self.shape[1:])
        start = self.position
        stop = start + len(data)
        if start < 0 or stop > self.shape[0]:
            raise ValueError(f'{self.writer.desc}: projection callback block [{start},{stop}) is out of range')
        if not self._batch_slices or not len(data) or self.writer.failed:
            self.writer.consume(start, data)
        else:
            try:
                for first in range(0, len(data), self._batch_slices):
                    count = min(self._batch_slices, len(data) - first)
                    fields, packed = encode_dense_fields(
                        data, first, count, output_first=start + first,
                        max_payload_bytes=_REVIEW_PACK_PAYLOAD_BYTES,
                    )
                    self.writer.consume_encoded_fields(start + first, fields, packed, packed=True)
                    del fields, packed  # Release this batch before allocating the next.
            except Exception as error:
                self.writer.abort(error)
                raise
        self.position = stop
        return data.nbytes

    def write_zeros(self, size):
        count, remainder = divmod(size, self.slice_bytes)
        if remainder:
            raise ValueError('Review staging expects complete raster planes')
        self.writer.consume_empty_range(self.position, count)
        self.position += count
        return size

    def finalize(self):
        return self.writer.finalize()

    def discard(self):
        self.writer.discard()


def stage_review_volume(destination, volume):
    """Store the mirror tee's exact output without writing an intermediate NRRD."""
    raster = PackedReviewRaster(destination, volume.shape)
    try:
        for start in range(0, len(volume), 16):
            raster.write(volume[start:start + 16])
        return raster.finalize()
    except BaseException:
        raster.discard()
        raise


def layer_file_identities(directory):
    from .tta_sweep_branch_manifest import layer_payload_identity

    return {name: layer_payload_identity(directory / name)
            for name in ('chunks.bin', 'index.bin', 'meta.json')}


def _layer_record(entry, cutoff, review_id, grid_id):
    from .sweep_publication import _identifier, _role

    role = _role(entry)
    logical_id = _identifier(entry['model_name'], entry['layer_key'] or entry['suffix'], entry['suffix'], role)
    return dict(entry, logical_id=logical_id, role=role, cutoff_cent=cutoff,
                segment_id='xta_' + _identifier(review_id, grid_id, cutoff, logical_id),
                color_rgb=entry['segment_color_rgb'])


def review_layers(entries, paths, shape, cutoff, review_id, grid_id):
    """Read private sealed stores one layer and one bounded Z band at a time."""
    from .review_archive import ReviewLayer

    for entry, path in zip(entries, paths, strict=True):
        record = _layer_record(entry, cutoff, review_id, grid_id)
        with closing(RawBBoxMaskStore.open(path, mmap_payload=True)) as store:
            if store.shape != tuple(shape) or store.meta['format'] != INTERNAL_PACKED_CVOL_FORMAT:
                raise ValueError(f'Review layer source has the wrong shape or raster format: {path}')
            x0, x1, y0, y1, z0, z1 = store.meta['segment_extent_ijk']
            if not store.meta['stats']['foreground_voxels']:
                yield ReviewLayer(record, lambda first, stop: np.zeros((stop - first, 1, 1), np.uint8),
                                  bounds_tyx=((0, 1), (0, 1), (0, 1)))
                continue
            bounds = ((z0, z1 + 1), (y0, y1 + 1), (x0, x1 + 1))
            yield ReviewLayer(record, _slab_reader(store, bounds), bounds_tyx=bounds)


def _slab_reader(store, bounds):
    (_z0, _z1), (y0, y1), (x0, x1) = bounds

    def read_slab(first, stop):
        slab = np.zeros((stop - first, y1 - y0, x1 - x0), dtype=np.uint8)
        for z in range(first, stop):
            record = store.decode_slice_crop(z)
            if record is None:
                continue
            top, left, bottom, right, crop = record
            if not (y0 <= top < bottom <= y1 and x0 <= left < right <= x1):
                raise ValueError('Review layer crop falls outside its settled extent')
            slab[z - first, top-y0:bottom-y0, left-x0:right-x0] = crop
        return slab

    return read_slab
