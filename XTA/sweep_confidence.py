"""Explicit per-cutoff observed-confidence publication for shared view inference."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import hashlib
from pathlib import Path
import threading
import time
import zlib

import numpy as np

from .native_cpu import kernels as _cpu
from .confidence_evidence import (
    SCHEMA,
    _json_value,
    _write_json_atomic,
    write_block_confidence_evidence,
)
from .confidence_retention import observed_confidence_contract
from .confidence_storage import BLOCK_DTYPE, checked_shape


_CROP_GRID_ERROR = 'Sweep confidence native crop differs from its grid'
_NO_RECORDS = np.empty(0, BLOCK_DTYPE)
_NO_RECORDS.setflags(write=False)


def _merge_masked_crop(plane, touched, mask, scores, y0, x0, block):
    """Max-merge one strip's masked scores into the frame plane; mark its grid cells."""
    return _cpu.merge_masked_crop(plane, touched, mask, scores, y0, x0, block)


def _take_frame_cells(plane, touched, block):
    """Pop every marked grid cell in row-major order as an exact tight crop."""
    return _cpu.take_frame_cells(plane, touched, block)


_NATIVE_LOCK = threading.Lock()
_NATIVE_COMPRESSORS = {}
_NATIVE_MAX_BLOCK = 128


def _probe_cell_compressor(kernel):
    """Exact-bytes qualification against ``zlib.compress(level=3)`` on varied cells."""
    rng = np.random.default_rng(20261112)
    cells = [np.zeros((1, 1), np.uint8) + 1, np.zeros((128, 128), np.uint8) + 255,
             rng.integers(0, 256, (128, 128), dtype=np.uint8),
             rng.integers(0, 256, (97, 113), dtype=np.uint8) * (rng.random((97, 113)) < 0.2),
             np.repeat(rng.integers(128, 256, (1, 128), dtype=np.uint8), 64, axis=0)]
    raw = np.concatenate([cell.astype(np.uint8).reshape(-1) for cell in cells])
    table = np.zeros((len(cells), 6), np.int64)
    start = 0
    for item, cell in enumerate(cells):
        table[item, 2:4] = cell.shape
        table[item, 5] = start
        start += cell.size
    payload, lengths, done, status = kernel(raw, table)
    expected = [zlib.compress(cell.astype(np.uint8).tobytes(), 3) for cell in cells]
    return (status == 0 and done == len(cells) and lengths.tolist() == [len(e) for e in expected]
            and payload.tobytes() == b''.join(expected))


def _native_cell_compressor():
    """Return ``(kernel, backend)`` for exact nogil cell compression.

    Uses the run's probed native codec only when it is Python's own zlib
    extension (or an explicit qualification library) at Python's zlib runtime
    version and reproduces ``zlib.compress`` bytes on probe cells; otherwise
    ``(None, 'python_zlib')`` keeps one Python zlib call per cell. The codec's
    configuration switch (YOLO_TTA_CONFIDENCE_NATIVE_CODEC) disables both.
    """
    import ctypes
    from .confidence_native_codec import native_confidence_codec
    codec, receipt = native_confidence_codec()
    if (codec is None or codec.version != zlib.ZLIB_RUNTIME_VERSION
            or receipt.get('library_source') not in ('python_zlib_extension', 'explicit_qualification')):
        return None, 'python_zlib'
    with _NATIVE_LOCK:
        entry = _NATIVE_COMPRESSORS.get(id(codec))
        if entry is None or entry[0] is not codec:
            address = ctypes.cast(codec._compressor, ctypes.c_void_p).value
            kernel = partial(_cpu.sweep_compress_cells, address)
            try:
                qualified = _probe_cell_compressor(kernel)
            except Exception:
                qualified = False
            entry = (codec, kernel if qualified else None)
            _NATIVE_COMPRESSORS[id(codec)] = entry
            print(f'Sweep confidence cell compression: '
                  f'{"native_compress2" if qualified else "python_zlib"} '
                  f'(library_source={receipt.get("library_source")}, version={codec.version}).', flush=True)
    return (entry[1], 'native_compress2') if entry[1] is not None else (None, 'python_zlib')


class _MaskedScorePlanes:
    """Read at most one native score plane and its exact surviving support."""

    def __init__(self, shape, read_mask, read_scores):
        if not callable(read_mask) or not callable(read_scores):
            raise TypeError('Observed confidence requires mask and score plane readers')
        self.shape = checked_shape(shape)
        self.read_mask, self.read_scores = read_mask, read_scores

    def iter_crops(self, z):
        """Yield the whole plane as one native strip (or nothing for empty support)."""
        mask = self.read_mask(z)
        if mask is None:
            return
        mask = np.asarray(mask)
        if mask.shape != self.shape[1:] or mask.dtype not in (np.dtype(bool), np.dtype(np.uint8)):
            raise ValueError('Sweep confidence mask plane differs from its native grid')
        if not np.any(mask):
            return
        values = self.read_scores(z)
        if values is None:
            return
        values = np.asarray(values)
        if values.dtype != np.uint8 or values.shape != self.shape[1:]:
            raise ValueError('Sweep confidence score plane differs from its native grid')
        yield 0, self.shape[1], 0, self.shape[2], mask, values


class _MaskedCropFrameEncoder:
    """Encode each frame's native strips as exact sparse scalar-confidence blocks.

    The strips are max-merged on their mask support into one reusable frame
    plane, then a single compiled call pops every nonempty global-grid cell in
    row-major order, trimmed to its nonzero box with its known-voxel count.
    Merge, scan and copy run without the GIL. Compression is one nogil call per
    frame when Python's own zlib library is exposed natively (Linux); otherwise
    one ``zlib.compress`` per cell, which releases the GIL while it deflates.
    The payload, index and metadata equal the generic ``iter_crops`` ->
    ``crop_blocks`` route of ``write_blocks``. One instance belongs to one
    ``write_blocks`` call (H x W plane plus one frame's cells live).
    """

    def __init__(self, shape, read_native_crops, block_size=128):
        if not callable(read_native_crops):
            raise TypeError('Observed confidence needs a native crop reader')
        self.shape = checked_shape(shape)
        self.read_native_crops = read_native_crops
        self.block_size = int(block_size)
        if not 1 <= self.block_size <= 65535:
            raise ValueError('Confidence block size must be in [1,65535]')
        height, width = self.shape[1:]
        block = self.block_size
        self._plane = np.zeros((height, width), np.uint8)
        self._touched = np.zeros(((height + block - 1) // block, (width + block - 1) // block), np.uint8)
        self._compress_cells, self.codec_backend = (
            _native_cell_compressor() if block <= _NATIVE_MAX_BLOCK else (None, 'python_zlib'))
        self.native_fallback_frames = 0

    def encode_caller_frame(self, z, block_size):
        """Return ``(records, payload, known_voxels, phase_metrics)`` for frame z.

        ``records`` are BLOCK_DTYPE index rows in global grid order whose
        offsets are relative to this frame's concatenated zlib ``payload``.
        """
        if int(block_size) != self.block_size:
            raise ValueError('Sweep confidence writer grid differs from its crop encoder')
        height, width = self.shape[1:]
        block = self.block_size
        plane, touched = self._plane, self._touched
        read_seconds = merge_seconds = 0.
        merged = False
        started = time.perf_counter()
        for y0, y1, x0, x1, mask, scores in self.read_native_crops(z):
            phase_started = time.perf_counter()
            read_seconds += phase_started - started
            if scores is not None:
                y0, y1, x0, x1 = map(int, (y0, y1, x0, x1))
                mask, scores = np.asarray(mask), np.asarray(scores)
                if (not 0 <= y0 < y1 <= height or not 0 <= x0 < x1 <= width
                        or mask.shape != (y1-y0, x1-x0) or scores.shape != mask.shape
                        or mask.dtype not in (np.dtype(bool), np.dtype(np.uint8))
                        or scores.dtype != np.uint8):
                    raise ValueError(_CROP_GRID_ERROR)
                if mask.dtype == np.dtype(bool):
                    mask = mask.view(np.uint8)
                if not mask.flags.c_contiguous:
                    mask = np.ascontiguousarray(mask)
                if not scores.flags.c_contiguous:
                    scores = np.ascontiguousarray(scores)
                _merge_masked_crop(plane, touched, mask, scores, y0, x0, block)
                merged = True
            mask = scores = None
            started = time.perf_counter()
            merge_seconds += started - phase_started
        read_seconds += time.perf_counter() - started
        if not merged:
            return _NO_RECORDS, b'', 0, dict(read_seconds=read_seconds, scan_seconds=merge_seconds,
                                             compression_seconds=0.)
        started = time.perf_counter()
        table, raw = _take_frame_cells(plane, touched, block)
        count = len(table)
        records = np.empty(count, BLOCK_DTYPE)
        records['z'] = z
        records['y'], records['x'] = table[:, 0], table[:, 1]
        records['h'], records['w'] = table[:, 2], table[:, 3]
        starts = table[:, 5]
        ends = starts + table[:, 2] * table[:, 3]
        known = int(table[:, 4].sum())
        scan_seconds = merge_seconds + time.perf_counter() - started
        started = time.perf_counter()
        payload = None
        if self._compress_cells is not None and count:
            # One GIL release for the whole frame's cells.
            compressed, lengths, done, status = self._compress_cells(raw, table)
            if status == 0 and done == count:
                payload = compressed.data
            else:
                self.native_fallback_frames += 1
        if payload is None:
            view = memoryview(raw)
            try:
                # Identical to zlib.compress(crop.tobytes(), level=3) per cell;
                # each call releases the GIL while it deflates.
                encoded = [zlib.compress(view[start:end], 3)
                           for start, end in zip(starts.tolist(), ends.tolist())]
            finally:
                view.release()
            lengths = np.fromiter(map(len, encoded), np.int64, count)
            payload = b''.join(encoded)
        if count and int(lengths.max()) > 0xFFFFFFFF:
            raise ValueError('Confidence block exceeds its uint32 compressed length field')
        records['length'] = lengths
        records['offset'] = np.cumsum(lengths) - lengths
        return records, payload, known, dict(read_seconds=read_seconds, scan_seconds=scan_seconds,
                                             compression_seconds=time.perf_counter() - started)


@dataclass(frozen=True)
class SweepEvidenceRequest:
    """Picklable inputs of one reserved sibling evidence layer (no reader, no lock).

    ``SweepObservedConfidenceWriter.begin_view`` creates it; the layer is then
    written by ``write_sweep_view_evidence`` in a writer thread or an evidence
    worker process, and adopted with ``complete_view`` (or released with
    ``abandon_view``).
    """
    output_dir: str
    identity: str
    model_name: str
    layer_key: str
    native_shape: tuple
    source_shape: tuple
    provenance: dict

    @property
    def token(self):
        return (self.model_name, self.layer_key)

    @property
    def path(self):
        return Path(self.output_dir) / self.identity


def write_sweep_view_evidence(request, read_native_crops, *, metrics=None):
    """Encode one reserved layer from its native crops and return its evidence reference.

    The bytes depend only on the request and the crops, so a thread of the
    producer and an evidence worker process write identical files.
    """
    reader = _MaskedCropFrameEncoder(request.native_shape, read_native_crops)
    try:
        return write_block_confidence_evidence(request.path, request.native_shape, reader,
            layer_key=request.layer_key, model_name=request.model_name,
            provenance=request.provenance, coordinate_space='native_view_processing',
            source_shape_tyx=request.source_shape, metrics=metrics)
    finally:
        # The encoder's frame plane is this call's only dense workspace.
        reader = None


class SweepObservedConfidenceWriter:
    """Retain each branch's numeric observations without the scalar singleton.

    Readers supplied to ``publish_view`` are already bound to this cutoff and
    return uint8 native-view planes after the ordinary scalar cleanup. ``None``
    denotes a declared empty plane. This writer is for deferred native evidence;
    confidence-reconciliation source projection keeps the existing scalar path.
    A layer can also be written elsewhere (an evidence worker process): reserve
    it with ``begin_view``, then ``complete_view`` or ``abandon_view``.
    """

    def __init__(self, output_dir, *, cutoff_hundredths, source_shape_tyx, run_id,
                 task='segment', min_conf=0.0, confidence_floor_by_backend=None,
                 defer_projection=True):
        if type(cutoff_hundredths) is not int or not 0 <= cutoff_hundredths <= 100:
            raise ValueError('Sweep confidence cutoff must be an integer hundredth in [0,100]')
        if not isinstance(run_id, str) or not run_id:
            raise ValueError('Sweep confidence needs its scalar branch run ID')
        if not defer_projection:
            raise ValueError('Source-projected confidence uses the ordinary scalar publication path')
        self.output = Path(output_dir) / 'reconciliation_evidence'
        self.source_shape = checked_shape(source_shape_tyx)
        self.cutoff, self.run_id = cutoff_hundredths, run_id
        self.contract = observed_confidence_contract(prediction_task=task,
            confidence_floor=cutoff_hundredths / 100., min_conf=min_conf,
            confidence_floor_by_backend=confidence_floor_by_backend)
        self._references, self._pending = {}, set()
        self._lock = threading.Lock()
        if self.manifest_path.exists():
            raise FileExistsError(f'Confidence catalogue already exists: {self.manifest_path}')

    @property
    def manifest_path(self):
        return self.output / 'manifest.json'

    def _publish_catalogue(self, references):
        layers = []
        for (model_name, layer_key), ref in sorted(references.items()):
            layers.append(dict(model_name=model_name, layer_key=layer_key,
                directory=ref.path.name, output_shape_tyx=list(ref.shape),
                storage_schema=ref.metadata['schema'], coordinate_space=ref.coordinate_space,
                stored_shape_tyx=list(ref.storage_shape), score_semantics=ref.metadata['score_semantics'],
                observed_confidence=ref.metadata['provenance']['observed_confidence'], unknown='score_zero'))
        _write_json_atomic(self.manifest_path, dict(schema=SCHEMA, run_id=self.run_id,
            purpose='observed_prediction_confidence', observed_confidence=self.contract, layers=layers))

    def begin_view(self, view, model_name, *, native_shape, layer_key=None, source='fullframe',
                   stage='pre_interpolation', tile_config_id='', tile_acceptance=''):
        """Reserve one layer and return its picklable ``SweepEvidenceRequest``.

        The layer stays pending (``finish`` refuses) until ``complete_view`` or
        ``abandon_view``; a second reservation of the same layer is rejected.
        """
        from .cuda_d1 import _nrrd_layer_key

        shape = checked_shape(native_shape)
        key = str(layer_key or _nrrd_layer_key(view_name=view.name, source=source,
            mask_kind='yolo', pass_index=0, tile_config_id=tile_config_id,
            tile_acceptance=tile_acceptance, stage=stage))
        token = (str(model_name), key)
        identity = hashlib.sha256((token[0] + '\0' + key).encode('utf-8')).hexdigest()
        with self._lock:
            if token in self._references or token in self._pending:
                raise RuntimeError(f'Duplicate sweep confidence layer {token!r}')
            self._pending.add(token)
        try:
            provenance = dict(view=_json_value(view), processing_shape_tyx=list(shape),
                source=source, mask_kind='yolo', stage=stage,
                tile_config_id=tile_config_id, tile_acceptance=tile_acceptance,
                confidence_scope='surviving observed prediction support before interpolation',
                observed_confidence=self.contract, run_id=self.run_id,
                confidence_sweep=dict(cutoff_hundredths=self.cutoff))
            return SweepEvidenceRequest(output_dir=str(self.output), identity=identity,
                model_name=token[0], layer_key=key, native_shape=shape,
                source_shape=self.source_shape, provenance=provenance)
        except BaseException:
            with self._lock:
                self._pending.remove(token)
            raise

    def complete_view(self, request, reference):
        """Adopt a durable layer written for ``request`` into this cutoff's catalogue."""
        with self._lock:
            if request.token not in self._pending:
                raise RuntimeError(f'Sweep confidence layer was not reserved: {request.token!r}')
            try:
                if (Path(reference.path) != request.path
                        or (reference.model_name, reference.layer_key) != request.token
                        or reference.storage_shape != tuple(request.native_shape)):
                    raise RuntimeError(f'Sweep confidence layer differs from its reservation: '
                                       f'{request.token!r}')
                updated = {**self._references, request.token: reference}
                self._publish_catalogue(updated)
                self._references = updated
            finally:
                self._pending.remove(request.token)
        return reference

    def abandon_view(self, request):
        """Release a reservation whose layer was not written (or failed)."""
        with self._lock:
            self._pending.discard(request.token)

    def publish_view(self, view, model_name, *, native_shape, read_scores=None, read_mask=None,
                     layer_key=None, source='fullframe', stage='pre_interpolation',
                     tile_config_id='', tile_acceptance='', read_native_crops=None,
                     metrics=None):
        """Write one layer; ``metrics`` (optional dict) receives the writer's phase timings."""
        request = self.begin_view(view, model_name, native_shape=native_shape, layer_key=layer_key,
            source=source, stage=stage, tile_config_id=tile_config_id,
            tile_acceptance=tile_acceptance)
        try:
            crops = (read_native_crops if read_native_crops is not None
                     else _MaskedScorePlanes(request.native_shape, read_mask, read_scores).iter_crops)
            reference = write_sweep_view_evidence(request, crops, metrics=metrics)
        except BaseException:
            self.abandon_view(request)
            raise
        finally:
            crops = None
        return self.complete_view(request, reference)

    def finish(self):
        """Publish an empty catalogue too when no observed view was requested."""
        with self._lock:
            if self._pending:
                raise RuntimeError('Sweep confidence publication still has unfinished views')
            self._publish_catalogue(self._references)
        return self.manifest_path
