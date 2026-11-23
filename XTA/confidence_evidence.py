"""Persistent, source-aligned prediction-confidence evidence for reconciliation.

Scores are uint8 maxima of instance scores or semantic foreground probabilities. Zero explicitly means
unknown: background, unobserved support, and scores quantized to zero are not
negative evidence. These sidecars never modify the prediction or bridge masks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Callable
import zlib

import numpy as np

from .json_publication import write_json_atomic
from .confidence_capture import (NativeConfidenceCapturePlan, plan_confidence_capture,
    confidence_capture_resources, current_confidence_capture_plan)


SCHEMA = 'xta.confidence_evidence/1'
SCORE_SEMANTICS = 'maximum_surviving_observed_prediction_score_u8'
_LEGACY_SCORE_SEMANTICS = 'maximum_surviving_instance_confidence_u8'
_LOCK = threading.RLock()
_MANIFEST_WRITE_LOCK = threading.Lock()
_BATCH_MANIFEST = False
_MANIFEST_DIRTY = False
_OUTPUT_DIR: Path | None = None
_ENABLED = False
_DEFER_PROJECTION = False
_REGISTRY: dict[tuple[str, str], 'ConfidenceEvidenceRef'] = {}
_PUBLICATION = None
_CAPTURE_QUEUE = None
_OBSERVED_CONFIDENCE = None
_RUN_ID = None
_ADOPTED_KEYS: set[tuple[str, str]] = set()


def configure_confidence_evidence(output_dir=None, *, enabled=False, min_conf=0.0,
                                  defer_projection=False, background_publication=False,
                                  prediction_task=None, confidence_floor=None,
                                  confidence_floor_by_backend=None, run_id=None,
                                  adopt_existing=False):
    """Start a run-owned collection; output_dir is the run's output directory."""
    from .confidence_retention import observed_confidence_contract
    contract = observed_confidence_contract(prediction_task=prediction_task,
        confidence_floor=confidence_floor, min_conf=min_conf,
        confidence_floor_by_backend=confidence_floor_by_backend)
    global _OUTPUT_DIR, _ENABLED, _DEFER_PROJECTION, _PUBLICATION, _OBSERVED_CONFIDENCE, _RUN_ID
    global _BATCH_MANIFEST, _MANIFEST_DIRTY
    shutdown_confidence_publication()
    with _MANIFEST_WRITE_LOCK, _LOCK:
        _ENABLED = bool(enabled)
        _DEFER_PROJECTION = bool(defer_projection)
        _BATCH_MANIFEST = bool(background_publication)
        _MANIFEST_DIRTY = False
        _OUTPUT_DIR = None if output_dir is None else Path(output_dir) / 'reconciliation_evidence'
        _OBSERVED_CONFIDENCE = contract
        _RUN_ID = run_id
        _REGISTRY.clear()
        _ADOPTED_KEYS.clear()
        if enabled and _OUTPUT_DIR is not None:
            manifest_path = _OUTPUT_DIR / 'manifest.json'
            if adopt_existing and manifest_path.is_file():
                _REGISTRY.update(_read_adopted_confidence(_OUTPUT_DIR, run_id, contract))
                _ADOPTED_KEYS.update(_REGISTRY)
            else:
                # Ordinary scalar runs begin a new collection. Shared inference
                # consumers explicitly adopt only their matching branch catalogue.
                manifest_path.unlink(missing_ok=True)
        if enabled and defer_projection and background_publication:
            from .confidence_publication import ConfidencePublicationQueue
            _PUBLICATION = ConfidencePublicationQueue()
            _PUBLICATION._automatic_terminal_bounds = True
    from .confidence_tiles import reset_tile_confidence
    reset_tile_confidence()


def drain_confidence_publication():
    """Join all compressed publications before reconciliation or run completion."""
    with _LOCK:
        publisher, captures = _PUBLICATION, _CAPTURE_QUEUE
    if captures is not None:
        captures.drain()
    result = {} if publisher is None else publisher.drain()
    _flush_confidence_manifest()
    return result


def confidence_publication_reserve_bytes():
    """Charge numeric stage pages and bounded copy buffers, including tmpfs scratch."""
    with _LOCK:
        return ((0 if _PUBLICATION is None else _PUBLICATION.workspace_reserve_bytes)
                + (0 if _CAPTURE_QUEUE is None else _CAPTURE_QUEUE.workspace_bytes))


def confidence_publication_snapshot():
    """Current numeric staging/backing state without retaining an old run owner."""
    with _LOCK:
        publisher, captures = _PUBLICATION, _CAPTURE_QUEUE
    result = dict(configured=False) if publisher is None else dict(configured=True, **publisher.snapshot())
    result['terminal_capture'] = None if captures is None else captures.snapshot()
    return result


def shutdown_confidence_publication(*, raise_errors=True):
    """Settle run-owned publishers and flush the manifest after a strict, clean shutdown."""
    global _PUBLICATION, _CAPTURE_QUEUE, _MANIFEST_DIRTY
    with _LOCK:
        publisher, captures = _PUBLICATION, _CAPTURE_QUEUE
        _CAPTURE_QUEUE = None
    error = None
    try:
        if captures is not None:
            captures.close(raise_errors=raise_errors)
    except BaseException as exc:
        error = exc
    finally:
        # Readers may still submit compressed stages until their join finishes.
        with _LOCK:
            _PUBLICATION = None
        try:
            if publisher is not None:
                publisher.close(raise_errors=raise_errors)
        except BaseException as exc:
            if error is None:
                error = exc
    if raise_errors and error is None:
        _flush_confidence_manifest()
    else:
        with _LOCK:
            # A later shutdown during reconfiguration cannot verify this run's queues.
            _MANIFEST_DIRTY = False
    if error is not None:
        raise error


def configure_terminal_confidence_capture(*, slots, workspace_bytes, source_bytes):
    """Install a startup-funded queue before any parent/publication is submitted."""
    global _CAPTURE_QUEUE, _PUBLICATION
    from .confidence_orchestration import ConfidenceCaptureQueue
    from .confidence_publication import ConfidencePublicationQueue
    with _LOCK:
        if not (_ENABLED and _DEFER_PROJECTION and _PUBLICATION is not None):
            return False
        if _CAPTURE_QUEUE is not None or _PUBLICATION.snapshot()['submitted']:
            raise RuntimeError('Confidence capture funding must precede source admission')
        if (not getattr(_PUBLICATION, '_automatic_terminal_bounds', False)
                or _PUBLICATION.max_pending_bytes != 256*1024**2
                or _PUBLICATION.stage_bytes != 64*1024**2):
            # An explicitly supplied queue keeps its original numeric limits.
            return False
        previous = _PUBLICATION
        captures = ConfidenceCaptureQueue(slots=slots, workspace_bytes=workspace_bytes,
                                         source_bytes=source_bytes)
        try:
            publisher = ConfidencePublicationQueue(max_pending_bytes=int(slots)*64*1024**2)
        except BaseException:
            captures.close(raise_errors=False)
            raise
        _PUBLICATION, _CAPTURE_QUEUE = publisher, captures
    previous.close()
    return _CAPTURE_QUEUE


def terminal_confidence_capture_available():
    """Whether score preparation is enabled and an async queue is funded."""
    from .workspace import _env_flag
    if not _env_flag('YOLO_TTA_TERMINAL_SCORE_ONLY', True):
        return False
    with _LOCK:
        return bool(_ENABLED and _DEFER_PROJECTION and _CAPTURE_QUEUE is not None)


def try_capture_prediction_confidence(mask, scores, *, retire, **kwargs):
    """Transfer immutable terminal sources only after bounded queue acceptance."""
    with _LOCK:
        queue = _CAPTURE_QUEUE
    plan = current_confidence_capture_plan()
    if queue is None or plan is None or scores is None or not confidence_projection_deferred():
        return None
    from .confidence_orchestration import RetiringConfidenceCapture
    task = RetiringConfidenceCapture(mask, scores, plan=plan,
        capture=capture_prediction_confidence, retire=retire, kwargs=kwargs)
    return queue.try_submit(task)


@dataclass(frozen=True)
class TerminalConfidenceScores:
    """One full score backing; the caller has transferred its sole writer."""
    scores: np.ndarray
    source_bytes: int
    reader: _MaskedNativeScoreReader | None = None


def _terminal_confidence_backing_identity(array, *, require_full=False):
    """Prove a heap root or registered memfd, without trusting mapping names."""
    if type(array) not in (np.ndarray, np.memmap):
        return None
    from .runtime import _root_memmap_for_array, capture_memfd_owner_proofs
    root = _root_memmap_for_array(array)
    if root is not None:
        if type(root) is not np.memmap or (require_full and array is not root):
            return None
        proofs = capture_memfd_owner_proofs({'source': root})
        if len(proofs) != 1:
            return None
        # The retained FD proof validates the full extent. Identity is device
        # and inode, even when two mappings use different addresses or keys.
        return ('mapped', int(proofs[0][2]), int(proofs[0][3]))
    root = array
    while type(root) is np.ndarray and root.base is not None:
        root = root.base
    if (type(root) is not np.ndarray or root.base is not None
            or not root.flags.owndata or (require_full and array is not root)):
        return None
    return ('heap', id(root))


def prepare_terminal_confidence_scores(mask, scores, *, known_slice_any=None,
                                       known_slice_bboxes=None):
    """Replace a terminal score source with its exact masked product in place.

    The terminal caller must already own the only score writer, after detector
    cleanup. Unsupported ownership returns None before any mutation. The full
    grid is masked when support metadata is missing. Otherwise one frozen reader
    owns the exact support domain through capture and every publication retry.
    """
    if not confidence_projection_deferred():
        return None
    plan = current_confidence_capture_plan()
    if (not isinstance(plan, NativeConfidenceCapturePlan)
            or type(scores) not in (np.ndarray, np.memmap)
            or type(mask) not in (np.ndarray, np.memmap)
            or scores.dtype != np.uint8 or scores.ndim != 3
            or mask.dtype not in (np.dtype(np.uint8), np.dtype(bool))
            or mask.shape != scores.shape or not mask.flags.c_contiguous
            or not scores.flags.c_contiguous or not scores.flags.writeable
            or plan.shape != tuple(scores.shape) or plan.consumer_bytes < 1):
        return None
    if plan != plan_confidence_capture(plan.shape, plan.requested_workers,
            workspace_bytes=plan.workspace_limit_bytes, block_size=plan.block_size):
        return None
    score_identity = _terminal_confidence_backing_identity(scores, require_full=True)
    mask_identity = _terminal_confidence_backing_identity(mask)
    if score_identity is None or mask_identity is None or score_identity == mask_identity:
        return None
    for metadata in (known_slice_any, known_slice_bboxes):
        if metadata is None:
            continue
        if (type(metadata) not in (np.ndarray, np.memmap)
                or metadata.dtype.kind not in 'biu'):
            return None
        metadata_identity = _terminal_confidence_backing_identity(metadata)
        if metadata_identity is None or metadata_identity in (mask_identity, score_identity):
            # Capture kwargs outlive this handoff. A mask alias would keep its
            # backing alive; a score alias would change support during masking.
            # Unknown/container metadata keeps the existing two-source path.
            return None
    started = time.perf_counter()
    reader = None
    if known_slice_any is not None and known_slice_bboxes is not None:
        # Validate and snapshot once, before the first source write. The plan's
        # initialization/fixed allowance already funds these immutable arrays.
        reader = _MaskedNativeScoreReader(mask, scores, known_slice_any, known_slice_bboxes)
    prepared = TerminalConfidenceScores(scores, int(scores.nbytes), reader)
    from .confidence_capture_cpu import zero_unmasked_scores_boxes, zero_unmasked_scores_flat
    threads = _terminal_score_prepare_threads()
    # From the first write onward, errors propagate to the parent's ownership
    # guard. A partially transformed source must never trigger a legacy retry.
    # Compiled nogil kernels need no scratch and write disjoint ranges.
    if reader is None:
        flat_mask, flat_scores = mask.reshape(-1), scores.reshape(-1)
        _run_terminal_score_masking(zero_unmasked_scores_flat, (flat_mask, flat_scores),
                                    int(scores.size), threads)
        del flat_mask, flat_scores
        visited = int(scores.size)
    else:
        active, boxes = reader.active, reader.boxes
        _run_terminal_score_masking(zero_unmasked_scores_boxes, (mask, scores, active, boxes),
                                    int(scores.shape[0]), threads)
        del active, boxes
        # All later reads are restricted by this same immutable support snapshot.
        # No original mask alias or alternate full-plane reader may survive.
        reader.mask = reader.scores
        visited = int(reader.capture_metrics['bounded_input_bytes']) // 2
    scores.setflags(write=False)
    prepare_seconds = time.perf_counter() - started
    from .runtime import runtime_telemetry
    telemetry = runtime_telemetry()
    telemetry.add('confidence.terminal_score_only.parents', 1)
    telemetry.add('confidence.terminal_score_only.full_score_bytes', int(scores.nbytes))
    telemetry.add('confidence.terminal_score_only.visited_score_bytes', visited)
    telemetry.add('confidence.terminal_score_only.prepare_seconds', prepare_seconds)
    telemetry.add('confidence.terminal_score_only.prepare_threads', threads)
    return prepared


def _terminal_score_prepare_threads():
    """Threads for terminal score masking (YOLO_TTA_TERMINAL_SCORE_PREPARE_THREADS).

    The default is one compiled nogil call: it needs the GIL once. Each extra
    thread must win the GIL to start and finish, which costs more than it saves
    while the main process's other threads are busy.
    """
    try:
        requested = int(os.environ.get('YOLO_TTA_TERMINAL_SCORE_PREPARE_THREADS', '1'))
    except ValueError:
        requested = 1
    return max(1, min(64, requested))


def _run_terminal_score_masking(kernel, arrays, count, threads):
    """Run kernel(*arrays, first, stop) over [0, count); return after every piece ends.

    Arrays travel as arguments, not closures, so a failure's cleared traceback
    frames release every dense alias.
    """
    pieces = max(1, min(count, threads))  # one piece per thread: fewest GIL handoffs
    bounds = [count * piece // pieces for piece in range(pieces + 1)]
    ranges = [(first, stop) for first, stop in zip(bounds, bounds[1:]) if stop > first]
    if threads <= 1 or len(ranges) <= 1:
        for first, stop in ranges:
            kernel(*arrays, first, stop)
        return
    from concurrent.futures import ThreadPoolExecutor
    # Leaving the block waits for every piece, so no write outlives a failure.
    with ThreadPoolExecutor(max_workers=min(threads, len(ranges)),
                            thread_name_prefix='terminal-score-mask') as pool:
        futures = [pool.submit(kernel, *arrays, first, stop) for first, stop in ranges]
        errors = [future.exception() for future in futures]
    for error in errors:
        if error is not None:
            raise error


def _capture_prepared_terminal_confidence_scores(mask, scores, *, prepared, **kwargs):
    """Module-level queue adapter; no closure can keep a completed source alive."""
    try:
        if mask is not prepared.scores or scores is not prepared.scores:
            raise ValueError('Terminal confidence capture changed its prepared score source')
        return capture_terminal_confidence_scores(prepared, **kwargs)
    finally:
        mask = scores = prepared = None
        kwargs.clear()


def _terminal_confidence_reader_plan(prepared):
    plan = current_confidence_capture_plan()
    if (prepared.reader is not None
            and (plan is None or plan != prepared.reader.capture_plan)):
        raise ValueError('Prepared terminal confidence reader requires its original capture plan')
    return plan


def try_capture_terminal_confidence_scores(prepared, *, retire, **kwargs):
    """Admit a score-only reader with an explicit single-backing source charge."""
    if not isinstance(prepared, TerminalConfidenceScores):
        raise TypeError('Terminal confidence capture requires prepared scores')
    task = task_kwargs = None
    try:
        plan = _terminal_confidence_reader_plan(prepared)
        with _LOCK:
            queue = _CAPTURE_QUEUE
        if queue is None or plan is None or not confidence_projection_deferred():
            return None
        if prepared.reader is not None:
            kwargs.pop('known_slice_any', None)
            kwargs.pop('known_slice_bboxes', None)
        from .confidence_orchestration import RetiringConfidenceCapture
        # The task owns its own dictionary. This caller frame can then release
        # every carrier reference even if submission raises after enqueueing.
        task_kwargs = dict(kwargs, prepared=prepared)
        task = RetiringConfidenceCapture(prepared.scores, prepared.scores, plan=plan,
            capture=_capture_prepared_terminal_confidence_scores, retire=retire,
            kwargs=task_kwargs, score_only_source=True)
        return queue.try_submit(task)
    finally:
        prepared = task = task_kwargs = None
        kwargs.clear()


def capture_terminal_confidence_scores(prepared, **kwargs):
    """Keep the same masked score product when asynchronous admission declines."""
    if not isinstance(prepared, TerminalConfidenceScores):
        raise TypeError('Terminal confidence capture requires prepared scores')
    reader = None
    try:
        _terminal_confidence_reader_plan(prepared)
        if not confidence_projection_deferred():
            raise RuntimeError('Prepared terminal confidence scores require deferred projection')
        reader = prepared.reader
        if reader is None:
            return capture_prediction_confidence(prepared.scores, prepared.scores, **kwargs)
        kwargs.pop('known_slice_any', None)
        kwargs.pop('known_slice_bboxes', None)
        return publish_confidence_scores(prepared.scores, native_reader=reader, **kwargs)
    except BaseException as error:
        # Match the asynchronous owner: a retained synchronous failed Future
        # must not keep finished nested reader/writer frames' source arguments.
        import traceback
        traceback.clear_frames(error.__traceback__)
        raise
    finally:
        prepared = reader = None
        kwargs.clear()


def configure_confidence_evidence_worker(*, enabled=False, min_conf=0.0):
    """Configure score-only inference semantics in an independent worker."""
    global _ENABLED
    _ENABLED = bool(enabled)


def confidence_evidence_enabled():
    return bool(_ENABLED)


def confidence_projection_deferred():
    return bool(_ENABLED and _DEFER_PROJECTION)


def _confidence_provenance(provenance):
    result = dict(provenance or {})
    with _LOCK:
        run_id = _RUN_ID
    if run_id is not None:
        result.setdefault('run_id', run_id)
    if 'observed_confidence' not in result:
        from .confidence_retention import observed_confidence_contract
        with _LOCK:
            contract = _OBSERVED_CONFIDENCE
        result['observed_confidence'] = contract or observed_confidence_contract()
    return result


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return {str(k): _json_value(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


def _write_json_atomic(path, value):
    write_json_atomic(path, _json_value(value), sort_keys=True, trailing_newline=True)


def _read_adopted_confidence(output, run_id, contract):
    catalogue = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    if (catalogue.get('schema') != SCHEMA or not run_id or catalogue.get('run_id') != run_id
            or catalogue.get('purpose') != 'observed_prediction_confidence'
            or catalogue.get('observed_confidence') != contract
            or not isinstance(catalogue.get('layers'), list)):
        raise ValueError('Prepublished confidence differs from the scalar branch identity or score contract')
    references = {}
    for row in catalogue['layers']:
        directory = (output / row['directory']).resolve()
        if not directory.is_relative_to(output.resolve()):
            raise ValueError('Prepublished confidence directory escapes its scalar branch')
        ref = ConfidenceEvidenceRef.open(directory)
        token = (ref.model_name, ref.layer_key)
        if (token in references or token != (str(row['model_name']), str(row['layer_key']))
                or ref.shape != tuple(row['output_shape_tyx'])
                or ref.storage_shape != tuple(row['stored_shape_tyx'])
                or ref.coordinate_space != row['coordinate_space']
                or ref.metadata.get('provenance', {}).get('run_id') != run_id
                or ref.metadata['provenance'].get('observed_confidence') != contract):
            raise ValueError('Prepublished confidence layer differs from its branch catalogue')
        references[token] = ref
    return references


def _adopted_confidence_reference(view, model_name, *, native_shape=None,
                                  source='fullframe', tile_config_id='', tile_acceptance='',
                                  stage='pre_interpolation', layer_key=None):
    from .cuda_d1 import _nrrd_layer_key
    key = str(layer_key or _nrrd_layer_key(view_name=view.name, source=source,
        mask_kind='yolo', pass_index=0, tile_config_id=tile_config_id,
        tile_acceptance=tile_acceptance, stage=stage))
    token = (str(model_name), key)
    with _LOCK:
        reference = _REGISTRY.get(token) if token in _ADOPTED_KEYS else None
    if (reference is not None and native_shape is not None
            and reference.storage_shape != tuple(native_shape)):
        raise ValueError('Adopted confidence differs from the native prediction grid')
    return reference


@dataclass(frozen=True)
class ConfidenceEvidenceRef:
    path: Path
    shape: tuple[int, int, int]
    layer_key: str
    model_name: str
    metadata: dict

    def reader(self):
        return ConfidenceEvidenceReader(self)

    def read(self, z0, z1):
        with self.reader() as reader:
            return reader(z0, z1)

    @property
    def coordinate_space(self):
        raw = self.metadata.get('coordinate_space', self.metadata.get('provenance',{}).get('coordinate_space','source'))
        return 'native_view_processing' if raw in ('native_view_processing','tile_native_processing') else raw

    @property
    def storage_shape(self):
        return tuple(map(int,self.metadata.get('stored_shape_tyx') or self.metadata['output_shape_tyx']))

    @property
    def source_shape(self):
        value = self.metadata.get('source_shape_tyx')
        if value is None and self.coordinate_space == 'source':
            value = self.metadata.get('output_shape_tyx')
        return None if value is None else tuple(map(int,value))

    def native_reader(self):
        """Explicit access in the declared payload grid, never source projection."""
        return ConfidenceEvidenceReader(self, native=True)

    def source_reader(self, workspace=None, *, memory_mib=512, max_staging_mib=32768):
        """Explicit, bounded one-layer conversion; native readers never do this implicitly."""
        from .confidence_native import source_reader
        return source_reader(self, workspace, memory_mib=memory_mib, max_staging_mib=max_staging_mib)

    @classmethod
    def open(cls, path):
        directory = Path(path)
        data = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
        from .confidence_storage import BLOCK_SCHEMA, checked_shape
        if data.get('schema') not in (SCHEMA,BLOCK_SCHEMA):
            raise ValueError('Unsupported confidence evidence schema')
        storage = checked_shape(data.get('stored_shape_tyx') or data['output_shape_tyx'])
        raw_shape = data.get('source_shape_tyx') or data.get('output_shape_tyx') or storage
        shape = checked_shape(raw_shape)
        if data.get('unknown') != 'score_zero':
            raise ValueError('Invalid confidence evidence geometry or support semantics')
        if data.get('dtype') != 'uint8' or data.get('score_semantics') not in (
                SCORE_SEMANTICS, _LEGACY_SCORE_SEMANTICS):
            raise ValueError('Invalid confidence evidence score semantics')
        result = cls(directory, shape, str(data['layer_key']), str(data['model_name']), data)
        if result.coordinate_space not in ('source','native_view_processing'):
            raise ValueError('Unsupported confidence evidence coordinate space')
        if result.coordinate_space=='source' and result.storage_shape!=result.shape:
            raise ValueError('Source confidence payload and source grids differ')
        return result


class ConfidenceEvidenceReader:
    """Bounded random-slice reader; no decompressed whole-volume cache."""
    def __init__(self, reference, *, native=False):
        self.reference = reference
        self._closed=False
        self._delegate = None
        if reference.coordinate_space != 'source' and not native:
            raise ValueError('Native confidence requires explicit source_reader(workspace, ...) conversion')
        self.shape = reference.storage_shape if native else reference.shape
        from .confidence_storage import BLOCK_SCHEMA, BLOCK_LAYOUT, PIECE_LAYOUT, BlockScoreReader
        if reference.metadata.get('schema') == BLOCK_SCHEMA:
            layout = reference.metadata.get('layout')
            if layout == BLOCK_LAYOUT:
                self._delegate = BlockScoreReader(reference)
            elif layout == PIECE_LAYOUT:
                from .confidence_native import NativePieceReader
                self._delegate = NativePieceReader(reference)
            else:
                raise ValueError('Unsupported confidence evidence storage layout')
            self.records = None
            return
        self.records = json.loads((reference.path / 'index.json').read_text(encoding='utf-8'))
        if len(self.records) != self.shape[0]:
            raise ValueError('Confidence evidence index depth differs from its output grid')
        payload_bytes = (reference.path / 'scores.u8.zlib').stat().st_size
        previous_end = 0
        for record in self.records:
            y0, y1, x0, x1, offset, length = map(int, record)
            if (not 0 <= y0 <= y1 <= self.shape[1] or not 0 <= x0 <= x1 <= self.shape[2]
                    or offset != previous_end or length < 0 or offset + length > payload_bytes
                    or (length == 0) != (y0 == y1 or x0 == x1)):
                raise ValueError('Malformed confidence evidence slice index')
            previous_end = offset + length
        if previous_end != payload_bytes:
            raise ValueError('Confidence payload has unindexed data')

    def __call__(self, z0, z1):
        if self._closed:
            raise RuntimeError('Confidence reader is closed')
        first, stop = int(z0), int(z1)
        if not 0 <= first <= stop <= self.shape[0]:
            raise IndexError('Confidence slab is outside its output grid')
        scores = np.zeros((stop - first, *self.shape[1:]), dtype=np.uint8)
        if self._delegate is not None:
            for z in range(first,stop):
                for y0,y1,x0,x1,crop in self.iter_crops(z):
                    target = scores[z-first,y0:y1,x0:x1]
                    np.maximum(target,crop,out=target)
            return scores,scores>0
        with (self.reference.path / 'scores.u8.zlib').open('rb') as stream:
            for z in range(first, stop):
                y0, y1, x0, x1, offset, length = map(int, self.records[z])
                if not length:
                    continue
                stream.seek(offset)
                compressed = stream.read(length)
                expected = (y1 - y0) * (x1 - x0)
                decoder = zlib.decompressobj()
                raw = decoder.decompress(compressed, expected + 1)
                if len(raw) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                    raise ValueError('Confidence score crop is truncated or malformed')
                scores[z - first, y0:y1, x0:x1] = np.frombuffer(raw, dtype=np.uint8).reshape(y1-y0, x1-x0)
        return scores, scores > 0

    def read_crop(self, z):
        """Return a known-support crop without allocating a native full plane."""
        if self._closed:
            raise RuntimeError('Confidence reader is closed')
        z = int(z)
        if not 0 <= z < self.shape[0]:
            raise IndexError('Confidence crop is outside its grid')
        if self._delegate is not None:
            from .confidence_storage import assembled_crop
            return assembled_crop(self.iter_crops(z))
        y0, y1, x0, x1, offset, length = map(int, self.records[z])
        if not length:
            return None
        with (self.reference.path / 'scores.u8.zlib').open('rb') as stream:
            stream.seek(offset)
            compressed = stream.read(length)
        expected = (y1-y0) * (x1-x0)
        decoder = zlib.decompressobj()
        raw = decoder.decompress(compressed, expected + 1)
        if len(raw) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError('Confidence score crop is truncated or malformed')
        return y0, y1, x0, x1, np.frombuffer(raw, dtype=np.uint8).reshape(y1-y0, x1-x0)

    def iter_crops(self,z):
        if self._closed:
            raise RuntimeError('Confidence reader is closed')
        if self._delegate is not None:
            yield from self._delegate.iter_crops(z)
        else:
            crop = self.read_crop(z)
            if crop is not None:
                yield crop

    def close(self):
        self._closed=True
        delegate = getattr(self,'_delegate',None)
        if delegate is not None:
            delegate.close()

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.close()

    def __del__(self):
        self.close()


def write_confidence_evidence(path, shape, slice_reader: Callable, *, layer_key, model_name,
                              provenance=None):
    """Write one source-space score sidecar using a single XY plane at a time."""
    provenance = _confidence_provenance(provenance)
    directory = Path(path)
    shape = tuple(map(int, shape))
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError('Confidence output grid must contain three positive dimensions')
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'metadata.json').exists():
        raise FileExistsError(f'Confidence evidence already exists: {directory}')
    payload = directory / 'scores.u8.zlib'
    temporary = directory / 'scores.u8.zlib.partial'
    records, digest, known_count = [], hashlib.sha256(), 0
    z_first, z_stop = map(int, getattr(slice_reader, 'known_z_bounds', (0, shape[0])))
    if not 0 <= z_first <= z_stop <= shape[0]:
        raise ValueError('Confidence known-support bounds are outside the output grid')
    try:
        with temporary.open('wb') as stream:
            for z in range(shape[0]):
                if not z_first <= z < z_stop:
                    records.append([0, 0, 0, 0, stream.tell(), 0])
                    continue
                value = slice_reader(z)
                if value is None:
                    records.append([0, 0, 0, 0, stream.tell(), 0])
                    continue
                plane = np.asarray(value)
                if plane.dtype != np.uint8 or plane.shape != shape[1:]:
                    raise ValueError('Confidence slice reader returned an invalid shape or dtype')
                rows = np.flatnonzero(np.any(plane, axis=1))
                offset = stream.tell()
                if not len(rows):
                    records.append([0, 0, 0, 0, offset, 0])
                    continue
                y0, y1 = int(rows[0]), int(rows[-1]) + 1
                columns = np.flatnonzero(np.any(plane[y0:y1], axis=0))
                x0, x1 = int(columns[0]), int(columns[-1]) + 1
                crop = np.ascontiguousarray(plane[y0:y1, x0:x1])
                known_count += int(np.count_nonzero(crop))
                encoded = zlib.compress(crop.tobytes(), level=3)
                stream.write(encoded)
                digest.update(encoded)
                records.append([y0, y1, x0, x1, offset, len(encoded)])
        temporary.replace(payload)
        _write_json_atomic(directory / 'index.json', records)
        metadata = dict(
            schema=SCHEMA, layer_key=str(layer_key), model_name=str(model_name),
            output_shape_tyx=list(shape), exported_axes='(X, Y, t)', dtype='uint8',
            score_semantics=SCORE_SEMANTICS, unknown='score_zero',
            quantization='round-half-even(clip(prediction_score,0,1)*255); quantized zero is unknown',
            projection='maximum over categorical source-address support',
            payload='scores.u8.zlib', index='index.json', payload_sha256=digest.hexdigest(),
            known_voxels=known_count, provenance=_json_value(provenance or {}))
        _write_json_atomic(directory / 'metadata.json', metadata)
        return ConfidenceEvidenceRef(directory, shape, str(layer_key), str(model_name), metadata)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_block_confidence_evidence(path, shape, slice_reader: Callable, *, layer_key, model_name,
                                    provenance=None, coordinate_space=None, source_shape_tyx=None,
                                    block_size=128, metrics=None, max_numeric_bytes=None):
    """Write schema2 numeric blocks without a whole-plane bounding rectangle."""
    from .confidence_storage import write_blocks
    provenance = _confidence_provenance(provenance)
    space = coordinate_space or provenance.get('coordinate_space','source')
    if space == 'tile_native_processing':
        space = 'native_view_processing'
    write_blocks(path,shape,slice_reader,layer_key=layer_key,model_name=model_name,
        provenance=provenance,coordinate_space=space,source_shape=source_shape_tyx,block_size=block_size,
        metrics=metrics,max_numeric_bytes=max_numeric_bytes)
    return ConfidenceEvidenceRef.open(path)


def _write_confidence_manifest(output, run_id, observed_confidence, references):
    _write_json_atomic(output/'manifest.json',dict(schema=SCHEMA, run_id=run_id,
        purpose='observed_prediction_confidence', observed_confidence=observed_confidence, layers=[
            dict(model_name=m,layer_key=k,directory=ref.path.name,output_shape_tyx=list(ref.shape),
                 storage_schema=ref.metadata['schema'],coordinate_space=ref.coordinate_space,
                 stored_shape_tyx=list(ref.storage_shape),score_semantics=ref.metadata['score_semantics'],
                 observed_confidence=ref.metadata.get('provenance', {}).get('observed_confidence'),
                 unknown='score_zero') for (m,k),ref in sorted(references.items())]))


def _flush_confidence_manifest():
    global _MANIFEST_DIRTY
    with _MANIFEST_WRITE_LOCK:
        with _LOCK:
            if not _MANIFEST_DIRTY:
                return
            snapshot = (_OUTPUT_DIR, _RUN_ID, _OBSERVED_CONFIDENCE, dict(_REGISTRY))
            _MANIFEST_DIRTY = False
        try:
            _write_confidence_manifest(*snapshot)
        except BaseException:
            with _LOCK:
                _MANIFEST_DIRTY = True
            raise


def _register_confidence_reference(reference, *, metrics=None):
    global _MANIFEST_DIRTY
    started = time.perf_counter()
    token = (reference.model_name, reference.layer_key)
    with _LOCK:
        batched = _BATCH_MANIFEST
        if batched:
            acquired = time.perf_counter()
            if _OUTPUT_DIR is None:
                raise RuntimeError('Confidence evidence has no configured output directory')
            if token in _REGISTRY:
                raise RuntimeError(f'Duplicate confidence evidence for {token!r}')
            _REGISTRY[token] = reference
            _MANIFEST_DIRTY = True
    if not batched:
        with _MANIFEST_WRITE_LOCK:
            acquired = time.perf_counter()
            with _LOCK:
                if _OUTPUT_DIR is None:
                    raise RuntimeError('Confidence evidence has no configured output directory')
                if token in _REGISTRY:
                    raise RuntimeError(f'Duplicate confidence evidence for {token!r}')
                snapshot = (_OUTPUT_DIR, _RUN_ID, _OBSERVED_CONFIDENCE, {**_REGISTRY, token: reference})
            _write_confidence_manifest(*snapshot)
            with _LOCK:
                _REGISTRY[token] = reference
    if metrics is not None:
        metrics.update(registry_wait_seconds=acquired-started,
                       registry_write_seconds=time.perf_counter()-acquired)
    return reference


def _confidence_destination(model_name,layer_key):
    with _LOCK:
        output = _OUTPUT_DIR
    if output is None:
        raise RuntimeError('Confidence evidence publication has no configured run output directory')
    identity = hashlib.sha256((str(model_name)+'\0'+str(layer_key)).encode()).hexdigest()
    return output/identity


def publish_confidence_scores(scores, *, view, model_name, temp_dir, output_shape=None,
                               source='fullframe', tile_config_id='', tile_acceptance='',
                               stage='pre_interpolation', layer_key=None, native_reader=None):
    """Project immutable native-view scores and persist one prediction companion."""
    from .assembly import final_source_output_shape
    from .confidence_projection import read_score_slice_in_output_shape, score_projection_reader
    from .cuda_d1 import _nrrd_layer_key
    if not confidence_evidence_enabled():
        return None
    adopted = _adopted_confidence_reference(view, model_name,
        native_shape=getattr(scores, 'shape', None), source=source,
        tile_config_id=tile_config_id, tile_acceptance=tile_acceptance,
        stage=stage, layer_key=layer_key)
    if adopted is not None:
        return adopted
    with _LOCK:
        output = _OUTPUT_DIR
        publisher = _PUBLICATION
    if output is None:
        raise RuntimeError('Confidence evidence publication has no configured run output directory')
    key = str(layer_key or _nrrd_layer_key(
        view_name=view.name, source=source, mask_kind='yolo', pass_index=0,
        tile_config_id=tile_config_id, tile_acceptance=tile_acceptance, stage=stage))
    identity = hashlib.sha256((str(model_name) + '\0' + key).encode('utf-8')).hexdigest()
    shape = tuple(output_shape or final_source_output_shape() or
                  (int(view.full_t), int(view.full_h), int(view.full_w)))
    work = Path(temp_dir) / 'confidence_evidence' / identity
    provenance = dict(view=_json_value(view), processing_shape_tyx=list(scores.shape),
                      source=source, mask_kind='yolo', stage=stage,
                      tile_config_id=tile_config_id, tile_acceptance=tile_acceptance,
                      confidence_scope='surviving observed prediction support before interpolation')
    started = time.perf_counter()
    operation = 'native retention' if confidence_projection_deferred() else 'source projection'
    print(f'Confidence {operation} start {model_name}/{view.name}: '
          f'native_shape={tuple(scores.shape)}, source_shape={shape}.', flush=True)
    metrics = {}
    read_native = native_reader if native_reader is not None else lambda z:scores[z]
    if confidence_projection_deferred():
        try:
            def write_native(destination, limit=None):
                result = write_block_confidence_evidence(destination,scores.shape,read_native,
                    layer_key=key,model_name=model_name,provenance=provenance,
                    coordinate_space='native_view_processing',source_shape_tyx=shape,
                    metrics=metrics,max_numeric_bytes=limit)
                # Per-frame execution diagnostics from the writer take precedence
                # over the reader's initialization/eligibility description.
                for field, value in getattr(read_native, 'capture_metrics', {}).items():
                    metrics.setdefault(field, value)
                return result
            if publisher is not None:
                from .confidence_publication import copy_staged_blocks
                # The background closure owns only compressed files and scalar metadata.
                # Neither the original score nor prediction workspace crosses this boundary.
                def publish_staged(path, queue_wait):
                    reference = copy_staged_blocks(path, output/identity, metrics=metrics)
                    result = _register_confidence_reference(reference, metrics=metrics)
                    print(f'Confidence native publication complete {model_name}/{view.name}: '
                          f'queue_wait_s={queue_wait:.3f}, metrics={json.dumps(metrics,sort_keys=True)}.', flush=True)
                    return result
                future = publisher.stage_and_submit(Path(temp_dir)/'confidence_publication',
                    write_native, publish_staged, metrics=metrics)
                if future is not None:
                    print(f'Confidence native retention staged {model_name}/{view.name}: '
                          f'elapsed_s={time.perf_counter()-started:.3f}, '
                          f'numeric_bytes={metrics["payload_bytes"]+metrics["index_bytes"]}.', flush=True)
                    return future
                print(f'Confidence native retention {model_name}/{view.name}: '
                      'compressed stage exceeds its slot; using direct streaming.', flush=True)
            reference = write_native(output/identity)
        finally:
            # Staging/direct encoding has joined its readers. Failed Futures can
            # retain closure cells even after traceback frames are cleared; only
            # compressed files and scalar metadata may survive this boundary.
            scores = native_reader = read_native = None
    elif native_reader is not None:
        if not _can_stream_eager_planes(view):
            raise ValueError('Eager confidence plane streaming requires a transverse view')
        reference = write_block_confidence_evidence(output / identity, shape,
            lambda z: read_score_slice_in_output_shape(scores, shape, z, read_native=native_reader),
            layer_key=key, model_name=model_name, provenance=provenance, metrics=metrics)
    elif not any(bool(np.any(scores[index])) for index in range(scores.shape[0])):
        reference = write_block_confidence_evidence(output / identity, shape, lambda _z: None,
            layer_key=key, model_name=model_name, provenance=provenance,metrics=metrics)
    else:
        with score_projection_reader(scores, view, shape, work) as reader:
            reference = write_block_confidence_evidence(output / identity, shape, reader,
                layer_key=key, model_name=model_name, provenance=provenance,metrics=metrics)
    result = _register_confidence_reference(reference,metrics=metrics)
    print(f'Confidence {operation} complete {model_name}/{view.name}: '
          f'elapsed_s={time.perf_counter()-started:.3f}, metrics={json.dumps(metrics,sort_keys=True)}.', flush=True)
    return result


class _MaskedNativeScoreReader:
    """Read observed scores only inside trusted, pre-interpolation mask bounds."""
    def __init__(self, mask, scores, known_slice_any, known_slice_bboxes, *, stop=None):
        self.stop = stop
        if stop is not None and stop.is_set():
            raise RuntimeError('Confidence capture cancelled')
        self.mask, self.scores = mask, scores
        typed = (isinstance(mask, np.ndarray) and isinstance(scores, np.ndarray)
                 and mask.dtype in (np.dtype(np.uint8), np.dtype(bool))
                 and scores.dtype == np.uint8)
        if typed and (scores.ndim != 3 or mask.shape != scores.shape):
            raise ValueError('Compiled confidence capture mask/score geometry differs')
        plan = current_confidence_capture_plan() if typed else None
        if plan is not None and plan.shape != tuple(scores.shape):
            raise ValueError('Confidence capture plan differs from retained source geometry')
        if typed and plan is None:
            # Admit snapshot/control ownership before making independent
            # active/bbox arrays. The writer resolves its actual grid later.
            plan_confidence_capture(scores.shape, 1)
        if typed:
            count = scores.shape[0]
            if isinstance(known_slice_any, np.ndarray):
                if (known_slice_any.shape != (count,)
                        or not (known_slice_any.dtype == np.bool_
                                or np.issubdtype(known_slice_any.dtype, np.integer))):
                    raise ValueError('Confidence active metadata has invalid shape or scalar dtype')
            elif isinstance(known_slice_any, (list, tuple)):
                if len(known_slice_any) != count or any(
                        not isinstance(value, (bool, int, np.integer, np.bool_))
                        or not -2**63 <= int(value) <= 2**64-1 for value in known_slice_any):
                    raise ValueError('Confidence active metadata needs bounded integer scalars')
            else:
                raise ValueError('Confidence active metadata must be an array or integer sequence')
            if isinstance(known_slice_bboxes, np.ndarray):
                if (known_slice_bboxes.shape != (count, 4)
                        or not np.issubdtype(known_slice_bboxes.dtype, np.integer)):
                    raise ValueError('Confidence bbox metadata has invalid shape or scalar dtype')
            elif isinstance(known_slice_bboxes, (list, tuple)):
                if len(known_slice_bboxes) != count:
                    raise ValueError('Confidence bbox metadata has invalid frame count')
                for row in known_slice_bboxes:
                    if (not isinstance(row, (list, tuple)) or len(row) != 4
                            or any(not isinstance(value, (int, np.integer))
                                   or not -2**63 <= int(value) <= 2**63-1 for value in row)):
                        raise ValueError('Confidence bbox metadata needs four bounded integer scalars')
            else:
                raise ValueError('Confidence bbox metadata must be an array or integer rows')
        active_values = np.asarray(known_slice_any)
        raw_boxes = np.asarray(known_slice_bboxes)
        if (active_values.shape != (scores.shape[0],) or raw_boxes.shape != (scores.shape[0],4)
                or not np.issubdtype(raw_boxes.dtype, np.integer)
                or (typed and not (np.issubdtype(active_values.dtype, np.integer)
                                   or active_values.dtype == np.bool_))):
            raise ValueError('Confidence support metadata has invalid shape or bounds dtype')
        self.active = np.array(active_values, dtype=bool, copy=True) if typed else np.asarray(active_values, dtype=bool)
        self.boxes = np.array(raw_boxes, dtype=np.int64, copy=True) if typed else raw_boxes.astype(np.int64, copy=False)
        boxes = self.boxes[self.active]
        if boxes.size and (np.any(boxes[:,0] < 0) or np.any(boxes[:,1] > scores.shape[1])
                or np.any(boxes[:,2] < 0) or np.any(boxes[:,3] > scores.shape[2])
                or np.any(boxes[:,0] >= boxes[:,1]) or np.any(boxes[:,2] >= boxes[:,3])):
            raise ValueError('Confidence support metadata exceeds its native grid')
        indices = np.flatnonzero(self.active)
        self.known_z_bounds = (int(indices[0]),int(indices[-1])+1) if indices.size else (0,0)
        self.capture_metrics = dict(dense_equivalent_input_bytes=2*int(scores.size),
            bounded_input_bytes=2*int(np.sum((boxes[:,1]-boxes[:,0])*(boxes[:,3]-boxes[:,2]))),
            empty_slices_skipped=int(scores.shape[0]-indices.size))
        self.capture_plan = None
        self.compiled_capture_capable = False
        if typed:
            self.mask, self.scores = mask.view(), scores.view()
            self.mask.setflags(write=False)
            self.scores.setflags(write=False)
            self.active.setflags(write=False)
            self.boxes.setflags(write=False)
            self.capture_plan = plan
            self.compiled_capture_capable = True

    def iter_crops(self, z):
        if self.active[z]:
            y0,y1,x0,x1 = map(int,self.boxes[z])
            values = np.where(self.mask[z,y0:y1,x0:x1] != 0,
                              self.scores[z,y0:y1,x0:x1],np.uint8(0))
            yield y0,y1,x0,x1,values

    def encode_frame(self, z, block_size):
        """Typed retiring inputs: fused bounds/count, bounded exact zlib cells."""
        from .confidence_capture_cpu import masked_cell_bounds, copy_masked_cell, copy_masked_cells_batch
        from .confidence_capture import COPY_BATCH_MAX_CELLS, COPY_BATCH_MAX_BYTES
        if self.stop is not None and self.stop.is_set():
            raise RuntimeError('Confidence capture cancelled')
        if not self.compiled_capture_capable:
            raise ValueError('Compiled confidence capture needs its admitted grid plan')
        if self.capture_plan is not None and int(block_size) != self.capture_plan.block_size:
            raise ValueError('Compiled confidence capture grid differs from its admitted task plan')
        if not 0 <= int(z) < self.scores.shape[0] or not 1 <= int(block_size) <= 65535:
            raise ValueError('Compiled confidence capture frame/grid is outside its source')
        if not self.active[z]:
            return [], dict(scan_seconds=0., compression_seconds=0., known_voxels=0)
        from .confidence_native_codec import native_confidence_codec
        codec, codec_receipt = native_confidence_codec()
        if codec is not None and not codec.supports(block_size):
            codec = None
            codec_receipt = dict(backend='python_zlib',codec_version=zlib.ZLIB_RUNTIME_VERSION,
                                 fallback_reason='unsupported_native_block_size')
        codec_metrics = dict(codec_backend=codec_receipt['backend'],
            codec_version=codec_receipt['codec_version'],
            codec_python_version=codec_receipt.get('python_zlib_version',zlib.ZLIB_RUNTIME_VERSION),
            codec_fallback_reason=codec_receipt.get('fallback_reason'))
        self.capture_metrics.update(codec_metrics)
        y0, y1, x0, x1 = map(int, self.boxes[z])
        started = time.perf_counter()
        bounds = masked_cell_bounds(self.mask, self.scores, z, y0, y1, x0, x1, block_size)
        scan_seconds = time.perf_counter() - started
        compression_seconds = 0.
        records = []
        cell_bytes = min(int(block_size), self.scores.shape[1]) * min(int(block_size), self.scores.shape[2])
        batch_cells = min(COPY_BATCH_MAX_CELLS, COPY_BATCH_MAX_BYTES // cell_bytes)
        copy_kernel_calls = copy_batch_peak_bytes = native_compression_batches = 0
        native_batch_seconds = 0.
        for first in range(0, len(bounds), max(1, batch_cells)):
            if self.stop is not None and self.stop.is_set():
                raise RuntimeError('Confidence capture cancelled')
            started = time.perf_counter()
            selected = bounds[first:first + max(1, batch_cells)]
            if codec is not None and batch_cells > 0:
                raw, offsets = codec.encode_batch(self.mask,self.scores,z,selected,block_size)
                elapsed = time.perf_counter() - started
                # Native copy+compression share one nogil entry. Keep their
                # combined time explicit rather than inventing a scan split.
                native_batch_seconds += elapsed
                compression_seconds += elapsed
                native_compression_batches += 1
                copy_batch_peak_bytes = max(copy_batch_peak_bytes, raw.nbytes + offsets.nbytes + 8)
            elif batch_cells > 0:
                raw, offsets = copy_masked_cells_batch(self.mask, self.scores, z, selected,
                                                       COPY_BATCH_MAX_BYTES)
                copy_batch_peak_bytes = max(copy_batch_peak_bytes, raw.nbytes + offsets.nbytes)
            else:
                # Large public grids retain the separately admitted single-cell
                # path; a 65535 block must never imply a 16-cell allocation.
                a, b, c, d, _ = map(int, selected[0])
                raw = copy_masked_cell(self.mask, self.scores, z, a, b, c, d).reshape(-1)
                offsets = (0, raw.size)
            copy_kernel_calls += 1
            if codec is None:
                scan_seconds += time.perf_counter() - started
            raw_view = memoryview(raw)
            try:
                for item, offset in enumerate(selected):
                    if self.stop is not None and self.stop.is_set():
                        raise RuntimeError('Confidence capture cancelled')
                    a, b, c, d, known = map(int, offset)
                    started = time.perf_counter()
                    cell_view = raw_view[int(offsets[item]):int(offsets[item + 1])]
                    try:
                        encoded = bytes(cell_view) if codec is not None else zlib.compress(cell_view, level=3)
                    finally:
                        # A failed encoder traceback can retain its arguments.
                        # Releasing even that view prevents it pinning a slab.
                        cell_view.release()
                        cell_view = None
                    if codec is None:
                        compression_seconds += time.perf_counter() - started
                    records.append((a, b, c, d, known, encoded))
            finally:
                # Do not let the last memoryview keep the previous slab alive
                # while the next batch allocates. Records own zlib bytes only.
                raw_view.release()
                raw_view = raw = offsets = None
        return records, dict(scan_seconds=scan_seconds,
                             compression_seconds=compression_seconds,
                              known_voxels=int(bounds[:, 4].sum()) if len(bounds) else 0,
                              copy_kernel_calls=copy_kernel_calls,
                              copy_batch_peak_bytes=copy_batch_peak_bytes,
                              native_compression_batches=native_compression_batches,
                              native_batch_seconds=native_batch_seconds,**codec_metrics)


def capture_prediction_confidence(mask, scores, *, view, model_name, temp_dir,
                                  known_slice_any=None, known_slice_bboxes=None, **kwargs):
    """Mask a retiring score workspace, then publish before it is closed/deleted."""
    if not confidence_evidence_enabled():
        return None
    adopted = _adopted_confidence_reference(view, model_name,
        native_shape=getattr(mask, 'shape', None), **{
            key: kwargs[key] for key in ('source', 'tile_config_id', 'tile_acceptance', 'stage', 'layer_key')
            if key in kwargs})
    if adopted is not None:
        return adopted
    if scores is None:
        raise RuntimeError(f'Missing retained confidence for {model_name}/{view.name}')
    values, retained = np.asarray(scores), np.asarray(mask)
    if values.dtype != np.uint8 or values.shape != retained.shape or values.ndim != 3:
        raise ValueError('Retained confidence and prediction mask geometry differ')
    if confidence_projection_deferred():
        # Fuse eligibility with compression's plane read; avoid rewriting the dense score map.
        reader = None
        try:
            reader = (lambda z:np.where(retained[z] != 0, values[z], np.uint8(0)))
            if known_slice_any is not None and known_slice_bboxes is not None:
                reader = _MaskedNativeScoreReader(retained,values,known_slice_any,known_slice_bboxes)
            return publish_confidence_scores(values, view=view, model_name=model_name, temp_dir=temp_dir,
                native_reader=reader, **kwargs)
        finally:
            # A failed nested writer can keep the lambda alive in a closure cycle
            # with GC disabled. Empty its captured arrays after its last read.
            reader = values = retained = None
    for index in range(values.shape[0]):
        values[index][np.asarray(retained[index]) == 0] = np.uint8(0)
    return publish_confidence_scores(values, view=view, model_name=model_name, temp_dir=temp_dir, **kwargs)


def lookup_confidence_evidence(layer, model_name=None):
    if hasattr(layer, 'key'):
        key = str(layer.key)
        model_name = str(layer.model_name)
    else:
        key = str(layer)
    with _LOCK:
        if model_name is not None:
            return _REGISTRY.get((str(model_name), key))
        matches = [ref for (model, stored_key), ref in _REGISTRY.items() if stored_key == key]
    if len(matches) > 1:
        raise ValueError('Confidence layer key requires a model identity')
    return matches[0] if matches else None


def publish_native_confidence_pieces(pieces, *, native_shape, view, model_name, output_shape=None,
                                     source='fullframe',tile_config_id='',tile_acceptance='',
                                     stage='pre_interpolation',layer_key=None,disjoint=False):
    """Persist native pieces and complete geometry without creating a dense map."""
    from .assembly import final_source_output_shape
    from .cuda_d1 import _nrrd_layer_key
    from .confidence_native import write_native_pieces, write_native_disjoint_leases
    adopted = _adopted_confidence_reference(view, model_name, native_shape=native_shape,
        source=source, tile_config_id=tile_config_id, tile_acceptance=tile_acceptance,
        stage=stage, layer_key=layer_key)
    if adopted is not None:
        return adopted
    key=str(layer_key or _nrrd_layer_key(view_name=view.name,source=source,mask_kind='yolo',
        pass_index=0,tile_config_id=tile_config_id,tile_acceptance=tile_acceptance,stage=stage))
    shape=tuple(output_shape or final_source_output_shape() or (view.full_t,view.full_h,view.full_w))
    provenance=_confidence_provenance(dict(view=_json_value(view),processing_shape_tyx=list(native_shape),source=source,
        mask_kind='yolo',stage=stage,tile_config_id=tile_config_id,tile_acceptance=tile_acceptance,
        confidence_scope='surviving observed prediction support before interpolation'))
    started = time.perf_counter()
    if disjoint:
        ref=write_native_disjoint_leases(_confidence_destination(model_name,key),pieces,native_shape=native_shape,
            source_shape=shape,layer_key=key,model_name=model_name,provenance=provenance)
    else:
        ref=write_native_pieces(_confidence_destination(model_name,key),pieces,native_shape=native_shape,
            source_shape=shape,layer_key=key,model_name=model_name,provenance=provenance,disjoint=False)
    metrics = dict(numeric_publication_seconds=time.perf_counter()-started)
    result = _register_confidence_reference(ref,metrics=metrics)
    print(f'Confidence native pieces complete {model_name}/{view.name}: '
          f'layout={ref.metadata["layout"]}, metrics={json.dumps(metrics,sort_keys=True)}.', flush=True)
    return result


def publish_confidence_shards(shards, *, view, model_name, temp_dir, output_shape=None):
    """Join exact disjoint D1 frame leases before source projection."""
    from .runtime import close_memmap_array_without_flush
    records = sorted(shards, key=lambda item: int(item['slice_start']))
    if not records:
        raise ValueError(f'No D1 confidence shards for {model_name}/{view.name}')
    shape = tuple(map(int, records[0]['view_shape_tyx']))
    if len(shape) != 3 or shape[0] != int(view.num_slices) or min(shape) <= 0:
        raise ValueError('D1 confidence native view geometry is invalid')
    expected_start = 0
    layer_key = str(records[0]['layer_key'])
    for record in records:
        count = int(record['slice_count'])
        if (str(record['model_name']) != str(model_name) or str(record['view_name']) != str(view.name)
                or tuple(map(int, record['view_shape_tyx'])) != shape
                or str(record['layer_key']) != layer_key
                or int(record['slice_start']) != expected_start or count <= 0
                or tuple(map(int, record['shape_tyx'])) != (count, *shape[1:])):
            raise ValueError('D1 confidence leases overlap, have gaps, or change identity/geometry')
        expected_start += count
    if expected_start != shape[0]:
        raise ValueError('D1 confidence leases do not cover the complete native view')
    metrics = {}
    for record in records:
        for name, value in record.get('capture_metrics', {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if name == 'host_reserved_bytes':
                    metrics['max_task_host_reserved_bytes'] = max(metrics.get('max_task_host_reserved_bytes',0),value)
                else:
                    metrics[name] = metrics.get(name, 0) + value
    if metrics:
        print(f'D1 confidence capture {model_name}/{view.name}: leases={len(records)}, '
              f'metrics={json.dumps(metrics,sort_keys=True)}.', flush=True)
    pieces = []
    for record in records:
        ref = ConfidenceEvidenceRef.open(record['path'])
        if (ref.storage_shape != tuple(map(int, record['shape_tyx']))
                or ref.model_name != str(model_name) or ref.layer_key != layer_key):
            raise ValueError('D1 confidence shard descriptor differs from its stored identity or grid')
        pieces.append(dict(reference=ref, offset_tyx=(int(record['slice_start']), 0, 0)))
    if confidence_projection_deferred():
        return publish_native_confidence_pieces(pieces,native_shape=shape,view=view,model_name=model_name,
            output_shape=output_shape,layer_key=layer_key,disjoint=True)
    if _can_stream_eager_planes(view):
        return _publish_eager_transverse_pieces(pieces, shape, view=view, model_name=model_name,
            temp_dir=temp_dir, output_shape=output_shape, layer_key=layer_key)
    identity = hashlib.sha256((str(model_name) + '\0' + str(view.name)).encode()).hexdigest()
    path = Path(temp_dir) / 'confidence_evidence' / identity / 'native_score_leases.u8.dat'
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = np.memmap(path, dtype=np.uint8, mode='w+', shape=shape)
    try:
        for piece in pieces:
            ref, (start, _, _) = piece['reference'], piece['offset_tyx']
            with ref.native_reader() as reader:
                for local in range(ref.storage_shape[0]):
                    for y0,y1,x0,x1,crop in reader.iter_crops(local):
                        merged[start + local, y0:y1, x0:x1] = crop
        return publish_confidence_scores(merged, view=view, model_name=model_name,
            temp_dir=temp_dir, output_shape=output_shape, layer_key=layer_key)
    finally:
        close_memmap_array_without_flush(merged, unlink_path=path)
        try:
            path.unlink(missing_ok=True)
        except PermissionError:
            pass  # The last mapping consumer completes deferred deletion.


def _can_stream_eager_planes(view):
    from .geometry import physical_view_name
    return view.family == 'orthogonal' and physical_view_name(view) == 'transverse'


def _publish_eager_transverse_pieces(pieces, shape, *, mask=None, **kwargs):
    from contextlib import closing
    from .confidence_native import NativePieceReader

    inputs = [(piece['reference'], piece['offset_tyx']) for piece in pieces]
    with closing(NativePieceReader(pieces=inputs, shape=shape)) as reader:
        def read(z):
            plane = None
            for y0, y1, x0, x1, crop in reader.iter_crops(z):
                if plane is None:
                    plane = np.zeros(shape[1:], np.uint8)
                dst = plane[y0:y1, x0:x1]
                np.maximum(dst, crop, out=dst)
            if plane is not None and mask is not None:
                plane[np.asarray(mask[z]) == 0] = np.uint8(0)
            return plane
        return publish_confidence_scores(reader, native_reader=read, **kwargs)


__all__ = [
    'ConfidenceEvidenceRef', 'ConfidenceEvidenceReader', 'configure_confidence_evidence',
    'configure_confidence_evidence_worker', 'confidence_evidence_enabled',
    'confidence_projection_deferred',
    'drain_confidence_publication', 'shutdown_confidence_publication',
    'confidence_publication_reserve_bytes', 'confidence_publication_snapshot',
    'capture_prediction_confidence',
    'TerminalConfidenceScores', 'prepare_terminal_confidence_scores',
    'try_capture_terminal_confidence_scores', 'capture_terminal_confidence_scores',
    'terminal_confidence_capture_available',
    'publish_confidence_scores', 'lookup_confidence_evidence', 'write_confidence_evidence',
    'publish_confidence_shards',
    'write_block_confidence_evidence','publish_native_confidence_pieces',
]
