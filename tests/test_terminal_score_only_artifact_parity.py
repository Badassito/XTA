"""Independent complete-artifact parity for terminal score-only confidence.

These are numerical/format tests, not performance measurements. Native cases
use an explicitly qualified local zlib when one is provided, never an automatic
search for an additional dependency.
"""
from __future__ import annotations

import ctypes
from contextlib import nullcontext
import hashlib
import json
import os
import weakref

import numpy as np
import pytest

from XTA import confidence_capture_cpu as cpu
from XTA import confidence_evidence as evidence
from XTA import confidence_native_codec as native
from XTA import confidence_storage as storage
from XTA import geometry
from XTA.confidence_capture import confidence_capture_resources, plan_confidence_capture


_ARTIFACTS = ('index.bin', 'scores.u8.zlib', 'metadata.json')
_QUALIFIED_CODECS = {}


@pytest.fixture(params=('python', 'native'))
def codec_backend(request, monkeypatch):
    if request.param == 'python':
        monkeypatch.setenv('YOLO_TTA_CONFIDENCE_NATIVE_CODEC', 'python')
        codec, receipt = native.native_confidence_codec()
        assert codec is None and receipt['backend'] == 'python_zlib'
        return request.param, None
    path = os.environ.get('XTA_TEST_NATIVE_ZLIB_LIBRARY')
    if path:
        if path not in _QUALIFIED_CODECS:
            codec = native.NativeConfidenceCodec(ctypes.CDLL(path), source='explicit_qualification')
            codec.warm()
            _QUALIFIED_CODECS[path] = codec
        codec = _QUALIFIED_CODECS[path]
        receipt = dict(backend='native_compress2', codec_version=codec.version,
                       python_zlib_version=__import__('zlib').ZLIB_RUNTIME_VERSION,
                       library_source='explicit_qualification', fallback_reason=None)
        monkeypatch.setattr(native, 'native_confidence_codec', lambda: (codec, receipt.copy()))
    else:
        codec, receipt = native.native_confidence_codec()
        if codec is None:
            pytest.skip('Native codec unavailable; an explicit qualification library is optional')
    assert receipt['backend'] == 'native_compress2'
    return request.param, codec


def _inputs(case):
    shape = (4, 137, 145)
    rng = np.random.default_rng(756204)
    mask = np.zeros(shape, np.uint8)
    scores = rng.integers(1, 256, shape, dtype=np.uint8)
    mask[1, 3:133, 5:140] = rng.choice(np.asarray([0, 1, 2, 17, 255], np.uint8), (130, 135))
    mask[2, 19:136, 7:144] = rng.choice(np.asarray([0, 1, 2, 17, 255], np.uint8), (117, 137))
    mask[3, 128:137, 128:145] = 255
    # A real binary foreground voxel may have quantized-zero confidence.
    mask[1, 17, 19] = 255
    scores[1, 17, 19] = 0
    scores[2, 29:31, 31:33] = 0
    if case == 'empty':
        mask[:] = 0
    elif case == 'all_zero_scores':
        scores[:] = 0
    elif case == 'dense_nonbinary':
        mask[:] = rng.choice(np.asarray([1, 2, 17, 255], np.uint8), shape)
    elif case == 'boolean_mask':
        mask = mask.astype(bool)
    active = np.any(mask != 0, axis=(1, 2))
    boxes = np.full((shape[0], 4), -1, np.int64)
    for z in np.flatnonzero(active):
        yy, xx = np.nonzero(mask[z])
        boxes[z] = (yy.min(), yy.max() + 1, xx.min(), xx.max() + 1)
    return mask, scores, active, boxes


def _capture(root, mask, scores, active, boxes, *, metadata=True,
             score_only=False, expect_prepared=True, view=None):
    if view is None:
        view = geometry.get_view_infos(*mask.shape, cartesian_views=('transverse',))[0]
    evidence.configure_confidence_evidence(root, enabled=True, defer_projection=True)
    plan = plan_confidence_capture(mask.shape, 2, workspace_bytes=32 * 1024**2)
    metrics = []
    original = storage.write_blocks

    def measured(*args, **kwargs):
        result = original(*args, **kwargs)
        metrics.append(dict(kwargs.get('metrics') or {}))
        return result

    from unittest.mock import patch
    try:
        with confidence_capture_resources(plan), patch.object(storage, 'write_blocks', measured):
            kwargs = (dict(known_slice_any=active, known_slice_bboxes=boxes) if metadata else {})
            prepared = None
            if score_only:
                prepared = evidence.prepare_terminal_confidence_scores(mask, scores,
                    **kwargs)
                assert (prepared is not None) is expect_prepared
                if prepared is not None:
                    assert prepared.scores is scores
                    assert prepared.source_bytes == scores.nbytes
                    assert not prepared.scores.flags.writeable
            if prepared is None:
                ref = evidence.capture_prediction_confidence(mask, scores, view=view, model_name='parity',
                    temp_dir=root / 'work', **kwargs)
            else:
                ref = evidence.capture_terminal_confidence_scores(prepared,
                    view=view, model_name='parity', temp_dir=root / 'work', **kwargs)
        assert ref is not None and len(metrics) == 1
        files = {name: (ref.path / name).read_bytes() for name in _ARTIFACTS}
        files['manifest.json'] = (root / 'reconciliation_evidence' / 'manifest.json').read_bytes()
        with ref.native_reader() as reader:
            decoded, known = reader(0, mask.shape[0])
        return files, decoded, known, metrics[0]
    finally:
        evidence.configure_confidence_evidence(None, enabled=False)


def _assert_artifacts(tmp_path, baseline, actual, *, route, backend):
    assert baseline[0].keys() == actual[0].keys()
    for name in baseline[0]:
        assert actual[0][name] == baseline[0][name], name
    np.testing.assert_array_equal(actual[1], baseline[1])
    np.testing.assert_array_equal(actual[2], baseline[2])
    receipt = dict(route=route, codec_backend=backend,
        files={name: dict(bytes=len(value), sha256=hashlib.sha256(value).hexdigest())
               for name, value in actual[0].items()},
        baseline_metrics=baseline[3], actual_metrics=actual[3])
    (tmp_path / 'parity-receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')


@pytest.mark.parametrize('case', ('ragged_nonbinary', 'empty', 'all_zero_scores',
                                  'dense_nonbinary', 'boolean_mask'))
@pytest.mark.parametrize('metadata', (True, False), ids=('trusted-bbox', 'full-plane-fallback'))
def test_score_only_publishes_identical_complete_artifacts(tmp_path, codec_backend, case, metadata):
    backend, codec = codec_backend
    mask, scores, active, boxes = _inputs(case)
    original_mask, original_scores = mask.copy(), scores.copy()
    expected = np.where(mask != 0, scores, np.uint8(0))
    baseline = _capture(tmp_path / 'baseline', mask.copy(), scores.copy(), active, boxes,
                        metadata=metadata)
    actual = _capture(tmp_path / 'score-only', mask, scores, active, boxes,
                      metadata=metadata, score_only=True)
    _assert_artifacts(tmp_path, baseline, actual, route='score-only', backend=backend)
    np.testing.assert_array_equal(actual[1], expected)
    np.testing.assert_array_equal(actual[2], expected != 0)
    np.testing.assert_array_equal(mask, original_mask)
    expected_source = original_scores.copy()
    if metadata:
        for z in np.flatnonzero(active):
            y0, y1, x0, x1 = boxes[z]
            expected_source[z, y0:y1, x0:x1] = expected[z, y0:y1, x0:x1]
    else:
        expected_source[:] = expected
    np.testing.assert_array_equal(scores, expected_source)
    # A prepared reader owns the exact bounded-support snapshot. Scores outside
    # it may stay positive, but must never appear in the stored evidence.
    if not metadata and np.any(mask == 0):
        if case != 'all_zero_scores':
            assert np.any(original_scores[mask == 0] != 0)
        assert not np.any(scores[mask == 0])
    if case not in ('empty', 'all_zero_scores'):
        assert mask[1, 17, 19] != 0 and scores[1, 17, 19] == 0
    if metadata and np.any(expected):
        assert cpu.masked_cell_bounds.nopython_signatures
        assert actual[3]['capture_backend'] == 'compiled_masked_cells_ordered_frames'
        if backend == 'native':
            assert codec._kernel.nopython_signatures
            assert codec._kernel.targetoptions['nogil']
            assert actual[3]['codec_backend'] == 'native_compress2'
            assert actual[3]['native_compression_batches'] > 0
        else:
            assert cpu.copy_masked_cells_batch.nopython_signatures
            assert actual[3]['codec_backend'] == 'python_zlib'


@pytest.mark.parametrize('unsupported', ('readonly', 'fortran', 'partial-root',
                                         'reversed-mask', 'shared-root', 'disk-memmap'))
def test_unsupported_ownership_keeps_original_route_and_artifacts(tmp_path, codec_backend, unsupported):
    backend, _ = codec_backend
    mask, scores, active, boxes = _inputs('ragged_nonbinary')
    mapping = None
    if unsupported == 'readonly':
        scores.setflags(write=False)
    elif unsupported == 'fortran':
        scores = np.asfortranarray(scores)
    elif unsupported == 'partial-root':
        root = np.empty((5, *scores.shape[1:]), np.uint8)
        root[:4] = scores
        scores = root[:4]
        assert scores.flags.c_contiguous and not scores.flags.owndata
    elif unsupported == 'reversed-mask':
        mask = mask[:, ::-1, :]
        active = np.any(mask != 0, axis=(1, 2))
        boxes = np.tile((0, mask.shape[1], 0, mask.shape[2]), (mask.shape[0], 1))
    elif unsupported == 'shared-root':
        mask = scores
        active = np.ones(mask.shape[0], bool)
        boxes = np.tile((0, mask.shape[1], 0, mask.shape[2]), (mask.shape[0], 1))
    elif unsupported == 'disk-memmap':
        mapping = np.memmap(tmp_path / 'original-score.u8', mode='w+', dtype=np.uint8,
                            shape=scores.shape)
        mapping[:] = scores
        scores = mapping
    original_mask, original_scores = mask.copy(), scores.copy()
    try:
        baseline = _capture(tmp_path / 'baseline', mask.copy(), scores.copy(), active, boxes)
        actual = _capture(tmp_path / 'fallback', mask, scores, active, boxes,
                          score_only=True, expect_prepared=False)
        _assert_artifacts(tmp_path, baseline, actual, route='unsupported-' + unsupported,
                          backend=backend)
        np.testing.assert_array_equal(mask, original_mask)
        np.testing.assert_array_equal(scores, original_scores)
    finally:
        if mapping is not None:
            mapping._mmap.close()


@pytest.mark.parametrize('family', ('tilted', 'azimuthal'))
def test_actual_terminal_view_metadata_and_manifest_parity(tmp_path, codec_backend, monkeypatch, family):
    """Actual eligible geometry metadata survives the changed source borrow."""
    from XTA.packed_tilted_projection import terminal_native_source_packing_eligible

    backend, codec = codec_backend
    source_shape = (5, 9, 13)
    if family == 'tilted':
        view = geometry._build_tilted_view_infos(*source_shape,
            tilt_views=('transverse',), tilt_angles=(31.,), tilt_directions=('vertical',))[0]
    else:
        view = geometry._build_azimuthal_view_info(*source_shape,
            base_view='transverse', azimuth_angle=37., azimuthal_native_raster=0,
            request_token='score-only-parity')
    shape = (view.num_slices, view.src_h, view.src_w)
    rng = np.random.default_rng(756137)
    mask = rng.choice(np.asarray([0, 1, 17, 255], np.uint8), shape)
    mask[0] = 0
    scores = rng.integers(1, 256, shape, dtype=np.uint8)
    mask[1, 1, 1], scores[1, 1, 1] = 255, 0
    before = mask.copy()
    expected = np.where(mask != 0, scores, np.uint8(0))
    active = np.any(mask != 0, axis=(1, 2))
    boxes = np.tile(np.asarray((0, shape[1], 0, shape[2]), np.int64), (shape[0], 1))
    monkeypatch.setenv('YOLO_TTA_PACKED_PARENT', '1')
    monkeypatch.setenv('YOLO_TTA_PACKED_PARENT_MIN_MIB', '0')
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_BACKEND', 'compiled')
    assert terminal_native_source_packing_eligible(view, native_bytes=mask.nbytes,
        retire_dense=True, component_layers=True, tiled=False, keep_temp=False,
        prepublished=False, interpolate=0, interpolation_passes=0, extrapolate=0)
    baseline = _capture(tmp_path / 'baseline', mask.copy(), scores.copy(), active, boxes, view=view)
    actual = _capture(tmp_path / 'score-only', mask, scores, active, boxes,
                      score_only=True, view=view)
    _assert_artifacts(tmp_path, baseline, actual, route='terminal-view-' + family, backend=backend)
    metadata = json.loads(actual[0]['metadata.json'])
    assert metadata['provenance']['view']['family'] == family
    assert metadata['source_shape_tyx'] == list(source_shape)
    assert metadata['stored_shape_tyx'] == list(shape)
    assert metadata['coordinate_space'] == 'native_view_processing'
    assert metadata['provenance']['view']['name'] == view.name
    np.testing.assert_array_equal(actual[1], expected)
    np.testing.assert_array_equal(mask, before)
    assert mask[1, 1, 1] == 255 and actual[1][1, 1, 1] == 0
    if backend == 'native':
        assert codec._kernel.nopython_signatures
        assert actual[3]['native_compression_batches'] > 0


@pytest.mark.parametrize('metadata_owner', ('scores', 'mask'))
def test_active_metadata_alias_declines_before_write_and_preserves_artifacts(
        tmp_path, codec_backend, metadata_owner):
    """Metadata must neither change during masking nor hide a dense mask borrow."""
    backend, _ = codec_backend
    if metadata_owner == 'scores':
        mask = np.asarray([[[0, 1, 1, 0], [0, 0, 0, 0]]], np.uint8)
        scores = np.asarray([[[200, 9, 11, 0], [0, 0, 0, 0]]], np.uint8)
        active = scores.reshape(-1)[:1]
    else:
        mask = np.ones((1, 2, 4), np.uint8)
        scores = np.full(mask.shape, 91, np.uint8)
        active = mask.reshape(-1)[:1]
    boxes = np.asarray([[0, 2, 0, 4]], np.int64)
    before_mask, before_scores, before_active = mask.copy(), scores.copy(), active.copy()
    baseline = _capture(tmp_path / 'baseline', mask.copy(), scores.copy(), active, boxes)
    actual = _capture(tmp_path / 'fallback', mask, scores, active, boxes,
                      score_only=True, expect_prepared=False)
    _assert_artifacts(tmp_path, baseline, actual,
                      route='metadata-alias-' + metadata_owner, backend=backend)
    np.testing.assert_array_equal(mask, before_mask)
    np.testing.assert_array_equal(scores, before_scores)
    np.testing.assert_array_equal(active, before_active)
    assert scores.flags.writeable  # Decline did not freeze or partially transform the source.
    if metadata_owner == 'scores':
        assert active[0] == 200
        assert actual[1][0, 0, 1] == 9 and actual[1][0, 0, 2] == 11
        assert np.count_nonzero(actual[2]) == 2
    else:
        # Prove why accepting a score-only charge would be incorrect: the
        # caller's active metadata itself retains M, independently of scores.
        mask_reference = weakref.ref(mask)
        del mask
        assert mask_reference() is not None
        del active
        assert mask_reference() is None


@pytest.mark.parametrize('handoff', ('sync-refusal', 'async', 'stage-retry'))
def test_v2_reuses_frozen_reader_after_metadata_changes_and_retries(
        tmp_path, codec_backend, monkeypatch, handoff):
    """Unmasked background must stay hidden through every prepared handoff."""
    from XTA.confidence_orchestration import ConfidenceCaptureQueue
    from XTA.confidence_publication import ConfidencePublicationQueue

    backend, _ = codec_backend
    mask, scores, active, boxes = _inputs('ragged_nonbinary')
    original_scores = scores.copy()
    original_mask = mask.copy()
    expected = np.where(mask != 0, scores, np.uint8(0))
    baseline = _capture(tmp_path / 'baseline', mask.copy(), scores.copy(), active.copy(), boxes.copy())
    root = tmp_path / 'prepared'
    evidence.configure_confidence_evidence(root, enabled=True, defer_projection=True)
    view = geometry.get_view_infos(*mask.shape, cartesian_views=('transverse',))[0]
    plan = plan_confidence_capture(mask.shape, 2, workspace_bytes=32 * 1024**2)
    created, scores_at_construction, seen_readers, metrics = [], [], [], []
    real_init = evidence._MaskedNativeScoreReader.__init__
    real_write = storage.write_blocks

    def initialized(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        created.append(self)
        source_scores = args[1] if len(args) > 1 else kwargs['scores']
        scores_at_construction.append(np.array(source_scores, copy=True))

    def written(directory, shape, reader, *args, **kwargs):
        seen_readers.append(reader)
        result = real_write(directory, shape, reader, *args, **kwargs)
        metrics.append(dict(kwargs.get('metrics') or {}))
        return result

    monkeypatch.setattr(evidence._MaskedNativeScoreReader, '__init__', initialized)
    monkeypatch.setattr(storage, 'write_blocks', written)
    try:
        with confidence_capture_resources(plan):
            prepared = evidence.prepare_terminal_confidence_scores(mask, scores,
                known_slice_any=active, known_slice_bboxes=boxes)
            assert prepared is not None
            assert len(created) == 1 and prepared.reader is created[0]
            reader = prepared.reader
            np.testing.assert_array_equal(scores_at_construction[0], original_scores)
            assert not reader.active.flags.writeable and not reader.boxes.flags.writeable
            assert reader.mask is reader.scores
            assert np.shares_memory(reader.mask, scores) and np.shares_memory(reader.scores, scores)
            assert not np.shares_memory(reader.mask, mask)
            # The frozen reader must still be used if original metadata is
            # subsequently altered or omitted by a refusal/retry continuation.
            active[:] = False
            boxes[:] = -999
            kwargs = dict(view=view, model_name='parity', temp_dir=root / 'work')
            if handoff == 'async':
                queue = ConfidenceCaptureQueue(slots=1, workspace_bytes=plan.workspace_bytes,
                                               source_bytes=scores.nbytes)
                monkeypatch.setattr(evidence, '_CAPTURE_QUEUE', queue)
                future = evidence.try_capture_terminal_confidence_scores(prepared,
                    retire=lambda _scores: None, known_slice_any=active,
                    known_slice_bboxes=boxes, **kwargs)
                assert future is not None
                assert future.result(timeout=10) is None
                assert queue.snapshot()['peak_source_bytes'] == scores.nbytes
            elif handoff == 'sync-refusal':
                assert evidence.try_capture_terminal_confidence_scores(prepared,
                    retire=lambda _scores: None, **kwargs) is None
                evidence.capture_terminal_confidence_scores(prepared, **kwargs)
            else:
                # A real too-small staging slot forces the second direct write.
                publisher = ConfidencePublicationQueue(max_pending_bytes=1, stage_bytes=1, workers=1)
                monkeypatch.setattr(evidence, '_PUBLICATION', publisher)
                evidence.capture_terminal_confidence_scores(prepared,
                    known_slice_any=active, known_slice_bboxes=boxes, **kwargs)
                assert len(seen_readers) >= 2
        assert len(created) == 1
        assert seen_readers and all(item is reader for item in seen_readers)
        assert scores[0, 0, 0] == original_scores[0, 0, 0] > 0
        manifest_path = root / 'reconciliation_evidence' / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        ref = evidence.ConfidenceEvidenceRef.open(manifest_path.parent / manifest['layers'][0]['directory'])
        files = {name: (ref.path / name).read_bytes() for name in _ARTIFACTS}
        files['manifest.json'] = manifest_path.read_bytes()
        with ref.native_reader() as native_reader:
            decoded, known = native_reader(0, mask.shape[0])
        actual = files, decoded, known, metrics[-1]
        _assert_artifacts(tmp_path, baseline, actual, route='v2-' + handoff, backend=backend)
        np.testing.assert_array_equal(decoded, expected)
        np.testing.assert_array_equal(mask, original_mask)
        assert not np.any(decoded[0]) and not np.any(known[0])
        (tmp_path / 'reader-proof.json').write_text(json.dumps(dict(
            constructors=len(created), writer_attempts=len(seen_readers),
            one_reader_identity=True, original_metadata_mutated=True,
            positive_unmasked_background_present=True), indent=2))
    finally:
        evidence.configure_confidence_evidence(None, enabled=False)


@pytest.mark.parametrize('handoff', ('sync', 'async'))
def test_prepared_reader_cannot_escape_its_funded_capture_plan(
        tmp_path, monkeypatch, handoff):
    """A retained larger reader must not consume a smaller or absent grant."""
    from XTA.confidence_orchestration import ConfidenceCaptureQueue

    mask, scores, active, boxes = _inputs('ragged_nonbinary')
    plan_a = plan_confidence_capture(mask.shape, 2, workspace_bytes=32 * 1024**2)
    plan_b = plan_confidence_capture(mask.shape, 1, workspace_bytes=32 * 1024**2)
    assert plan_a != plan_b and plan_a.workspace_bytes > plan_b.workspace_bytes
    evidence.configure_confidence_evidence(tmp_path / 'output', enabled=True, defer_projection=True)
    try:
        with confidence_capture_resources(plan_a):
            prepared = evidence.prepare_terminal_confidence_scores(mask, scores,
                known_slice_any=active, known_slice_bboxes=boxes)
        assert prepared is not None and prepared.reader is not None
        reader = prepared.reader
        before = scores.copy()
        queue = ConfidenceCaptureQueue(slots=1, workspace_bytes=plan_b.workspace_bytes,
                                       source_bytes=scores.nbytes)
        monkeypatch.setattr(evidence, '_CAPTURE_QUEUE', queue)

        def forbidden(*_args, **_kwargs):
            pytest.fail('A mismatched prepared reader reached reconstruction, queue or writer')

        monkeypatch.setattr(queue, 'try_submit', forbidden)
        monkeypatch.setattr(evidence, 'publish_confidence_scores', forbidden)
        monkeypatch.setattr(evidence._MaskedNativeScoreReader, '__init__', forbidden)
        view = geometry.get_view_infos(*mask.shape, cartesian_views=('transverse',))[0]
        kwargs = dict(view=view, model_name='plan-guard', temp_dir=tmp_path / 'work')
        for context in ('different-plan', 'missing-plan'):
            grant = confidence_capture_resources(plan_b) if context == 'different-plan' else nullcontext()
            with grant:
                with pytest.raises(ValueError, match='original capture plan'):
                    if handoff == 'async':
                        evidence.try_capture_terminal_confidence_scores(prepared,
                            retire=lambda _scores: pytest.fail('Unsubmitted source retired'), **kwargs)
                    else:
                        evidence.capture_terminal_confidence_scores(prepared, **kwargs)
        assert queue.snapshot()['submitted'] == 0
        assert prepared.reader is reader and reader.capture_plan == plan_a
        np.testing.assert_array_equal(scores, before)
        assert not list(tmp_path.rglob('metadata.json'))
        assert not list(tmp_path.rglob('scores.u8.zlib'))
    finally:
        evidence.configure_confidence_evidence(None, enabled=False)

