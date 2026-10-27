"""Actual source death, scheduler credit, and admission during score-only capture."""
from contextlib import nullcontext
from functools import partial
import gc
import json
from pathlib import Path
import queue
from threading import Event
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from XTA import assembly, confidence_evidence as evidence, pipeline
from XTA import packed_tilted_projection as encoded
from XTA.confidence_capture import plan_confidence_capture
from XTA.interpolation import _DirectUnionBackingLease, materialize_raw_bbox_mask_store_workspace
from XTA.view_prepare import AdmittedViewPrepare, ViewPrepareLeaseState, _array_lifetime_owner
from tests.test_packed_tilted_projection import _VIEWS
from tests.test_terminal_component_refs import _function
from tests.test_tta_scheduler_boundary import _scheduler, _state, _view


def _owned_prepare(root, mask_callback, confidence_callback, *, with_metadata):
    """Use genuine owning heap allocations, without claiming disk maps are RAM."""
    view = _VIEWS[0]
    shape = (view.num_slices, view.src_h, view.src_w)
    mask = np.zeros(shape, np.uint8)
    mask[::2, 2:4, 2:4] = 1
    mask[::2, 2, 2] = 0
    scores = np.full(shape, 91, np.uint8)
    scores[mask != 0] = 213
    assert mask.flags.owndata and scores.flags.owndata
    references = tuple(weakref.ref(_array_lifetime_owner(value)) for value in (mask, scores))
    expected_mask, expected_scores = mask.copy(), scores.copy()
    active = np.any(mask, axis=(1, 2))
    boxes = np.zeros((shape[0], 4), np.int64)
    boxes[active] = (2, 4, 2, 4)
    if with_metadata:
        for z in np.flatnonzero(active):
            expected_scores[z, 2:4, 2:4][mask[z, 2:4, 2:4] == 0] = 0
        slice_meta = dict(valid=True, slice_any=active, slice_bboxes=boxes)
    else:
        expected_scores[mask == 0] = 0
        slice_meta = None
    plan = plan_confidence_capture(shape, 2)
    task = AdmittedViewPrepare(
        admission=SimpleNamespace(reserve=lambda *_args: nullcontext()), transient_bytes=1,
        model_name='model', view=view, union_mm=mask, confmap_mm=scores,
        d1_shadow_path=None, union_path=root/'owned-mask.u8.dat',
        confmap_path=None, temp_dir=root, dense_tiling_active=False,
        min_conf=0., min_radius=0., interpolation_distance=0, interpolation_walk_back=1,
        interpolation_candidates=1, interpolation_passes=1, interpolation_min_radius=0.,
        interpolation_search_angle=0., keep_temp_artifacts=False, slice_workers=2,
        interpolation_task_workers=1, component_layers_needed=True,
        precleaned_slice_cleanup=True, hole_fill_done_on_device=True, slice_meta=slice_meta,
        fuse_azimuthal_component_layers=lambda: False, component_ref_dense_retirement_active=True,
        preinterpolation_layer_already_published=False, parent_mask_ready_callback=None,
        submit_component_projection=lambda *_args, **_kwargs: None,
        materialize_workspace=materialize_raw_bbox_mask_store_workspace,
        prepare=partial(pipeline._prepare_parent_with_confidence_capture,
            assembly.prepare_view_volume_after_fullframe, capture_plan=plan),
        mask_retired_callback=mask_callback, confidence_retired_callback=confidence_callback)
    return task, plan, references, expected_mask, expected_scores


@pytest.mark.parametrize('external_alias', ['none', 'slice', 'frombuffer'])
@pytest.mark.parametrize('with_metadata', [False, True], ids=['full-grid', 'metadata-roi'])
def test_mask_death_returns_real_admission_credit_while_score_capture_is_blocked(
        tmp_path, monkeypatch, external_alias, with_metadata):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT', '0')
    monkeypatch.setenv('YOLO_TTA_PACKED_PARENT_MIN_MIB', '0')
    mask_events, confidence_events = queue.SimpleQueue(), queue.SimpleQueue()
    state, wake = _state(), Event()
    namespace = dict(vars(pipeline), parent_mask_retired_events=mask_events,
        parent_confidence_retired_events=confidence_events,
        scheduler_state=SimpleNamespace(scheduler_wake=wake))
    source = Path(pipeline.__file__).read_text(encoding='utf-8')
    publish_mask = _function(source, '_publish_parent_mask_retired', namespace)
    publish_confidence = _function(source, '_publish_parent_confidence_retired', namespace)
    task, plan, references, expected_mask, expected_scores = _owned_prepare(
        tmp_path, None, None, with_metadata=with_metadata)
    mask_ref, score_ref = references
    native_bytes = expected_mask.nbytes
    key = ('model', task.view.name)
    lease = _DirectUnionBackingLease(key, 2*native_bytes, phase='postprocess')
    state.direct_union_backing_leases[key] = lease
    state.direct_union_postprocess_views.add(key)
    state.direct_union_postprocess_bytes[key] = lease.nbytes
    credits = ViewPrepareLeaseState(state.direct_union_backing_leases,
        state.direct_union_inference_views, state.direct_union_inference_bytes,
        state.direct_union_postprocess_views, state.direct_union_postprocess_bytes)
    task.mask_retired_callback = partial(publish_mask, expected_lease=lease)
    task.confidence_retired_callback = partial(publish_confidence, expected_lease=lease)
    alias = None
    if external_alias == 'slice':
        alias = task.union_mm[:]
    elif external_alias == 'frombuffer':
        alias = np.frombuffer(task.union_mm, dtype=np.uint8).reshape(task.union_mm.shape)

    # A real, separately named parent needs 64 bytes while evidence is enabled.
    # It exceeds this limit by one byte until an authentic mask-death receipt.
    pending = dict(kind='fullframe', model_name='model', view=_view(),
        result_mode='direct_union', processing_shape=(32, 1, 1))
    scheduler = _scheduler(tmp_path, state=state, input_overrides=dict(
        direct_union_total_dense_byte_limit=2*native_bytes + 63))
    entered, release = Event(), Event()
    observations, mask_receipts, packed_resident_bytes = [], [], []
    actual_publish = evidence.publish_confidence_scores
    actual_project = encoded.project_tilted_sparse_store

    def apply_mask_receipt():
        event_key, expected_lease, nbytes = mask_events.get_nowait()
        assert event_key == key and expected_lease is lease
        assert nbytes == native_bytes - packed_resident_bytes[0] > 0
        assert credits.retire_input_bytes(event_key, expected_lease, nbytes, token='mask')
        assert not credits.retire_input_bytes(event_key, expected_lease, nbytes, token='mask')
        mask_receipts.append((event_key, expected_lease, nbytes))
        assert state.direct_union_postprocess_bytes[key] == 2*native_bytes - nbytes
        assert state.direct_union_postprocess_bytes[key] == native_bytes + packed_resident_bytes[0]
        assert scheduler.direct_union_task_admissible(pending)

    def blocked_publication(scores, **kwargs):
        # Both the ROI direct-publish path and the full-grid fallback cross
        # this gate. Their actual readers still encode after the gate opens.
        assert _array_lifetime_owner(scores) is score_ref()
        assert not scores.flags.writeable
        np.testing.assert_array_equal(scores, expected_scores)
        reader = kwargs['native_reader']
        if with_metadata:
            assert isinstance(reader, evidence._MaskedNativeScoreReader)
            assert reader.mask is reader.scores
            assert _array_lifetime_owner(reader.scores) is score_ref()
            assert not reader.scores.flags.writeable
            assert not reader.active.flags.writeable and not reader.boxes.flags.writeable
            np.testing.assert_array_equal(reader.active, task.slice_meta['slice_any'])
            np.testing.assert_array_equal(reader.boxes, task.slice_meta['slice_bboxes'])
            assert reader.capture_metrics['bounded_input_bytes'] < 2*native_bytes
        else:
            assert not isinstance(reader, evidence._MaskedNativeScoreReader)
        entered.set()
        assert release.wait(30), 'test did not release its confidence reader'
        return actual_publish(scores, **kwargs)

    def observe_projection(*args, **kwargs):
        packed_stats = json.loads((Path(args[0])/'meta.json').read_text(encoding='utf-8'))['stats']
        resident_bytes = int(packed_stats['index_bytes'])
        if packed_stats['payload_backing'] == 'planned_memfd':
            resident_bytes += int(packed_stats['raw_payload_bytes'])
        packed_resident_bytes.append(resident_bytes)
        assert entered.wait(5)
        assert task.union_mm is task.confmap_mm is None
        assert score_ref() is not None and confidence_events.empty()
        assert not release.is_set()
        if external_alias == 'none':
            assert mask_ref() is None
            assert wake.is_set()
            apply_mask_receipt()
        else:
            assert mask_ref() is not None
            assert mask_events.empty()
            assert not scheduler.direct_union_task_admissible(pending)
        observations.append('projection_started_while_capture_blocked')
        return actual_project(*args, **kwargs)

    monkeypatch.setattr(evidence, 'publish_confidence_scores', blocked_publication)
    monkeypatch.setattr(encoded, 'project_tilted_sparse_store', observe_projection)
    evidence.configure_confidence_evidence(tmp_path/'output', enabled=True,
        defer_projection=True, background_publication=True)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        capture_queue = evidence.configure_terminal_confidence_capture(slots=1,
            workspace_bytes=plan.workspace_bytes, source_bytes=2*native_bytes)
        assert capture_queue
        assert not scheduler.direct_union_task_admissible(pending)
        result = task()
        assert observations == ['projection_started_while_capture_blocked']
        assert result._packed_native_source_retired
        capture_future, = result.pending_component_layers
        assert not capture_future.done() and score_ref() is not None
        assert capture_queue.snapshot()['source_bytes'] == native_bytes
        if external_alias != 'none':
            np.testing.assert_array_equal(alias, expected_mask)
            assert mask_ref() is not None and mask_events.empty()
            assert not scheduler.direct_union_task_admissible(pending)
            alias = None
            assert mask_ref() is None  # Refcount death, with cyclic GC disabled.
            assert wake.is_set()
            apply_mask_receipt()
        assert len(mask_receipts) == 1 and mask_events.empty()
        assert score_ref() is not None and confidence_events.empty()
        release.set()
        capture_future.result(timeout=15)
        capture_queue.drain()
        assert score_ref() is None
        assert mask_events.empty()
        confidence_key, confidence_lease, nbytes = confidence_events.get_nowait()
        assert confidence_key == key and confidence_lease is lease and nbytes == native_bytes
        assert credits.retire_input_bytes(confidence_key, confidence_lease, nbytes,
            token='confidence')
        assert not credits.retire_input_bytes(confidence_key, confidence_lease, nbytes,
            token='confidence')
        assert confidence_events.empty()
        evidence.drain_confidence_publication()
        assert capture_queue.snapshot()['source_bytes'] == 0
        reference, = evidence._REGISTRY.values()
        with reference.native_reader() as read_native:
            # ROI preparation deliberately leaves positive background outside
            # its frozen bounds untouched. The persisted reader must exclude it.
            values, known = read_native(0, expected_mask.shape[0])
            np.testing.assert_array_equal(values, expected_mask * np.uint8(213))
            np.testing.assert_array_equal(known, expected_mask != 0)
        # The remaining packed continuation stays charged until ordinary
        # parent completion. Real delayed receipts cannot credit a replacement.
        if key in credits.leases:
            assert credits.postprocess_bytes[key] == packed_resident_bytes[0]
            assert credits.complete(key, retain_for_dense_retirement=False)
        replacement = _DirectUnionBackingLease(key, 2*native_bytes, phase='postprocess')
        credits.leases[key] = replacement
        credits.postprocess_views.add(key)
        credits.postprocess_bytes[key] = replacement.nbytes
        event_key, old_lease, mask_bytes = mask_receipts[0]
        assert not credits.retire_input_bytes(event_key, old_lease, mask_bytes, token='mask')
        assert not credits.retire_input_bytes(confidence_key, confidence_lease, nbytes,
            token='confidence')
        assert replacement.nbytes == credits.postprocess_bytes[key] == 2*native_bytes
        assert credits.complete(key, retain_for_dense_retirement=False)
    finally:
        release.set()
        alias = None
        evidence.configure_confidence_evidence(None, enabled=False)
        if was_enabled:
            gc.enable()
