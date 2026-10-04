"""Confidence credit follows the real last owner, before parent projection."""
from contextlib import nullcontext
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
import gc
from pathlib import Path
import queue
from threading import Event
from types import SimpleNamespace
from unittest import mock
import weakref

import numpy as np
import pytest

from XTA import assembly, confidence_evidence, pipeline
from XTA.interpolation import _DirectUnionBackingLease
from XTA.runtime import wait_for_retired_memmap_unlinks
from XTA.view_prepare import AdmittedViewPrepare, ViewPrepareLeaseState
from tests.test_tta_scheduler_boundary import _view, _scheduler, _state
from tests.test_terminal_component_refs import _function


def owned_prepare(tmp_path, callback, *, derived_alias=False):
    shape = (16, 8, 8)
    scores = np.memmap(tmp_path/'scores.dat', mode='w+', dtype=np.uint8, shape=shape)
    scores[:] = 255
    mapping = weakref.ref(scores._mmap)
    aliases = [scores[1:3, 2:4, 2:4]] if derived_alias else []
    task = AdmittedViewPrepare(
        admission=SimpleNamespace(reserve=lambda *_args:nullcontext()), transient_bytes=1,
        model_name='model', view=_view(), union_mm=np.ones(shape, np.uint8),
        confmap_mm=scores, d1_shadow_path=None, union_path=tmp_path/'mask.dat',
        confmap_path=tmp_path/'scores.dat', temp_dir=tmp_path, dense_tiling_active=False,
        min_conf=0, min_radius=0, interpolation_distance=0, interpolation_walk_back=1,
        interpolation_candidates=1, interpolation_passes=1, interpolation_min_radius=0,
        interpolation_search_angle=15, keep_temp_artifacts=False, slice_workers=1,
        interpolation_task_workers=1, component_layers_needed=True,
        precleaned_slice_cleanup=True, hole_fill_done_on_device=True, slice_meta=None,
        fuse_azimuthal_component_layers=lambda:False, component_ref_dense_retirement_active=True,
        preinterpolation_layer_already_published=False, parent_mask_ready_callback=None,
        submit_component_projection=lambda *_args,**_kwargs:None,
        materialize_workspace=lambda *_args,**_kwargs:None,
        prepare=partial(pipeline._prepare_parent_with_confidence_capture,
            assembly.prepare_view_volume_after_fullframe, capture_plan=None),
        confidence_retired_callback=callback,
    )
    return task, mapping, aliases


@pytest.mark.parametrize('derived_alias', [False, True])
def test_capture_retires_mapping_before_projection_without_invalidating_live_alias(tmp_path, derived_alias):
    events = []
    task, mapping, aliases = owned_prepare(tmp_path,
        lambda *values:events.append(values), derived_alias=derived_alias)
    def projection(*_args, **_kwargs):
        # Runs while the admitted parent and confidence-budget wrapper are live.
        # No test Mock may hold capture's array arguments beyond their call.
        assert task.confmap_mm is None
        if aliases:
            assert not events and mapping() is not None and not mapping().closed
            assert int(aliases[0].sum()) == 8*255
            aliases.clear()
            gc.collect()
        assert mapping() is None
        assert events == [('model', task.view.name, 16*8*8)]
        wait_for_retired_memmap_unlinks(path=tmp_path/'scores.dat')
        assert not (tmp_path/'scores.dat').exists()
        return None
    with mock.patch.object(assembly, 'cleanup_view_volume_after_prediction_inplace',
                           new=lambda *_args,**_kwargs:None), \
         mock.patch.object(confidence_evidence, 'capture_prediction_confidence',
                           new=lambda *_args,**_kwargs:None), \
         mock.patch.object(assembly, 'materialize_nrrd_view_layer', new=projection):
        result = task()
    assert result.final_view_volume_mm is task.union_mm
    assert len(events) == 1


def test_confidence_event_returns_only_matching_postprocess_credit_once():
    key = ('model', 'view')
    lease = _DirectUnionBackingLease(key, 300)
    state = ViewPrepareLeaseState({key:lease}, {key}, {key:300}, set(), {})
    state.handoff(key)
    assert state.retire_input_bytes(key, lease, 100, token='confidence')
    assert lease.nbytes == 200 and state.postprocess_bytes[key] == 200
    assert not state.retire_input_bytes(key, lease, 100, token='confidence')
    assert state.complete(key, retain_for_dense_retirement=False)
    assert not state.retire_input_bytes(key, lease, 100, token='confidence')
    replacement = _DirectUnionBackingLease(key, 500, phase='postprocess')
    state.leases[key] = replacement
    state.postprocess_views.add(key)
    state.postprocess_bytes[key] = 500
    assert not state.retire_input_bytes(key, lease, 100, token='confidence')
    assert replacement.nbytes == 500 and state.postprocess_bytes[key] == 500


def test_checkpoint_resume_rebinds_confidence_credit_to_restored_lease(tmp_path):
    from XTA.sam_parent_staging import DeferredSamParentQueue
    ready, queued, events = [False], [], []
    task, _mapping, _aliases = owned_prepare(tmp_path, lambda *_args:None)
    key = ('model', task.view.name)
    old_lease = _DirectUnionBackingLease(key, 2048, phase='postprocess')
    state = ViewPrepareLeaseState({key:old_lease}, set(), {}, {key}, {key:2048})
    task.confidence_retired_callback_factory = lambda lease:lambda *values:events.append((lease, *values))
    task.rebind_confidence_retirement(old_lease)
    with ThreadPoolExecutor(max_workers=1) as writer:
        stage = DeferredSamParentQueue(temp_dir=tmp_path, output_dir=tmp_path/'output',
            checkpoint_executor=writer,
            prepare_executor=SimpleNamespace(submit=lambda value:queued.append(value) or Future()),
            leases=state, dense_limit=4096, ready=lambda:ready[0])
        stage.defer(task, 2048)
        next(iter(stage.checkpoint_futures)).result()
        stage.pump()
        assert key not in state.leases
        ready[0] = True
        stage.pump()
        restored_lease = state.leases[key]
        assert restored_lease is not old_lease and queued == [task]
        def projection(*_args, **_kwargs):
            assert len(events) == 1 and events[0][0] is restored_lease
            lease, model, view, nbytes = events[0]
            assert state.retire_input_bytes((model, view), lease, nbytes, token='confidence')
            assert restored_lease.nbytes == 1024
            assert not state.retire_input_bytes(key, old_lease, nbytes, token='confidence')
            return None
        with mock.patch.object(assembly, 'cleanup_view_volume_after_prediction_inplace',
                               new=lambda *_args,**_kwargs:None), \
             mock.patch.object(confidence_evidence, 'capture_prediction_confidence',
                               new=lambda *_args,**_kwargs:None), \
             mock.patch.object(assembly, 'materialize_nrrd_view_layer', new=projection):
            task()
        stage.close()


def test_early_confidence_callback_wakes_and_reopens_dense_admission(tmp_path):
    key = ('model', 'completed')
    lease = _DirectUnionBackingLease(key, 300, phase='postprocess')
    state = _state()
    state.direct_union_backing_leases[key] = lease
    state.direct_union_postprocess_views.add(key)
    state.direct_union_postprocess_bytes[key] = 300
    scheduler = _scheduler(tmp_path, state=state,
        input_overrides=dict(direct_union_total_dense_byte_limit=350))
    pending = dict(kind='fullframe', model_name='model', view=_view(),
        result_mode='direct_union', processing_shape=(100, 1, 1))
    with mock.patch.object(confidence_evidence, 'confidence_evidence_enabled', return_value=False):
        assert not scheduler.direct_union_task_admissible(pending)
        events, wake = queue.SimpleQueue(), Event()
        namespace = dict(vars(pipeline), parent_confidence_retired_events=events,
            scheduler_state=SimpleNamespace(scheduler_wake=wake))
        callback = _function(Path(pipeline.__file__).read_text(encoding='utf-8'),
            '_publish_parent_confidence_retired', namespace)
        callback(*key, 100, expected_lease=lease)
        assert wake.is_set()  # No local scheduler_wake alias has been bound.
        event_key, event_lease, nbytes = events.get_nowait()
        credits = ViewPrepareLeaseState(state.direct_union_backing_leases,
            state.direct_union_inference_views, state.direct_union_inference_bytes,
            state.direct_union_postprocess_views, state.direct_union_postprocess_bytes)
        assert credits.retire_input_bytes(event_key, event_lease, nbytes, token='confidence')
        assert scheduler.direct_union_task_admissible(pending)


@pytest.mark.parametrize('invalid_bytes', [0, -1, 300, 400])
def test_confidence_event_cannot_release_parent_mask_or_create_credit(invalid_bytes):
    key = ('model', 'view')
    lease = _DirectUnionBackingLease(key, 300, phase='postprocess')
    state = ViewPrepareLeaseState({key:lease}, set(), {}, {key}, {key:300})
    with pytest.raises(RuntimeError, match='invalid dense credit'):
        state.retire_input_bytes(key, lease, invalid_bytes, token='confidence')
    assert state.postprocess_bytes[key] == 300 and lease.nbytes == 300
