"""Finished SAM work returns its ordinary credit before output backpressure."""
from concurrent.futures import CancelledError, FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing, contextmanager
import gc
from pathlib import Path
from threading import Event, get_ident
import time
import weakref

import numpy as np
import pytest

from XTA import assembly, geometry, sam_resources
from XTA.config import GIB
from XTA.interpolation import RawBBoxMaskStore, _ByteAdmissionPool, _DirectUnionBackingLease
from XTA.projection_queue import settle_prepared_view_components
from XTA.runtime import close_memmap_array_without_flush, wait_for_retired_memmap_unlinks
from XTA.view_prepare import ViewPrepareLeaseState, _array_lifetime_owner
from tests.test_sam_async_layer_publication import (
    SelectedContext, make_queue, make_submitter, owned_task, read_layer,
)


class ObservedContext(SelectedContext):
    def __init__(self, root, shape, *, hold=False):
        super().__init__(root, shape)
        self.hold = hold
        self.release_hold = lambda: None
        self.scope_exited = Event()
        self.created_entries = []
        self._cancel = Event()

    @contextmanager
    def resource_scope(self, profile):
        profile._validate_owner()
        self.profile = profile
        self.prepare_thread = get_ident()
        if self.hold:
            self.release_hold = sam_resources.retain_sam_parent_image_credit(profile)
        try:
            yield
        finally:
            self.scope_exited.set()

    def entries(self, *args):
        entries = super().entries(*args)
        self.created_entries.extend(entries)
        return entries


class LegacyPublicationError(OSError):
    add_note = None


@pytest.fixture
def cases(tmp_path, monkeypatch):
    shape = (5, 7, 9)
    monkeypatch.setattr(sam_resources, 'physical_sam_headroom', lambda: 128 * GIB)
    assembly.set_final_source_output_shape(shape)
    created = []

    def make(*, hold=False, directional=None, **queue_settings):
        root = tmp_path / str(len(created))
        context = ObservedContext(root / 'evidence', shape, hold=hold)
        view = geometry.get_view_infos(*shape, cartesian_views=('transverse',))[0]
        queue = make_queue(**queue_settings)
        pool = _ByteAdmissionPool(17 * GIB, 'handoff parent')
        task = owned_task(root / 'parent', view, context,
                          make_submitter(queue, shape, directional), pool)
        task.preinterpolation_layer_already_published = True
        created.append((context, queue))
        return context, queue, pool, task

    yield make
    for context, queue in created:
        context.release_hold()
        queue.abort()
    for _context, queue in created:
        queue.shutdown(cancel_futures=True)
    assembly.set_final_source_output_shape(None)


def assert_selected_sources_readable(context):
    assert len(context.created_entries) == 4
    for entry in context.created_entries:
        with closing(RawBBoxMaskStore.open(Path(entry['path']), mmap_payload=True)) as store:
            assert sum(int(np.count_nonzero(store.decode_slice(z)))
                       for z in range(store.shape[0])) == 1


def drain_publications(result):
    deadline = time.monotonic() + 10
    while not settle_prepared_view_components(result):
        assert time.monotonic() < deadline, 'publication admission stopped progressing'
        wait([future for future in result.pending_component_layers if not future.done()],
             timeout=.05, return_when=FIRST_COMPLETED)


@pytest.mark.parametrize('pressure', ('pending', 'source_bytes'))
@pytest.mark.parametrize('hold', (False, True))
def test_saturated_publication_refunds_only_settled_sam_credit(cases, monkeypatch, pressure, hold):
    context, queue, pool, task = cases(
        hold=hold, max_pending=1 if pressure == 'pending' else 8,
        max_source_bytes=1024**2)
    occupied, release, flushing = Event(), Event(), Event()

    def unrelated():
        occupied.set()
        assert release.wait(10), 'unrelated publication was not released'

    queue.submit(unrelated, source_bytes=1024**2, working_bytes=1)
    assert occupied.wait(3)
    actual_submit = queue.try_submit
    submitted, flush_threads = [], []

    def submit(*args, **kwargs):
        flush_threads.append(get_ident())
        flushing.set()
        future = actual_submit(*args, **kwargs)
        if future is not None:
            submitted.append(future)
        return future

    monkeypatch.setattr(queue, 'try_submit', submit)
    key = (task.model_name, task.view.name)
    amount = task.union_mm.nbytes
    lease = _DirectUnionBackingLease(key, amount, phase='postprocess')
    task.backing_lease = lease
    dense = ViewPrepareLeaseState({key: lease}, set(), {}, {key}, {key: amount})
    try:
        with ThreadPoolExecutor(1) as parents:
            future = parents.submit(task)
            try:
                assert flushing.wait(5)
                assert context.extrapolation_entered.is_set()
                assert context.scope_exited.is_set()
                assert not context.profile._lease.active
                result = future.result(3)
                assert result.pending_component_layers and not result.nrrd_layers
                assert dense.postprocess_bytes[key] == amount
                assert_selected_sources_readable(context)
                if hold:
                    assert pool.in_use == context.profile._lease.charged_bytes
                    assert context.profile._lease.scope_holds == 1
                    context.release_hold()
                assert pool.in_use == 0
                with sam_resources.admit_sam_parent_resources(pool, GIB, 'next-parent',
                        headroom_probe=lambda: 128 * GIB):
                    assert pool.in_use > 0
            finally:
                release.set()
            result = future.result(5)
        proxies = tuple(result.pending_component_layers)
        drain_publications(result)
        queue.shutdown()
        assert flush_threads[0] == context.prepare_thread
        assert set(flush_threads) <= {context.prepare_thread, get_ident()}
        assert len(submitted) == 4
        assert [future.result() for future in proxies] == [future.result() for future in submitted]
        assert settle_prepared_view_components(result)
        assert [(ref.mask_kind, ref.interpolation_direction or ref.extrapolation_provenance['direction'])
                for ref in result.nrrd_layers] == [
                    ('bridge', 'forward'), ('bridge', 'backward'),
                    ('extrapolation', 'forward'), ('extrapolation', 'backward')]
        for ref in result.nrrd_layers:
            expected = np.zeros((5, 7, 9), np.uint8)
            direction = ref.interpolation_direction or ref.extrapolation_provenance['direction']
            expected[2 if direction == 'forward' else 4, 3, 4] = 1
            np.testing.assert_array_equal(read_layer(ref, expected.shape), expected)
            assert ref.sam_bundle_identity == 'sam-bundle' or ref.extrapolation_provenance['sam_bundle_identity'] == 'sam-bundle'
        assert_selected_sources_readable(context)
        state = queue.snapshot()
        assert state['pending'] == state['source_bytes'] == state['working_bytes'] == pool.in_use == 0
        assert state['peak_pending'] <= state['max_pending']
        assert state['peak_source_bytes'] <= state['max_source_bytes']
    finally:
        release.set()


@pytest.mark.parametrize('stage', ('prepare', 'scope_exit'))
def test_failure_before_flush_submits_no_publication(cases, monkeypatch, stage):
    context, queue, pool, task = cases()
    failure = RuntimeError('controlled before-flush failure')
    if stage == 'prepare':
        original = task.prepare

        def prepare(**kwargs):
            original(**kwargs)
            raise failure

        task.prepare = prepare
    else:
        original = context.resource_scope

        @contextmanager
        def scope(profile):
            with original(profile):
                yield
                raise failure

        context.resource_scope = scope
    monkeypatch.setattr(queue, 'try_submit', lambda *_a, **_kw: pytest.fail('publication escaped failed prepare'))
    with pytest.raises(RuntimeError) as caught:
        task()
    assert caught.value is failure
    assert pool.in_use == 0
    assert_selected_sources_readable(context)


def test_recorded_publication_keeps_immutable_identity_after_scope_cleanup(cases):
    context, queue, pool, task = cases()
    original = context.resource_scope

    @contextmanager
    def scope(profile):
        with original(profile):
            yield
            context.bundle_identity = 'changed after preparation'
            context.detector_identity = 'changed after preparation'
            for entry in context.created_entries:
                entry['policy_hash'] = 'changed after preparation'

    context.resource_scope = scope
    result = task()
    drain_publications(result)
    queue.shutdown()
    assert pool.in_use == 0 and settle_prepared_view_components(result)
    for ref in result.nrrd_layers:
        if ref.mask_kind == 'bridge':
            assert ref.sam_bundle_identity == 'sam-bundle'
            assert ref.seed_detector_identity == 'detector'
            assert ref.interpolation_policy_identity == 'policy'
        else:
            assert ref.extrapolation_provenance['sam_bundle_identity'] == 'sam-bundle'
            assert ref.extrapolation_provenance['policy_identity'] == 'policy'


def test_custom_submitter_keeps_ordinary_synchronous_call_contract(cases):
    context, queue, pool, task = cases()
    production_submitter = task.submit_sam_layer_projection
    live_calls = []

    def custom(*args, **kwargs):
        live_calls.append(context.profile._lease.active)
        return production_submitter(*args, **kwargs)

    task.submit_sam_layer_projection = custom
    result = task()
    drain_publications(result)
    queue.shutdown()
    assert live_calls == [True] * 4
    assert pool.in_use == 0 and settle_prepared_view_components(result)


@pytest.mark.parametrize('passes,extrapolation', ((0, 2), (2, 0), (3, 2)))
def test_handoff_preserves_configured_directional_work(cases, passes, extrapolation):
    context, queue, pool, task = cases()
    task.interpolation_passes = passes
    task.interpolation_distance = 2 if passes else 0
    task.extrapolation_distance = extrapolation
    result = task()
    drain_publications(result)
    queue.shutdown()
    assert pool.in_use == 0 and settle_prepared_view_components(result)
    # SAM consumes the configured policy passes internally and emits one pair.
    assert len(result.nrrd_layers) == (2 if passes else 0) + (2 if extrapolation else 0)
    assert len(context.created_entries) == len(result.nrrd_layers)
    assert queue.snapshot()['submitted'] == len(result.nrrd_layers)


@pytest.mark.parametrize('cleanup_failure', (False, True))
@pytest.mark.parametrize('failure_type', (OSError, LegacyPublicationError))
def test_partial_submission_failure_preserves_accepted_source_and_error(
        cases, monkeypatch, cleanup_failure, failure_type):
    entered, release = Event(), Event()

    def publish(entry, **kwargs):
        entered.set()
        assert release.wait(10), 'accepted publisher was not released'
        return assembly.materialize_sam_directional_view_layer(entry, **kwargs)

    context, queue, pool, task = cases(directional=publish)
    actual_submit = queue.try_submit
    accepted, results = [], []
    failure = failure_type('controlled second submission failure')
    prepare, close_dense = task.prepare, task.close_dense

    def capture_result(**kwargs):
        result = prepare(**kwargs)
        results.append(result)
        return result

    def close(value, **kwargs):
        if cleanup_failure and context.scope_exited.is_set():
            raise RuntimeError('controlled cleanup failure')
        return close_dense(value, **kwargs)

    task.prepare, task.close_dense = capture_result, close

    def submit(*args, **kwargs):
        if accepted:
            raise failure
        future = actual_submit(*args, **kwargs)
        if future is not None:
            accepted.append(future)
        return future

    monkeypatch.setattr(queue, 'try_submit', submit)
    try:
        with pytest.raises(OSError) as caught:
            task()
        assert caught.value is failure
        assert entered.wait(3)
        assert pool.in_use == 0 and not context.profile._lease.active
        assert not task._cancelled_before_prepare
        assert task.union_mm is None
        assert results[0].native_support_mm is results[0].final_view_volume_mm is None
        assert results[0].pending_component_layers[1].exception() is failure
        assert all(future.cancelled() for future in results[0].pending_component_layers[2:])
        if cleanup_failure and callable(getattr(failure, 'add_note', None)):
            assert any('controlled cleanup failure' in note for note in failure.__notes__)
        assert_selected_sources_readable(context)
    finally:
        release.set()
    queue.shutdown()
    assert len(accepted) == 1 and accepted[0].result().mask_kind == 'bridge'
    state = queue.snapshot()
    assert state['pending'] == state['source_bytes'] == state['working_bytes'] == 0


def test_cancellation_at_scope_exit_submits_nothing_and_keeps_started_ownership(cases, monkeypatch):
    context, _queue, pool, task = cases()
    original = context.resource_scope

    @contextmanager
    def scope(profile):
        with original(profile):
            yield
            context._cancel.set()

    context.resource_scope = scope
    monkeypatch.setattr(_queue, 'try_submit', lambda *_a, **_kw: pytest.fail('cancelled publication submitted'))
    with pytest.raises(CancelledError):
        task()
    assert context.scope_exited.is_set() and pool.in_use == 0
    assert not task._cancelled_before_prepare and task.union_mm is None
    assert_selected_sources_readable(context)


def test_abort_wakes_flush_without_claiming_prepare_was_unstarted(cases, monkeypatch):
    context, queue, pool, task = cases(max_pending=1)
    occupied, release, flushing = Event(), Event(), Event()

    def unrelated():
        occupied.set()
        assert release.wait(10), 'unrelated publication was not released'

    queue.submit(unrelated, source_bytes=1, working_bytes=1)
    assert occupied.wait(3)
    actual_submit = queue.try_submit

    def submit(*args, **kwargs):
        flushing.set()
        return actual_submit(*args, **kwargs)

    monkeypatch.setattr(queue, 'try_submit', submit)
    try:
        with ThreadPoolExecutor(1) as parents:
            future = parents.submit(task)
            try:
                assert flushing.wait(5)
                assert pool.in_use == 0 and not context.profile._lease.active
                prepared = future.result(3)
                context._cancel.set()
                queue.abort()
                with pytest.raises((RuntimeError, CancelledError)):
                    settle_prepared_view_components(prepared)
                assert future.exception() is None
                assert not task._cancelled_before_prepare
                assert_selected_sources_readable(context)
            finally:
                release.set()
    finally:
        release.set()


@pytest.mark.parametrize('materialized', (False, True))
@pytest.mark.parametrize('failure_kind', ('submit', 'abort'))
def test_failed_flush_retires_own_dense_aliases_with_failed_future_retained(
        cases, monkeypatch, materialized, failure_kind):
    publishing, release, second_submit = Event(), Event(), Event()

    def publish(entry, **kwargs):
        publishing.set()
        assert release.wait(10), 'accepted publication was not released'
        return assembly.materialize_sam_directional_view_layer(entry, **kwargs)

    context, queue, pool, task = cases(directional=publish, max_pending=1)
    task.component_ref_dense_retirement_active = False
    task.preinterpolation_layer_already_published = False
    task.extrapolation_distance = 0
    outside, owners, results, accepted = [], [], [], []

    def keep_outside_alias(volume):
        outside.append(volume[:])
        owners.append(weakref.ref(_array_lifetime_owner(volume)))

    if materialized:
        close_memmap_array_without_flush(task.union_mm)
        task.union_mm = None
        task.d1_shadow_path = task.temp_dir / 'shadow.cvol'

        def materialize(_source, destination, **_kwargs):
            volume = np.memmap(destination, mode='w+', dtype=np.uint8, shape=(5, 7, 9))
            volume[:] = 0
            volume[0, 3, 4] = 1
            keep_outside_alias(volume)
            return volume

        task.materialize_workspace = materialize
    else:
        keep_outside_alias(task.union_mm)

    def interpolate(observations, *, work_dir, **_kwargs):
        context.interpolation_entered.set()
        return observations, dict(skipped=False, added_voxels=2), context.entries(
            observations, work_dir, 'bridge')

    context.interpolate = interpolate
    prepare = task.prepare

    def prepare_with_shared_support(**kwargs):
        result = prepare(**kwargs)
        assert result.native_support_mm is result.final_view_volume_mm is kwargs['union_mm']
        result.parent_mask_support_mm = RawBBoxMaskStore.open(
            Path(context.created_entries[0]['path']), mmap_payload=True)
        results.append(result)
        return result

    task.prepare = prepare_with_shared_support
    actual_submit = queue.try_submit
    failure = OSError('controlled dense-alias flush failure')

    def submit(*args, **kwargs):
        if accepted:
            second_submit.set()
            assert publishing.wait(3), 'first accepted reader did not start'
            if failure_kind == 'submit':
                raise failure
            queue.abort()
        future = actual_submit(*args, **kwargs)
        if future is not None:
            accepted.append(future)
        return future

    monkeypatch.setattr(queue, 'try_submit', submit)
    try:
        with ThreadPoolExecutor(1) as parents:
            failed_future = parents.submit(task)
            try:
                assert publishing.wait(5) and second_submit.wait(5)
                if failure_kind == 'abort':
                    queue.abort()
                with pytest.raises(OSError if failure_kind == 'submit' else RuntimeError) as caught:
                    failed_future.result(3)
                if failure_kind == 'submit':
                    assert caught.value is failure
                assert failed_future.exception().__traceback__ is not None
                assert task.union_mm is None
                assert results[0].native_support_mm is results[0].final_view_volume_mm is None
                assert not task._cancelled_before_prepare and pool.in_use == 0
                assert owners[0]() is not None and task.union_path.exists()
                assert int(outside[0][0, 3, 4]) == 1
                outside[0][0, 3, 4] = 7
                assert int(outside[0][0, 3, 4]) == 7
                support = results[0].parent_mask_support_mm
                assert np.count_nonzero(support.decode_slice(2)) == 1
                for entry in context.created_entries:
                    with closing(RawBBoxMaskStore.open(Path(entry['path']))) as source:
                        assert np.count_nonzero(source.decode_slice(2 if entry['direction'] == 'forward' else 4)) == 1
                outside.clear()
                gc.collect()
                wait_for_retired_memmap_unlinks(path=task.union_path)
                assert owners[0]() is None and task.union_path.exists()
                assert failed_future.exception().__traceback__ is not None
                support.close()
            finally:
                release.set()
        assert accepted[0].result(3).mask_kind == 'bridge'
    finally:
        release.set()
        outside.clear()
        for result in results:
            result.parent_mask_support_mm.close()
