"""Private cache credit and donor leases remain bounded under concurrent callers."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import geometry
from XTA.sam_resources import admit_sam_parent_resources
from tests.test_sam_view_image_cache import context_for, crop_from, demand

GIB = 1024**3


def wait_until(predicate, timeout=5.):
    end = time.monotonic()+timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.005)
    assert predicate(), 'cache ownership state did not reach its expected barrier'


def proof_runtime(callback=lambda _reference: None):
    def retire(reference):
        callback(reference)
        return dict(status='retired', workers_finished=True, gray_mappings_retired=True)
    return SimpleNamespace(release_source_cache=retire, close=lambda: None, dispatch_stats={})


def test_donor_retirement_waits_copy_pin_without_blocking_independent_view(tmp_path, monkeypatch):
    source = np.arange(5*12*17, dtype=np.uint16).reshape(5, 12, 17).astype(np.uint8)
    context = context_for(tmp_path, source)
    transverse, sagittal = geometry.get_view_infos(*source.shape, cartesian_views=('transverse', 'sagittal'))
    ready, exit_cohort, copying, finish_copy, barrier = (threading.Event() for _ in range(5))
    context._runtime = proof_runtime(lambda _reference: barrier.set())
    old_copy = context._copy_cached_pixels
    references = []
    def gated_copy(*args, **kwargs):
        if kwargs.get('cache_entries'):
            copying.set()
            assert finish_copy.wait(5)
        return old_copy(*args, **kwargs)
    monkeypatch.setattr(context, '_copy_cached_pixels', gated_copy)
    def cohort():
        with context.image_cohort_provider(transverse, source.shape,
                demand(source.shape, {0:(1, 1, 7, 9)}), max_cache_bytes=1000) as reference:
            references.append(reference)
            ready.set()
            assert exit_cohort.wait(5)
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            first = executor.submit(cohort)
            assert ready.wait(5)
            second = executor.submit(context.image_provider, transverse, source.shape,
                demand(source.shape, {0:(0, 0, 10, 12)}))
            assert copying.wait(5)
            exit_cohort.set()
            wait_until(lambda: context._cache_owners[str(references[0].path)].get('retiring'))
            assert references[0].path.exists() and not barrier.is_set()
            shape = (sagittal.num_slices, sagittal.src_h, sagittal.src_w)
            independent = executor.submit(context.image_provider, sagittal, shape,
                demand(shape, {0:(0, 0, shape[1], shape[2])}))
            reference = independent.result(timeout=5)
            np.testing.assert_array_equal(crop_from(reference, 0, (0, 0, shape[1], shape[2])), source[:, 0, :])
            finish_copy.set()
            expanded = second.result(timeout=5)
            first.result(timeout=5)
            np.testing.assert_array_equal(crop_from(expanded, 0, (0, 0, 10, 12)), source[0, :10, :12])
            assert barrier.is_set() and not references[0].path.exists()
        snapshot = context.image_cache_lifetime_snapshot()
        assert snapshot['image_cache_borrower_pins'] == snapshot['active_image_build_credit_bytes'] == 0
        assert snapshot['image_build_peak_count'] == 2
    finally:
        exit_cohort.set()
        finish_copy.set()
        context.close()


def test_identical_cohort_waiter_survives_capacity_wait_and_single_retirement(tmp_path, monkeypatch):
    source = np.arange(5*12*17, dtype=np.uint16).reshape(5, 12, 17).astype(np.uint8)
    context = context_for(tmp_path, source)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    gates = (threading.Event(), threading.Event())
    entered = (threading.Event(), threading.Event())
    counts, counter_lock, retirements = {}, threading.Lock(), []
    promised_thread, retirement_calls = [], []
    promise_parked, claim_promise, retirement_started, finish_retirement = (threading.Event() for _ in range(4))
    context._runtime = proof_runtime(lambda reference: retirements.append(reference.path))
    old_render = context._render_demand_crop
    def render(current_view, index, *args, **kwargs):
        with counter_lock:
            counts[index] = counts.get(index, 0)+1
        if index < 2:
            entered[index].set()
            assert gates[index].wait(5)
        return old_render(current_view, index, *args, **kwargs)
    monkeypatch.setattr(context, '_render_demand_crop', render)
    original_wait = context._idle.wait
    original_retire = context._retire_image_cohort_owner
    def park_promised_waiter(timeout=None):
        result = original_wait(timeout)
        if (promised_thread and threading.get_ident() == promised_thread[0]
                and any(owner.get('pins', 0) for owner in context._cache_owners.values())):
            # Return from Condition.wait only after the producer has committed
            # and begun retiring; a promised consumer still owns its cache pin.
            context._idle.release()
            try:
                promise_parked.set()
                assert claim_promise.wait(5)
            finally:
                context._idle.acquire()
        return result
    def gate_retirement(*args, **kwargs):
        retirement_calls.append(args[0].path)
        retirement_started.set()
        assert finish_retirement.wait(5)
        return original_retire(*args, **kwargs)
    monkeypatch.setattr(context._idle, 'wait', park_promised_waiter)
    monkeypatch.setattr(context, '_retire_image_cohort_owner', gate_retirement)
    bbox = (1, 1, 7, 9)
    planned = demand(source.shape, {2:bbox})
    def cohort(promised=False):
        if promised:
            promised_thread.append(threading.get_ident())
        with context.image_cohort_provider(view, source.shape, planned, max_cache_bytes=1000) as reference:
            return reference, crop_from(reference, 2, bbox)
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            blockers = [executor.submit(context.image_provider, view, source.shape,
                demand(source.shape, {index:bbox})) for index in range(2)]
            assert all(event.wait(5) for event in entered)
            producer = executor.submit(cohort)
            wait_until(lambda: context.image_cache_lifetime_snapshot()['pending_image_builds'] == 3)
            waiter = executor.submit(cohort, True)
            def pending_waiter():
                with context._lock:
                    return any(not ticket.get('admitted') and ticket['waiters'] == 1
                        for ticket in context._image_builds.values())
            wait_until(pending_waiter)
            for gate in gates:
                gate.set()
            for blocker in blockers:
                blocker.result(timeout=5)
            assert promise_parked.wait(5)
            assert retirement_started.wait(5)
            claim_promise.set()
            second, second_pixels = waiter.result(timeout=5)
            # A quick promised-consumer enter/exit must not claim a second
            # retirement while the first retirement is waiting for its pin.
            assert len(retirement_calls) == 1
            finish_retirement.set()
            first, first_pixels = producer.result(timeout=5)
            assert first is second
            np.testing.assert_array_equal(first_pixels, source[2, 1:7, 1:9])
            np.testing.assert_array_equal(second_pixels, first_pixels)
        assert counts == {0:1, 1:1, 2:1}
        assert retirements == [first.path]
        assert not first.path.exists()
        snapshot = context.image_cache_lifetime_snapshot()
        assert snapshot['image_cache_borrower_pins'] == snapshot['active_image_build_credit_bytes'] == 0
        assert snapshot['retirement_unproven_count'] == 0
    finally:
        for gate in gates:
            gate.set()
        claim_promise.set()
        finish_retirement.set()
        context.close()


@pytest.mark.parametrize('credited', (False, True))
def test_two_realistic_large_targets_require_distinct_live_additive_credit(tmp_path, credited):
    # Reservation-only geometry avoids allocating >1GiB merely to verify the
    # admission arithmetic. Pixel parity and real map ownership run above.
    context = context_for(tmp_path, np.zeros((2, 3, 4), np.uint8))
    pool = SimpleNamespace(condition=threading.Condition(), capacity=192*GIB, in_use=0)
    ready, release = (threading.Event(), threading.Event()), threading.Event()
    reservations = []
    def reserve(index):
        admission = (admit_sam_parent_resources(pool, 4*GIB, f'producer-{index}',
            worker_count=1, base_allowance_bytes=4*GIB, headroom_probe=lambda:256*GIB)
            if credited else nullcontext(None))
        with admission as profile:
            with context.resource_scope(profile) if profile is not None else nullcontext():
                key = ('quota', index)
                with context._idle:
                    ticket = dict(admitted=False)
                    context._image_builds[key] = ticket
                    reservation = context._admit_image_build(685_588_645, 256*1024**2, GIB)
                    ticket.update(reservation, admitted=True)
                    reservations.append(reservation)
                try:
                    ready[index].set()
                    assert release.wait(5)
                finally:
                    with context._idle:
                        context._image_builds.pop(key)
                        context._idle.notify_all()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(reserve, 0)
            assert ready[0].wait(5)
            second = executor.submit(reserve, 1)
            if credited:
                assert ready[1].wait(5), 'independent admitted large target was serialized'
                assert reservations[0]['lease_id'] != reservations[1]['lease_id']
                assert all(row['credited'] and row['credit_bytes'] <= 2*GIB for row in reservations)
            else:
                assert not ready[1].wait(.1), 'uncredited callers exceeded the old aggregate target/scratch peak'
            release.set()
            first.result(timeout=5)
            second.result(timeout=5)
        assert pool.in_use == 0 and not context._image_builds
    finally:
        release.set()
        context.close()


@pytest.mark.parametrize('ending', ('completed', 'render_error', 'cancel', 'close'))
def test_four_funded_parent_builds_overlap_and_retire_all_owners(tmp_path, monkeypatch, ending):
    source = np.arange(5*12*17, dtype=np.uint16).reshape(5, 12, 17).astype(np.uint8)
    context = context_for(tmp_path, source)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    pool = SimpleNamespace(condition=threading.Condition(), capacity=192*GIB, in_use=0)
    context._runtime = proof_runtime()
    entered = [threading.Event() for _ in range(4)]
    release = threading.Event()
    original = context._render_demand_crop
    bbox = (1, 1, 7, 9)

    def render(current_view, index, *args, **kwargs):
        entered[index].set()
        assert release.wait(10), 'four admitted renderers were not released'
        if ending == 'render_error' and index == 1:
            raise ValueError('controlled parallel renderer failure')
        return original(current_view, index, *args, **kwargs)

    def parent(index):
        with admit_sam_parent_resources(pool, 4*GIB, f'parent-{index}', worker_count=1,
                base_allowance_bytes=4*GIB, headroom_probe=lambda:256*GIB) as profile:
            with context.resource_scope(profile):
                with context.image_cohort_provider(view, source.shape,
                        demand(source.shape, {index:bbox})) as reference:
                    return reference, crop_from(reference, index, bbox)

    monkeypatch.setattr(context, '_render_demand_crop', render)
    try:
        # These are the existing bounded parent executor lanes. Each thread
        # mints and owns its real grant; copying a profile is not permission.
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(parent, index) for index in range(4)]
            closing = None
            try:
                assert all(event.wait(5) for event in entered), 'funded parents hit a global render-count cap'
                snapshot = context.image_cache_lifetime_snapshot()
                assert snapshot['active_image_builders'] == snapshot['image_build_peak_count'] == 4
                with context._lock:
                    tickets = tuple(context._image_builds.values())
                    assert all(ticket['credited'] for ticket in tickets)
                    assert len({ticket['lease_id'] for ticket in tickets}) == 4
                    assert sum(ticket['credit_bytes'] for ticket in tickets) <= pool.in_use
                if ending == 'cancel':
                    context.cancel('controlled parallel cancellation')
                elif ending == 'close':
                    closing = executor.submit(context.close)
                    assert context._cancel.wait(5)
                    assert not closing.done() and context.source_volume is source
            finally:
                release.set()
            references = []
            for index, future in enumerate(futures):
                if ending in ('cancel', 'close'):
                    with pytest.raises(RuntimeError, match='cancellation|closing|lifetime|ended'):
                        future.result(timeout=5)
                elif ending == 'render_error' and index == 1:
                    with pytest.raises(ValueError, match='controlled parallel renderer failure'):
                        future.result(timeout=5)
                else:
                    reference, pixels = future.result(timeout=5)
                    references.append(reference)
                    np.testing.assert_array_equal(pixels, source[index, 1:7, 1:9])
                    assert not reference.path.exists()
            if closing is not None:
                closing.result(timeout=5)
        assert pool.in_use == 0
        snapshot = context.image_cache_lifetime_snapshot()
        assert snapshot['active_image_builders'] == snapshot['pending_image_builds'] == 0
        assert snapshot['active_image_build_credit_bytes'] == snapshot['image_cache_borrower_pins'] == 0
        assert not context._cache_owners
        assert not list((context.temp_dir/'sam_image_cache').glob('*.dat'))
    finally:
        release.set()
        context.close()
    assert not list((context.temp_dir/'sam_image_cache').glob('*.dat'))


def test_one_parent_image_allowance_cannot_be_spent_twice(tmp_path):
    context = context_for(tmp_path, np.zeros((2, 3, 4), np.uint8))
    pool = SimpleNamespace(condition=threading.Condition(), capacity=192*GIB, in_use=0)
    waiting, release_first = threading.Event(), threading.Event()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            def release_ticket():
                assert waiting.wait(5)
                with context._idle:
                    assert not release_first.is_set()
                    context._image_builds.pop('first')
                    release_first.set()
                    context._idle.notify_all()
            sibling = executor.submit(release_ticket)
            with admit_sam_parent_resources(pool, 4*GIB, 'same-parent', worker_count=1,
                    base_allowance_bytes=4*GIB, headroom_probe=lambda:256*GIB) as profile:
                with context.resource_scope(profile), context._idle:
                    context._image_builds['first'] = dict(context._admit_image_build(100, 100, 1000), admitted=True)
                    original_wait = context._idle.wait
                    def mark_wait(timeout=None):
                        waiting.set()
                        return original_wait(timeout)
                    context._idle.wait = mark_wait
                    second = context._admit_image_build(100, 100, 1000)
                    assert waiting.is_set() and release_first.is_set()
                    assert second['credited'] and second['lease_id'] == profile._lease.lease_id
            sibling.result(timeout=5)
        assert pool.in_use == 0 and not context._image_builds
    finally:
        context.close()


def test_batch_initializer_failure_discards_private_file_and_credit(tmp_path, monkeypatch):
    source = np.zeros((3, 7, 9), np.uint8)
    context = context_for(tmp_path, source)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    def fail(*args, **kwargs):
        raise RuntimeError('controlled batch initializer failure')
    monkeypatch.setattr(context, '_transverse_batch_iterator', fail)
    try:
        with pytest.raises(RuntimeError, match='batch initializer failure'):
            context.image_provider(view, source.shape, demand(source.shape, {0:(0, 0, 7, 9)}))
        assert not list((context.temp_dir/'sam_image_cache').glob('*.dat'))
        snapshot = context.image_cache_lifetime_snapshot()
        assert snapshot['active_image_build_credit_bytes'] == snapshot['image_cache_borrower_pins'] == 0
        assert not context._caches
    finally:
        context.close()


def test_lower_cap_same_demand_waiter_cannot_borrow_oversized_owned_commit(tmp_path, monkeypatch):
    source = np.zeros((3, 7, 9), np.uint8)
    context = context_for(tmp_path, source)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    entered, release = threading.Event(), threading.Event()
    old_render = context._render_demand_crop
    def gated(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return old_render(*args, **kwargs)
    monkeypatch.setattr(context, '_render_demand_crop', gated)
    planned = demand(source.shape, {0:(0, 0, 7, 9)})
    def lower_cap():
        with context.image_cohort_provider(view, source.shape, planned, max_cache_bytes=10):
            pytest.fail('lower-cap waiter bypassed admission')
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            producer = executor.submit(context.image_provider, view, source.shape, planned)
            assert entered.wait(5)
            waiter = executor.submit(lower_cap)
            def waiter_joined():
                with context._lock:
                    return any(ticket['waiters'] for ticket in context._image_builds.values())
            wait_until(waiter_joined)
            release.set()
            reference = producer.result(timeout=5)
            with pytest.raises(RuntimeError, match='exceeds cache budget'):
                waiter.result(timeout=5)
            assert reference.path.exists()
        assert context.image_cache_lifetime_snapshot()['image_cache_borrower_pins'] == 0
    finally:
        release.set()
        context.close()


def test_custom_provider_cannot_return_a_bare_reference_without_cohort_claim(tmp_path, monkeypatch):
    source = np.zeros((3, 7, 9), np.uint8)
    context = context_for(tmp_path, source)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    planned = demand(source.shape, {0:(0, 0, 7, 9)})
    reference = context.image_provider(view, source.shape, planned)
    monkeypatch.setattr(context, 'image_provider', lambda *_args, **_kwargs: reference)
    try:
        with pytest.raises(RuntimeError, match='missing atomic cohort lease'):
            with context.image_cohort_provider(view, source.shape, planned):
                pytest.fail('bare reference bypassed the cohort owner contract')
        assert context._cache_owners[str(reference.path)]['leases'] == 0
        assert reference.path.exists()
        assert context._active_image_cohorts == 0
    finally:
        context.close()


def test_corrupted_lease_cannot_underflow_or_hide_primary_sdk_failure(tmp_path):
    source = np.zeros((3, 7, 9), np.uint8)
    context = context_for(tmp_path, source)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    planned = demand(source.shape, {0:(0, 0, 7, 9)})
    retirements = []
    context._runtime = proof_runtime(lambda reference: retirements.append(reference.path))
    try:
        with pytest.raises(ValueError, match='original SDK failure') as caught:
            with context.image_cohort_provider(view, source.shape, planned) as reference:
                context._cache_owners[str(reference.path)]['leases'] = 0
                raise ValueError('original SDK failure')
        owner = context._cache_owners[str(reference.path)]
        assert owner['leases'] == 0 and owner['retirement_unproven']
        assert reference.path.exists() and not retirements
        assert any('lease underflow' in note for note in caught.value.__notes__)
        with pytest.raises(RuntimeError, match='unproven'):
            with context.image_cohort_provider(view, source.shape, planned):
                pytest.fail('unproven ownership admitted another cache')
    finally:
        context.close()


def test_bare_custom_provider_cannot_consume_another_cohorts_existing_lease(tmp_path, monkeypatch):
    source = np.zeros((3, 7, 9), np.uint8)
    context = context_for(tmp_path, source)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    planned = demand(source.shape, {0:(0, 0, 7, 9)})
    retirements = []
    context._runtime = proof_runtime(lambda reference: retirements.append(reference.path))
    try:
        with context.image_cohort_provider(view, source.shape, planned) as reference:
            monkeypatch.setattr(context, 'image_provider', lambda *_args, **_kwargs: reference)
            with pytest.raises(RuntimeError, match='missing atomic cohort lease'):
                with context.image_cohort_provider(view, source.shape, planned):
                    pytest.fail('invalid consumer reused another cohort lease')
            assert context._cache_owners[str(reference.path)]['leases'] == 1
            assert reference.path.exists() and not retirements
        assert retirements == [reference.path] and not reference.path.exists()
    finally:
        context.close()


def test_provider_failure_after_real_claim_retires_its_unused_owned_reference(tmp_path, monkeypatch):
    source = np.zeros((3, 7, 9), np.uint8)
    context = context_for(tmp_path, source)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    planned = demand(source.shape, {0:(0, 0, 7, 9)})
    retirements, references = [], []
    context._runtime = proof_runtime(lambda reference: retirements.append(reference.path))
    original = context.image_provider
    def late_failure(*args, **kwargs):
        references.append(original(*args, **kwargs))
        raise ValueError('late provider failure after claim')
    monkeypatch.setattr(context, 'image_provider', late_failure)
    try:
        with pytest.raises(ValueError, match='late provider failure'):
            with context.image_cohort_provider(view, source.shape, planned):
                pytest.fail('failed factory exposed a cohort')
        assert retirements == [references[0].path] and not references[0].path.exists()
        assert not context._cache_owners and context._active_image_cohorts == 0
    finally:
        context.close()
