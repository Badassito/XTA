"""Continuous next-cohort rendering owns fresh credit and retires on its producer."""
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import geometry, sam_resources
from tests.test_sam_view_image_cache import context_for, crop_from, demand

GIB = 1024**3


def account(capacity=12*GIB):
    return SimpleNamespace(capacity=capacity, in_use=0, condition=threading.Condition(threading.RLock()))


def setup(tmp_path, monkeypatch, retire=None):
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', '4096')
    source = np.arange(5*12*13, dtype=np.uint16).reshape(5, 12, 13).astype(np.uint8)
    context = context_for(tmp_path, source)
    view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    receipts = []
    def proof(ref):
        assert ref.path.exists()
        receipts.append((ref.path, threading.get_ident()))
        if retire is not None:
            return retire(ref)
        return dict(status='retired', workers_finished=True, gray_mappings_retired=True)
    context._runtime = SimpleNamespace(release_source_cache=proof, close=lambda:None,
        cancel=lambda _reason:None, residency_released=True, dispatch_stats={})
    bbox = (2, 3, 8, 9)
    plans = [demand(source.shape, {frame:bbox}) for frame in range(4)]
    return source, context, view, bbox, plans, receipts


def test_four_cohorts_each_next_builds_while_current_is_consumed_with_two_live_grants(tmp_path, monkeypatch):
    source, context, view, bbox, plans, retirements = setup(tmp_path, monkeypatch)
    pool = account()
    owner = threading.get_ident()
    producers, paths, holders = [], [], []
    original = context._render_demand_crop
    def render(*args, **kwargs):
        profile = context._resource_local.profile
        profile._validate_owner()
        assert profile._lease.owner_thread == threading.get_ident() != owner
        assert profile.assigned_cpu_wave_bytes == profile.assigned_session_cpu_bytes == 0
        assert profile.non_cpu_base_allowance_bytes >= 4096+36
        producers.append((threading.get_ident(), profile._lease.lease_id))
        return original(*args, **kwargs)
    monkeypatch.setattr(context, '_render_demand_crop', render)
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'cohort-parent', worker_count=4,
                base_allowance_bytes=4*GIB, headroom_probe=lambda:16*GIB) as profile:
            with context.resource_scope(profile):
                pending = context.prefetch_image_cohort(view, source.shape, plans[0], max_cache_bytes=1000)
                for index in range(4):
                    current, pending = pending, None
                    holders.append(current)
                    with current as ref:
                        paths.append(ref.path)
                        np.testing.assert_array_equal(crop_from(ref, index, bbox), source[index, 2:8, 3:9])
                        if index+1 < 4:
                            pending = context.prefetch_image_cohort(view, source.shape, plans[index+1], max_cache_bytes=1000)
                            assert pending is not None
                            next_ref = pending.future.result(timeout=5)
                            assert next_ref is not None and next_ref.path.exists()
                            assert ref.path.exists(), 'next must build before current cache retires'
                            assert len(context._image_prefetches) == 2
                            assert pool.in_use == 4*GIB
                            assert sam_resources.sam_parent_promised_bytes(pool) == 4*GIB+current.phase_bytes+pending.phase_bytes
                            assert context.prefetch_image_cohort(view, source.shape, plans[-1], max_cache_bytes=1000) is None
                    assert not ref.path.exists()
                assert pool.in_use == 4*GIB
                assert not context._image_prefetches
        assert pool.in_use == sam_resources.sam_parent_promised_bytes(pool) == 0
        assert len(producers) == len(retirements) == 4
        assert len({lease for _thread, lease in producers}) == 4
        assert all(thread != owner for _path, thread in retirements)
        assert len(set(paths)) == 4 and all(not path.exists() for path in paths)
        assert context.cache_logical_bytes == 0
        assert not list((tmp_path/'runtime'/'sam_image_cache').glob('*.gray8.dat'))
    finally:
        for holder in holders:
            holder.close()
        if 'pending' in locals() and pending is not None:
            pending.close()
        context.close()


def test_cpu_fallback_current_does_not_block_next_gpu_cohort_or_other_parent(tmp_path, monkeypatch):
    from XTA.sam_gpu_rendering import live_image_sampling
    from tests.test_sam_gpu_rendering import FakeRenderer
    source, context, view, bbox, plans, _retirements = setup(tmp_path, monkeypatch)
    pool = account(64*GIB)
    cpu_started, cpu_release, finish = (threading.Event() for _ in range(3))
    ready = [threading.Event(), threading.Event()]
    records = {}
    original = context._render_demand_crop
    def render(*args, **kwargs):
        if int(args[1]) == 0 and context._resource_local.gpu_image_renderer is None:
            cpu_started.set()
            assert cpu_release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(context, '_render_demand_crop', render)
    def gpu(_context, _view, _shape, prepared):
        frame = next(iter(prepared.frame_crop_bounds))
        return FakeRenderer(context, fail_after=0 if frame == 0 else None)
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', gpu)
    def parent(index):
        holders = []
        try:
            with sam_resources.admit_sam_parent_resources(pool, 4*GIB, f'parent-{index}',
                    worker_count=1, base_allowance_bytes=4*GIB, headroom_probe=lambda:128*GIB) as profile:
                with context.resource_scope(profile):
                    for frame in (2*index, 2*index+1):
                        holders.append(context.prefetch_image_cohort(view, source.shape,
                            plans[frame], max_cache_bytes=1000))
                    assert all(holders)
                    # Both wrappers may be pending; a third cannot spend this
                    # same parent's image pipeline slots again.
                    assert context.prefetch_image_cohort(view, source.shape, plans[-1], max_cache_bytes=1000) is None
                    ref = holders[-1].future.result(timeout=5)
                    assert live_image_sampling(ref)['backend'] == 'cuda'
                    records[index] = (profile, tuple(holders), ref)
                    ready[index].set()
                    assert finish.wait(5)
                    for frame, holder in zip((2*index,2*index+1),holders):
                        with holder as current:
                            expected = source[frame, 2:8, 3:9] if frame == 0 else 201+frame%7
                            np.testing.assert_array_equal(crop_from(current, frame, bbox), expected)
                            assert current.path.exists()
        finally:
            for holder in holders:
                holder.close()
    try:
        with ThreadPoolExecutor(2) as executor:
            old = executor.submit(parent, 0)
            assert cpu_started.wait(5) and ready[0].wait(5)
            other = executor.submit(parent, 1)
            assert ready[1].wait(5)
            assert not records[0][1][0].future.done(), 'the original CPU build remains blocked'
            assert records[0][2].path.exists() and records[1][2].path.exists()
            with context._idle:
                assert len(context._image_prefetches) == 4
            with pool.condition:
                parents = sum(record[0]._lease.charged_bytes for record in records.values())
                images = sum(holder.phase_bytes for record in records.values() for holder in record[1])
                assert pool.in_use == parents
                assert sam_resources.sam_parent_promised_bytes(pool) == parents+images
            cpu_release.set()
            finish.set()
            old.result(timeout=5)
            other.result(timeout=5)
        assert pool.in_use == sam_resources.sam_parent_promised_bytes(pool) == 0 and not context._image_prefetches
        assert context.cache_logical_bytes == 0
    finally:
        cpu_release.set()
        finish.set()
        context.close()


def test_no_physical_headroom_declines_profile_and_uses_original_thread_synchronous_provider(tmp_path, monkeypatch):
    source, context, view, bbox, plans, _retirements = setup(tmp_path, monkeypatch)
    pool = account(4*GIB)
    render_threads = []
    original = context._render_demand_crop
    def render(*args, **kwargs):
        render_threads.append(threading.get_ident())
        return original(*args, **kwargs)
    monkeypatch.setattr(context, '_render_demand_crop', render)
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'cohort-parent', worker_count=4,
                base_allowance_bytes=4*GIB, headroom_probe=lambda:16*GIB) as profile:
            profile._lease.headroom_probe = lambda:4*GIB
            with context.resource_scope(profile):
                holder = context.prefetch_image_cohort(view, source.shape, plans[0], max_cache_bytes=1000)
                assert holder.future.result(timeout=5) is None
                with holder as ref:
                    assert pool.in_use == 4*GIB
                    np.testing.assert_array_equal(crop_from(ref, 0, bbox), source[0, 2:8, 3:9])
                assert not ref.path.exists()
                holder.close()
        assert render_threads == [threading.get_ident()] and pool.in_use == 0
    finally:
        context.close()


@pytest.mark.parametrize('during_build', [False, True])
def test_abandoned_next_cohort_joins_and_returns_image_credit_without_expired_profile_use(tmp_path, monkeypatch, during_build):
    source, context, view, _bbox, plans, _retirements = setup(tmp_path, monkeypatch)
    pool = account()
    entered, release = threading.Event(), threading.Event()
    original = context._render_demand_crop
    if during_build:
        def render(*args, **kwargs):
            entered.set()
            assert release.wait(5)
            return original(*args, **kwargs)
        monkeypatch.setattr(context, '_render_demand_crop', render)
    holder = None
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'cohort-parent', worker_count=4,
                base_allowance_bytes=4*GIB, headroom_probe=lambda:16*GIB) as profile:
            with context.resource_scope(profile):
                holder = context.prefetch_image_cohort(view, source.shape, plans[0], max_cache_bytes=1000)
                if during_build:
                    assert entered.wait(5)
                    with ThreadPoolExecutor(1) as cleanup:
                        done = cleanup.submit(holder.close)
                        assert holder.cancelled.wait(5)
                        release.set()
                        done.result(timeout=5)
                else:
                    assert holder.future.result(timeout=5).path.exists()
                    holder.close()
                holder.close()
                assert not holder._thread.is_alive() and pool.in_use == 4*GIB
                assert not context._image_prefetches and context.cache_logical_bytes == 0
                assert not list((tmp_path/'runtime'/'sam_image_cache').glob('*.gray8.dat'))
                with pytest.raises(RuntimeError, match='consumed once'):
                    holder.__enter__()
        assert pool.in_use == sam_resources.sam_parent_promised_bytes(pool) == 0
    finally:
        release.set()
        if holder is not None:
            holder.close()
        context.close()


def test_unproven_prefetched_gray_retirement_preserves_file_and_fresh_image_credit_until_run_settles(tmp_path, monkeypatch):
    def refused(_ref):
        raise RuntimeError('controlled uncertain prefetched gray mapping')
    source, context, view, _bbox, plans, _retirements = setup(tmp_path, monkeypatch, refused)
    pool = account()
    holder = None
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'cohort-parent', worker_count=4,
                base_allowance_bytes=4*GIB, headroom_probe=lambda:16*GIB) as profile:
            with context.resource_scope(profile):
                holder = context.prefetch_image_cohort(view, source.shape, plans[0], max_cache_bytes=1000)
                with pytest.raises(RuntimeError, match='controlled uncertain prefetched gray mapping'):
                    with holder as ref:
                        assert ref.path.exists()
                assert ref.path.exists() and sam_resources.sam_parent_promised_bytes(pool) == 4*GIB+holder.phase_bytes
                assert len(context._retained_image_prefetch_credits) == 1
                assert context.image_cache_lifetime_snapshot()['retirement_unproven_count'] == 1
                assert context.prefetch_image_cohort(view, source.shape, plans[1], max_cache_bytes=1000) is None
            context.close()  # Fake runtime here proves settlement; gray stays diagnostic data.
            assert pool.in_use == 4*GIB and not context._retained_image_prefetch_credits
        assert pool.in_use == sam_resources.sam_parent_promised_bytes(pool) == 0 and ref.path.exists()
    finally:
        if holder is not None:
            holder.close()
        context.close()


@pytest.mark.parametrize('consumed', [False, True])
def test_resistant_join_preserves_primary_failure_and_retains_producer_ownership(tmp_path, monkeypatch, consumed):
    retiring, finish_retiring = threading.Event(), threading.Event()
    def gate_retirement(_ref):
        retiring.set()
        assert finish_retiring.wait(5)
        return dict(status='retired', workers_finished=True, gray_mappings_retired=True)
    source, context, view, _bbox, plans, _retirements = setup(tmp_path, monkeypatch, gate_retirement)
    pool = account()
    holder = None
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'cohort-parent', worker_count=4,
                base_allowance_bytes=4*GIB, headroom_probe=lambda:16*GIB) as profile:
            with context.resource_scope(profile):
                holder = context.prefetch_image_cohort(view, source.shape, plans[0], max_cache_bytes=1000)
                ref = holder.future.result(timeout=5)
                original_join = holder._join
                def resistant_join():
                    raise RuntimeError('controlled resistant prefetch join')
                monkeypatch.setattr(holder, '_join', resistant_join)
                if consumed:
                    with pytest.raises(ValueError, match='original SDK failure') as failure:
                        with holder:
                            raise ValueError('original SDK failure')
                else:
                    try:
                        raise ValueError('original SDK failure')
                    except ValueError as original:
                        holder.close()
                        failure = SimpleNamespace(value=original)
                assert any('controlled resistant prefetch join' in note for note in failure.value.__notes__)
                assert retiring.wait(5)
                assert holder._thread.is_alive() and ref.path.exists()
                assert sam_resources.sam_parent_promised_bytes(pool) == 4*GIB+holder.phase_bytes
                monkeypatch.setattr(holder, '_join', original_join)
                finish_retiring.set()
                holder.close()
                assert not holder._thread.is_alive() and pool.in_use == 4*GIB
        assert pool.in_use == sam_resources.sam_parent_promised_bytes(pool) == 0
    finally:
        finish_retiring.set()
        if holder is not None:
            holder.close()
        context.close()


@pytest.mark.parametrize('building', [False, True])
@pytest.mark.parametrize('action', ['cancel', 'close'])
def test_context_cancel_or_close_settles_unused_ready_and_building_prefetch(tmp_path, monkeypatch, building, action):
    source, context, view, _bbox, plans, _retirements = setup(tmp_path, monkeypatch)
    pool = account()
    entered, release = threading.Event(), threading.Event()
    original = context._render_demand_crop
    if building:
        def render(*args, **kwargs):
            entered.set()
            assert release.wait(5)
            return original(*args, **kwargs)
        monkeypatch.setattr(context, '_render_demand_crop', render)
    holder = None
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'cohort-parent', worker_count=4,
                base_allowance_bytes=4*GIB, headroom_probe=lambda:16*GIB) as profile:
            with context.resource_scope(profile):
                holder = context.prefetch_image_cohort(view, source.shape, plans[0], max_cache_bytes=1000)
                if building:
                    assert entered.wait(5)
                else:
                    assert holder.future.result(timeout=5).path.exists()
                with ThreadPoolExecutor(1) as cleanup:
                    operation = cleanup.submit(context.close if action == 'close' else context.cancel)
                    assert context._cancel.wait(5)
                    release.set()
                    operation.result(timeout=5)
                holder.close()
                assert not holder._thread.is_alive() and not context._image_prefetches
                assert pool.in_use == 4*GIB and context.cache_logical_bytes == 0
                assert not list((tmp_path/'runtime'/'sam_image_cache').glob('*.gray8.dat'))
        assert pool.in_use == sam_resources.sam_parent_promised_bytes(pool) == 0
    finally:
        release.set()
        if holder is not None:
            holder.close()
        context.close()


def test_context_close_cannot_retire_entered_gray_lease_before_its_consumer_exits(tmp_path, monkeypatch):
    source, context, view, bbox, plans, _retirements = setup(tmp_path, monkeypatch)
    pool = account()
    holder = None
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'cohort-parent', worker_count=4,
                base_allowance_bytes=4*GIB, headroom_probe=lambda:16*GIB) as profile:
            with context.resource_scope(profile):
                holder = context.prefetch_image_cohort(view, source.shape, plans[0], max_cache_bytes=1000)
                with ThreadPoolExecutor(1) as cleanup:
                    with holder as ref:
                        closing = cleanup.submit(context.close)
                        assert context._cancel.wait(5)
                        assert not closing.done()
                        assert ref.path.exists() and holder._thread.is_alive()
                        np.testing.assert_array_equal(crop_from(ref, 0, bbox), source[0, 2:8, 3:9])
                    closing.result(timeout=5)
                assert not ref.path.exists() and context.source_volume is None
                assert pool.in_use == 4*GIB and not context._image_prefetches
        assert pool.in_use == sam_resources.sam_parent_promised_bytes(pool) == 0
    finally:
        if holder is not None:
            holder.close()
        context.close()


def test_partial_provider_enter_failure_retains_claimed_gray_and_image_grant(tmp_path, monkeypatch):
    def refused(_ref):
        raise RuntimeError('controlled unproven claimed gray')
    source, context, view, _bbox, plans, _retirements = setup(tmp_path, monkeypatch, refused)
    pool = account()
    original = context.image_provider
    claimed = []
    def fail_after_claim(*args, **kwargs):
        claimed.append(original(*args, **kwargs))
        raise RuntimeError('controlled partial provider enter failure')
    monkeypatch.setattr(context, 'image_provider', fail_after_claim)
    holder = None
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'cohort-parent', worker_count=4,
                base_allowance_bytes=4*GIB, headroom_probe=lambda:16*GIB) as profile:
            with context.resource_scope(profile):
                holder = context.prefetch_image_cohort(view, source.shape, plans[0], max_cache_bytes=1000)
                with pytest.raises(RuntimeError, match='controlled partial provider enter failure'):
                    with holder:
                        pytest.fail('a partially failed provider returned a consumer reference')
                assert len(claimed) == 1 and claimed[0].path.exists()
                assert sam_resources.sam_parent_promised_bytes(pool) == 4*GIB+holder.phase_bytes
                assert len(context._retained_image_prefetch_credits) == 1
            context.close()
            assert pool.in_use == 4*GIB and claimed[0].path.exists()
        assert pool.in_use == sam_resources.sam_parent_promised_bytes(pool) == 0
    finally:
        if holder is not None:
            holder.close()
        context.close()


def test_cancel_during_accepted_enter_future_wait_signals_producer_without_lexical_exit(tmp_path, monkeypatch):
    source, context, view, _bbox, plans, _retirements = setup(tmp_path, monkeypatch)
    pool = account()
    rendering, release, entering = threading.Event(), threading.Event(), threading.Event()
    original_render = context._render_demand_crop
    def render(*args, **kwargs):
        rendering.set()
        assert release.wait(5)
        return original_render(*args, **kwargs)
    monkeypatch.setattr(context, '_render_demand_crop', render)
    holder = None
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'cohort-parent', worker_count=4,
                base_allowance_bytes=4*GIB, headroom_probe=lambda:16*GIB) as profile:
            with context.resource_scope(profile):
                holder = context.prefetch_image_cohort(view, source.shape, plans[0], max_cache_bytes=1000)
                assert rendering.wait(5)
                original_result = holder.future.result
                def result(*args, **kwargs):
                    entering.set()
                    return original_result(*args, **kwargs)
                monkeypatch.setattr(holder.future, 'result', result)
                def cancel():
                    assert entering.wait(5)
                    context.cancel('cancelled during accepted image entry')
                    release.set()
                with ThreadPoolExecutor(1) as controller:
                    stopped = controller.submit(cancel)
                    with pytest.raises(RuntimeError, match='cancelled|abandoned'):
                        holder.__enter__()
                    stopped.result(timeout=5)
                assert holder.finish.is_set() and not holder._thread.is_alive()
                assert not context._image_prefetches and pool.in_use == 4*GIB
        assert pool.in_use == sam_resources.sam_parent_promised_bytes(pool) == 0
    finally:
        release.set()
        if holder is not None:
            holder.close()
        context.close()


def test_reference_verification_failure_after_accepted_enter_does_not_wait_for_missing_exit(tmp_path, monkeypatch):
    source, context, view, _bbox, plans, _retirements = setup(tmp_path, monkeypatch)
    pool = account()
    holder = None
    caller = threading.get_ident()
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'cohort-parent', worker_count=4,
                base_allowance_bytes=4*GIB, headroom_probe=lambda:16*GIB) as profile:
            with context.resource_scope(profile):
                holder = context.prefetch_image_cohort(view, source.shape, plans[0], max_cache_bytes=1000)
                reference = holder.future.result(timeout=5)
                original = type(reference).revalidate
                def invalid(current):
                    if current is reference and threading.get_ident() == caller:
                        raise RuntimeError('controlled consumer reference verification failure')
                    return original(current)
                monkeypatch.setattr(type(reference), 'revalidate', invalid)
                with pytest.raises(RuntimeError, match='consumer reference verification failure'):
                    holder.__enter__()
                assert holder.finish.is_set() and not holder._thread.is_alive()
                assert not reference.path.exists() and pool.in_use == 4*GIB
        assert pool.in_use == sam_resources.sam_parent_promised_bytes(pool) == 0
    finally:
        if holder is not None:
            holder.close()
        context.close()


def test_rejected_foreign_or_second_enter_cannot_retire_existing_or_future_consumer(tmp_path, monkeypatch):
    source, context, view, bbox, plans, _retirements = setup(tmp_path, monkeypatch)
    pool = account()
    holder = None
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'cohort-parent', worker_count=4,
                base_allowance_bytes=4*GIB, headroom_probe=lambda:16*GIB) as profile:
            with context.resource_scope(profile):
                holder = context.prefetch_image_cohort(view, source.shape, plans[0], max_cache_bytes=1000)
                reference = holder.future.result(timeout=5)
                with ThreadPoolExecutor(1) as foreign:
                    with pytest.raises(RuntimeError, match='original preparation thread'):
                        foreign.submit(holder.__enter__).result(timeout=5)
                    assert not holder.finish.is_set() and reference.path.exists()
                    with holder as ref:
                        with pytest.raises(RuntimeError, match='consumed once'):
                            holder.__enter__()
                        with pytest.raises(RuntimeError, match='original preparation thread'):
                            foreign.submit(holder.__enter__).result(timeout=5)
                        assert not holder.finish.is_set() and ref.path.exists()
                        np.testing.assert_array_equal(crop_from(ref, 0, bbox), source[0, 2:8, 3:9])
                assert not reference.path.exists() and pool.in_use == 4*GIB
        assert pool.in_use == sam_resources.sam_parent_promised_bytes(pool) == 0
    finally:
        if holder is not None:
            holder.close()
        context.close()


def test_normal_exit_waits_for_late_natural_retirement_without_refunding_early(tmp_path, monkeypatch):
    from XTA import sam_image_prefetch
    retiring, release, joining = threading.Event(), threading.Event(), threading.Event()
    snapshots = []
    def retirement(ref):
        retiring.set()
        assert release.wait(5)
        return dict(status='retired', workers_finished=True, gray_mappings_retired=True)
    source, context, view, _bbox, plans, _receipts = setup(tmp_path, monkeypatch, retirement)
    pool = account()
    holder = None
    monkeypatch.setattr(sam_image_prefetch, 'PREFETCH_CANCEL_JOIN_SECONDS', .02, raising=False)
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'cohort-parent', worker_count=4,
                base_allowance_bytes=4*GIB, headroom_probe=lambda:16*GIB) as profile:
            with context.resource_scope(profile):
                holder = context.prefetch_image_cohort(view, source.shape, plans[0], max_cache_bytes=1000)
                ref = holder.future.result(timeout=5)
                original_join = holder._thread.join
                def join(timeout=None):
                    joining.set()
                    # Simulate the old 30s expiry without a 30s test delay.
                    original_join(timeout=.005 if timeout is not None and timeout >= 1 else timeout)
                monkeypatch.setattr(holder._thread, 'join', join)
                def finish_later():
                    assert retiring.wait(5) and joining.wait(5)
                    snapshots.append((ref.path.exists(), holder._thread.is_alive(),
                        sam_resources.sam_parent_promised_bytes(pool)))
                    release.wait(.06)  # Healthy wait outlives the cancellation deadline.
                    release.set()
                with ThreadPoolExecutor(1) as controller:
                    finished = controller.submit(finish_later)
                    try:
                        with holder:
                            assert ref.path.exists()
                    finally:
                        release.set()
                        finished.result(timeout=5)
                assert snapshots == [(True, True, 4*GIB+holder.phase_bytes)]
                assert not holder._thread.is_alive() and not ref.path.exists()
                assert not context._image_prefetches and pool.in_use == 4*GIB
        assert sam_resources.sam_parent_promised_bytes(pool) == 0
    finally:
        release.set()
        if holder is not None:
            holder.close()
        context.close()


@pytest.mark.parametrize('signal', ['holder', 'context'])
def test_cancellation_during_normal_retirement_wait_is_bounded_and_retains_ownership(tmp_path, monkeypatch, signal):
    from XTA import sam_image_prefetch
    retiring, release, joining = threading.Event(), threading.Event(), threading.Event()
    def retirement(ref):
        retiring.set()
        assert release.wait(5)
        return dict(status='retired', workers_finished=True, gray_mappings_retired=True)
    source, context, view, _bbox, plans, _receipts = setup(tmp_path, monkeypatch, retirement)
    pool = account()
    holder = None
    monkeypatch.setattr(sam_image_prefetch, 'PREFETCH_CANCEL_JOIN_SECONDS', .02, raising=False)
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'cohort-parent', worker_count=4,
                base_allowance_bytes=4*GIB, headroom_probe=lambda:16*GIB) as profile:
            with context.resource_scope(profile):
                holder = context.prefetch_image_cohort(view, source.shape, plans[0], max_cache_bytes=1000)
                ref = holder.future.result(timeout=5)
                original_join = holder._thread.join
                def join(timeout=None):
                    joining.set()
                    original_join(timeout=min(.005, timeout) if timeout is not None else .005)
                monkeypatch.setattr(holder._thread, 'join', join)
                def cancel_wait():
                    assert retiring.wait(5) and joining.wait(5)
                    if signal == 'holder':
                        holder.cancel_builder()
                    else:
                        context._cancel.set()
                with ThreadPoolExecutor(1) as controller:
                    stopped = controller.submit(cancel_wait)
                    with pytest.raises(RuntimeError, match='producer remains active; image/profile ownership retained'):
                        with holder:
                            pass
                    stopped.result(timeout=5)
                assert retiring.is_set() and holder._thread.is_alive() and ref.path.exists()
                assert holder in context._image_prefetches
                assert sam_resources.sam_parent_promised_bytes(pool) == 4*GIB+holder.phase_bytes
                release.set()
                holder.close()
                assert not holder._thread.is_alive() and not ref.path.exists()
                assert not context._image_prefetches and pool.in_use == 4*GIB
        assert sam_resources.sam_parent_promised_bytes(pool) == 0
    finally:
        release.set()
        if holder is not None:
            holder.close()
        context.close()


def test_lookahead_donor_pin_releases_before_lookahead_consumer_entry(tmp_path, monkeypatch):
    from XTA import sam_image_prefetch
    source, context, view, bbox, plans, _receipts = setup(tmp_path, monkeypatch)
    pool = account()
    copying, release_copy, joining = threading.Event(), threading.Event(), threading.Event()
    current = lookahead = None
    monkeypatch.setattr(sam_image_prefetch, 'PREFETCH_CANCEL_JOIN_SECONDS', .02)
    try:
        with sam_resources.admit_sam_parent_resources(pool, 4*GIB, 'cohort-parent', worker_count=4,
                base_allowance_bytes=4*GIB, headroom_probe=lambda:16*GIB) as profile:
            with context.resource_scope(profile):
                current = context.prefetch_image_cohort(view, source.shape, plans[0], max_cache_bytes=1000)
                ref = current.future.result(timeout=5)
                original_render = context._render_demand_crop
                def paused_render(*args, **kwargs):
                    with context._idle:
                        assert context._cache_owners[str(ref.path)]['pins'] > 0
                    copying.set()
                    assert release_copy.wait(5)
                    return original_render(*args, **kwargs)
                monkeypatch.setattr(context, '_render_demand_crop', paused_render)
                original_join = current._thread.join
                def join(timeout=None):
                    joining.set()
                    original_join(timeout=timeout)
                monkeypatch.setattr(current._thread, 'join', join)
                expanded = (1, 2, 9, 10)
                def finish_copy():
                    assert copying.wait(5) and joining.wait(5)
                    with context._idle:
                        assert context._idle.wait_for(
                            lambda: context._cache_owners[str(ref.path)]['retiring'], timeout=5)
                        assert context._cache_owners[str(ref.path)]['pins'] > 0
                        assert ref.path.exists() and not lookahead.future.done()
                        assert not lookahead._entered
                    release_copy.wait(.06)
                    release_copy.set()
                with ThreadPoolExecutor(1) as controller:
                    try:
                        with current:
                            lookahead = context.prefetch_image_cohort(view, source.shape,
                                demand(source.shape, {0:expanded}), max_cache_bytes=1000)
                            assert copying.wait(5)
                            finished = controller.submit(finish_copy)
                        finished.result(timeout=5)
                    finally:
                        release_copy.set()
                assert not current._thread.is_alive() and not ref.path.exists()
                next_ref = lookahead.future.result(timeout=5)
                assert not lookahead._entered and next_ref.path.exists()
                with lookahead:
                    np.testing.assert_array_equal(crop_from(next_ref, 0, expanded), source[0, 1:9, 2:10])
                assert not next_ref.path.exists() and not lookahead._thread.is_alive()
                assert not context._image_prefetches and pool.in_use == 4*GIB
        assert sam_resources.sam_parent_promised_bytes(pool) == 0
    finally:
        release_copy.set()
        if current is not None:
            current.close()
        if lookahead is not None:
            lookahead.close()
        context.close()
