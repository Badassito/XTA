"""Final family accounting follows worker and raw-consumer completion."""
from __future__ import annotations

import pytest

from XTA.sam_tracker_runtime import SamTrackerFamily, _FamilyDispatch
from tests.test_sam_family_dispatch import families_for, tracker_for
from tests.test_sam_tracker_runtime import _CompletionPool


def test_exhausted_family_is_retired_only_on_its_free_worker_without_new_request():
    calls = []
    family = SamTrackerFamily('only', (7,), ('run-7',),
        lambda index: calls.append(index) or {'run_id': 'run-7'})
    dispatch = _FamilyDispatch((family,), 'fifo')
    assert dispatch.next_for({3})[:3] == (3, 7, 'only')
    dispatch.retire_completed({0, 1, 2})
    assert dispatch.completed_families == 0
    assert 3 in dispatch.active
    dispatch.retire_completed({3})
    dispatch.retire_completed({3})
    assert dispatch.completed_families == 1
    assert dispatch.active == {}
    assert calls == [7]


@pytest.mark.parametrize('completed', (False, True))
def test_final_family_counter_waits_for_last_raw_consumer_to_finish(tmp_path, monkeypatch, completed):
    pool = _CompletionPool()
    tracker, cache = tracker_for(tmp_path, pool, monkeypatch, devices=(0, 1))
    calls, refs = [], []
    families = families_for([('only', (7,))], calls, refs)
    stream = tracker.iter_family_results(families, source_cache_ref=cache,
        defer_refill_until_consumed=True)
    try:
        index, result = next(stream)
        assert index == 7
        assert 'family_completed' not in tracker.dispatch_stats
        tracker.release_result(result)
        del result
        if completed:
            with pytest.raises(StopIteration):
                next(stream)
            assert tracker.dispatch_stats['family_completed'] == 1
        else:
            stream.close()
            assert 'family_completed' not in tracker.dispatch_stats
            assert tracker.workers_settled
        assert calls == [7]
        assert list(tracker.artifact_root.glob('run-*')) == []
    finally:
        stream.close()
        tracker.close()


def test_empty_family_inventory_finishes_without_submitting_a_job(tmp_path, monkeypatch):
    pool = _CompletionPool()
    tracker, cache = tracker_for(tmp_path, pool, monkeypatch, devices=(0, 1))
    try:
        assert list(tracker.iter_family_results((), source_cache_ref=cache)) == []
        assert tracker.dispatch_stats['family_completed'] == 0
        assert pool.submissions == []
    finally:
        tracker.close()
