"""A completed intrinsic lane returns credit without waiting for an older run."""
from contextlib import contextmanager
import threading
from types import SimpleNamespace

from XTA import sam_policy


class Reader:
    def __init__(self):
        self.active = 0
        self.lock = threading.Lock()

    @contextmanager
    def fork(self, *, max_cache_bytes):
        with self.lock:
            self.active += 1
        try:
            yield SimpleNamespace(stats={}, borrowed_filter_snapshot=lambda value: value)
        finally:
            with self.lock:
                self.active -= 1


def execution():
    return dict(parallel_credit_bytes=2, requested_workers=2, lane_cache_bytes=0,
        maximum_run_charge_bytes=0, serial_group_count=0, serial_reasons={},
        parallel_group_count=0, peak_pending_runs=0, peak_charged_bytes=0,
        parallel_run_count=0, oversized_serial_runs=0,
        reader_totals={name: 0 for name in (
            'mask_decodes', 'filter_computations', 'effective_candidate_computations',
            'cache_hits', 'cache_misses', 'cache_evictions', 'compact_filter_expansions',
            'compact_filter_parent_hits', 'compact_filter_parent_exports')})


def test_completed_younger_lane_refills_before_oldest_finishes(monkeypatch):
    reader, audit = Reader(), execution()
    oldest_entered, replacement_entered = threading.Event(), threading.Event()
    monkeypatch.setattr(sam_policy, '_measurement_charge', lambda *_args: 1)

    def measure(_reader, key, **_kwargs):
        if key == 'oldest':
            oldest_entered.set()
            assert replacement_entered.wait(3), 'A completed younger lane did not return its credit'
        elif key == 'second':
            assert oldest_entered.wait(3)
        elif key == 'replacement':
            replacement_entered.set()
        return dict(run_id=key)

    monkeypatch.setattr(sam_policy, 'measure_sam_run', measure)
    order = ['oldest', 'second', 'replacement', 'last']
    measured = sam_policy._measure_group_intrinsic(reader, {}, order, None, audit)
    assert list(measured) == order
    assert [row['run_id'] for row in measured.values()] == order
    assert audit['parallel_run_count'] == 4
    assert audit['peak_pending_runs'] == audit['peak_charged_bytes'] == 2
    assert reader.active == 0


def test_oversized_measurement_waits_for_all_lane_owners_before_serial_fallback(monkeypatch):
    reader, audit = Reader(), execution()
    paired = threading.Barrier(2)
    owner_thread = threading.get_ident()
    calls = []
    charges = dict(first=1, second=1, oversized=3, last=1)
    monkeypatch.setattr(sam_policy, '_measurement_charge', lambda _bundle, _group, key, _cache: charges[key])

    def measure(active_reader, key, **_kwargs):
        if key in ('first', 'second'):
            paired.wait(3)
        if key == 'oversized':
            assert active_reader is reader and reader.active == 0
            assert threading.get_ident() == owner_thread
        else:
            assert active_reader is not reader
        calls.append(key)
        return dict(run_id=key)

    monkeypatch.setattr(sam_policy, 'measure_sam_run', measure)
    order = list(charges)
    measured = sam_policy._measure_group_intrinsic(reader, {}, order, None, audit)
    assert list(measured) == order
    assert sorted(calls) == sorted(order)
    assert calls.index('oversized') > calls.index('first')
    assert calls.index('oversized') > calls.index('second')
    assert audit['oversized_serial_runs'] == 1 and audit['parallel_run_count'] == 3
    assert audit['peak_charged_bytes'] <= audit['parallel_credit_bytes']
    assert reader.active == 0
