"""Family submission provenance belongs to one consumed scope, never shared stats."""
from concurrent.futures import ThreadPoolExecutor
import json
import threading

import numpy as np
import pytest

from XTA.sam_interpolation import SamInterpolationInfrastructureError
from tests.test_sam_family_dispatch import families_for, request, tracker_for
from tests.test_sam_interpolation import RepeatedSeedTracker, _close, _generate
from tests.test_sam_tracker_runtime import _CompletionPool


def test_scope_owned_receipt_survives_next_scope_overwriting_global_diagnostics(tmp_path, monkeypatch):
    tracker, cache = tracker_for(tmp_path, _CompletionPool(), monkeypatch, devices=(0,))
    first_drained, second_drained = threading.Event(), threading.Event()
    received = {}

    def first():
        receipts = []
        stream = tracker.iter_family_results(families_for([('first', (7,))], [], []),
            source_cache_ref=cache, execution_order_callback=receipts.append)
        results = list(stream)
        first_drained.set()
        assert second_drained.wait(10)
        received['first'] = receipts
        received['late_shared_stats'] = list(tracker.dispatch_stats['family_execution_order'])
        assert results[0][0] == 7
        assert all(np.array_equal(mask, request(7)['seed_mask']) for mask in results[0][1].frames.values())

    def second():
        assert first_drained.wait(10)
        receipts = []
        results = list(tracker.iter_family_results(families_for([('second', (11,))], [], []),
            source_cache_ref=cache, execution_order_callback=receipts.append))
        received['second'] = receipts
        assert results[0][0] == 11
        second_drained.set()

    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            one, two = workers.submit(first), workers.submit(second)
            one.result(timeout=15)
            two.result(timeout=15)
        assert received == {'first': [(7,)], 'second': [(11,)], 'late_shared_stats': [11]}
    finally:
        first_drained.set()
        second_drained.set()
        tracker.close()


@pytest.mark.parametrize('consume_final_result', [False, True])
def test_submission_receipt_requires_successful_final_consumer_completion(tmp_path, monkeypatch, consume_final_result):
    tracker, cache = tracker_for(tmp_path, _CompletionPool(), monkeypatch, devices=(0,))
    receipts = []
    stream = tracker.iter_family_results(families_for([('only', (7,))], [], []),
        source_cache_ref=cache, execution_order_callback=receipts.append)
    try:
        _index, result = next(stream)
        assert receipts == []
        tracker.release_result(result)
        if consume_final_result:
            with pytest.raises(StopIteration):
                next(stream)
            assert receipts == [(7,)]
        else:
            stream.close()
            assert receipts == [] and tracker.workers_settled
    finally:
        stream.close()
        tracker.close()


class _ScopeTracker(RepeatedSeedTracker):
    device_ids = (0, 1, 2, 3)

    def __init__(self):
        super().__init__()
        self.dispatch_stats = {}

    def _iterate(self, families):
        order = []
        for family in reversed(families):
            for index in reversed(family.input_indices):
                order.append(index)
                req = family.request_factory(index)
                result = self.run(**req)
                result.receipt['run_id'] = req['run_id']
                yield index, result
        self.dispatch_stats['family_execution_order'] = [999]
        self.order = tuple(order)


class _ReceiptTracker(_ScopeTracker):
    def __init__(self, mutation=None):
        super().__init__()
        self.mutation = mutation

    def iter_family_results(self, families, *, source_cache_ref=None, max_in_flight=None,
            defer_refill_until_consumed=False, execution_order_callback=None):
        yield from self._iterate(families)
        if self.mutation == 'missing':
            return
        order = self.order
        if self.mutation == 'different':
            order = tuple(reversed(order))
        elif self.mutation == 'duplicate-index':
            order = (order[0], order[0])
        elif self.mutation == 'omitted':
            order = order[:1]
        elif self.mutation == 'boolean':
            order = (True, 0)
        elif self.mutation == 'mutable':
            order = list(order)
        execution_order_callback(order)
        if self.mutation == 'repeated':
            execution_order_callback(order)


class _LegacyTracker(_ScopeTracker):
    def iter_family_results(self, families, *, source_cache_ref=None, max_in_flight=None,
            defer_refill_until_consumed=False):
        yield from self._iterate(families)


class _IgnoredCallbackTracker(_ScopeTracker):
    def iter_family_results(self, families, **options):
        yield from self._iterate(families)


@pytest.mark.parametrize('kind', ['receipt', 'legacy', 'ignored-callback'])
def test_real_generator_uses_owned_family_order_and_preserves_selected_pixels(tmp_path, monkeypatch, kind):
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE', 'fifo')
    tracker = {'receipt': _ReceiptTracker, 'legacy': _LegacyTracker,
        'ignored-callback': _IgnoredCallbackTracker}[kind]()
    merged, stats, components = _generate(tmp_path, tracker)
    try:
        assert tracker.dispatch_stats['family_execution_order'] == [999]
        assert stats['sam_execution_order'] == stats['sam_family_completion_order'] == [1, 0]
        assert stats['sam_execution_order_provenance'] == ('completed_iterator_submission_receipt'
            if kind == 'receipt' else 'scope_owned_request_factory')
        expected = np.zeros(merged.shape, np.uint8)
        expected[:, 9:15, 10:16] = 1
        np.testing.assert_array_equal(merged, expected)
        assert len(components) == 2 and [entry['voxel_count'] for entry in components] == [108, 108]
    finally:
        _close(merged)


@pytest.mark.parametrize('mutation', ['missing', 'different', 'duplicate-index', 'omitted', 'boolean', 'mutable', 'repeated'])
def test_malformed_submission_receipt_refuses_complete_evidence_publication(tmp_path, monkeypatch, mutation):
    monkeypatch.setenv('YOLO_TTA_SAM_FAMILY_SCHEDULE', 'fifo')
    with pytest.raises(SamInterpolationInfrastructureError, match='submission order'):
        _generate(tmp_path, _ReceiptTracker(mutation))
    manifests = list(tmp_path.glob('sam_*/evidence/manifest.json'))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text(encoding='utf-8'))['complete'] is False
    failure = json.loads(next(tmp_path.glob('sam_*/failure.json')).read_text(encoding='utf-8'))
    assert failure['complete'] is False
