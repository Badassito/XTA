"""Intrinsic lanes preserve ordered policy decisions and outer integrity."""
import threading

import numpy as np
import pytest

from XTA import sam_policy
from XTA.sam_filtering import build_mask_filter
from XTA.sam_resources import admit_sam_parent_resources
from XTA.sam_mask_reader import SamMaskReader
from tests.test_sam_guarded_rescue import _bundle
from tests.test_sam_selection_resources import Pool, GIB


def test_live_parallel_measurements_are_exact_and_really_concurrent(tmp_path, monkeypatch):
    bundle, _, _, _ = _bundle(tmp_path)
    serial = sam_policy.select_sam_proposals(bundle)
    original = sam_policy.measure_sam_run
    barrier = threading.Barrier(2)
    threads = set()
    def measured(*args, **kwargs):
        threads.add(threading.get_ident())
        barrier.wait(timeout=10)
        return original(*args, **kwargs)
    monkeypatch.setattr(sam_policy, 'measure_sam_run', measured)
    pool = Pool()
    with admit_sam_parent_resources(pool, GIB, 'parallel', headroom_probe=lambda:64*GIB) as profile:
        parallel = sam_policy.select_sam_proposals(bundle, workers=4, resource_profile=profile)
    assert len(threads) == 2 and threading.get_ident() not in threads
    assert parallel['selected_run_ids'] == serial['selected_run_ids']
    assert parallel['run_receipts'] == serial['run_receipts']
    assert parallel['group_receipts'] == serial['group_receipts']
    assert parallel['guarded_rescue'] == serial['guarded_rescue']
    assert parallel['resolved_policy'] == serial['resolved_policy']
    assert parallel['policy_hash'] == serial['policy_hash']
    execution = parallel['selection_resources']['intrinsic_measurements']
    assert execution['parallel_run_count'] == 2 and execution['peak_pending_runs'] == 2
    assert execution['peak_charged_bytes'] <= execution['parallel_credit_bytes']
    assert parallel['reader_cache']['integrity_checks'] == 2
    assert pool.in_use == 0


def test_worker_hint_without_live_capacity_stays_serial(tmp_path, monkeypatch):
    bundle, _, _, _ = _bundle(tmp_path)
    original = sam_policy.measure_sam_run
    threads = []
    def measured(*args, **kwargs):
        threads.append(threading.get_ident())
        return original(*args, **kwargs)
    monkeypatch.setattr(sam_policy, 'measure_sam_run', measured)
    receipt = sam_policy.select_sam_proposals(bundle, workers=128)
    execution = receipt['selection_resources']['intrinsic_measurements']
    assert execution['fallback_reason'] == 'no_authenticated_extra_capacity'
    assert execution['peak_pending_runs'] == execution['parallel_run_count'] == 0
    assert set(threads) == {threading.get_ident()}


def test_custom_hook_keeps_same_thread_and_order_even_with_credit(tmp_path, monkeypatch):
    bundle, _, _, _ = _bundle(tmp_path)
    original = sam_policy.measure_sam_run
    calls = []
    def measured(*args, **kwargs):
        calls.append(('measure', args[1], threading.get_ident()))
        return original(*args, **kwargs)
    def hook(context):
        calls.append(('hook', context['group']['group_id'], threading.get_ident()))
        return []
    monkeypatch.setattr(sam_policy, 'measure_sam_run', measured)
    with admit_sam_parent_resources(Pool(), GIB, 'hook', headroom_probe=lambda:64*GIB) as profile:
        receipt = sam_policy.select_sam_proposals(bundle,
            {'proposal_api_version':1, 'select_proposals':hook}, workers=4, resource_profile=profile)
    assert [(row[0], row[1]) for row in calls] == [('measure','backward'),('measure','forward'),('hook','family')]
    assert all(row[2] == threading.get_ident() for row in calls)
    assert receipt['selection_resources']['intrinsic_measurements']['fallback_reason'] == 'custom_hook_order_preserved'


def test_fork_has_private_cursor_and_parent_bound_snapshot(tmp_path):
    bundle, _, _, _ = _bundle(tmp_path)
    with bundle.reader(max_cache_bytes=1024) as parent:
        snapshot = parent.filter_snapshot(build_mask_filter(bundle))
        with parent.fork(max_cache_bytes=1024) as child:
            assert child.bundle is bundle and child._stream is not parent._stream
            mask = child.effective_raw_mask('forward', 2, child.borrowed_filter_snapshot(snapshot))
            assert not mask.flags.writeable
            with pytest.raises(ValueError):
                mask.setflags(write=True)
            with pytest.raises(RuntimeError, match='active reader lanes'):
                parent.close()
            assert parent.active
        assert not parent._children
        assert child.stats['integrity_checks'] == 0
    assert parent.stats['integrity_checks'] == 2


def test_cached_lane_cannot_waive_outer_payload_corruption(tmp_path):
    bundle, _, _, _ = _bundle(tmp_path)
    with pytest.raises(ValueError, match='changed'):
        with bundle.reader() as parent:
            with parent.fork(max_cache_bytes=1024**2) as child:
                cached = child.raw_mask('forward', 2)
                path = bundle.directory/'masks.bin'
                data = path.read_bytes()
                path.write_bytes(bytes([data[0]^1])+data[1:])
                assert child.raw_mask('forward', 2) is cached


def test_measurement_failure_joins_all_lanes_before_integrity_exit(tmp_path, monkeypatch):
    bundle, _, _, _ = _bundle(tmp_path)
    original = sam_policy.measure_sam_run
    barrier = threading.Barrier(2)
    lanes = []
    def measured(reader, key, **kwargs):
        lanes.append(reader)
        barrier.wait(timeout=10)
        if key == 'backward':
            raise OSError('intrinsic worker failed')
        return original(reader, key, **kwargs)
    monkeypatch.setattr(sam_policy, 'measure_sam_run', measured)
    with admit_sam_parent_resources(Pool(), GIB, 'failure', headroom_probe=lambda:64*GIB) as profile:
        with pytest.raises(OSError, match='intrinsic worker failed'):
            sam_policy.select_sam_proposals(bundle, workers=4, resource_profile=profile)
    assert len(lanes) == 2 and all(not lane.active and lane._stream is None for lane in lanes)
    assert all(not lane._integrity_parent._children for lane in lanes)


@pytest.mark.parametrize('workers', (True, 0, -1, 1.5))
def test_invalid_worker_hint_refused_before_evidence_read(tmp_path, workers):
    with pytest.raises(ValueError, match='positive integer'):
        sam_policy.select_sam_proposals(tmp_path/'missing', workers=workers)
