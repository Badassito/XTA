"""Native descriptors preserve order, capabilities and scientific ranges."""
import pytest
from XTA.native_scheduler import CPU, GPU
from tests.native_scheduler_support import NativeRun

def test_priority_stable_ties_preserve_all_ranges():
    with NativeRun() as run:
        stage = run.owner.try_acquire_stage(0)
        run.declare(1, tasks=3)
        run.prepare(1)
        for task, priority, start, count in [(1, 5, 0, 4), (2, 1, 4, 3), (3, 1, 7, 2)]:
            run.submit(task, 1, priority=priority,
                       descriptor={'slice_start': start, 'slice_count': count, 'view': 'radial'})
        run.owner.release_stage(0, stage)
        seen = []
        for _ in range(3):
            lease, descriptor = run.take()
            seen.append(descriptor)
            run.finish(lease)
        assert [d['task_id'] for d in seen] == [2, 3, 1]
        assert sorted(i for d in seen for i in range(d['slice_start'], d['slice_start']+d['slice_count'])) == list(range(9))
        run.published(1)

def test_incompatible_required_worker_rejected_before_enqueue():
    with NativeRun() as run:
        run.add_worker(1, kind=CPU, device=-1)
        run.declare(1)
        run.prepare(1)
        with pytest.raises(RuntimeError, match='incompatible'):
            run.submit(1, 1, kind=GPU, required_worker=1)
        assert run.owner.snapshot()['ready'] == 0
        assert run.owner.snapshot()['dispatched'] == 0
        run.submit(1, 1, kind=GPU, required_worker=0)
        assert run.take(0)[0].worker_id == 0

def test_exclusive_parent_keeps_every_range_on_one_actor():
    with NativeRun(workers=2) as run:
        run.declare(1, tasks=3, exclusive=GPU)
        run.prepare(1)
        for task in range(3):
            run.submit(task, 1, descriptor={'slice_start': 4*task, 'slice_count': 4})
        leases = []
        for _ in range(3):
            lease, descriptor = run.take(0)
            leases.append(lease)
            run.finish(lease)
        assert {lease.worker_id for lease in leases} == {0}
        assert run.take(1, timeout=0.03) is None
        run.published(1)

def test_duplicate_identity_cannot_replace_pending_range():
    with NativeRun() as run:
        stage = run.owner.try_acquire_stage(0)
        run.declare(1, tasks=2)
        run.prepare(1)
        run.submit(9, 1, descriptor={'slice_start': 0, 'slice_count': 4})
        with pytest.raises(RuntimeError, match='duplicate task'):
            run.submit(9, 1, descriptor={'slice_start': 999, 'slice_count': 1})
        run.submit(10, 1, descriptor={'slice_start': 4, 'slice_count': 4})
        run.owner.release_stage(0, stage)
        lease, descriptor = run.take()
        assert descriptor['slice_start'] == 0
        run.finish(lease)
        assert run.take()[1]['slice_start'] == 4
