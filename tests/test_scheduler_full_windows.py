"""Native worker windows without Python refill callbacks."""
import pytest
from XTA.native_scheduler import CPU, GPU
from tests.native_scheduler_support import NativeRun

@pytest.mark.parametrize('kind', [CPU, GPU])
@pytest.mark.parametrize('window', [1, 2, 4])
def test_compute_refills_exact_window_before_publication(kind, window):
    with NativeRun(kind=kind, window=window) as run:
        run.declare(1, tasks=window + 1, bytes=64)
        run.prepare(1)
        for task in range(window + 1):
            run.submit(task, 1, kind=kind, descriptor={'slice_start': task, 'slice_count': 1})
        admitted = [run.take() for _ in range(window)]
        assert {item[1]['slice_start'] for item in admitted} == set(range(window))
        assert run.take(timeout=0.03) is None
        first = admitted[0][0]
        run.receipt(first, 'installed')
        run.receipt(first, 'compute')
        following = run.take()
        assert following[1]['slice_start'] == window
        assert run.owner.wait_publication(0) is None
        run.receipt(first, 'final', payload=b'first')
        assert run.owner.wait_task_final(first.task_id, first.generation, 2)
        for lease, _ in admitted[1:] + [following]:
            run.finish(lease)
        run.published(1)
        assert run.wait_snapshot(lambda s: s['active'] == 0)['final'] == window + 1
        assert len(run.consume_events()) == window + 1
        for lease, _ in admitted + [following]:
            run.retire(lease)
        run.owner.retire_parent(1, run.parents[1])
        assert run.owner.snapshot()['bytes_used'] == 0

def test_windows_and_stage_guards_are_per_device():
    with NativeRun(workers=2) as run:
        run.declare(1, tasks=2)
        run.prepare(1)
        run.submit(1, 1, required_worker=0)
        run.submit(2, 1, required_worker=1)
        first, second = run.take(0), run.take(1)
        assert first[0].worker_id == 0 and second[0].worker_id == 1
        assert run.owner.try_acquire_stage(0) is None
        run.finish(first[0])
        stage = run.owner.try_acquire_stage(0)
        assert stage is not None
        assert run.owner.try_acquire_stage(1) is None
        run.owner.release_stage(0, stage)
        run.finish(second[0])
        run.published(1)
