"""Native device releases wake prepared inference without Python polling."""
import pytest
from tests.native_scheduler_support import NativeRun

def test_release_dispatches_without_python_refill():
    with NativeRun() as run:
        stage = run.owner.try_acquire_stage(0)
        run.declare(1)
        run.prepare(1)
        run.submit(5, 1, descriptor={'slice_start': 3, 'slice_count': 7})
        assert run.take(timeout=0.03) is None
        run.owner.release_stage(0, stage)
        lease, task = run.take()
        assert (task['slice_start'], task['slice_count']) == (3, 7)
        run.finish(lease)
        run.published(1)

def test_wrong_stage_generation_cannot_open_device():
    with NativeRun() as run:
        stage = run.owner.try_acquire_stage(0)
        run.declare(1)
        run.prepare(1)
        run.submit(1, 1)
        with pytest.raises(RuntimeError, match='generation'):
            run.owner.release_stage(0, stage + 1)
        assert run.owner.snapshot()['stage_claims'] == 1
        assert run.take(timeout=0.03) is None
        run.owner.release_stage(0, stage)
        assert run.take()[0].task_id == 1

def test_stage_blocks_only_its_device():
    with NativeRun(workers=2) as run:
        stage = run.owner.try_acquire_stage(0)
        run.declare(1, tasks=2)
        run.prepare(1)
        run.submit(1, 1, required_worker=0)
        run.submit(2, 1, required_worker=1)
        other = run.take(1)
        assert other[0].task_id == 2
        assert run.take(0, timeout=0.03) is None
        run.finish(other[0])
        run.owner.release_stage(0, stage)
        first = run.take(0)
        assert first[0].task_id == 1
        run.finish(first[0])
        run.published(1)

def test_cancelled_run_cannot_dispatch_after_stage_release():
    with NativeRun() as run:
        stage = run.owner.try_acquire_stage(0)
        run.declare(1)
        run.prepare(1)
        run.submit(1, 1)
        run.owner.cancel('injected publication failure')
        # Abort must permit actual resource retirement without reviving work.
        run.owner.release_stage(0, stage)
        snapshot = run.owner.snapshot()
        assert snapshot['failed'] and snapshot['stage_claims'] == 0
        assert snapshot['dispatched'] == snapshot['active'] == 0
        with pytest.raises(RuntimeError, match='cancelled'):
            run.owner.try_acquire_stage(0)
