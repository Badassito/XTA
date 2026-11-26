"""Native receipt ordering and independent compute/payload credits."""
import pytest
from tests.native_scheduler_support import NativeRun

@pytest.mark.parametrize('final_first', [False, True])
def test_legal_completion_orders_return_window_once(final_first):
    with NativeRun() as run:
        run.declare(1, tasks=2, bytes=32)
        run.prepare(1)
        run.submit(1, 1, retained_bytes=8)
        run.submit(2, 1, retained_bytes=8)
        first = run.take()[0]
        run.finish(first, final_first=final_first)
        second = run.take()[0]
        assert second.task_id == 2
        run.finish(second, final_first=not final_first)
        run.published(1)
        snapshot = run.wait_snapshot(lambda s: s['active'] == 0)
        assert snapshot['final'] == 2 and snapshot['bytes_used'] == 48
        assert len(run.consume_events()) == 2
        run.retire(first)
        assert run.owner.snapshot()['bytes_used'] == 40
        run.retire(second, parent=True)
        assert run.owner.snapshot()['bytes_used'] == 0

def test_compute_dispatches_next_before_python_consumes_final():
    with NativeRun() as run:
        run.declare(1, tasks=2)
        run.prepare(1)
        run.submit(1, 1)
        run.submit(2, 1)
        first = run.take()[0]
        run.receipt(first, 'installed')
        run.receipt(first, 'compute')
        second = run.take()[0]
        assert second.task_id == 2
        run.receipt(first, 'final', payload=b'unconsumed-result')
        assert run.owner.wait_task_final(1, first.generation, 2)
        assert run.owner.snapshot()['events'] == 1
        run.finish(second)
        run.published(1)

def test_duplicate_compute_aborts_without_double_credit():
    with NativeRun() as run:
        run.declare(1, bytes=20)
        run.prepare(1)
        run.submit(1, 1, retained_bytes=10)
        lease = run.take()[0]
        run.receipt(lease, 'installed')
        run.receipt(lease, 'compute')
        run.receipt(lease, 'compute')
        snapshot = run.wait_snapshot(lambda s: s['failed'])
        assert snapshot['active'] == 0 and snapshot['bytes_used'] == 30
        with pytest.raises(RuntimeError, match='duplicate compute'):
            run.owner.wait_publication(0)

def test_failed_actor_stops_queue_without_publication_or_refund():
    with NativeRun() as run:
        run.declare(1, tasks=2, bytes=40)
        run.prepare(1)
        run.submit(1, 1, retained_bytes=10)
        run.submit(2, 1, retained_bytes=10)
        lease = run.take()[0]
        run.receipt(lease, 'installed')
        run.receipt(lease, 'failed', reason='injected model failure')
        snapshot = run.wait_snapshot(lambda s: s['failed'])
        assert snapshot['dispatched'] == 1 and snapshot['final'] == 0
        assert snapshot['bytes_used'] == 50 and snapshot['publication_tickets'] == 0
