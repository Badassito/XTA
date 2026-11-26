"""Actual native byte and publication-ticket backpressure."""
import pytest
from tests.native_scheduler_support import NativeRun

def test_held_publication_does_not_block_another_funded_parent():
    with NativeRun(capacity=128, tickets=2) as run:
        run.declare(1, bytes=64)
        run.prepare(1)
        run.submit(1, 1)
        first = run.take()[0]
        run.finish(first)
        run.published(1)
        run.declare(2, bytes=64)
        run.prepare(2)
        run.submit(2, 2)
        second = run.take()[0]
        assert run.owner.snapshot()['bytes_used'] == 128
        run.finish(second)
        run.published(2)
        assert len(run.consume_events()) == 2
        run.retire(first, parent=True)
        run.retire(second, parent=True)
        assert run.owner.snapshot()['bytes_used'] == 0

@pytest.mark.parametrize('capacity,tickets', [(64, 2), (128, 1)])
def test_actual_cap_blocks_next_parent_until_retirement(capacity, tickets):
    with NativeRun(capacity=capacity, tickets=tickets) as run:
        run.declare(1, bytes=64)
        run.prepare(1)
        run.submit(1, 1)
        first = run.take()[0]
        run.finish(first)
        run.published(1)
        run.declare(2, bytes=64)
        assert run.owner.wait_preparation(0.03) is None
        assert run.owner.snapshot()['blocked_parents'] == 1
        assert len(run.consume_events()) == 1
        run.retire(first, parent=True)
        run.prepare(2)
        run.submit(2, 2)
        assert run.take()[0].parent_id == 2

def test_final_does_not_refund_live_task_bytes():
    with NativeRun(capacity=100) as run:
        run.declare(1, bytes=40, headroom=20)
        run.prepare(1)
        run.submit(1, 1, retained_bytes=30)
        first = run.take()[0]
        run.finish(first)
        run.published(1)
        assert run.owner.snapshot()['bytes_used'] == 70
        assert len(run.consume_events()) == 1
        with pytest.raises(RuntimeError, match='retained task'):
            run.owner.retire_parent(1, run.parents[1])
        assert run.owner.snapshot()['bytes_used'] == 70
        run.retire(first, parent=True)
        assert run.owner.snapshot()['bytes_used'] == 0

def test_successful_empty_is_not_missing_inventory():
    with NativeRun() as run:
        run.declare(1, tasks=2)
        run.prepare(1)
        run.submit(1, 1, descriptor={'mask_voxels': 0})
        run.submit(2, 1, descriptor={'mask_voxels': 4})
        empty = run.take()
        run.finish(empty[0], payload=b'computed-empty')
        assert run.owner.wait_publication(0) is None
        nonempty = run.take()
        run.finish(nonempty[0])
        run.published(1)
        assert run.owner.snapshot()['final'] == 2
