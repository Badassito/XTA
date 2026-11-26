"""Python telemetry consumption cannot gate native dispatch progress."""
from tests.native_scheduler_support import NativeRun

def test_undrained_events_do_not_gate_compute_credit():
    with NativeRun() as run:
        run.declare(1, tasks=3)
        run.prepare(1)
        for task in range(3):
            run.submit(task, 1)
        observed = []
        for _ in range(3):
            lease, _ = run.take()
            observed.append(lease.task_id)
            run.finish(lease, payload=str(lease.task_id).encode())
        run.published(1)
        assert observed == [0, 1, 2]
        assert run.owner.snapshot()['events'] == 3
        events = [run.owner.wait_event(1) for _ in range(3)]
        assert [event[2] for event in events] == [b'0', b'1', b'2']
        assert run.owner.wait_event(0) is None
