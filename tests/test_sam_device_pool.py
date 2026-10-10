"""Fixed cohorts join one route owner without exposing unready workers."""
from collections import deque
import threading
from types import SimpleNamespace

import pytest

from XTA.sam_device_pool import SamDeviceWorkerPools


class Pool:
    def __init__(self, device, sessions=2):
        self.device_ids = (device,)
        self.worker_slots = tuple((device,index) for index in range(sessions))
        self.ready_events = tuple(SimpleNamespace(execution_device_id=device, worker_index=index,
            worker_pid=100+10*device+index) for index in range(sessions))
        self.pids_by_slot = {slot:event.worker_pid for slot,event in zip(self.worker_slots,self.ready_events)}
        self.results, self.submitted = deque(), []
        self.closed = False
        self.workers_settled = False
        self.fail_shutdown = False

    def submit(self, task, **route):
        self.submitted.append((task,route))

    def wait_result(self, timeout):
        if not self.results:
            raise TimeoutError()
        return self.results.popleft()

    def check_liveness(self):
        if self.closed:
            raise RuntimeError('closed child')

    def shutdown(self, *, timeout, force):
        self.closed = True
        if self.fail_shutdown:
            raise RuntimeError('unsettled child shutdown')
        self.workers_settled = True
        return ()

    def force_close(self, *, timeout):
        self.workers_settled = True
        return self.device_ids


def test_registration_publishes_only_admitted_slots_and_routes_exactly():
    router = SamDeviceWorkerPools((0,2))
    first, later = Pool(0), Pool(2,1)
    router.retain(later)
    assert router.worker_slots == () and not router.workers_settled
    router.register(0,first)
    assert router.worker_slots == ((0,0),(0,1))
    with pytest.raises(RuntimeError, match='admitted'):
        router.submit('premature',execution_device_id=2)
    router.submit('first',execution_device_id=0,worker_index=1)
    router.register(2,later)
    router.submit('later',execution_device_id=2)
    assert first.submitted == [('first',{'execution_device_id':0,'worker_index':1})]
    assert later.submitted == [('later',{'execution_device_id':2,'worker_index':0})]
    assert router.pids_by_slot == {(0,0):100,(0,1):101,(2,0):120}
    assert len(router.ready_events) == 3
    router.shutdown()
    assert router.workers_settled


def test_result_polling_is_fair_and_does_not_wait_on_first_idle_cohort():
    router = SamDeviceWorkerPools((0,1))
    first, later = Pool(0), Pool(1)
    router.register(0,first)
    router.register(1,later)
    later.results.extend(('late-first','late-second'))
    assert router.wait_result(timeout=0.) == 'late-first'
    first.results.append('first-ready')
    assert router.wait_result(timeout=0.) == 'first-ready'
    assert router.wait_result(timeout=0.) == 'late-second'
    with pytest.raises(TimeoutError):
        router.wait_result(timeout=0.)
    router.shutdown()


def test_rejected_child_still_retains_cleanup_ownership_and_failure_does_not_skip_peer():
    router = SamDeviceWorkerPools((0,1))
    first, failed = Pool(0), Pool(1)
    first.fail_shutdown = True
    router.register(0,first)
    router.retain(failed)
    with pytest.raises(RuntimeError, match='unsettled child'):
        router.shutdown()
    assert failed.workers_settled and not router.workers_settled
    assert router.force_close() == (0,1)
    assert router.workers_settled


def test_duplicate_registration_cancellation_and_closed_registry_are_not_usable():
    cancelled = threading.Event()
    router = SamDeviceWorkerPools((0,),cancel_event=cancelled)
    first = Pool(0)
    router.register(0,first)
    with pytest.raises(RuntimeError, match='identity'):
        router.register(0,Pool(0))
    cancelled.set()
    with pytest.raises(RuntimeError, match='cancelled'):
        router.wait_result(timeout=1.)
    router.shutdown()
    late = Pool(0)
    router.retain(late)
    assert not router.workers_settled
    with pytest.raises(RuntimeError, match='identity'):
        router.register(0,late)
    router.force_close()
    assert router.workers_settled


def test_shutdown_cannot_claim_settlement_while_late_constructor_is_running():
    router = SamDeviceWorkerPools((0,))
    router.begin_boot(0)
    router.shutdown()
    assert not router.workers_settled
    late = Pool(0)
    router.retain(late)
    router.finish_boot(0)
    assert not router.workers_settled
    router.force_close()
    assert router.workers_settled
