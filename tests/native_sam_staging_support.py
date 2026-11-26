"""Synchronous test transport around the real native owner for scientific staging tests.

No admission algorithm is implemented here: the compiled run owns all restored
bytes and tickets. These small tests manually schedule scientific executors;
the production-adjacent suite separately exercises the threaded actor ports.
"""
from __future__ import annotations

import weakref

from tests.native_detector_test_support import DetectorRun, detector_inputs, detector_state


class NativeStagingOwner:
    def __init__(self, root, leases, capacity):
        capacity = int(capacity() if callable(capacity) else capacity)
        state = detector_state()
        state.direct_union_backing_leases = leases.leases
        state.direct_union_inference_views = leases.inference_views
        state.direct_union_inference_bytes = leases.inference_bytes
        state.direct_union_postprocess_views = leases.postprocess_views
        state.direct_union_postprocess_bytes = leases.postprocess_bytes
        inputs = detector_inputs(root, direct_union_total_dense_byte_limit=capacity,
                                 direct_union_inference_byte_limit=capacity,
                                 direct_union_inference_view_limit=4096)
        self.run = DetectorRun(root, state=state, inputs=inputs)
        self.run.declare(ticket_limit=4096)
        self.failed = False
        self._fixture_owners = {}

    def retain_for(self, stage):
        # A test can close staging while its returned scientific arrays remain
        # live. The run owner therefore lives until the stage itself is gone.
        weakref.finalize(stage, self.close)
        return self

    def _drain(self):
        for channel, take in (('prepared', self.run.native.wait_preparation),
                              ('publication', self.run.native.wait_publication)):
            while (event := take(0)) is not None:
                self.run.scheduler._handle_event(channel, event)
        self.run.scheduler.service_preparation()

    def _synchronize_fixture_owners(self):
        """Charge manually installed initial science inputs in the same native run.

        Scientific unit fixtures install arrays directly instead of running a
        detector. Those real lease objects must still receive native grants;
        otherwise a restore test would hide competing bytes from the arbiter.
        """
        self._drain()
        registry = self.run.state.direct_union_backing_leases
        for identity, (key, lease, namespace) in tuple(self._fixture_owners.items()):
            if registry.get(key) is not lease and lease.phase == 'released' and lease.owner_count == 0:
                self.run.scheduler.host_workspace_retired(key, namespace=namespace, owner=lease)
                del self._fixture_owners[identity]
        self._drain()
        bound = {id(lease) for parent in self.run.scheduler._parents.values()
                 for _key, lease in parent.owners}
        for key, lease in tuple(registry.items()):
            if id(lease) in bound or id(lease) in self._fixture_owners:
                continue
            namespace = ('fixture_initial_owner', id(lease))
            options = dict(namespace=namespace, allow_oversize=(
                lease.nbytes > self.run.native.snapshot()['byte_capacity']))
            ready = self.run.scheduler.request_host_workspace(key, lease.nbytes, **options)
            while not ready:
                event = self.run.native.wait_preparation(1)
                if event is None:
                    raise AssertionError('Fixture installed input owners without fundable native capacity')
                self.run.scheduler._handle_event('prepared', event)
                ready = self.run.scheduler.request_host_workspace(key, lease.nbytes, **options)
            self.run.scheduler.host_workspace_bound(key, namespace=namespace, owner=lease)
            self.run.scheduler.host_workspace_adopted(key, namespace=namespace, owner=lease)
            self._fixture_owners[id(lease)] = (key, lease, namespace)
            self._drain()

    def request_host_workspace(self, key, nbytes, **options):
        self._synchronize_fixture_owners()
        ready = self.run.scheduler.request_host_workspace(key, nbytes, **options)
        if not ready:
            # A bounded wait in this test adapter lets a real native grant reach
            # the caller before a manually queued executor is advanced.
            event = self.run.native.wait_preparation(0.1)
            if event is not None:
                self.run.scheduler._handle_event('prepared', event)
            self._drain()
            ready = self.run.scheduler.request_host_workspace(key, nbytes, **options)
        return ready

    def host_workspace_bound(self, key, **options):
        return self.run.scheduler.host_workspace_bound(key, **options)

    def host_workspace_adopted(self, key, **options):
        result = self.run.scheduler.host_workspace_adopted(key, **options)
        self._drain()
        return result

    def host_workspace_retired(self, key, **options):
        self._drain()
        return self.run.scheduler.host_workspace_retired(key, **options)

    def host_workspace_failed(self, key, error, **options):
        self.failed = True
        return self.run.scheduler.host_workspace_failed(key, error, **options)

    def wake_scheduler(self):
        self.run.scheduler.wake_scheduler()

    def close(self):
        self.run.native.close()
