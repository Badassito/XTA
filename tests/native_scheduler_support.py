"""Small actor/observer fixtures for the real native scheduler, never a selector model."""
from __future__ import annotations

import itertools
import os
import pickle
import select
import socket
import time

from XTA.native_scheduler import CPU, GPU, PreparedWorkScheduler, recv_dispatch, send_receipt


_RUN_IDS = itertools.count(10000)


class NativeRun:
    """Own real native threads and socket actors; tests decide only when actors reply."""

    def __init__(self, *, capacity=1024, tickets=8, inference_capacity=None,
                 preparation_window=None, workers=1, kind=GPU, window=1):
        self.owner = PreparedWorkScheduler(
            next(_RUN_IDS), capacity, tickets,
            inference_byte_capacity=inference_capacity, preparation_window=preparation_window,
        )
        self.channels = {}
        self.host_channels = []
        self.parents = {}
        for worker in range(workers):
            self.add_worker(worker, kind=kind, window=window,
                            device=worker if kind == GPU else -1)

    def add_worker(self, worker, *, kind=GPU, window=1, device=None, epoch=1):
        host, actor = socket.socketpair()
        self.host_channels.append(host)
        self.channels[worker] = actor
        self.owner.register_worker(worker, epoch, os.getpid(), kind, window, host,
                                   device=worker if device is None else device)

    def declare(self, parent, *, tasks=1, bytes=0, priority=0, exclusive=0, headroom=0):
        self.owner.declare_parent(parent, tasks, bytes, priority, exclusive, headroom)

    def prepare(self, parent, *, timeout=2):
        actual, generation = self.owner.wait_preparation(timeout)
        assert actual == parent, (actual, parent, self.owner.snapshot())
        self.parents[parent] = generation
        return generation

    def submit(self, task, parent, *, descriptor=None, kind=GPU, priority=0,
               required_worker=-1, retained_bytes=0):
        descriptor = dict(descriptor or {}, task_id=task)
        self.owner.submit_prepared(task, parent, pickle.dumps(descriptor), kind,
                                   priority, required_worker, retained_bytes)

    def take(self, worker=0, *, timeout=2):
        channel = self.channels[worker]
        if not select.select([channel], [], [], timeout)[0]:
            return None
        lease, payload = recv_dispatch(channel)
        return lease, pickle.loads(payload)

    def receipt(self, lease, kind, *, payload=b'', reason=''):
        send_receipt(self.channels[lease.worker_id], lease, kind, payload=payload, reason=reason)

    def finish(self, lease, *, final_first=False, payload=b'result'):
        self.receipt(lease, 'installed')
        if final_first:
            self.receipt(lease, 'final', payload=payload)
            self.receipt(lease, 'compute')
        else:
            self.receipt(lease, 'compute')
            self.receipt(lease, 'final', payload=payload)
        assert self.owner.wait_task_final(lease.task_id, lease.generation, 2)

    def published(self, parent, *, timeout=2):
        actual, generation = self.owner.wait_publication(timeout)
        assert actual == parent, (actual, parent, self.owner.snapshot())
        assert generation == self.parents[parent]
        return generation

    def retire(self, lease, *, parent=False):
        self.owner.retire_task(lease.task_id, lease.generation)
        if parent:
            self.owner.retire_parent(lease.parent_id, self.parents[lease.parent_id])

    def consume_events(self):
        """Explicitly observe finished science payloads before their owner can retire."""
        events = []
        while (event := self.owner.wait_event(0)) is not None:
            events.append(event)
        return events

    def wait_snapshot(self, predicate, *, timeout=2):
        deadline = time.monotonic() + timeout
        while True:
            value = self.owner.snapshot()
            if predicate(value):
                return value
            if time.monotonic() >= deadline:
                raise AssertionError(f'native state did not settle: {value}')
            time.sleep(0.001)

    def close(self):
        self.owner.close()
        for channel in (*self.channels.values(), *self.host_channels):
            channel.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
