"""Worker-side endpoint for native prepared inference and its scientific results."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler

from .native_scheduler import Lease, recv_dispatch, send_receipt
from .native_detector_handles import close_received_handles, install_detector_handles


@dataclass
class _ActorTask:
    lease: Lease
    received: dict[int, int]
    materialized: bool
    computed: bool = False
    finalized: bool = False


class DetectorActorPort:
    """One process owns the socket. Deferred publishers may finish on other threads."""

    def __init__(self, stream, control_results):
        self.stream = stream
        self.control_results = control_results
        self.lock = threading.Lock()
        self.leases = {}
        self.controls = {}

    def get(self):
        received = {}
        try:
            lease, payload = recv_dispatch(self.stream, received_handles=received)
        except EOFError:
            return None
        try:
            message = ForkingPickler.loads(payload)
            if lease is not None and lease.control:
                with self.lock:
                    self.controls[message['task_id']] = lease
            elif lease is not None:
                if lease.pid != os.getpid() or message['task_id'] != lease.task_id:
                    raise RuntimeError('Native inference envelope and worker descriptor disagree')
                with self.lock:
                    if lease.task_id in self.leases:
                        raise RuntimeError(f'Actor received overlapping lease for task {lease.task_id}')
                    self.leases[lease.task_id] = _ActorTask(
                        lease, received, not received and '_detector_handle_table' not in message)
            return message
        except BaseException as exc:
            close_received_handles(received)
            if lease is not None:
                with self.lock:
                    send_receipt(self.stream, lease, 'failed', reason=str(exc))
            raise

    def materialize(self, task, persistent_sources):
        with self.lock:
            entry = self.leases[task['task_id']]
        transient = install_detector_handles(task, persistent_sources, entry.received)
        with self.lock:
            entry.materialized = True
        return transient

    def installed(self, task_id):
        with self.lock:
            entry = self.leases[task_id]
            if not entry.materialized:
                raise RuntimeError('Worker cannot acknowledge uninstalled detector mappings')
            send_receipt(self.stream, entry.lease, 'installed')

    def compute_done(self, task_id):
        with self.lock:
            entry = self.leases[task_id]
            send_receipt(self.stream, entry.lease, 'compute')
            entry.computed = True
            if entry.finalized:
                del self.leases[task_id]

    def put(self, message):
        event = message['type']
        task_id = message.get('task_id')
        if event == 'compute_released':
            self.compute_done(task_id)
            return
        if event != 'result' and not (event == 'fatal' and task_id is not None):
            message = dict(message, worker_pid=os.getpid())
            with self.lock:
                control = self.controls.pop(task_id, None)
                if control is not None:
                    send_receipt(self.stream, control, 'control',
                                 payload=bytes(ForkingPickler.dumps(message)))
                else:
                    self.control_results.put(message)
            return
        try:
            with self.lock:
                lease = self.leases[task_id].lease
            cpu = lease.worker_id >= 2**32
            index = message.get('cpu_index' if cpu else 'gpu_index')
            if (message.get('worker_kind', 'gpu') != ('cpu' if cpu else 'gpu')
                    or type(index) is not int or index != lease.worker_id - (2**32 if cpu else 0)
                    or (event == 'result' and type(message.get('ok')) is not bool)):
                raise RuntimeError('Worker result identity conflicts with its native actor lease')
            payload = bytes(ForkingPickler.dumps(message)) if message.get('ok') else None
        except BaseException as exc:
            with self.lock:
                entry = self.leases.pop(task_id)
                close_received_handles(entry.received)
                send_receipt(self.stream, entry.lease, 'failed', reason=str(exc))
            raise
        with self.lock:
            entry = self.leases[task_id]
            if payload is None:
                close_received_handles(entry.received)
                reason = f'{message.get("error")}\n{message.get("traceback", "")}'
                send_receipt(self.stream, entry.lease, 'failed', reason=reason)
                del self.leases[task_id]
                return
            send_receipt(self.stream, entry.lease, 'final', payload=payload)
            entry.finalized = True
            if entry.computed:
                del self.leases[task_id]
