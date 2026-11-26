"""Bindings and isolated model actor wire protocol for the native TTA scheduler."""
from __future__ import annotations

import importlib
import importlib.util
import math
import os
import socket
import struct
import sys
import threading
from dataclasses import dataclass
from weakref import WeakKeyDictionary


CPU = 1
GPU = 2
_MAGIC = 0x315343415458
_HEADER = struct.Struct("<10Q")
_MAX_PAYLOAD = 64 << 20
_DISPATCH, _CONTROL = 1, 2
_RECEIPTS = {"installed": 3, "compute": 4, "final": 5, "failed": 6, "control": 7}
_NAMES = {number: name for name, number in _RECEIPTS.items()}
_native_module = None
_load_lock = threading.Lock()
_send_locks = WeakKeyDictionary()
_send_locks_guard = threading.Lock()


def _load_native():
    global _native_module
    with _load_lock:
        if _native_module is not None:
            return _native_module
        module_name = "XTA._prepared_scheduler"
        explicit = os.environ.get("XTA_NATIVE_SCHEDULER_BINARY")
        if explicit:
            from .native_scheduler_build import validate_native_scheduler_binary
            path = validate_native_scheduler_binary(explicit)
        else:
            try:
                module = importlib.import_module(module_name)
            except ModuleNotFoundError as error:
                if error.name != module_name:
                    raise
                from .native_scheduler_build import build_native_scheduler
                path = build_native_scheduler()
            else:
                if module.ABI_VERSION != 1:
                    raise RuntimeError("Unsupported native prepared scheduler ABI")
                _native_module = module
                return module
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load native prepared scheduler binary at {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if module.ABI_VERSION != 1:
            raise RuntimeError("Unsupported native prepared scheduler ABI")
        sys.modules[module_name] = module
        _native_module = module
        return module


def require_native_scheduler() -> None:
    """Validate the mandatory compiled backend before preparing a study or opening products."""
    _load_native()


@dataclass(frozen=True)
class Lease:
    run_id: int
    parent_id: int
    task_id: int
    worker_id: int
    epoch: int
    pid: int
    generation: int
    control: bool = False


def _read_exact(channel: socket.socket, length: int) -> bytes:
    data = bytearray(length)
    view = memoryview(data)
    while view:
        received = channel.recv_into(view)
        if received == 0:
            raise EOFError("Native scheduler command stream closed")
        view = view[received:]
    return bytes(data)


def recv_dispatch(channel: socket.socket, *, received_handles=None) -> tuple[Lease | None, bytes]:
    """Receive the next command in an isolated actor. A control can request a reply."""
    from .native_detector_handles import close_received_handles, receive_detector_files
    received = {}
    try:
        header = _read_exact(channel, _HEADER.size)
        magic, wire_kind, run, parent, task, worker, epoch, pid, generation, size = _HEADER.unpack(header)
        kind, files = wire_kind & 255, wire_kind >> 8
        if magic != _MAGIC or kind not in {_DISPATCH, _CONTROL} or size > _MAX_PAYLOAD:
            raise RuntimeError("Invalid native scheduler command header")
        if pid != os.getpid() or not run or not epoch or (kind == _DISPATCH and not generation):
            raise RuntimeError("Native scheduler command addressed to a different actor incarnation")
        if files and (files > _MAX_PAYLOAD or received_handles is None or received_handles or kind != _DISPATCH):
            raise RuntimeError("Unexpected detector file descriptors on actor command")
        if files:
            receive_detector_files(channel, files, _read_exact, received)
        lease = Lease(run, parent, task, worker, epoch, pid, generation, kind == _CONTROL) if generation else None
        payload = _read_exact(channel, size)
        if received:
            received_handles.update(received)
            received.clear()
        return lease, payload
    finally:
        close_received_handles(received)


def send_receipt(channel: socket.socket, lease: Lease, kind: str, *, payload: bytes = b"", reason: str = "") -> None:
    """Return one sender-bound receipt. Deferred completion threads may share the channel."""
    message_type = _RECEIPTS[kind]
    if lease.pid != os.getpid() or lease.control != (kind == "control"):
        raise RuntimeError("Native receipt does not belong to this process or command type")
    if kind == "failed":
        payload = str(reason).encode("utf-8", "replace")[:8192]
    if len(payload) > _MAX_PAYLOAD:
        raise ValueError("Native actor receipt exceeds the transport limit")
    header = _HEADER.pack(_MAGIC, message_type, lease.run_id, lease.parent_id, lease.task_id,
                          lease.worker_id, lease.epoch, os.getpid(), lease.generation, len(payload))
    with _send_locks_guard:
        lock = _send_locks.get(channel)
        if lock is None:
            lock = threading.Lock()
            _send_locks[channel] = lock
    with lock:
        channel.sendall(header)
        if payload:
            channel.sendall(payload)


def _milliseconds(timeout: float | None) -> int:
    if timeout is None:
        return -1
    if not math.isfinite(timeout) or timeout < 0:
        raise ValueError("Native scheduler timeout must be finite and nonnegative")
    return math.ceil(timeout * 1000)


class PreparedWorkScheduler:
    """One native run owner. Waits release the GIL. Model actors use their private socket."""

    def __init__(self, run_id: int, byte_capacity: int, ticket_capacity: int, *,
                 inference_byte_capacity: int | None = None, preparation_window: int | None = None):
        self._native = _load_native()
        self._owner = self._native.create(
            run_id, byte_capacity, ticket_capacity,
            byte_capacity if inference_byte_capacity is None else inference_byte_capacity,
            ticket_capacity if preparation_window is None else preparation_window,
        )

    def configure_capacity(self, byte_capacity, ticket_capacity, inference_byte_capacity, preparation_window,
                           *, pressure_byte_capacity=None):
        self._native.configure(self._owner, byte_capacity, ticket_capacity, inference_byte_capacity, preparation_window,
                               byte_capacity if pressure_byte_capacity is None else pressure_byte_capacity)

    def try_resize_capacity(self, byte_capacity, inference_byte_capacity, *, pressure_byte_capacity=None):
        """Transfer settled dense capacity after detector drain. An unsafe transfer returns False."""
        return self._native.try_resize_capacity(self._owner, byte_capacity, inference_byte_capacity,
                                                byte_capacity if pressure_byte_capacity is None else pressure_byte_capacity)

    def register_worker(self, worker_id, epoch, pid, kind_mask, window, socket, device=-1):
        if socket.gettimeout() is not None:
            raise ValueError("Native scheduler requires a blocking actor endpoint")
        self._native.register_worker(self._owner, worker_id, epoch, pid, kind_mask, window, socket.fileno(), device)

    def declare_parent(self, parent_id, expected_tasks, byte_cost, priority=0, exclusive_worker_kind=0,
                       task_headroom=0, *, allow_oversize_host=False, retirement_pressure_bytes=None,
                       descriptor_only=False, admission_reserve=0):
        return self._native.declare_parent(self._owner, parent_id, expected_tasks, byte_cost, priority,
                                           exclusive_worker_kind, task_headroom, allow_oversize_host,
                                           byte_cost if retirement_pressure_bytes is None else retirement_pressure_bytes,
                                           descriptor_only, admission_reserve)

    def update_host_reserve(self, parent_id, generation, reserve):
        """Refresh the free bytes a pending restore must leave for another preparation lane."""
        return self._native.update_host_reserve(self._owner, parent_id, generation, reserve)

    def wait_preparation(self, timeout=None):
        return self._native.next_parent(self._owner, False, _milliseconds(timeout))

    def wait_publication(self, timeout=None):
        return self._native.next_parent(self._owner, True, _milliseconds(timeout))

    def submit_prepared(self, task_id, parent_id, payload, allowed_kinds, priority=0, required_worker=-1,
                        retained_bytes=0, *, handles=()):
        self._native.submit(self._owner, task_id, parent_id, bytes(payload), allowed_kinds,
                            priority, required_worker, retained_bytes, tuple(handles))

    def wait_task_final(self, task_id, generation, timeout=None):
        return self._native.wait_final(self._owner, task_id, generation, _milliseconds(timeout))

    def submit_planned(self, task_id, parent_id, payload, allowed_kinds, priority=0, required_worker=-1,
                       retained_bytes=0, *, handles=()):
        """Register an immutable command. Dispatch requires complete inventory and actual funding."""
        self._native.submit_planned(self._owner, task_id, parent_id, bytes(payload), allowed_kinds,
                                    priority, required_worker, retained_bytes, tuple(handles))

    def host_prepared(self, parent_id, generation):
        self._native.host_prepared(self._owner, parent_id, generation)

    def adopt_postprocess(self, parent_id, generation):
        self._native.adopt_postprocess(self._owner, parent_id, generation)

    def wait_event(self, timeout=None):
        result = self._native.next_event(self._owner, _milliseconds(timeout))
        if result is None:
            return None
        kind, fields, payload = result
        name = _NAMES[kind]
        return name, Lease(*fields, control=(name == "control")), payload

    def retire_task(self, task_id, generation):
        self._native.retire_task(self._owner, task_id, generation)

    def retire_parent(self, parent_id, generation):
        self._native.retire_parent(self._owner, parent_id, generation)

    def set_parent_remaining_bytes(self, parent_id, generation, remaining_bytes):
        self._native.set_parent_remaining(self._owner, parent_id, generation, remaining_bytes)

    def send_control(self, worker_id, payload, *, expect_reply=False, stop=False):
        return self._native.send_control(self._owner, worker_id, bytes(payload), stop, expect_reply)

    def try_acquire_stage(self, device):
        return self.stage("acquire_exclusive", (device,))[0] or None

    def release_stage(self, device, lease):
        if not self.stage("release_exclusive", (device, lease))[0]:
            raise RuntimeError("Device stage lease generation mismatch")

    def stage(self, operation, fields=()):
        return self._native.stage(self._owner, operation, tuple(fields))

    def wait_stage_change(self, epoch, timeout=None):
        return self._native.wait_stage_change(self._owner, epoch, _milliseconds(timeout))

    def snapshot(self):
        return self._native.snapshot(self._owner)

    def cancel(self, reason):
        self._native.cancel(self._owner, str(reason))

    def close(self):
        self._native.close_run(self._owner)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
